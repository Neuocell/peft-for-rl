"""RL policy-gradient sketches and gradient-subspace LoRA initialization.

The probe uses a zero-output LoRA parameterization with ``A = 0`` and a
random Gaussian ``B``.  For a linear-layer policy gradient ``G`` this gives

``grad_A = scaling * B.T @ G``.

Consequently ``grad_A.T / scaling`` is a randomized right-subspace sketch of
the otherwise dense base-weight gradient. The optimizer never updates A or B;
A stays zero while B is explicitly refreshed between observations, so the
policy remains exactly equal to the base policy while sketches accumulate.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def normalize_target_name(name: str) -> str:
    """Return the stable PEFT rank-pattern suffix for one target module."""

    import re

    match = re.search(r"(model\.layers\.\d+\.(?:self_attn|mlp)\.[^.]+)$", name)
    if match is not None:
        return match.group(1)
    cleaned = name.replace("_fsdp_wrapped_module.", "")
    for prefix in ("base_model.model.", "base_model."):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned


def _stable_seed(base_seed: int, name: str, observation: int) -> int:
    digest = hashlib.sha256(f"{name}:{observation}".encode()).digest()
    offset = int.from_bytes(digest[:8], "little")
    return (int(base_seed) + offset) % (2**63 - 1)


@dataclass(frozen=True)
class GradientProbeInitStats:
    num_layers: int
    rank: int
    a_abs_max: float
    b_rms_mean: float


@torch.no_grad()
def _fill_probe_factors(
    module,
    *,
    name: str,
    adapter_name: str,
    seed: int,
    observation: int,
) -> tuple[int, float]:
    a = module.lora_A[adapter_name].weight
    b = module.lora_B[adapter_name].weight
    rank = a.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, name, observation))
    projection = torch.randn(tuple(b.shape), generator=generator, dtype=torch.float32) / math.sqrt(rank)
    a.zero_()
    b.copy_(projection.to(device=b.device, dtype=b.dtype))
    return rank, float(projection.square().mean().sqrt().item())


@torch.no_grad()
def initialize_gradient_probe(
    model,
    config,
    adapter_name: str = "default",
) -> GradientProbeInitStats:
    """Initialize a function-preserving random gradient probe."""

    from peft.tuners.lora.layer import Linear as LoraLinear

    seed = int(_get(config, "gradient_probe_seed", 42))
    expected_width = int(_get(config, "gradient_probe_width", 0))
    if expected_width <= 0:
        raise ValueError("gradient_probe_width must be positive")

    count = 0
    a_abs_max = 0.0
    b_rms = []
    observed_rank = None
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear) or adapter_name not in module.lora_A:
            continue
        stable_name = normalize_target_name(name)
        rank, rms = _fill_probe_factors(
            module,
            name=stable_name,
            adapter_name=adapter_name,
            seed=seed,
            observation=0,
        )
        if rank != expected_width:
            raise ValueError(f"Probe layer {stable_name} has rank {rank}, expected {expected_width}")
        observed_rank = rank
        count += 1
        a_abs_max = max(a_abs_max, float(module.lora_A[adapter_name].weight.abs().max().item()))
        b_rms.append(rms)

    if not count:
        raise ValueError("Gradient probe found no PEFT LoRA Linear layers")
    return GradientProbeInitStats(
        num_layers=count,
        rank=int(observed_rank),
        a_abs_max=a_abs_max,
        b_rms_mean=sum(b_rms) / len(b_rms),
    )


def _compress_factor(block: torch.Tensor, capacity: int) -> torch.Tensor:
    """Return an orthogonal-times-singular-value factor for ``block block.T``."""

    if block.ndim != 2 or block.shape[1] == 0:
        raise ValueError(f"Expected a non-empty matrix, got {tuple(block.shape)}")
    q, r = torch.linalg.qr(block.float(), mode="reduced")
    u_small, singular_values, _ = torch.linalg.svd(r, full_matrices=False)
    keep = min(capacity, singular_values.numel())
    return (q @ u_small[:, :keep]) * singular_values[:keep].unsqueeze(0)


def _factor_basis_and_values(factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.linalg.vector_norm(factor.float(), dim=0)
    nonzero = values > 1e-12
    if not bool(nonzero.any()):
        return factor[:, :0].float(), values[:0]
    values = values[nonzero]
    basis = factor[:, nonzero].float() / values.unsqueeze(0)
    order = torch.argsort(values, descending=True)
    return basis[:, order], values[order]


def _quantized_energy_rank(
    singular_values: torch.Tensor,
    *,
    total_energy: float,
    target_energy: float,
    rank_bins: tuple[int, ...],
) -> tuple[int, int, float]:
    if singular_values.numel() == 0 or total_energy <= 0:
        return rank_bins[0], 0, 0.0
    cumulative = singular_values.square().cumsum(0)
    target = target_energy * total_energy
    hits = torch.nonzero(cumulative >= target, as_tuple=False)
    raw_rank = int(hits[0].item() + 1) if hits.numel() else int(singular_values.numel() + 1)
    selected = next((rank for rank in rank_bins if rank >= raw_rank), rank_bins[-1])
    used = min(selected, singular_values.numel())
    retained = float(cumulative[used - 1].item() / total_energy) if used else 0.0
    return selected, raw_rank, min(retained, 1.0)


def _principal_overlap(previous: torch.Tensor, current: torch.Tensor) -> float:
    if previous.numel() == 0 or current.numel() == 0:
        return 0.0
    numerator = (previous.T @ current).square().sum().item()
    return float(numerator / min(previous.shape[1], current.shape[1]))


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return float(ordered[index])


class GradientSubspaceAccumulator:
    """Incrementally compress randomized policy-gradient sketches on CPU."""

    def __init__(
        self,
        *,
        capacity: int,
        target_energy: float,
        rank_bins: Iterable[int],
        scaling_ratio: float,
    ) -> None:
        bins = tuple(sorted(set(int(rank) for rank in rank_bins)))
        if capacity <= 0:
            raise ValueError("gradient_probe_capacity must be positive")
        if not 0 < target_energy <= 1:
            raise ValueError("gradient_probe_target_energy must be in (0, 1]")
        if not bins or bins[0] <= 0 or bins[-1] > capacity:
            raise ValueError("gradient_probe_rank_bins must be positive and no larger than capacity")
        if scaling_ratio <= 0:
            raise ValueError("LoRA scaling ratio must be positive")
        if any(
            not math.isclose(rank * scaling_ratio, round(rank * scaling_ratio), rel_tol=0.0, abs_tol=1e-12)
            for rank in bins
        ):
            raise ValueError("gradient_subspace_scaling must produce integral alpha for every rank bin")
        self.capacity = int(capacity)
        self.target_energy = float(target_energy)
        self.rank_bins = bins
        self.scaling_ratio = float(scaling_ratio)
        self.factors: dict[str, torch.Tensor] = {}
        self.pending: dict[str, list[torch.Tensor]] = {}
        self.total_energy: dict[str, float] = {}
        self.observations: dict[str, int] = {}
        self.previous_bases: dict[str, torch.Tensor] = {}
        self.previous_ranks: dict[str, int] = {}

    def add(self, name: str, sketch: torch.Tensor) -> None:
        value = sketch.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim != 2 or not value.shape[1]:
            raise ValueError(f"Invalid gradient sketch for {name}: {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite gradient sketch for {name}")
        self.pending.setdefault(name, []).append(value)
        self.total_energy[name] = self.total_energy.get(name, 0.0) + float(value.square().sum().item())
        self.observations[name] = self.observations.get(name, 0) + 1

    def compress(self) -> None:
        for name, values in self.pending.items():
            blocks = ([self.factors[name]] if name in self.factors else []) + values
            self.factors[name] = _compress_factor(torch.cat(blocks, dim=1), self.capacity)
        self.pending.clear()

    def snapshot(self) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor], dict[str, float]]:
        if self.pending:
            raise RuntimeError("Call compress before requesting a snapshot")
        diagnostics: dict[str, dict[str, Any]] = {}
        selected_bases: dict[str, torch.Tensor] = {}
        overlaps = []
        rank_differences = []
        for name, factor in self.factors.items():
            basis, values = _factor_basis_and_values(factor)
            selected_rank, raw_rank, retained = _quantized_energy_rank(
                values,
                total_energy=self.total_energy[name],
                target_energy=self.target_energy,
                rank_bins=self.rank_bins,
            )
            available = min(selected_rank, basis.shape[1])
            selected_basis = basis[:, :available].contiguous()
            if available < selected_rank:
                # Complete a numerically degenerate sketch deterministically.
                completion = torch.linalg.qr(
                    torch.cat([selected_basis, torch.eye(basis.shape[0], dtype=torch.float32)], dim=1),
                    mode="reduced",
                ).Q[:, :selected_rank]
                selected_basis = completion
            selected_bases[name] = selected_basis
            overlap = None
            if name in self.previous_bases:
                overlap = _principal_overlap(self.previous_bases[name], selected_basis)
                overlaps.append(overlap)
                rank_differences.append(abs(selected_rank - self.previous_ranks[name]))
            diagnostics[name] = {
                "selected_rank": selected_rank,
                "raw_energy_rank": raw_rank,
                "retained_energy": retained,
                "observations": self.observations[name],
                "total_sketch_energy": self.total_energy[name],
                "stability_overlap": overlap,
            }

        ranks = [item["selected_rank"] for item in diagnostics.values()]
        retained = [item["retained_energy"] for item in diagnostics.values()]
        metrics = {
            "module_count": float(len(diagnostics)),
            "rank_mean": float(sum(ranks) / len(ranks)) if ranks else 0.0,
            "rank_min": float(min(ranks)) if ranks else 0.0,
            "rank_max": float(max(ranks)) if ranks else 0.0,
            "retained_energy_mean": float(sum(retained) / len(retained)) if retained else 0.0,
            "retained_energy_min": float(min(retained)) if retained else 0.0,
            "overlap_mean": float(sum(overlaps) / len(overlaps)) if overlaps else 0.0,
            "overlap_p10": _percentile(overlaps, 0.10),
            "overlap_min": float(min(overlaps)) if overlaps else 0.0,
            "rank_mae": float(sum(rank_differences) / len(rank_differences)) if rank_differences else 0.0,
        }
        self.previous_bases = {name: basis.clone() for name, basis in selected_bases.items()}
        self.previous_ranks = {name: item["selected_rank"] for name, item in diagnostics.items()}
        return diagnostics, selected_bases, metrics


def _complete_orthonormal_basis(basis: torch.Tensor, target_rank: int) -> torch.Tensor:
    """Complete a thin basis without materializing a potentially huge identity matrix."""

    if basis.ndim != 2:
        raise ValueError(f"Expected a matrix basis, got {tuple(basis.shape)}")
    dimension = basis.shape[0]
    if not 0 < target_rank <= dimension:
        raise ValueError(f"target_rank must be in [1, {dimension}], got {target_rank}")
    if basis.shape[1] >= target_rank:
        return basis[:, :target_rank].contiguous()

    columns = [basis[:, index].float().clone() for index in range(basis.shape[1])]
    for coordinate in range(dimension):
        if len(columns) >= target_rank:
            break
        candidate = torch.zeros(dimension, dtype=torch.float32)
        candidate[coordinate] = 1.0
        # Two passes keep the deterministic completion stable in float32.
        for _ in range(2):
            for column in columns:
                candidate -= torch.dot(column, candidate) * column
        norm = torch.linalg.vector_norm(candidate)
        if norm > 1e-6:
            columns.append(candidate / norm)
    if len(columns) != target_rank:
        raise RuntimeError(f"Could only construct {len(columns)}/{target_rank} orthonormal directions")
    return torch.stack(columns, dim=1).contiguous()


def _stable_snr_basis(
    signal_factors: list[torch.Tensor],
    noise_factors: list[torch.Tensor],
    *,
    max_rank: int,
    ridge_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    """Solve a small signal/noise generalized eigenproblem in the signal span."""

    if not signal_factors:
        raise ValueError("Stable-SNR basis requires at least one signal factor")
    signal = torch.cat(signal_factors, dim=1).float()
    dimension = signal.shape[0]
    nonzero = torch.linalg.vector_norm(signal, dim=0) > 1e-12
    signal = signal[:, nonzero]
    if signal.shape[1] == 0:
        empty = torch.empty((dimension, 0), dtype=torch.float32)
        return _complete_orthonormal_basis(empty, max_rank), torch.zeros(max_rank), 0.0, 0

    signal_q, _ = torch.linalg.qr(signal, mode="reduced")
    projected_signal = signal_q.T @ signal
    signal_core = projected_signal @ projected_signal.T
    if noise_factors:
        noise = torch.cat(noise_factors, dim=1).float()
        projected_noise = signal_q.T @ noise
        noise_core = projected_noise @ projected_noise.T
    else:
        noise_core = torch.zeros_like(signal_core)

    core_size = signal_core.shape[0]
    signal_scale = float(torch.trace(signal_core).item() / max(core_size, 1))
    noise_scale = float(torch.trace(noise_core).item() / max(core_size, 1))
    reference_scale = max(noise_scale, signal_scale * 1e-3, 1e-12)
    ridge = max(float(ridge_ratio) * reference_scale, 1e-12)
    regularized_noise = noise_core + ridge * torch.eye(core_size, dtype=torch.float32)
    cholesky = torch.linalg.cholesky(regularized_noise)
    whitened = torch.linalg.solve_triangular(cholesky, signal_core, upper=False)
    whitened = torch.linalg.solve_triangular(cholesky, whitened.T, upper=False).T
    whitened = (whitened + whitened.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    generalized = torch.linalg.solve_triangular(cholesky.T, eigenvectors, upper=True)
    directions = signal_q @ generalized
    # QR preserves every ordered prefix span while producing a PEFT-friendly A A^T = I.
    ordered_basis = torch.linalg.qr(directions, mode="reduced").Q
    available = min(max_rank, ordered_basis.shape[1])
    completed = _complete_orthonormal_basis(ordered_basis[:, :available], max_rank)
    padded_values = torch.zeros(max_rank, dtype=torch.float32)
    padded_values[: min(max_rank, eigenvalues.numel())] = eigenvalues[:max_rank]
    relative_threshold = max(float(eigenvalues[0].item()) * 1e-8, 1e-12)
    stable_rank = min(max_rank, int((eigenvalues > relative_threshold).sum().item()))
    return completed, padded_values, ridge, stable_rank


def _capture_by_rank(
    basis: torch.Tensor,
    heldout: torch.Tensor,
    rank_bins: tuple[int, ...],
    *,
    valid_rank: int,
) -> dict[int, float]:
    energy = float(heldout.float().square().sum().item())
    if energy <= 1e-20:
        return {rank: 0.0 for rank in rank_bins}
    if valid_rank <= 0:
        return {rank: 0.0 for rank in rank_bins}
    atom_energy = (basis[:, :valid_rank].T @ heldout.float()).square().sum(dim=1).cumsum(0)
    return {rank: float(atom_energy[min(rank, valid_rank) - 1].item() / energy) for rank in rank_bins}


def _allocate_rank_budget(
    captures: dict[str, dict[int, float]],
    parameter_costs: dict[str, int],
    rank_bins: tuple[int, ...],
    target_mean_rank: float,
    *,
    normalize_by_max_capture: bool = False,
    allocation_groups: dict[str, str] | None = None,
) -> tuple[dict[str, int], int, int]:
    """Greedily allocate ranks under a parameter-matched uniform-rank budget.

    Normalizing each module by its maximum-rank capture prevents modules with a
    larger absolute sketch capture from monopolizing the budget. Optional groups
    reserve an equal-rank budget per module family while retaining heterogeneous
    ranks inside each family.
    """

    if set(captures) != set(parameter_costs):
        raise ValueError("Rank captures and parameter costs must cover the same modules")
    if allocation_groups is not None and set(allocation_groups) != set(captures):
        raise ValueError("Rank allocation groups must cover every module exactly once")

    if allocation_groups is not None:
        ranks: dict[str, int] = {}
        used = 0
        budget = 0
        for group in sorted(set(allocation_groups.values())):
            names = [name for name in captures if allocation_groups[name] == group]
            group_ranks, group_used, group_budget = _allocate_rank_budget(
                {name: captures[name] for name in names},
                {name: parameter_costs[name] for name in names},
                rank_bins,
                target_mean_rank,
                normalize_by_max_capture=normalize_by_max_capture,
            )
            ranks.update(group_ranks)
            used += group_used
            budget += group_budget
        return ranks, used, budget

    utilities = captures
    if normalize_by_max_capture:
        utilities = {}
        maximum_rank = rank_bins[-1]
        for name, values in captures.items():
            denominator = max(float(values[maximum_rank]), 1e-12)
            utilities[name] = {rank: min(max(float(values[rank]) / denominator, 0.0), 1.0) for rank in rank_bins}

    minimum = rank_bins[0]
    budget = int(round(sum(parameter_costs[name] * target_mean_rank for name in captures)))
    used = sum(parameter_costs[name] * minimum for name in captures)
    if used > budget:
        raise ValueError("Target mean rank is below the minimum rank bin")
    indices = {name: 0 for name in captures}
    heap: list[tuple[float, str, int, int]] = []

    def push_next(name: str) -> None:
        current_index = indices[name]
        if current_index + 1 >= len(rank_bins):
            return
        current_rank = rank_bins[current_index]
        next_rank = rank_bins[current_index + 1]
        added_parameters = (next_rank - current_rank) * parameter_costs[name]
        gain = max(0.0, utilities[name][next_rank] - utilities[name][current_rank])
        heapq.heappush(heap, (-gain / max(added_parameters, 1), name, current_index + 1, added_parameters))

    for module_name in sorted(captures):
        push_next(module_name)
    deferred: list[tuple[float, str, int, int]] = []
    while heap:
        item = heapq.heappop(heap)
        _, name, next_index, added_parameters = item
        if next_index != indices[name] + 1:
            continue
        if used + added_parameters > budget:
            deferred.append(item)
            continue
        used += added_parameters
        indices[name] = next_index
        push_next(name)
    ranks = {name: rank_bins[index] for name, index in indices.items()}
    return ranks, used, budget


class StableSNRGradientSubspaceAccumulator:
    """Robust signed-window gradient probe with held-out global rank allocation."""

    def __init__(
        self,
        *,
        window_size: int,
        num_windows: int,
        rank_bins: Iterable[int],
        scaling_ratio: float,
        target_mean_rank: float,
        snr_ridge: float,
        clip_factor: float,
        parameter_costs: dict[str, int],
        calibration_windows: int = 0,
        validation_windows: int = 0,
        normalize_rank_utility: bool = False,
        balance_rank_by_module_type: bool = False,
    ) -> None:
        self.rank_bins = tuple(sorted(set(int(rank) for rank in rank_bins)))
        if window_size < 2:
            raise ValueError("gradient_probe_window_size must be at least 2")
        if num_windows < 3:
            raise ValueError("gradient_probe_num_windows must be at least 3")
        if calibration_windows < 0 or validation_windows < 0:
            raise ValueError("Stable-SNR holdout window counts must be non-negative")
        if bool(calibration_windows) != bool(validation_windows):
            raise ValueError("Independent calibration and validation windows must both be enabled")
        if not self.rank_bins or self.rank_bins[0] <= 0:
            raise ValueError("gradient_probe_rank_bins must be positive")
        if not parameter_costs or any(cost <= 0 for cost in parameter_costs.values()):
            raise ValueError("Stable-SNR probe requires non-empty positive parameter costs")
        if target_mean_rank < self.rank_bins[0] or target_mean_rank > self.rank_bins[-1]:
            raise ValueError("gradient_probe_target_mean_rank must lie inside the rank bins")
        if snr_ridge <= 0:
            raise ValueError("gradient_probe_snr_ridge must be positive")
        if clip_factor < 1:
            raise ValueError("gradient_probe_clip_factor must be at least 1")
        if scaling_ratio <= 0:
            raise ValueError("gradient_subspace_scaling must be positive")
        if any(
            not math.isclose(rank * scaling_ratio, round(rank * scaling_ratio), rel_tol=0.0, abs_tol=1e-12)
            for rank in self.rank_bins
        ):
            raise ValueError("gradient_subspace_scaling must produce integral alpha for every rank bin")
        self.window_size = int(window_size)
        self.num_windows = int(num_windows)
        self.scaling_ratio = float(scaling_ratio)
        self.target_mean_rank = float(target_mean_rank)
        self.snr_ridge = float(snr_ridge)
        self.clip_factor = float(clip_factor)
        self.parameter_costs = dict(parameter_costs)
        self.calibration_windows = int(calibration_windows)
        self.validation_windows = int(validation_windows)
        self.normalize_rank_utility = bool(normalize_rank_utility)
        self.balance_rank_by_module_type = bool(balance_rank_by_module_type)
        self.pending: dict[str, list[torch.Tensor]] = {}
        self.windows: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.observations = 0
        self.batch_clips: dict[str, int] = {}

    @property
    def complete_windows(self) -> int:
        return self.observations // self.window_size

    @property
    def required_windows(self) -> int:
        return self.num_windows + self.calibration_windows + self.validation_windows

    @property
    def ready(self) -> bool:
        return self.complete_windows >= self.required_windows

    def add(self, name: str, sketch: torch.Tensor) -> None:
        value = sketch.detach().to(device="cpu", dtype=torch.float32)
        if value.ndim != 2 or not value.shape[1]:
            raise ValueError(f"Invalid gradient sketch for {name}: {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite gradient sketch for {name}")
        self.pending.setdefault(name, []).append(value)

    def finish_observation(self) -> bool:
        self.observations += 1
        if self.observations % self.window_size:
            return False
        expected_modules = set(self.parameter_costs)
        if set(self.pending) != expected_modules:
            missing = sorted(expected_modules - set(self.pending))
            extra = sorted(set(self.pending) - expected_modules)
            raise RuntimeError(f"Stable-SNR window module mismatch: missing={missing[:5]}, extra={extra[:5]}")
        for name, values in self.pending.items():
            if len(values) != self.window_size:
                raise RuntimeError(f"Stable-SNR window for {name} has {len(values)}/{self.window_size} observations")
            norms = torch.tensor([float(value.square().sum().sqrt().item()) for value in values])
            threshold = self.clip_factor * float(norms.median().item())
            clipped = []
            clipped_count = 0
            for value, norm in zip(values, norms.tolist(), strict=True):
                scale = min(1.0, threshold / max(norm, 1e-12)) if threshold > 0 else 1.0
                clipped.append(value * scale)
                clipped_count += int(scale < 1.0)
            stacked = torch.stack(clipped, dim=0)
            mean = stacked.mean(dim=0)
            signal = mean * math.sqrt(self.window_size)
            residuals = stacked - mean.unsqueeze(0)
            noise = residuals.permute(1, 0, 2).reshape(mean.shape[0], -1) / math.sqrt(self.window_size - 1)
            self.windows.setdefault(name, []).append((signal.contiguous(), noise.contiguous()))
            self.batch_clips[name] = self.batch_clips.get(name, 0) + clipped_count
        self.pending.clear()
        return True

    def _clipped_windows(self, name: str) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        windows = self.windows[name][: self.required_windows]
        norms = torch.tensor([float(signal.square().sum().sqrt().item()) for signal, _ in windows])
        threshold = self.clip_factor * float(norms.median().item())
        signals = []
        noises = []
        clipped = 0
        for (signal, noise), norm in zip(windows, norms.tolist(), strict=True):
            scale = min(1.0, threshold / max(norm, 1e-12)) if threshold > 0 else 1.0
            signals.append(signal * scale)
            noises.append(noise * scale)
            clipped += int(scale < 1.0)
        return signals, noises, clipped

    def snapshot(self) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor], dict[str, float]]:
        if not self.ready:
            raise RuntimeError(f"Stable-SNR probe has {self.complete_windows}/{self.required_windows} complete windows")
        module_state: dict[str, dict[str, Any]] = {}
        full_bases: dict[str, torch.Tensor] = {}
        aggregate_captures: dict[str, dict[int, float]] = {}
        fold_captures_by_module: dict[str, list[dict[int, float]]] = {}
        max_rank = self.rank_bins[-1]

        for name in sorted(self.parameter_costs):
            all_signals, all_noises, window_clips = self._clipped_windows(name)
            signals = all_signals[: self.num_windows]
            noises = all_noises[: self.num_windows]
            calibration_signals = all_signals[self.num_windows : self.num_windows + self.calibration_windows]
            validation_signals = all_signals[self.num_windows + self.calibration_windows :]
            full_basis, eigenvalues, ridge, stable_rank = _stable_snr_basis(
                signals, noises, max_rank=max_rank, ridge_ratio=self.snr_ridge
            )
            fold_captures = []
            fold_overlaps = []
            for heldout_index in range(self.num_windows):
                train_signals = [value for index, value in enumerate(signals) if index != heldout_index]
                train_noises = [value for index, value in enumerate(noises) if index != heldout_index]
                fold_basis, _, _, fold_stable_rank = _stable_snr_basis(
                    train_signals, train_noises, max_rank=max_rank, ridge_ratio=self.snr_ridge
                )
                fold_captures.append(
                    _capture_by_rank(
                        fold_basis,
                        signals[heldout_index],
                        self.rank_bins,
                        valid_rank=fold_stable_rank,
                    )
                )
                overlap_rank = min(stable_rank, fold_stable_rank)
                fold_overlaps.append(
                    _principal_overlap(full_basis[:, :overlap_rank], fold_basis[:, :overlap_rank])
                    if overlap_rank
                    else 0.0
                )
            loo_captures = {
                rank: float(torch.tensor([fold[rank] for fold in fold_captures]).median().item())
                for rank in self.rank_bins
            }
            calibration_captures_by_window = [
                _capture_by_rank(
                    full_basis,
                    heldout,
                    self.rank_bins,
                    valid_rank=stable_rank,
                )
                for heldout in calibration_signals
            ]
            captures = (
                {
                    rank: float(torch.tensor([fold[rank] for fold in calibration_captures_by_window]).median().item())
                    for rank in self.rank_bins
                }
                if calibration_captures_by_window
                else loo_captures
            )
            validation_captures_by_window = [
                _capture_by_rank(
                    full_basis,
                    heldout,
                    self.rank_bins,
                    valid_rank=stable_rank,
                )
                for heldout in validation_signals
            ]
            aggregate_captures[name] = captures
            fold_captures_by_module[name] = calibration_captures_by_window or fold_captures
            full_bases[name] = full_basis
            module_state[name] = {
                "heldout_capture_by_rank": {str(rank): captures[rank] for rank in self.rank_bins},
                "loo_discovery_capture_by_rank": {str(rank): loo_captures[rank] for rank in self.rank_bins},
                "heldout_capture_mean_at_max_rank": float(
                    sum(fold[self.rank_bins[-1]] for fold in (calibration_captures_by_window or fold_captures))
                    / len(calibration_captures_by_window or fold_captures)
                ),
                "validation_capture_by_rank": {
                    str(rank): float(
                        sum(fold[rank] for fold in validation_captures_by_window) / len(validation_captures_by_window)
                    )
                    for rank in self.rank_bins
                }
                if validation_captures_by_window
                else {},
                "fold_overlap_mean": float(sum(fold_overlaps) / len(fold_overlaps)),
                "signal_energy": float(sum(value.square().sum().item() for value in signals)),
                "noise_energy": float(sum(value.square().sum().item() for value in noises)),
                "snr_eigenvalue_top": float(eigenvalues[0].item()),
                "snr_eigenvalue_at_max_rank": float(eigenvalues[-1].item()),
                "stable_numerical_rank": stable_rank,
                "ridge": ridge,
                "batch_clips": self.batch_clips.get(name, 0),
                "window_clips": window_clips,
                "observations": self.required_windows * self.window_size,
            }

        allocation_groups = (
            {name: name.rsplit(".", 1)[-1] for name in aggregate_captures} if self.balance_rank_by_module_type else None
        )
        selected_ranks, used_parameters, parameter_budget = _allocate_rank_budget(
            aggregate_captures,
            self.parameter_costs,
            self.rank_bins,
            self.target_mean_rank,
            normalize_by_max_capture=self.normalize_rank_utility,
            allocation_groups=allocation_groups,
        )
        fold_rank_maps = []
        rank_stability_folds = len(next(iter(fold_captures_by_module.values())))
        for fold_index in range(rank_stability_folds):
            fold_captures = {name: fold_captures_by_module[name][fold_index] for name in fold_captures_by_module}
            fold_rank_map, _, _ = _allocate_rank_budget(
                fold_captures,
                self.parameter_costs,
                self.rank_bins,
                self.target_mean_rank,
                normalize_by_max_capture=self.normalize_rank_utility,
                allocation_groups=allocation_groups,
            )
            fold_rank_maps.append(fold_rank_map)

        selected_bases = {}
        rank_deviations = []
        for name, rank in selected_ranks.items():
            selected_bases[name] = full_bases[name][:, :rank].contiguous()
            fold_ranks = [rank_map[name] for rank_map in fold_rank_maps]
            rank_deviations.extend(abs(value - rank) for value in fold_ranks)
            module_state[name].update(
                {
                    "selected_rank": rank,
                    "heldout_capture_selected": aggregate_captures[name][rank],
                    "fold_selected_ranks": fold_ranks,
                    "parameter_cost_per_rank": self.parameter_costs[name],
                    "rank_allocation_group": allocation_groups[name] if allocation_groups else "global",
                }
            )
            validation_by_rank = module_state[name]["validation_capture_by_rank"]
            if validation_by_rank:
                selected_validation = float(validation_by_rank[str(rank)])
                max_validation = float(validation_by_rank[str(self.rank_bins[-1])])
                module_state[name]["validation_capture_selected"] = selected_validation
                module_state[name]["validation_capture_at_max_rank"] = max_validation
                module_state[name]["validation_retention_of_max"] = selected_validation / max(max_validation, 1e-12)

        ranks = list(selected_ranks.values())
        captures = [module_state[name]["heldout_capture_selected"] for name in module_state]
        fold_overlaps = [module_state[name]["fold_overlap_mean"] for name in module_state]
        validation_selected = [
            module_state[name]["validation_capture_selected"]
            for name in module_state
            if "validation_capture_selected" in module_state[name]
        ]
        validation_max = [
            module_state[name]["validation_capture_at_max_rank"]
            for name in module_state
            if "validation_capture_at_max_rank" in module_state[name]
        ]
        validation_retention = [
            module_state[name]["validation_retention_of_max"]
            for name in module_state
            if "validation_retention_of_max" in module_state[name]
        ]
        metrics = {
            "module_count": float(len(module_state)),
            "rank_mean": float(sum(ranks) / len(ranks)),
            "rank_min": float(min(ranks)),
            "rank_max": float(max(ranks)),
            "heldout_capture_mean": float(sum(captures) / len(captures)),
            "heldout_capture_min": float(min(captures)),
            "fold_overlap_mean": float(sum(fold_overlaps) / len(fold_overlaps)),
            "fold_overlap_p10": _percentile(fold_overlaps, 0.10),
            "rank_loo_mae": float(sum(rank_deviations) / len(rank_deviations)),
            "parameter_budget_used_ratio": float(used_parameters / parameter_budget),
            "equivalent_uniform_rank": float(used_parameters / sum(self.parameter_costs.values())),
            "target_equivalent_uniform_rank": self.target_mean_rank,
            "complete_windows": float(self.complete_windows),
            "observations": float(self.observations),
            "discovery_windows": float(self.num_windows),
            "calibration_windows": float(self.calibration_windows),
            "validation_windows": float(self.validation_windows),
            "validation_capture_selected_mean": (
                float(sum(validation_selected) / len(validation_selected)) if validation_selected else 0.0
            ),
            "validation_capture_selected_min": float(min(validation_selected)) if validation_selected else 0.0,
            "validation_capture_at_max_rank_mean": (
                float(sum(validation_max) / len(validation_max)) if validation_max else 0.0
            ),
            "validation_retention_of_max_mean": (
                float(sum(validation_retention) / len(validation_retention)) if validation_retention else 0.0
            ),
            "validation_retention_of_max_min": (float(min(validation_retention)) if validation_retention else 0.0),
        }
        return module_state, selected_bases, metrics


def _reconstruct_window_observations(
    signal: torch.Tensor,
    noise: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Reconstruct the clipped observations encoded by one signal/noise window."""

    if noise.shape[1] % window_size:
        raise ValueError(f"Noise width {noise.shape[1]} is not divisible by window_size={window_size}")
    probe_width = noise.shape[1] // window_size
    residuals = noise.reshape(signal.shape[0], window_size, probe_width).permute(1, 0, 2)
    residuals = residuals * math.sqrt(window_size - 1)
    mean = signal / math.sqrt(window_size)
    return residuals + mean.unsqueeze(0)


def _dominant_window_atoms(
    signal: torch.Tensor,
    noise: torch.Tensor,
    *,
    window_size: int,
    local_atoms: int,
    gap_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Extract locally dominant right-gradient atoms and robust scalar weights."""

    u, singular_values, _ = torch.linalg.svd(signal.float(), full_matrices=False)
    keep = min(local_atoms, singular_values.numel())
    atoms = u[:, :keep].contiguous()
    values = singular_values[:keep]
    total_energy = max(float(singular_values.square().sum().item()), 1e-20)
    observations = _reconstruct_window_observations(signal.float(), noise.float(), window_size)

    weights = []
    dominance_values = []
    agreement_values = []
    gap_values = []
    for index in range(keep):
        atom = atoms[:, index]
        dominance = float(values[index].square().item() / total_energy)
        captures = []
        for observation in observations:
            energy = max(float(observation.square().sum().item()), 1e-20)
            captures.append(float((atom @ observation).square().sum().item() / energy))
        agreement = float(torch.tensor(captures).median().item())
        if index + 1 < singular_values.numel():
            gap = float(values[index].item() / max(float(singular_values[index + 1].item()), 1e-12))
            gap = min(max(gap, 1.0), gap_cap)
        else:
            gap = 1.0
        dominance_values.append(dominance)
        agreement_values.append(agreement)
        gap_values.append(gap)
        weights.append(dominance * agreement * gap)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if float(weight_tensor.sum().item()) <= 1e-20:
        weight_tensor = torch.tensor(dominance_values, dtype=torch.float32)
    diagnostics = {
        "singular_values": [float(value) for value in singular_values.tolist()],
        "dominance": dominance_values,
        "agreement": agreement_values,
        "spectral_gap": gap_values,
        "weights": [float(value) for value in weight_tensor.tolist()],
    }
    return atoms, weight_tensor, diagnostics


def _dominant_consensus_basis(
    signals: list[torch.Tensor],
    noises: list[torch.Tensor],
    *,
    window_size: int,
    local_atoms: int,
    gap_cap: float,
    max_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], list[dict[str, Any]]]:
    """Order a basis by recurrent atoms, then complete it from the pooled sketch span."""

    if not signals or len(signals) != len(noises):
        raise ValueError("Dominant-atom consensus requires matching non-empty signal/noise windows")
    raw_window_atoms = []
    window_metrics = []
    for signal, noise in zip(signals, noises, strict=True):
        atoms, weights, diagnostics = _dominant_window_atoms(
            signal,
            noise,
            window_size=window_size,
            local_atoms=local_atoms,
            gap_cap=gap_cap,
        )
        raw_window_atoms.append((atoms, weights))
        window_metrics.append(diagnostics)

    window_atoms = []
    weighted_atoms = []
    for window_index, (atoms, weights) in enumerate(raw_window_atoms):
        supports = []
        for atom_index in range(atoms.shape[1]):
            atom = atoms[:, atom_index]
            cross_window = []
            for other_index, (other_atoms, _) in enumerate(raw_window_atoms):
                if other_index == window_index:
                    continue
                cross_window.append(float((atom @ other_atoms).square().max().item()))
            supports.append(sum(cross_window) / len(cross_window))
        support_tensor = torch.tensor(supports, dtype=torch.float32)
        # Squared support makes recurrence, rather than one-window magnitude,
        # the primary criterion while retaining a small floor for novel atoms.
        consensus_weights = weights * support_tensor.clamp_min(0.05).square()
        window_metrics[window_index]["recurrence_support"] = supports
        window_metrics[window_index]["consensus_weights"] = [float(value) for value in consensus_weights.tolist()]
        window_atoms.append((atoms, consensus_weights))
        positive = consensus_weights > 1e-20
        if bool(positive.any()):
            weighted_atoms.append(atoms[:, positive] * consensus_weights[positive].sqrt().unsqueeze(0))
    if not weighted_atoms:
        raise RuntimeError("Dominant-atom probe found no finite positive-weight atoms")

    consensus_u, consensus_s, _ = torch.linalg.svd(torch.cat(weighted_atoms, dim=1), full_matrices=False)
    numerical = consensus_s > max(float(consensus_s[0].item()) * 1e-7, 1e-10)
    recurrence_rank = min(max_rank, int(numerical.sum().item()))
    recurrent = consensus_u[:, :recurrence_rank].contiguous()

    pooled = torch.cat(signals, dim=1).float()
    if recurrent.shape[1]:
        pooled = pooled - recurrent @ (recurrent.T @ pooled)
    residual_basis, _ = _factor_basis_and_values(_compress_factor(pooled, max_rank))
    needed = max_rank - recurrent.shape[1]
    if residual_basis.shape[1] < needed:
        available = recurrent.shape[1] + residual_basis.shape[1]
        message = f"Dominant discovery span only supplies {available}/{max_rank} directions"
        raise RuntimeError(message)
    combined = torch.cat([recurrent, residual_basis[:, :needed]], dim=1)
    basis = torch.linalg.qr(combined, mode="reduced").Q[:, :max_rank].contiguous()
    recurrence_values = torch.zeros(max_rank, dtype=torch.float32)
    recurrence_values[: min(max_rank, consensus_s.numel())] = consensus_s[:max_rank].square()
    return basis, recurrence_values, window_atoms, window_metrics


def _dominant_capture_by_rank(
    basis: torch.Tensor,
    signal: torch.Tensor,
    rank_bins: tuple[int, ...],
    *,
    local_atoms: int,
) -> dict[int, float]:
    """Measure held-out top-atom energy captured by each ordered basis prefix."""

    u, singular_values, _ = torch.linalg.svd(signal.float(), full_matrices=False)
    keep = min(local_atoms, singular_values.numel())
    atoms = u[:, :keep]
    atom_energy = singular_values[:keep].square()
    total = max(float(atom_energy.sum().item()), 1e-20)
    projections = (basis.T @ atoms).square()
    cumulative = projections.cumsum(dim=0)
    return {
        rank: float((cumulative[min(rank, basis.shape[1]) - 1] * atom_energy).sum().item() / total)
        for rank in rank_bins
    }


class DominantAtomGradientSubspaceAccumulator(StableSNRGradientSubspaceAccumulator):
    """Find recurrent local gradient atoms instead of fitting full sketch energy."""

    def __init__(
        self,
        *,
        window_size: int,
        num_windows: int,
        rank_bins: Iterable[int],
        scaling_ratio: float,
        target_mean_rank: float,
        clip_factor: float,
        parameter_costs: dict[str, int],
        local_atoms: int = 2,
        gap_cap: float = 4.0,
        calibration_windows: int = 1,
        validation_windows: int = 1,
        normalize_rank_utility: bool = True,
        balance_rank_by_module_type: bool = False,
    ) -> None:
        if local_atoms <= 0:
            raise ValueError("gradient_probe_local_atoms must be positive")
        if gap_cap < 1:
            raise ValueError("gradient_probe_gap_cap must be at least 1")
        super().__init__(
            window_size=window_size,
            num_windows=num_windows,
            rank_bins=rank_bins,
            scaling_ratio=scaling_ratio,
            target_mean_rank=target_mean_rank,
            snr_ridge=1.0,
            clip_factor=clip_factor,
            parameter_costs=parameter_costs,
            calibration_windows=calibration_windows,
            validation_windows=validation_windows,
            normalize_rank_utility=normalize_rank_utility,
            balance_rank_by_module_type=balance_rank_by_module_type,
        )
        self.local_atoms = int(local_atoms)
        self.gap_cap = float(gap_cap)
        self._window_sketch_tensors: dict[str, torch.Tensor] = {}
        self._window_atom_tensors: dict[str, torch.Tensor] = {}
        self._window_metrics: dict[str, list[dict[str, Any]]] = {}

    def snapshot(self) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor], dict[str, float]]:
        if not self.ready:
            raise RuntimeError(
                f"Dominant-atom probe has {self.complete_windows}/{self.required_windows} complete windows"
            )
        module_state: dict[str, dict[str, Any]] = {}
        full_bases: dict[str, torch.Tensor] = {}
        aggregate_captures: dict[str, dict[int, float]] = {}
        fold_captures_by_module: dict[str, list[dict[int, float]]] = {}
        max_rank = self.rank_bins[-1]
        self._window_sketch_tensors.clear()
        self._window_atom_tensors.clear()
        self._window_metrics.clear()

        for name in sorted(self.parameter_costs):
            all_signals, all_noises, window_clips = self._clipped_windows(name)
            signals = all_signals[: self.num_windows]
            noises = all_noises[: self.num_windows]
            calibration_signals = all_signals[self.num_windows : self.num_windows + self.calibration_windows]
            validation_signals = all_signals[self.num_windows + self.calibration_windows :]
            full_basis, recurrence_values, window_atoms, window_metrics = _dominant_consensus_basis(
                signals,
                noises,
                window_size=self.window_size,
                local_atoms=self.local_atoms,
                gap_cap=self.gap_cap,
                max_rank=max_rank,
            )

            fold_captures = []
            fold_overlaps = []
            for heldout_index in range(self.num_windows):
                train_signals = [value for index, value in enumerate(signals) if index != heldout_index]
                train_noises = [value for index, value in enumerate(noises) if index != heldout_index]
                fold_basis, _, _, _ = _dominant_consensus_basis(
                    train_signals,
                    train_noises,
                    window_size=self.window_size,
                    local_atoms=self.local_atoms,
                    gap_cap=self.gap_cap,
                    max_rank=max_rank,
                )
                fold_captures.append(
                    _dominant_capture_by_rank(
                        fold_basis,
                        signals[heldout_index],
                        self.rank_bins,
                        local_atoms=self.local_atoms,
                    )
                )
                fold_overlaps.append(_principal_overlap(full_basis, fold_basis))

            loo_captures = {
                rank: float(torch.tensor([fold[rank] for fold in fold_captures]).median().item())
                for rank in self.rank_bins
            }
            calibration_captures = [
                _dominant_capture_by_rank(
                    full_basis,
                    heldout,
                    self.rank_bins,
                    local_atoms=self.local_atoms,
                )
                for heldout in calibration_signals
            ]
            captures = (
                {
                    rank: float(torch.tensor([fold[rank] for fold in calibration_captures]).median().item())
                    for rank in self.rank_bins
                }
                if calibration_captures
                else loo_captures
            )
            validation_captures = [
                _dominant_capture_by_rank(
                    full_basis,
                    heldout,
                    self.rank_bins,
                    local_atoms=self.local_atoms,
                )
                for heldout in validation_signals
            ]

            aggregate_captures[name] = captures
            fold_captures_by_module[name] = fold_captures
            full_bases[name] = full_basis
            recurrence_total = max(float(recurrence_values.sum().item()), 1e-20)
            recurrence_distribution = recurrence_values / recurrence_total
            recurrence_rank = int((recurrence_values > max(float(recurrence_values[0].item()) * 1e-7, 1e-10)).sum())
            eigengap = (
                float(recurrence_values[0].item() / max(float(recurrence_values[1].item()), 1e-12))
                if recurrence_values.numel() > 1
                else 1.0
            )
            module_state[name] = {
                "heldout_dominant_capture_by_rank": {str(rank): captures[rank] for rank in self.rank_bins},
                "loo_dominant_capture_by_rank": {str(rank): loo_captures[rank] for rank in self.rank_bins},
                "validation_dominant_capture_by_rank": {
                    str(rank): float(sum(fold[rank] for fold in validation_captures) / len(validation_captures))
                    for rank in self.rank_bins
                }
                if validation_captures
                else {},
                "fold_overlap_mean": float(sum(fold_overlaps) / len(fold_overlaps)),
                "recurrence_rank": recurrence_rank,
                "recurrence_top1_share": float(recurrence_distribution[0].item()),
                "recurrence_top2_share": float(recurrence_distribution[:2].sum().item()),
                "recurrence_eigengap_top1": eigengap,
                "signal_energy": float(sum(value.square().sum().item() for value in signals)),
                "noise_energy": float(sum(value.square().sum().item() for value in noises)),
                "batch_clips": self.batch_clips.get(name, 0),
                "window_clips": window_clips,
                "observations": self.required_windows * self.window_size,
            }

            artifact_metrics = []
            for index, (signal, noise) in enumerate(zip(all_signals, all_noises, strict=True)):
                if index < self.num_windows:
                    atoms, weights = window_atoms[index]
                    item_metrics = dict(window_metrics[index])
                    role = "discovery"
                else:
                    atoms, weights, item_metrics = _dominant_window_atoms(
                        signal,
                        noise,
                        window_size=self.window_size,
                        local_atoms=self.local_atoms,
                        gap_cap=self.gap_cap,
                    )
                    role = "calibration" if index < self.num_windows + self.calibration_windows else "validation"
                item_metrics["role"] = role
                artifact_metrics.append(item_metrics)
                key = f"{name}.window_{index:02d}"
                observations = _reconstruct_window_observations(signal, noise, self.window_size)
                self._window_sketch_tensors[key] = (
                    observations.permute(1, 0, 2).reshape(signal.shape[0], -1).contiguous()
                )
                self._window_atom_tensors[f"{key}.atoms"] = atoms.contiguous()
                self._window_atom_tensors[f"{key}.weights"] = weights.contiguous()
            self._window_metrics[name] = artifact_metrics

        allocation_groups = (
            {name: name.rsplit(".", 1)[-1] for name in aggregate_captures} if self.balance_rank_by_module_type else None
        )
        selected_ranks, used_parameters, parameter_budget = _allocate_rank_budget(
            aggregate_captures,
            self.parameter_costs,
            self.rank_bins,
            self.target_mean_rank,
            normalize_by_max_capture=self.normalize_rank_utility,
            allocation_groups=allocation_groups,
        )
        fold_rank_maps = []
        rank_stability_folds = len(next(iter(fold_captures_by_module.values())))
        for fold_index in range(rank_stability_folds):
            fold_map, _, _ = _allocate_rank_budget(
                {name: values[fold_index] for name, values in fold_captures_by_module.items()},
                self.parameter_costs,
                self.rank_bins,
                self.target_mean_rank,
                normalize_by_max_capture=self.normalize_rank_utility,
                allocation_groups=allocation_groups,
            )
            fold_rank_maps.append(fold_map)

        selected_bases = {}
        rank_deviations = []
        for name, rank in selected_ranks.items():
            selected_bases[name] = full_bases[name][:, :rank].contiguous()
            fold_ranks = [rank_map[name] for rank_map in fold_rank_maps]
            rank_deviations.extend(abs(value - rank) for value in fold_ranks)
            module_state[name].update(
                {
                    "selected_rank": rank,
                    "heldout_dominant_capture_selected": aggregate_captures[name][rank],
                    "fold_selected_ranks": fold_ranks,
                    "parameter_cost_per_rank": self.parameter_costs[name],
                    "rank_allocation_group": allocation_groups[name] if allocation_groups else "global",
                }
            )
            validation_by_rank = module_state[name]["validation_dominant_capture_by_rank"]
            if validation_by_rank:
                module_state[name]["validation_dominant_capture_selected"] = float(validation_by_rank[str(rank)])
                module_state[name]["validation_dominant_capture_at_max_rank"] = float(
                    validation_by_rank[str(self.rank_bins[-1])]
                )

        ranks = list(selected_ranks.values())
        selected_capture = [module_state[name]["heldout_dominant_capture_selected"] for name in module_state]
        validation_selected = [
            item["validation_dominant_capture_selected"]
            for item in module_state.values()
            if "validation_dominant_capture_selected" in item
        ]
        recurrence_top1 = [item["recurrence_top1_share"] for item in module_state.values()]
        recurrence_top2 = [item["recurrence_top2_share"] for item in module_state.values()]
        metrics = {
            "module_count": float(len(module_state)),
            "rank_mean": float(sum(ranks) / len(ranks)),
            "rank_min": float(min(ranks)),
            "rank_max": float(max(ranks)),
            "heldout_dominant_capture_mean": float(sum(selected_capture) / len(selected_capture)),
            "heldout_dominant_capture_min": float(min(selected_capture)),
            "validation_dominant_capture_mean": (
                float(sum(validation_selected) / len(validation_selected)) if validation_selected else 0.0
            ),
            "validation_dominant_capture_min": float(min(validation_selected)) if validation_selected else 0.0,
            "recurrence_top1_share_mean": float(sum(recurrence_top1) / len(recurrence_top1)),
            "recurrence_top2_share_mean": float(sum(recurrence_top2) / len(recurrence_top2)),
            "fold_overlap_mean": float(
                sum(item["fold_overlap_mean"] for item in module_state.values()) / len(module_state)
            ),
            "rank_loo_mae": float(sum(rank_deviations) / len(rank_deviations)),
            "parameter_budget_used_ratio": float(used_parameters / parameter_budget),
            "equivalent_uniform_rank": float(used_parameters / sum(self.parameter_costs.values())),
            "target_equivalent_uniform_rank": self.target_mean_rank,
            "complete_windows": float(self.complete_windows),
            "observations": float(self.observations),
            "discovery_windows": float(self.num_windows),
            "calibration_windows": float(self.calibration_windows),
            "validation_windows": float(self.validation_windows),
        }
        return module_state, selected_bases, metrics

    def artifact_tensors(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self._window_sketch_tensors, self._window_atom_tensors

    def window_metrics(self) -> dict[str, list[dict[str, Any]]]:
        return self._window_metrics


def _leaf_fsdp_modules(module: torch.nn.Module) -> list[FSDP]:
    units = []
    for candidate in module.modules():
        if not isinstance(candidate, FSDP):
            continue
        nested = any(isinstance(child, FSDP) for child in list(candidate.modules())[1:])
        if not nested:
            units.append(candidate)
    return units


class GradientProbeCollector:
    """Extract LoRA-A gradient sketches at the FSDP optimizer boundary."""

    def __init__(self, model: torch.nn.Module, config: Any, adapter_name: str = "default") -> None:
        from peft.tuners.lora.layer import Linear as LoraLinear

        self.model = model
        self.config = config
        self.adapter_name = adapter_name
        self.seed = int(_get(config, "gradient_probe_seed", 42))
        self.output_dir = Path(str(_get(config, "gradient_probe_output_dir", ""))).expanduser().resolve()
        if not str(_get(config, "gradient_probe_output_dir", "")):
            raise ValueError("gradient_probe_output_dir is required for peft_type=grad_probe")
        self.min_steps = int(_get(config, "gradient_probe_min_steps", 6))
        self.max_steps = int(_get(config, "gradient_probe_max_steps", 12))
        self.patience = int(_get(config, "gradient_probe_stability_patience", 3))
        self.overlap_threshold = float(_get(config, "gradient_probe_overlap_threshold", 0.98))
        self.rank_tolerance = float(_get(config, "gradient_probe_rank_tolerance", 1.0))
        self.probe_method = str(_get(config, "gradient_probe_method", "energy")).strip().lower()
        if self.probe_method not in {"energy", "stable_snr", "dominant_atoms"}:
            raise ValueError(f"Unsupported gradient_probe_method={self.probe_method!r}")
        if self.min_steps <= 0 or self.max_steps < self.min_steps:
            raise ValueError("gradient probe requires 0 < min_steps <= max_steps")
        if self.patience <= 0:
            raise ValueError("gradient_probe_stability_patience must be positive")
        if not 0 <= self.overlap_threshold <= 1:
            raise ValueError("gradient_probe_overlap_threshold must be in [0, 1]")
        if self.rank_tolerance < 0:
            raise ValueError("gradient_probe_rank_tolerance must be non-negative")
        rank_bins = tuple(int(value) for value in _get(config, "gradient_probe_rank_bins", [8, 12, 16, 20, 24, 28, 32]))
        capacity = int(_get(config, "gradient_probe_capacity", 64))
        target_energy = float(_get(config, "gradient_probe_target_energy", 0.95))
        scaling_ratio = float(_get(config, "gradient_subspace_scaling", 2.0))
        self.module_names = {
            id(candidate): normalize_target_name(name)
            for name, candidate in model.named_modules()
            if isinstance(candidate, LoraLinear) and adapter_name in candidate.lora_A
        }
        if not self.module_names:
            raise ValueError("Gradient probe collector found no PEFT LoRA Linear layers")
        parameter_costs = {
            self.module_names[id(candidate)]: int(candidate.in_features + candidate.out_features)
            for candidate in model.modules()
            if isinstance(candidate, LoraLinear) and id(candidate) in self.module_names
        }
        if self.probe_method in {"stable_snr", "dominant_atoms"}:
            window_size = int(_get(config, "gradient_probe_window_size", 3))
            num_windows = int(_get(config, "gradient_probe_num_windows", 5))
            probe_width = int(_get(config, "gradient_probe_width", 8))
            if (num_windows - 1) * probe_width < max(rank_bins):
                raise ValueError(
                    "Windowed probe leave-one-window-out requires "
                    "(gradient_probe_num_windows - 1) * gradient_probe_width >= max(rank_bins)"
                )
            common = {
                "window_size": window_size,
                "num_windows": num_windows,
                "rank_bins": rank_bins,
                "scaling_ratio": scaling_ratio,
                "target_mean_rank": float(_get(config, "gradient_probe_target_mean_rank", 16.0)),
                "clip_factor": float(_get(config, "gradient_probe_clip_factor", 2.5)),
                "parameter_costs": parameter_costs,
                "calibration_windows": int(_get(config, "gradient_probe_calibration_windows", 0)),
                "validation_windows": int(_get(config, "gradient_probe_validation_windows", 0)),
                "normalize_rank_utility": bool(_get(config, "gradient_probe_normalize_rank_utility", False)),
                "balance_rank_by_module_type": bool(_get(config, "gradient_probe_balance_rank_by_module_type", False)),
            }
            if self.probe_method == "dominant_atoms":
                self.accumulator = DominantAtomGradientSubspaceAccumulator(
                    **common,
                    local_atoms=int(_get(config, "gradient_probe_local_atoms", 2)),
                    gap_cap=float(_get(config, "gradient_probe_gap_cap", 4.0)),
                )
            else:
                self.accumulator = StableSNRGradientSubspaceAccumulator(
                    **common,
                    snr_ridge=float(_get(config, "gradient_probe_snr_ridge", 0.05)),
                )
        else:
            self.accumulator = GradientSubspaceAccumulator(
                capacity=capacity,
                target_energy=target_energy,
                rank_bins=rank_bins,
                scaling_ratio=scaling_ratio,
            )
        self.optimizer_updates = 0
        self.stable_windows = 0
        self.ready = False
        self.last_diagnostics: dict[str, dict[str, Any]] = {}
        self.last_bases: dict[str, torch.Tensor] = {}
        self.last_snapshot_metrics: dict[str, float] = {}

    def _units_and_contexts(self):
        if isinstance(self.model, FSDP):
            units = _leaf_fsdp_modules(self.model)
            if not units:
                units = [self.model]
            return [(unit, FSDP.summon_full_params(unit, writeback=True, with_grads=True)) for unit in units]
        return [(self.model, nullcontext())]

    @torch.no_grad()
    def capture_and_refresh(self) -> dict[str, float]:
        from peft.tuners.lora.layer import Linear as LoraLinear

        self.optimizer_updates += 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        found_names: set[str] = set()
        skipped = 0
        refresh_projection = self.probe_method == "energy" or (
            self.optimizer_updates % self.accumulator.window_size == 0
        )
        for unit, context in self._units_and_contexts():
            with context:
                for candidate in unit.modules():
                    if not isinstance(candidate, LoraLinear) or id(candidate) not in self.module_names:
                        continue
                    a = candidate.lora_A[self.adapter_name].weight
                    if a.grad is None or a.grad.shape != a.shape:
                        continue
                    name = self.module_names[id(candidate)]
                    if name in found_names:
                        raise RuntimeError(f"Gradient probe encountered module {name} more than once")
                    found_names.add(name)
                    scaling = float(candidate.scaling[self.adapter_name])
                    sketch = a.grad.detach().float().T / scaling
                    if bool(torch.isfinite(sketch).all()):
                        if rank == 0:
                            self.accumulator.add(name, sketch)
                    else:
                        skipped += 1
                    if refresh_projection:
                        _fill_probe_factors(
                            candidate,
                            name=name,
                            adapter_name=self.adapter_name,
                            seed=self.seed,
                            observation=self.optimizer_updates,
                        )
        if len(found_names) != len(self.module_names):
            missing = sorted(set(self.module_names.values()) - found_names)
            raise RuntimeError(
                f"Gradient probe captured {len(found_names)}/{len(self.module_names)} LoRA gradients; "
                f"missing examples={missing[:5]}. The current FSDP wrapping is unsupported"
            )
        if skipped:
            raise FloatingPointError(f"Gradient probe found non-finite sketches in {skipped} modules")
        if rank == 0 and self.probe_method in {"stable_snr", "dominant_atoms"}:
            self.accumulator.finish_observation()
        return {
            "gradient_probe/optimizer_updates": float(self.optimizer_updates),
            "gradient_probe/nonfinite_modules": float(skipped),
            "gradient_probe/complete_windows": float(
                self.optimizer_updates // self.accumulator.window_size
                if self.probe_method in {"stable_snr", "dominant_atoms"}
                else 0
            ),
        }

    def finish_trainer_step(self, trainer_step: int) -> dict[str, float]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        metrics: dict[str, float] = {}
        if rank == 0:
            if self.probe_method in {"stable_snr", "dominant_atoms"}:
                self.ready = self.accumulator.ready
                metrics = {
                    "gradient_probe/trainer_step": float(trainer_step),
                    "gradient_probe/artifact_ready": float(self.ready),
                    "gradient_probe/module_coverage": float(len(self.accumulator.windows) / len(self.module_names)),
                    "gradient_probe/complete_windows": float(self.accumulator.complete_windows),
                    "gradient_probe/target_windows": float(self.accumulator.required_windows),
                    "gradient_probe/pending_observations": float(
                        self.accumulator.observations % self.accumulator.window_size
                    ),
                }
                if self.ready:
                    diagnostics, bases, snapshot = self.accumulator.snapshot()
                    self.last_diagnostics = diagnostics
                    self.last_bases = bases
                    self.last_snapshot_metrics = snapshot
                    metrics.update({f"gradient_probe/{key}": value for key, value in snapshot.items()})
                    self._export(trainer_step, stopped_by_stability=False)
            else:
                self.accumulator.compress()
                diagnostics, bases, snapshot = self.accumulator.snapshot()
                self.last_diagnostics = diagnostics
                self.last_bases = bases
                self.last_snapshot_metrics = snapshot
                complete = len(diagnostics) == len(self.module_names)
                stable = (
                    trainer_step >= self.min_steps
                    and complete
                    and snapshot["overlap_p10"] >= self.overlap_threshold
                    and snapshot["rank_mae"] <= self.rank_tolerance
                )
                self.stable_windows = self.stable_windows + 1 if stable else 0
                self.ready = complete and (self.stable_windows >= self.patience or trainer_step >= self.max_steps)
                metrics = {
                    "gradient_probe/trainer_step": float(trainer_step),
                    "gradient_probe/stable_window": float(stable),
                    "gradient_probe/stable_windows": float(self.stable_windows),
                    "gradient_probe/artifact_ready": float(self.ready),
                    "gradient_probe/module_coverage": float(len(diagnostics) / len(self.module_names)),
                    **{f"gradient_probe/{key}": value for key, value in snapshot.items()},
                }
                if self.ready:
                    self._export(trainer_step, stopped_by_stability=self.stable_windows >= self.patience)
        if dist.is_initialized():
            payload = [metrics]
            dist.broadcast_object_list(payload, src=0)
            metrics = payload[0]
        return metrics

    def _export(self, trainer_step: int, *, stopped_by_stability: bool) -> None:
        from safetensors.torch import save_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: basis[:, : self.last_diagnostics[name]["selected_rank"]].T.contiguous()
            for name, basis in self.last_bases.items()
        }
        tensor_path = self.output_dir / "subspaces.safetensors"
        temporary_tensor_path = tensor_path.with_suffix(".safetensors.tmp")
        save_file(tensors, str(temporary_tensor_path))
        os.replace(temporary_tensor_path, tensor_path)

        rank_pattern = {name: item["selected_rank"] for name, item in self.last_diagnostics.items()}
        alpha_pattern = {name: int(round(rank * self.accumulator.scaling_ratio)) for name, rank in rank_pattern.items()}
        rank_map = {
            "schema_version": 3
            if self.probe_method == "dominant_atoms"
            else 2
            if self.probe_method == "stable_snr"
            else 1,
            "method": {
                "dominant_atoms": "rl_policy_gradient_dominant_atom_subspace",
                "stable_snr": "rl_policy_gradient_stable_snr_subspace",
                "energy": "rl_policy_gradient_subspace",
            }[self.probe_method],
            "rank_pattern": rank_pattern,
            "alpha_pattern": alpha_pattern,
            "constant_scaling": self.accumulator.scaling_ratio,
            "subspace_path": str(tensor_path),
        }
        summary = {
            **rank_map,
            "probe_method": self.probe_method,
            "trainer_step": trainer_step,
            "optimizer_updates": self.optimizer_updates,
            "stopped_by_stability": stopped_by_stability,
            "stable_windows": self.stable_windows,
            "target_energy": getattr(self.accumulator, "target_energy", None),
            "rank_bins": list(self.accumulator.rank_bins),
            "capacity": getattr(self.accumulator, "capacity", self.accumulator.rank_bins[-1]),
            "metrics": self.last_snapshot_metrics,
            "modules": self.last_diagnostics,
        }
        if self.probe_method == "stable_snr":
            summary["stable_snr"] = {
                "window_size": self.accumulator.window_size,
                "num_windows": self.accumulator.num_windows,
                "calibration_windows": self.accumulator.calibration_windows,
                "validation_windows": self.accumulator.validation_windows,
                "target_mean_rank": self.accumulator.target_mean_rank,
                "snr_ridge": self.accumulator.snr_ridge,
                "clip_factor": self.accumulator.clip_factor,
                "normalize_rank_utility": self.accumulator.normalize_rank_utility,
                "balance_rank_by_module_type": self.accumulator.balance_rank_by_module_type,
            }
        if self.probe_method == "dominant_atoms":
            summary["dominant_atoms"] = {
                "window_size": self.accumulator.window_size,
                "num_windows": self.accumulator.num_windows,
                "calibration_windows": self.accumulator.calibration_windows,
                "validation_windows": self.accumulator.validation_windows,
                "target_mean_rank": self.accumulator.target_mean_rank,
                "clip_factor": self.accumulator.clip_factor,
                "local_atoms": self.accumulator.local_atoms,
                "gap_cap": self.accumulator.gap_cap,
                "normalize_rank_utility": self.accumulator.normalize_rank_utility,
                "balance_rank_by_module_type": self.accumulator.balance_rank_by_module_type,
            }
            sketches, atoms = self.accumulator.artifact_tensors()
            for filename, tensors_to_save in (
                ("window_sketches.safetensors", sketches),
                ("window_atoms.safetensors", atoms),
            ):
                artifact_path = self.output_dir / filename
                temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
                save_file(tensors_to_save, str(temporary))
                os.replace(temporary, artifact_path)
            metrics_path = self.output_dir / "window_metrics.json"
            temporary_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
            temporary_metrics.write_text(
                json.dumps(self.accumulator.window_metrics(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_metrics, metrics_path)
        for filename, payload in (("rank_map.json", rank_map), ("summary.json", summary)):
            path = self.output_dir / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, path)


@dataclass(frozen=True)
class GradientSubspaceInitStats:
    num_layers: int
    rank_mean: float
    rank_min: int
    rank_max: int
    orthogonality_error_max: float
    b_abs_max: float


@dataclass(frozen=True)
class FixedALoRAStats:
    num_layers: int
    a_parameters: int
    b_parameters: int
    unexpected_trainable_parameters: int


def freeze_lora_a_factors(model, adapter_name: str = "default") -> FixedALoRAStats:
    """Freeze PEFT LoRA-A factors and strictly validate a B-only actor."""

    from peft.tuners.lora.layer import Linear as LoraLinear

    a_parameter_ids = set()
    b_parameter_ids = set()
    a_parameters = 0
    b_parameters = 0
    layers = 0
    for module in model.modules():
        if not isinstance(module, LoraLinear) or adapter_name not in module.lora_A:
            continue
        a = module.lora_A[adapter_name].weight
        b = module.lora_B[adapter_name].weight
        a.requires_grad_(False)
        if not b.requires_grad:
            raise RuntimeError("Fixed-A LoRA expected every lora_B factor to remain trainable")
        a_parameter_ids.add(id(a))
        b_parameter_ids.add(id(b))
        a_parameters += a.numel()
        b_parameters += b.numel()
        layers += 1
    if not layers:
        raise ValueError("Fixed-A LoRA found no PEFT LoRA Linear layers")

    unexpected = 0
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in b_parameter_ids:
            unexpected += parameter.numel()
    if unexpected:
        raise RuntimeError(f"Fixed-A B-only actor has {unexpected} unexpected trainable parameters")
    if any(parameter.requires_grad for parameter in model.parameters() if id(parameter) in a_parameter_ids):
        raise RuntimeError("Fixed-A LoRA failed to freeze every lora_A factor")
    return FixedALoRAStats(
        num_layers=layers,
        a_parameters=a_parameters,
        b_parameters=b_parameters,
        unexpected_trainable_parameters=unexpected,
    )


@torch.no_grad()
def apply_gradient_subspace_initialization(
    model,
    config,
    adapter_name: str = "default",
) -> GradientSubspaceInitStats:
    """Load gradient-derived orthonormal A factors and reset B to zero."""

    from peft.tuners.lora.layer import Linear as LoraLinear
    from safetensors.torch import load_file

    configured_path = str(_get(config, "gradient_subspace_path", ""))
    if not configured_path:
        raise ValueError("gradient_subspace_path is required for peft_type=grad_subspace")
    path = Path(configured_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Gradient subspace artifact does not exist: {path}")
    tensors = load_file(str(path), device="cpu")
    used = set()
    ranks = []
    orthogonality_error_max = 0.0
    b_abs_max = 0.0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear) or adapter_name not in module.lora_A:
            continue
        stable_name = normalize_target_name(name)
        if stable_name not in tensors:
            raise ValueError(f"Gradient subspace artifact is missing module {stable_name}")
        basis = tensors[stable_name].float()
        a = module.lora_A[adapter_name].weight
        b = module.lora_B[adapter_name].weight
        if basis.shape != a.shape:
            raise ValueError(
                f"Gradient subspace shape mismatch for {stable_name}: artifact={tuple(basis.shape)}, "
                f"LoRA A={tuple(a.shape)}"
            )
        a.copy_(basis.to(device=a.device, dtype=a.dtype))
        b.zero_()
        gram = basis @ basis.T
        error = (gram - torch.eye(basis.shape[0])).abs().max().item()
        orthogonality_error_max = max(orthogonality_error_max, float(error))
        b_abs_max = max(b_abs_max, float(b.abs().max().item()))
        ranks.append(basis.shape[0])
        used.add(stable_name)
    unused = set(tensors) - used
    if unused:
        raise ValueError(f"Gradient subspace artifact has {len(unused)} unmatched modules")
    if not ranks:
        raise ValueError("Gradient subspace initialization found no PEFT LoRA Linear layers")
    return GradientSubspaceInitStats(
        num_layers=len(ranks),
        rank_mean=sum(ranks) / len(ranks),
        rank_min=min(ranks),
        rank_max=max(ranks),
        orthogonality_error_max=orthogonality_error_max,
        b_abs_max=b_abs_max,
    )
