"""Full-weight signed-GRPO probe for static LoRA subspace discovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.peft_gradient_subspace import (
    _get,
    _leaf_fsdp_modules,
    normalize_target_name,
)


_ALL_LINEAR_FAMILIES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
_CANDIDATE_METHODS = ("mean", "covariance", "hybrid")
_COVARIANCE_ESTIMATORS = (
    "centered_population_covariance",
    "uncentered_second_moment",
)
_TOKEN_MASK_SCOPES = ("discovery", "discovery_calibration", "all")
_TOKEN_MASK_MODES = (
    "none",
    "random",
    "top_surprisal",
    "advantage_entropy_stable_band",
)
_WINDOWED_CANDIDATE_METHODS = (
    "raw_momentum",
    "adam_update",
    "consensus_hybrid",
)


def probe_token_keep_count(
    valid_tokens: int,
    *,
    keep_ratio: float,
    min_keep: int,
    final_tokens: int,
) -> int:
    """Return the exact number of valid response tokens retained by a probe mask."""

    if valid_tokens < 0:
        raise ValueError("valid_tokens must be non-negative")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("full_gradient_probe_token_keep_ratio must be in (0, 1]")
    if min_keep < 0 or final_tokens < 0:
        raise ValueError("probe token minimums must be non-negative")
    requested = max(math.ceil(valid_tokens * keep_ratio), min_keep, final_tokens)
    return min(valid_tokens, requested)


def build_probe_token_mask(
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    mode: str,
    keep_ratio: float,
    min_keep: int,
    final_tokens: int,
    entropy: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    seed: int = 0,
    sample_key: str = "",
    surprisal_upper_quantile: float = 0.95,
) -> torch.Tensor:
    """Build a detached, deterministic token mask for probe-gradient discovery.

    All modes keep an exact per-response budget. ``top_surprisal`` reserves the
    final tokens and fills the remainder by descending sampled-token surprisal.
    ``random`` draws a deterministic density-matched control. The stable-band
    selector ranks by ``abs(advantage) * entropy`` after excluding the extreme
    sampled-surprisal tail whenever the requested budget permits it.
    """

    if log_prob.shape != response_mask.shape:
        raise ValueError("log_prob and response_mask must have identical shapes")
    if log_prob.ndim != 2:
        raise ValueError("probe token masking expects [batch, response_length] tensors")
    normalized_mode = str(mode).lower()
    if normalized_mode not in _TOKEN_MASK_MODES:
        raise ValueError(f"Unknown full-gradient probe token mask mode: {mode}")
    if normalized_mode == "none":
        return response_mask.detach().clone()

    output = torch.zeros_like(response_mask).detach()
    if not 0.0 < surprisal_upper_quantile <= 1.0:
        raise ValueError("surprisal_upper_quantile must be in (0, 1]")
    if normalized_mode == "advantage_entropy_stable_band":
        if entropy is None or advantages is None:
            raise ValueError(
                "advantage_entropy_stable_band requires entropy and advantages"
            )
        if entropy.shape != log_prob.shape or advantages.shape != log_prob.shape:
            raise ValueError("entropy and advantages must match log_prob shape")

    detached_surprisal = -log_prob.detach().float()
    valid = response_mask.detach() > 0
    for row in range(valid.shape[0]):
        valid_indices = torch.nonzero(valid[row], as_tuple=False).flatten()
        keep_count = probe_token_keep_count(
            int(valid_indices.numel()),
            keep_ratio=keep_ratio,
            min_keep=min_keep,
            final_tokens=final_tokens,
        )
        if keep_count == 0:
            continue
        if normalized_mode == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_seed(seed, f"{sample_key}:{row}", "random_token_mask")
            )
            order = torch.randperm(int(valid_indices.numel()), generator=generator)
            selected_indices = valid_indices.index_select(0, order[:keep_count].to(valid_indices.device))
            output[row, selected_indices] = response_mask[row, selected_indices].detach()
            continue
        if normalized_mode == "advantage_entropy_stable_band":
            row_surprisal = detached_surprisal[row, valid_indices]
            cutoff = torch.quantile(row_surprisal, surprisal_upper_quantile)
            stable = valid_indices[row_surprisal <= cutoff]
            if stable.numel() < keep_count:
                stable = valid_indices
            score = (
                advantages.detach().float()[row, stable].abs()
                * entropy.detach().float()[row, stable]
            )
            order = torch.argsort(score, descending=True, stable=True)
            selected_indices = stable[order[:keep_count]]
            output[row, selected_indices] = response_mask[row, selected_indices].detach()
            continue
        suffix_count = min(final_tokens, keep_count)
        suffix_indices = (
            valid_indices[-suffix_count:]
            if suffix_count
            else valid_indices.new_empty((0,))
        )
        remaining_budget = keep_count - suffix_count
        prefix_indices = (
            valid_indices[:-suffix_count] if suffix_count else valid_indices
        )
        if remaining_budget:
            order = torch.argsort(
                detached_surprisal[row, prefix_indices],
                descending=True,
                stable=True,
            )
            selected_indices = torch.cat(
                (prefix_indices[order[:remaining_budget]], suffix_indices)
            )
        else:
            selected_indices = suffix_indices
        output[row, selected_indices] = response_mask[row, selected_indices].detach()
    return output


def deterministic_rollout_halves(
    seed: int,
    prompt_id: str,
    response_count: int,
    split_count: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return reproducible, balanced rollout half-splits for one prompt."""

    if response_count < 2 or response_count % 2:
        raise ValueError("cross-fit requires a positive even response count")
    if split_count <= 0:
        raise ValueError("split_count must be positive")
    midpoint = response_count // 2
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(splits) < split_count:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(seed + attempts, prompt_id, "crossfit_rollout_split")
        )
        first = tuple(sorted(torch.randperm(response_count, generator=generator)[:midpoint].tolist()))
        canonical = min(first, tuple(index for index in range(response_count) if index not in first))
        attempts += 1
        if canonical in seen:
            if attempts > split_count * 32:
                raise ValueError(
                    f"Requested {split_count} unique half-splits from only {response_count} responses"
                )
            continue
        seen.add(canonical)
        second = tuple(index for index in range(response_count) if index not in first)
        splits.append((first, second))
    return splits


def principal_subspace_overlap(
    first: torch.Tensor, second: torch.Tensor
) -> dict[str, float]:
    """Summarize canonical-angle overlap for two row-orthonormal bases."""

    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("principal overlap expects compatible rank-by-width bases")
    singular_values = torch.linalg.svdvals(first.float() @ second.float().T).clamp(0, 1)
    if not singular_values.numel():
        return {"mean_cosine": 0.0, "mean_squared_cosine": 0.0, "min_cosine": 0.0}
    return {
        "mean_cosine": float(singular_values.mean().item()),
        "mean_squared_cosine": float(singular_values.square().mean().item()),
        "min_cosine": float(singular_values.min().item()),
    }


def covariance_effective_ranks(values: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    """Return stable and entropy ranks for a non-negative covariance spectrum."""

    spectrum = values.detach().float().clamp_min(0)
    total = spectrum.sum()
    if float(total.item()) <= eps:
        return {"stable_rank": 0.0, "entropy_rank": 0.0, "positive_rank": 0.0}
    probabilities = spectrum / total
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return {
        "stable_rank": float((total / spectrum.max().clamp_min(eps)).item()),
        "entropy_rank": float(entropy.exp().item()),
        "positive_rank": float((spectrum > eps * spectrum.max()).sum().item()),
    }


def probe_token_mask_active(*, mode: str, scope: str, phase: str) -> bool:
    """Return whether token masking is active for one probe phase."""

    if mode == "none":
        return False
    if scope not in _TOKEN_MASK_SCOPES:
        raise ValueError(f"Unknown full-gradient token mask scope: {scope}")
    if phase not in {"discovery", "calibration", "audit"}:
        raise ValueError(f"Unknown full-gradient probe phase: {phase}")
    if scope == "all":
        return True
    if scope == "discovery_calibration":
        return phase in {"discovery", "calibration"}
    return phase == "discovery"


def covariance_sketch(
    second_moment_y: torch.Tensor,
    mean: torch.Tensor,
    omega: torch.Tensor,
    count: int,
    *,
    estimator: str,
) -> torch.Tensor:
    """Build ``C Omega`` for centered covariance or an uncentered second moment."""

    if count <= 0:
        raise ValueError("covariance sketch requires at least one observation")
    normalized = str(estimator).lower()
    if normalized not in _COVARIANCE_ESTIMATORS:
        raise ValueError(f"Unknown full-gradient covariance estimator: {estimator}")
    result = second_moment_y / count
    if normalized == "centered_population_covariance":
        result = result - mean.T @ (mean @ omega)
    return result


def deterministic_response_offset(
    seed: int, prompt_id: str, window_index: int, response_count: int
) -> int:
    """Choose one rollout reproducibly without depending on process hash state."""

    if response_count <= 0:
        raise ValueError("response_count must be positive")
    digest = hashlib.sha256(
        f"{seed}:{window_index}:{prompt_id}:single_response".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % response_count


def unbiased_single_response_scale(response_count: int, window_prompts: int) -> float:
    """Scale one uniformly sampled response into a prompt-mean window estimate."""

    if response_count <= 0 or window_prompts <= 0:
        raise ValueError("response_count and window_prompts must be positive")
    return float(response_count) / float(window_prompts)


def virtual_adam_update(
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    gradient: torch.Tensor,
    *,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    """Update full-weight moments in place and return a bias-corrected direction."""

    if step <= 0:
        raise ValueError("Virtual Adam step must be positive")
    first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    corrected_first = first_moment / (1.0 - beta1**step)
    corrected_second = second_moment / (1.0 - beta2**step)
    return corrected_first / (corrected_second.sqrt() + eps)


def _stable_seed(seed: int, name: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{name}:{purpose}".encode()).digest()
    return (int(seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def _target_families(target_modules: Any) -> frozenset[str]:
    if target_modules is None or target_modules == "all-linear":
        return _ALL_LINEAR_FAMILIES
    if isinstance(target_modules, str):
        values = [item.strip() for item in target_modules.split(",") if item.strip()]
    else:
        values = [str(item) for item in target_modules]
    unknown = set(values) - _ALL_LINEAR_FAMILIES
    if unknown:
        raise ValueError(
            f"Full-gradient probe only supports transformer linear families; unknown={sorted(unknown)}"
        )
    return frozenset(values)


def configure_full_gradient_probe_parameters(
    model: torch.nn.Module, config: Any
) -> dict[str, float]:
    """Freeze the model except target base-weight matrices used by the probe."""

    families = _target_families(_get(config, "target_modules", "all-linear"))
    model.requires_grad_(False)
    modules = 0
    parameters = 0
    for name, module in model.named_modules():
        if (
            not isinstance(module, torch.nn.Linear)
            or name.rsplit(".", 1)[-1] not in families
        ):
            continue
        module.weight.requires_grad_(True)
        modules += 1
        parameters += module.weight.numel()
    if not modules:
        raise ValueError("Full-gradient probe found no target Linear modules")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return {
        "num_layers": float(modules),
        "trainable_weight_parameters": float(parameters),
    }


def _orthonormal_completion(
    rows: torch.Tensor, rank: int, *, seed: int
) -> torch.Tensor:
    """Return exactly ``rank`` deterministic orthonormal rows."""

    width = rows.shape[1]
    if rank > width:
        raise ValueError(f"Cannot construct {rank} orthogonal rows in width {width}")
    kept = rows[:rank].float()
    if kept.numel():
        kept = torch.linalg.qr(kept.T, mode="reduced").Q.T
    missing = rank - kept.shape[0]
    if missing:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_rows = torch.randn(
            (width, rank + missing), generator=generator, dtype=torch.float32
        )
        if kept.numel():
            random_rows = random_rows - kept.T @ (kept @ random_rows)
        completion = torch.linalg.qr(random_rows, mode="reduced").Q[:, :missing].T
        kept = torch.cat((kept, completion), dim=0)
    return kept.contiguous()


def positive_spectrum_rank(
    values: torch.Tensor,
    *,
    relative_threshold: float = 1e-7,
    eps: float = 1e-12,
) -> int:
    """Count numerically supported directions in a non-negative spectrum."""

    if values.ndim != 1:
        raise ValueError("spectrum must be one-dimensional")
    if relative_threshold < 0 or eps < 0:
        raise ValueError("spectrum thresholds must be non-negative")
    spectrum = values.detach().float().clamp_min(0)
    if not spectrum.numel():
        return 0
    threshold = max(float(spectrum.max().item()) * relative_threshold, eps)
    return int((spectrum > threshold).sum().item())


def stability_supported_hybrid_basis(
    stable_basis: torch.Tensor,
    stable_values: torch.Tensor,
    fallback_basis: torch.Tensor,
    fallback_values: torch.Tensor,
    rank: int,
    *,
    stable_directions: int,
    seed: int,
    relative_threshold: float = 1e-7,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Keep supported stable rows and fill the remaining rank from fallback rows.

    Candidate rows are accepted in order with re-orthogonalization. Zero-spectrum
    completion rows in ``stable_basis`` are therefore never treated as signal.
    The fallback is expected to be ordered by discovery importance.
    """

    if rank <= 0 or stable_directions < 0:
        raise ValueError("rank must be positive and stable_directions non-negative")
    if stable_basis.ndim != 2 or fallback_basis.ndim != 2:
        raise ValueError("hybrid inputs must be rank-by-width matrices")
    if stable_basis.shape[1] != fallback_basis.shape[1]:
        raise ValueError("hybrid inputs must have the same width")
    if stable_values.ndim != 1 or fallback_values.ndim != 1:
        raise ValueError("hybrid spectra must be one-dimensional")
    if stable_values.numel() < stable_basis.shape[0]:
        raise ValueError("stable spectrum is shorter than its basis")
    if fallback_values.numel() < fallback_basis.shape[0]:
        raise ValueError("fallback spectrum is shorter than its basis")
    if rank > stable_basis.shape[1]:
        raise ValueError("hybrid rank cannot exceed basis width")

    supported = min(
        stable_directions,
        positive_spectrum_rank(
            stable_values,
            relative_threshold=relative_threshold,
            eps=eps,
        ),
        stable_basis.shape[0],
        rank,
    )
    accepted: list[torch.Tensor] = []
    source_values: list[torch.Tensor] = []
    selected_stable = 0

    def append_rows(
        rows: torch.Tensor,
        values: torch.Tensor,
        limit: int,
        *,
        stable: bool,
    ) -> None:
        nonlocal selected_stable
        for index in range(limit):
            if len(accepted) == rank:
                return
            row = rows[index].float()
            if accepted:
                current = torch.stack(accepted)
                row = row - current.T @ (current @ row)
                row = row - current.T @ (current @ row)
            norm = row.norm()
            if float(norm.item()) <= eps:
                continue
            accepted.append(row / norm)
            source_values.append(values[index].detach().float().clamp_min(0))
            if stable:
                selected_stable += 1

    append_rows(stable_basis, stable_values, supported, stable=True)
    append_rows(fallback_basis, fallback_values, fallback_basis.shape[0], stable=False)

    rows = (
        torch.stack(accepted)
        if accepted
        else torch.empty((0, stable_basis.shape[1]), dtype=torch.float32)
    )
    if rows.shape[0] < rank:
        rows = _orthonormal_completion(rows, rank, seed=seed)
        source_values.extend(
            torch.tensor(0.0, dtype=torch.float32)
            for _ in range(rank - len(source_values))
        )
    values = torch.stack(source_values[:rank]).to(dtype=torch.float32)
    return rows[:rank].contiguous(), values.contiguous(), selected_stable


def signal_random_hybrid_basis(
    signal_basis: torch.Tensor,
    rank: int,
    *,
    signal_directions: int,
    seed: int,
    eps: float = 1e-8,
    orthogonality_atol: float = 3e-4,
) -> torch.Tensor:
    """Preserve an ordered signal prefix and fill its orthogonal complement randomly.

    Rows represent LoRA A input directions.  The random draw always has the same
    ``rank x width`` shape, so comparisons at different signal counts share the
    same underlying random matrix before residualization.
    """

    if signal_basis.ndim != 2:
        raise ValueError("signal_basis must be a rank-by-width matrix")
    width = int(signal_basis.shape[1])
    if not 0 < rank <= width:
        raise ValueError(f"rank must be in [1, {width}]")
    if not 0 <= signal_directions <= min(rank, signal_basis.shape[0]):
        raise ValueError("signal_directions exceeds the available signal basis")
    signal = signal_basis[:signal_directions].detach().float().cpu().contiguous()
    if signal.numel():
        gram_error = float(
            (signal @ signal.T - torch.eye(signal_directions)).abs().max().item()
        )
        if gram_error > orthogonality_atol:
            raise ValueError(
                f"Signal prefix is not orthonormal: {gram_error} > {orthogonality_atol}"
            )
    if signal_directions == rank:
        return signal.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_rows = torch.randn((rank, width), generator=generator, dtype=torch.float32)
    accepted = [signal[index].clone() for index in range(signal_directions)]
    for candidate in random_rows:
        if len(accepted) == rank:
            break
        row = candidate.clone()
        if accepted:
            current = torch.stack(accepted)
            row -= current.T @ (current @ row)
            row -= current.T @ (current @ row)
        norm = torch.linalg.vector_norm(row)
        if float(norm.item()) > eps:
            accepted.append(row / norm)
    if len(accepted) < rank:
        for coordinate in range(width):
            if len(accepted) == rank:
                break
            row = torch.zeros(width, dtype=torch.float32)
            row[coordinate] = 1.0
            current = torch.stack(accepted)
            row -= current.T @ (current @ row)
            row -= current.T @ (current @ row)
            norm = torch.linalg.vector_norm(row)
            if float(norm.item()) > eps:
                accepted.append(row / norm)
    if len(accepted) != rank:
        raise RuntimeError(f"Could only construct {len(accepted)}/{rank} hybrid rows")
    result = torch.stack(accepted).contiguous()
    error = float((result @ result.T - torch.eye(rank)).abs().max().item())
    if error > orthogonality_atol:
        raise RuntimeError(
            f"Signal/random hybrid is not orthonormal: {error} > {orthogonality_atol}"
        )
    return result


def _right_singular_basis(
    matrix: torch.Tensor,
    rank: int,
    *,
    oversample: int,
    niter: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute a deterministic randomized truncated right SVD."""

    matrix = matrix.float().to(device)
    out_features, in_features = matrix.shape
    q = min(max(rank + oversample, rank), out_features, in_features)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    omega = torch.randn((in_features, q), generator=generator, dtype=torch.float32).to(
        device
    )
    y = matrix @ omega
    for _ in range(max(niter, 0)):
        y = torch.linalg.qr(y, mode="reduced").Q
        y = matrix @ (matrix.T @ y)
    q_left = torch.linalg.qr(y, mode="reduced").Q
    small = q_left.T @ matrix
    _, singular_values, vh = torch.linalg.svd(small, full_matrices=False)
    available = min(rank, vh.shape[0])
    basis = _orthonormal_completion(vh[:available].cpu(), rank, seed=seed + 1)
    values = torch.zeros(rank, dtype=torch.float32)
    values[:available] = singular_values[:available].cpu()
    denominator = max(
        float(matrix.square().sum().item()), torch.finfo(torch.float32).tiny
    )
    captured = float(values.square().sum().item()) / denominator
    return basis, values, captured


def _nystrom_covariance_basis(
    y: torch.Tensor,
    omega: torch.Tensor,
    rank: int,
    *,
    seed: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Recover a right eigenspace from the one-pass sketch ``Y=C Omega``."""

    y = y.float()
    omega = omega.float()
    w = (omega.T @ y + y.T @ omega) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(w)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order)
    eigenvectors = eigenvectors.index_select(1, order)
    threshold = max(
        float(eigenvalues[0].item()) * 1e-7 if eigenvalues.numel() else 0.0, eps
    )
    positive = eigenvalues > threshold
    if bool(positive.any()):
        whitener = eigenvectors[:, positive] * eigenvalues[positive].rsqrt().unsqueeze(
            0
        )
        factor = y @ whitener
        u, singular_values, _ = torch.linalg.svd(factor, full_matrices=False)
        available = min(rank, u.shape[1])
        raw_basis = u[:, :available].T
        values = torch.zeros(rank, dtype=torch.float32)
        values[:available] = singular_values[:available].square()
    else:
        raw_basis = torch.empty((0, y.shape[0]), dtype=torch.float32)
        values = torch.zeros(rank, dtype=torch.float32)
    basis = _orthonormal_completion(raw_basis, rank, seed=seed)
    residual = float((y - basis.T @ (basis @ y)).norm().item()) / max(
        float(y.norm().item()), eps
    )
    return basis, values, residual


def _hybrid_basis(
    mean_basis: torch.Tensor,
    mean_values: torch.Tensor,
    covariance_basis: torch.Tensor,
    covariance_values: torch.Tensor,
    rank: int,
    *,
    seed: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_weight = mean_values.square()
    covariance_weight = covariance_values.clamp_min(0)
    mean_weight = mean_weight / (mean_weight.sum() + eps)
    covariance_weight = covariance_weight / (covariance_weight.sum() + eps)
    pooled = torch.cat(
        (
            mean_basis * mean_weight.sqrt().unsqueeze(1),
            covariance_basis * covariance_weight.sqrt().unsqueeze(1),
        ),
        dim=0,
    )
    _, singular_values, vh = torch.linalg.svd(pooled, full_matrices=False)
    basis = _orthonormal_completion(vh[:rank], rank, seed=seed)
    values = torch.zeros(rank, dtype=torch.float32)
    available = min(rank, singular_values.numel())
    values[:available] = singular_values[:available].square()
    return basis, values


class FullGradientRLProbeCollector:
    """Collect prompt-group full gradients and cross-fit static LoRA atoms."""

    def __init__(self, model: torch.nn.Module, config: Any) -> None:
        self.model = model
        self.config = config
        self.rank = int(_get(config, "full_gradient_probe_rank", 32))
        self.sketch_width = int(
            _get(config, "full_gradient_probe_sketch_width", self.rank + 8)
        )
        self.oversample = int(_get(config, "full_gradient_probe_svd_oversample", 8))
        self.svd_niter = int(_get(config, "full_gradient_probe_svd_niter", 1))
        configured_svd_device = str(
            _get(config, "full_gradient_probe_svd_device", "auto")
        ).lower()
        if configured_svd_device == "auto":
            configured_svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        if configured_svd_device not in {"cpu", "cuda"}:
            raise ValueError("full_gradient_probe_svd_device must be auto, cpu or cuda")
        self.svd_device = torch.device(configured_svd_device)
        self.discovery_target = int(
            _get(config, "full_gradient_probe_discovery_prompts", 16)
        )
        self.calibration_target = int(
            _get(config, "full_gradient_probe_calibration_prompts", 16)
        )
        self.audit_target = int(_get(config, "full_gradient_probe_audit_prompts", 8))
        self.clip_factor = float(_get(config, "full_gradient_probe_clip_factor", 2.5))
        self.confidence_z = float(_get(config, "full_gradient_probe_confidence_z", 1.0))
        self.seed = int(_get(config, "gradient_probe_seed", 42))
        self.eps = float(_get(config, "full_gradient_probe_eps", 1e-12))
        self.covariance_estimator = str(
            _get(
                config,
                "full_gradient_probe_covariance_estimator",
                "centered_population_covariance",
            )
        ).lower()
        if self.covariance_estimator not in _COVARIANCE_ESTIMATORS:
            raise ValueError(
                "full_gradient_probe_covariance_estimator must be "
                "centered_population_covariance or uncentered_second_moment"
            )
        self.token_mask_mode = str(
            _get(config, "full_gradient_probe_token_mask_mode", "none")
        ).lower()
        self.token_keep_ratio = float(
            _get(config, "full_gradient_probe_token_keep_ratio", 0.5)
        )
        self.token_min_keep = int(
            _get(config, "full_gradient_probe_token_min_keep", 128)
        )
        self.token_keep_final = int(
            _get(config, "full_gradient_probe_token_keep_final", 128)
        )
        self.token_mask_discovery_only = bool(
            _get(config, "full_gradient_probe_token_mask_discovery_only", True)
        )
        configured_mask_scope = str(
            _get(config, "full_gradient_probe_token_mask_scope", "legacy")
        ).lower()
        if configured_mask_scope == "legacy":
            configured_mask_scope = (
                "discovery" if self.token_mask_discovery_only else "all"
            )
        if configured_mask_scope not in _TOKEN_MASK_SCOPES:
            raise ValueError(
                "full_gradient_probe_token_mask_scope must be legacy, discovery, "
                "discovery_calibration or all"
            )
        self.token_mask_scope = configured_mask_scope
        if self.token_mask_mode not in {"none", "top_surprisal"}:
            raise ValueError(
                "full_gradient_probe_token_mask_mode must be none or top_surprisal"
            )
        probe_token_keep_count(
            0,
            keep_ratio=self.token_keep_ratio,
            min_keep=self.token_min_keep,
            final_tokens=self.token_keep_final,
        )
        output = str(_get(config, "gradient_probe_output_dir", ""))
        if not output:
            raise ValueError(
                "gradient_probe_output_dir is required for peft_type=full_gradient_probe"
            )
        self.output_dir = Path(output).expanduser().resolve()
        if self.rank <= 0 or self.sketch_width < self.rank:
            raise ValueError("Full-gradient probe requires sketch_width >= rank > 0")
        if (
            min(self.discovery_target, self.calibration_target) <= 0
            or self.audit_target < 0
        ):
            raise ValueError(
                "Invalid full-gradient discovery/calibration/audit prompt targets"
            )

        families = _target_families(_get(config, "target_modules", "all-linear"))
        self.module_names: dict[int, str] = {}
        self.module_shapes: dict[str, tuple[int, int]] = {}
        for name, module in model.named_modules():
            if (
                not isinstance(module, torch.nn.Linear)
                or name.rsplit(".", 1)[-1] not in families
            ):
                continue
            stable_name = normalize_target_name(name)
            self.module_names[id(module)] = stable_name
            self.module_shapes[stable_name] = (
                int(module.out_features),
                int(module.in_features),
            )
        if not self.module_names:
            raise ValueError("Full-gradient collector found no target Linear modules")
        if any(min(shape) < self.rank for shape in self.module_shapes.values()):
            raise ValueError(
                "full_gradient_probe_rank exceeds a target module dimension"
            )

        self.mean_sums: dict[str, torch.Tensor] = {}
        self.omegas: dict[str, torch.Tensor] = {}
        self.covariance_y: dict[str, torch.Tensor] = {}
        for name, (out_features, in_features) in self.module_shapes.items():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(self.seed, name, "covariance_omega"))
            omega = torch.randn(
                (in_features, self.sketch_width),
                generator=generator,
                dtype=torch.float32,
            )
            omega = torch.linalg.qr(omega, mode="reduced").Q
            self.mean_sums[name] = torch.zeros(
                (out_features, in_features), dtype=torch.float32
            )
            self.omegas[name] = omega
            self.covariance_y[name] = torch.zeros(
                (in_features, omega.shape[1]), dtype=torch.float32
            )

        self.candidate_sets: dict[str, dict[str, torch.Tensor]] = {}
        self.spectra: dict[str, dict[str, torch.Tensor]] = {}
        self.discovery_projected: dict[str, dict[str, torch.Tensor]] = {}
        self.split_stats: dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]] = {}
        self.discovery_count = 0
        self.calibration_count = 0
        self.audit_count = 0
        self.prompt_ids: list[str] = []
        self.losses: list[float] = []
        self.response_counts: list[int] = []
        self.response_tokens: list[int] = []
        self.probe_tokens: list[int] = []
        self.token_mask_active: list[bool] = []
        self.selected_surprisal: list[float | None] = []
        self.unselected_surprisal: list[float | None] = []
        self.advantage_rms: list[float] = []
        self.raw_norms: list[float] = []
        self.scales: list[float] = []
        self.total_prompts = 0
        self.total_rollouts = 0
        self.positive_rollouts = 0
        self.ready = False

    @property
    def phase(self) -> str:
        if self.discovery_count < self.discovery_target:
            return "discovery"
        if self.calibration_count < self.calibration_target:
            return "calibration"
        if self.audit_count < self.audit_target:
            return "audit"
        return "complete"

    def _units_and_contexts(self):
        if isinstance(self.model, FSDP):
            units = _leaf_fsdp_modules(self.model) or [self.model]
            return [
                (unit, FSDP.summon_full_params(unit, writeback=False, with_grads=True))
                for unit in units
            ]
        return [(self.model, nullcontext())]

    def record_rollout_batch(self, meta: dict[str, Any]) -> None:
        self.total_prompts += int(meta.get("full_gradient_total_prompts", 0))
        self.total_rollouts += int(meta.get("full_gradient_total_rollouts", 0))
        self.positive_rollouts += int(meta.get("full_gradient_positive_rollouts", 0))

    def _collect_observation(self, *, discovery: bool) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, dict[str, torch.Tensor]],
        float,
    ]:
        """Collect dense discovery means and GPU-computed thin statistics."""

        dense_gradients: dict[str, torch.Tensor] = {}
        covariance_updates: dict[str, torch.Tensor] = {}
        projected_gradients: dict[str, dict[str, torch.Tensor]] = {
            method: {} for method in _CANDIDATE_METHODS
        }
        found: set[str] = set()
        norm_square = 0.0
        for unit, context in self._units_and_contexts():
            with context:
                for module in unit.modules():
                    if id(module) not in self.module_names:
                        continue
                    name = self.module_names[id(module)]
                    if name in found:
                        raise RuntimeError(
                            f"Full-gradient probe encountered {name} more than once"
                        )
                    found.add(name)
                    gradient = module.weight.grad
                    if (
                        gradient is None
                        or tuple(gradient.shape) != self.module_shapes[name]
                    ):
                        raise RuntimeError(
                            f"Full-gradient probe is missing weight gradient for {name}"
                        )
                    value = gradient.detach().float()
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"Non-finite full gradient for {name}")
                    norm_square += float(value.square().sum().item())
                    if discovery:
                        dense_gradients[name] = value.cpu().contiguous()
                        omega = self.omegas[name].to(
                            device=value.device, non_blocking=True
                        )
                        covariance_updates[name] = (
                            (value.T @ (value @ omega)).cpu().contiguous()
                        )
                    else:
                        for method in _CANDIDATE_METHODS:
                            basis = self.candidate_sets[method][name].to(
                                device=value.device, non_blocking=True
                            )
                            projected_gradients[method][name] = (
                                (value @ basis.T).cpu().contiguous()
                            )
        if len(found) != len(self.module_names):
            missing = sorted(set(self.module_names.values()) - found)
            raise RuntimeError(
                f"Full-gradient probe captured {len(found)}/{len(self.module_names)} modules; missing={missing[:5]}"
            )
        return (
            dense_gradients,
            covariance_updates,
            projected_gradients,
            math.sqrt(norm_square),
        )

    def _scale_for_norm(self, raw_norm: float, *, discovery: bool) -> float:
        reference = self.raw_norms or [raw_norm]
        threshold = max(
            statistics.median(reference[: self.discovery_target]) * self.clip_factor,
            self.eps,
        )
        scale = min(1.0, threshold / max(raw_norm, self.eps))
        self.raw_norms.append(raw_norm)
        self.scales.append(scale)
        return scale

    @torch.no_grad()
    def _finish_discovery(self) -> None:
        self.candidate_sets = {method: {} for method in _CANDIDATE_METHODS}
        self.spectra = {method: {} for method in _CANDIDATE_METHODS}
        self.discovery_projected = {method: {} for method in _CANDIDATE_METHODS}
        for name in sorted(self.mean_sums):
            mean = self.mean_sums[name] / self.discovery_count
            mean_basis, mean_values, mean_capture = _right_singular_basis(
                mean,
                self.rank,
                oversample=self.oversample,
                niter=self.svd_niter,
                seed=_stable_seed(self.seed, name, "mean_svd"),
                device=self.svd_device,
            )
            covariance_y = covariance_sketch(
                self.covariance_y[name],
                mean,
                self.omegas[name],
                self.discovery_count,
                estimator=self.covariance_estimator,
            )
            covariance_basis, covariance_values, covariance_residual = (
                _nystrom_covariance_basis(
                    covariance_y,
                    self.omegas[name],
                    self.rank,
                    seed=_stable_seed(self.seed, name, "covariance_completion"),
                    eps=self.eps,
                )
            )
            hybrid_basis, hybrid_values = _hybrid_basis(
                mean_basis,
                mean_values,
                covariance_basis,
                covariance_values,
                self.rank,
                seed=_stable_seed(self.seed, name, "hybrid_completion"),
                eps=self.eps,
            )
            candidates = {
                "mean": mean_basis,
                "covariance": covariance_basis,
                "hybrid": hybrid_basis,
            }
            values = {
                "mean": mean_values,
                "covariance": covariance_values,
                "hybrid": hybrid_values,
            }
            for method in _CANDIDATE_METHODS:
                self.candidate_sets[method][name] = candidates[method].contiguous()
                self.spectra[method][name] = values[method].contiguous()
                self.discovery_projected[method][name] = (
                    mean @ candidates[method].T
                ).contiguous()
            self.spectra["mean"][name + ".capture"] = torch.tensor([mean_capture])
            self.spectra["covariance"][name + ".residual"] = torch.tensor(
                [covariance_residual]
            )

        self.mean_sums.clear()
        self.omegas.clear()
        self.covariance_y.clear()
        for split in ("calibration", "audit"):
            self.split_stats[split] = {}
            for method in _CANDIDATE_METHODS:
                self.split_stats[split][method] = {}
                for name, discovery_value in self.discovery_projected[method].items():
                    self.split_stats[split][method][name] = {
                        "sum": torch.zeros_like(discovery_value, dtype=torch.float64),
                        "f": torch.zeros(self.rank, dtype=torch.float64),
                        "gain_sum": torch.zeros(self.rank, dtype=torch.float64),
                        "gain_square_sum": torch.zeros(self.rank, dtype=torch.float64),
                        "adam_gain_sum": torch.zeros(self.rank, dtype=torch.float64),
                    }

    @torch.no_grad()
    def _add_held_out(
        self,
        split: str,
        projected_gradients: dict[str, dict[str, torch.Tensor]],
        scale: float,
    ) -> None:
        for method in _CANDIDATE_METHODS:
            for name, raw_projected in projected_gradients[method].items():
                projected = raw_projected * scale
                discovery_value = self.discovery_projected[method][name]
                gain = (projected * discovery_value).sum(dim=0).double()
                adam_direction = discovery_value / (discovery_value.abs() + self.eps)
                stats = self.split_stats[split][method][name]
                stats["sum"].add_(projected.double())
                stats["f"].add_(projected.double().square().sum(dim=0))
                stats["gain_sum"].add_(gain)
                stats["gain_square_sum"].add_(gain.square())
                stats["adam_gain_sum"].add_(
                    (projected * adam_direction).sum(dim=0).double()
                )

    def capture_group(
        self,
        *,
        loss: float,
        prompt_id: str,
        response_count: int,
        response_tokens: int,
        advantage_rms: float,
        probe_tokens: int | None = None,
        token_mask_active: bool = False,
        selected_surprisal: float | None = None,
        unselected_surprisal: float | None = None,
    ) -> dict[str, float]:
        if self.ready:
            return self.metrics()
        current_phase = self.phase
        gradients, covariance_updates, projected_gradients, raw_norm = (
            self._collect_observation(discovery=current_phase == "discovery")
        )
        scale = self._scale_for_norm(raw_norm, discovery=current_phase == "discovery")
        self.prompt_ids.append(str(prompt_id))
        self.losses.append(float(loss))
        self.response_counts.append(int(response_count))
        self.response_tokens.append(int(response_tokens))
        self.probe_tokens.append(
            int(response_tokens if probe_tokens is None else probe_tokens)
        )
        self.token_mask_active.append(bool(token_mask_active))
        self.selected_surprisal.append(
            None if selected_surprisal is None else float(selected_surprisal)
        )
        self.unselected_surprisal.append(
            None if unselected_surprisal is None else float(unselected_surprisal)
        )
        self.advantage_rms.append(float(advantage_rms))

        if current_phase == "discovery":
            for name, gradient in gradients.items():
                clipped = gradient * scale
                self.mean_sums[name].add_(clipped)
                self.covariance_y[name].add_(covariance_updates[name] * (scale * scale))
            self.discovery_count += 1
            if self.discovery_count == self.discovery_target:
                self._finish_discovery()
        elif current_phase == "calibration":
            self._add_held_out("calibration", projected_gradients, scale)
            self.calibration_count += 1
        elif current_phase == "audit":
            self._add_held_out("audit", projected_gradients, scale)
            self.audit_count += 1

        if self.phase == "complete":
            self.ready = True
            distributed = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            if not distributed or torch.distributed.get_rank() == 0:
                self._export()
            if distributed:
                torch.distributed.barrier()
        return self.metrics()

    def metrics(self) -> dict[str, float]:
        return {
            "full_gradient_probe/discovery_prompts": float(self.discovery_count),
            "full_gradient_probe/calibration_prompts": float(self.calibration_count),
            "full_gradient_probe/audit_prompts": float(self.audit_count),
            "full_gradient_probe/total_prompts": float(self.total_prompts),
            "full_gradient_probe/total_rollouts": float(self.total_rollouts),
            "full_gradient_probe/positive_rate": float(
                self.positive_rollouts / max(self.total_rollouts, 1)
            ),
            "full_gradient_probe/artifact_ready": float(self.ready),
        }

    def _score_tensors(
        self, split: str, method: str, name: str, count: int
    ) -> dict[str, torch.Tensor]:
        stats = self.split_stats[split][method][name]
        f_score = stats["f"] / count
        mean_projected = stats["sum"] / count
        s_score = mean_projected.square().sum(dim=0)
        r_score = s_score / (f_score + self.eps)
        p_score = f_score / (f_score.sum() + self.eps)
        gain = stats["gain_sum"] / count
        if count > 1:
            variance = (
                stats["gain_square_sum"] - stats["gain_sum"].square() / count
            ) / (count - 1)
            gain_se = variance.clamp_min(0).sqrt() / math.sqrt(count)
        else:
            gain_se = torch.zeros_like(gain)
        gain_lcb = gain - self.confidence_z * gain_se
        return {
            "F": f_score.float(),
            "S": s_score.float(),
            "R": r_score.float(),
            "P": p_score.float(),
            "gain": gain.float(),
            "gain_se": gain_se.float(),
            "gain_lcb": gain_lcb.float(),
            "adam_gain": (stats["adam_gain_sum"] / count).float(),
            "U": gain_lcb.clamp_min(0).float(),
        }

    def _export(self) -> None:
        from safetensors.torch import save_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths: dict[str, dict[str, str]] = {}
        module_summary: dict[str, dict[str, Any]] = {}
        for method in _CANDIDATE_METHODS:
            candidate_path = self.output_dir / f"candidates_{method}.safetensors"
            score_path = self.output_dir / f"atom_scores_{method}.safetensors"
            score_tensors: dict[str, torch.Tensor] = {}
            for name in sorted(self.candidate_sets[method]):
                calibration = self._score_tensors(
                    "calibration", method, name, self.calibration_count
                )
                audit = (
                    self._score_tensors("audit", method, name, self.audit_count)
                    if self.audit_count
                    else {
                        key: torch.zeros_like(value)
                        for key, value in calibration.items()
                    }
                )
                for label, value in calibration.items():
                    score_tensors[f"{name}.{label}"] = value.contiguous()
                for label, value in audit.items():
                    score_tensors[f"{name}.audit_{label}"] = value.contiguous()
                score_tensors[f"{name}.spectrum"] = (
                    self.spectra[method][name].float().contiguous()
                )
                if method == "mean":
                    score_tensors[f"{name}.discovery_capture"] = self.spectra[method][
                        name + ".capture"
                    ]
                if method == "covariance":
                    score_tensors[f"{name}.nystrom_residual"] = self.spectra[method][
                        name + ".residual"
                    ]

                if name not in module_summary:
                    out_features, in_features = self.module_shapes[name]
                    module_summary[name] = {
                        "shape": [out_features, in_features],
                        "candidate_rank": self.rank,
                    }
                basis = self.candidate_sets[method][name]
                module_summary[name][f"{method}_orthogonality_error"] = float(
                    (basis @ basis.T - torch.eye(self.rank)).abs().max().item()
                )
                module_summary[name][f"{method}_calibration_positive_lcb"] = int(
                    (calibration["gain_lcb"] > 0).sum().item()
                )

            for path, tensors in (
                (candidate_path, self.candidate_sets[method]),
                (score_path, score_tensors),
            ):
                temporary = path.with_suffix(path.suffix + ".tmp")
                save_file(
                    {
                        key: value.float().cpu().contiguous()
                        for key, value in tensors.items()
                    },
                    str(temporary),
                )
                os.replace(temporary, path)
            artifact_paths[method] = {
                "candidates": str(candidate_path),
                "atom_scores": str(score_path),
            }

        discovery_end = self.discovery_count
        calibration_end = discovery_end + self.calibration_count
        summary = {
            "schema_version": 1,
            "method": "full_gradient_signed_grpo_nystrom_v1",
            "covariance_estimator": self.covariance_estimator,
            "candidate_methods": list(_CANDIDATE_METHODS),
            "r_max": self.rank,
            "constant_scaling": 2.0,
            "loss": "prompt-group signed GRPO policy gradient at ratio=1",
            "aggregation": "token-mean within prompt group",
            "token_mask": {
                "mode": self.token_mask_mode,
                "keep_ratio": self.token_keep_ratio,
                "min_keep_per_response": self.token_min_keep,
                "keep_final_tokens_per_response": self.token_keep_final,
                "discovery_only": self.token_mask_discovery_only,
                "scope": self.token_mask_scope,
                "selection_score": "negative detached current-policy log probability",
            },
            "discovery_prompts": self.discovery_count,
            "calibration_prompts": self.calibration_count,
            "audit_prompts": self.audit_count,
            "discovery_prompt_ids": self.prompt_ids[:discovery_end],
            "calibration_prompt_ids": self.prompt_ids[discovery_end:calibration_end],
            "audit_prompt_ids": self.prompt_ids[calibration_end:],
            "prompt_splits_disjoint": len(set(self.prompt_ids)) == len(self.prompt_ids),
            "total_prompts": self.total_prompts,
            "total_rollouts": self.total_rollouts,
            "positive_rollouts": self.positive_rollouts,
            "positive_rate": self.positive_rollouts / max(self.total_rollouts, 1),
            "confidence_z": self.confidence_z,
            "sample_norm_control": {
                "method": "global-full-gradient discovery-median-clip",
                "clip_factor": self.clip_factor,
                "raw_norms": self.raw_norms,
                "scales": self.scales,
            },
            "samples": {
                "losses": self.losses,
                "response_counts": self.response_counts,
                "response_tokens": self.response_tokens,
                "probe_tokens": self.probe_tokens,
                "token_mask_active": self.token_mask_active,
                "selected_surprisal": self.selected_surprisal,
                "unselected_surprisal": self.unselected_surprisal,
                "advantage_rms": self.advantage_rms,
            },
            "artifacts": artifact_paths,
            "modules": module_summary,
        }
        path = self.output_dir / "probe_summary.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)


class _Phase0MomentAccumulator:
    """CPU statistics for one selector over a fixed prompt subset."""

    def __init__(
        self,
        module_shapes: dict[str, tuple[int, int]],
        omegas: dict[str, torch.Tensor],
    ) -> None:
        self.omegas = omegas
        self.mean_sums = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in module_shapes.items()
        }
        self.second_y = {
            name: torch.zeros(
                (shape[1], omegas[name].shape[1]), dtype=torch.float32
            )
            for name, shape in module_shapes.items()
        }
        self.energy_sums = {name: 0.0 for name in module_shapes}
        self.count = 0

    def add(
        self,
        gradients: dict[str, torch.Tensor],
        second_updates: dict[str, torch.Tensor] | None,
        scale: float,
    ) -> None:
        for name, gradient in gradients.items():
            scaled = gradient * scale
            self.mean_sums[name].add_(scaled)
            if second_updates is None:
                omega = self.omegas[name]
                update = gradient.T @ (gradient @ omega)
            else:
                update = second_updates[name]
            self.second_y[name].add_(update, alpha=scale * scale)
            self.energy_sums[name] += float(gradient.square().sum().item()) * scale * scale
        self.count += 1

    def mean(self, name: str) -> torch.Tensor:
        return self.mean_sums[name] / self.count

    def sketch(self, name: str, estimator: str) -> torch.Tensor:
        return covariance_sketch(
            self.second_y[name],
            self.mean(name),
            self.omegas[name],
            self.count,
            estimator=estimator,
        )

    def raw_second_moment(self, name: str) -> torch.Tensor:
        return self.second_y[name] / self.count


class _Phase0CrossAccumulator:
    """Symmetric cross-half covariance statistics for one rollout split."""

    def __init__(
        self,
        module_shapes: dict[str, tuple[int, int]],
        omegas: dict[str, torch.Tensor],
    ) -> None:
        self.omegas = omegas
        self.mean_a = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in module_shapes.items()
        }
        self.mean_b = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in module_shapes.items()
        }
        sketch_shapes = {
            name: (shape[1], omegas[name].shape[1])
            for name, shape in module_shapes.items()
        }
        self.second_a = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in sketch_shapes.items()
        }
        self.second_b = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in sketch_shapes.items()
        }
        self.cross_y = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in sketch_shapes.items()
        }
        self.full_energy_sums = {name: 0.0 for name in module_shapes}
        self.count = 0

    def add(
        self,
        gradients_a: dict[str, torch.Tensor],
        gradients_b: dict[str, torch.Tensor],
        scale: float,
        device: torch.device,
    ) -> None:
        for name in gradients_a:
            a_cpu = gradients_a[name] * scale
            b_cpu = gradients_b[name] * scale
            self.mean_a[name].add_(a_cpu)
            self.mean_b[name].add_(b_cpu)
            a = a_cpu.to(device)
            b = b_cpu.to(device)
            omega = self.omegas[name].to(device)
            a_omega = a @ omega
            b_omega = b @ omega
            second_a = a.T @ a_omega
            second_b = b.T @ b_omega
            cross_ab = a.T @ b_omega
            cross_ba = b.T @ a_omega
            self.second_a[name].add_(second_a.cpu())
            self.second_b[name].add_(second_b.cpu())
            self.cross_y[name].add_(((cross_ab + cross_ba) * 0.5).cpu())
            full = (a_cpu + b_cpu) * 0.5
            self.full_energy_sums[name] += float(full.square().sum().item())
        self.count += 1

    def cross_sketch(self, name: str, *, centered: bool = True) -> torch.Tensor:
        result = self.cross_y[name] / self.count
        if centered:
            mean_a = self.mean_a[name] / self.count
            mean_b = self.mean_b[name] / self.count
            omega = self.omegas[name]
            result = result - 0.5 * (
                mean_a.T @ (mean_b @ omega) + mean_b.T @ (mean_a @ omega)
            )
        return result

    def half_sketch(self, name: str, half: str) -> torch.Tensor:
        if half == "a":
            second, mean = self.second_a[name], self.mean_a[name] / self.count
        elif half == "b":
            second, mean = self.second_b[name], self.mean_b[name] / self.count
        else:
            raise ValueError(f"Unknown cross-fit half: {half}")
        return second / self.count - mean.T @ (mean @ self.omegas[name])

    def raw_full_second_moment(self, name: str) -> torch.Tensor:
        return (
            self.second_a[name] + self.second_b[name] + 2.0 * self.cross_y[name]
        ) / (4.0 * self.count)


def _nystrom_projected_energy(
    y: torch.Tensor,
    omega: torch.Tensor,
    basis: torch.Tensor,
    eps: float,
) -> float:
    """Estimate ``trace(P C P^T)`` from the same Nyström sketch used for C."""

    w = (omega.T @ y + y.T @ omega) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(w.float())
    threshold = max(
        float(eigenvalues.max().item()) * 1e-7 if eigenvalues.numel() else 0.0,
        eps,
    )
    positive = eigenvalues > threshold
    if not bool(positive.any()):
        return 0.0
    whitener = eigenvectors[:, positive] * eigenvalues[positive].rsqrt().unsqueeze(0)
    factor = y.float() @ whitener
    return float((basis.float() @ factor).square().sum().item())


def _consensus_basis(
    bases: list[torch.Tensor],
    values: list[torch.Tensor],
    rank: int,
    *,
    seed: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted = []
    for basis, spectrum in zip(bases, values):
        weights = spectrum.clamp_min(0)
        weights = weights / (weights.sum() + eps)
        weighted.append(basis * weights.sqrt().unsqueeze(1))
    pooled = torch.cat(weighted, dim=0)
    _, singular_values, vh = torch.linalg.svd(pooled, full_matrices=False)
    basis = _orthonormal_completion(vh[:rank], rank, seed=seed)
    spectrum = torch.zeros(rank, dtype=torch.float32)
    available = min(rank, singular_values.numel())
    spectrum[:available] = singular_values[:available].square()
    return basis, spectrum


class Phase0GradientDiagnosticsCollector:
    """Run all Phase-0 selectors on shared rollout groups without training."""

    mode = "phase0_diagnostics"
    is_phase0 = True

    def __init__(self, model: torch.nn.Module, config: Any) -> None:
        self.model = model
        self.config = config
        self.rank = int(_get(config, "full_gradient_probe_rank", 32))
        self.sketch_width = int(
            _get(config, "full_gradient_probe_sketch_width", self.rank + 8)
        )
        self.discovery_target = int(
            _get(config, "full_gradient_probe_discovery_prompts", 64)
        )
        self.calibration_target = int(
            _get(config, "full_gradient_probe_calibration_prompts", 32)
        )
        self.audit_target = int(_get(config, "full_gradient_probe_audit_prompts", 16))
        self.crossfit_splits = int(
            _get(config, "full_gradient_probe_crossfit_splits", 3)
        )
        self.keep_ratio = float(
            _get(config, "full_gradient_probe_token_keep_ratio", 0.5)
        )
        self.min_keep = int(_get(config, "full_gradient_probe_token_min_keep", 128))
        self.keep_final = int(
            _get(config, "full_gradient_probe_token_keep_final", 128)
        )
        self.stable_surprisal_quantile = float(
            _get(config, "full_gradient_probe_stable_surprisal_quantile", 0.95)
        )
        default_hybrid_ranks = ",".join(
            str(value) for value in (0, 2, 4, 8, 16) if value <= self.rank
        )
        hybrid_ranks = str(
            _get(
                config,
                "full_gradient_probe_hybrid_ranks",
                default_hybrid_ranks or "0",
            )
        )
        self.hybrid_ranks = tuple(
            sorted({int(value.strip()) for value in hybrid_ranks.split(",") if value.strip()})
        )
        self.hybrid_support_relative_threshold = float(
            _get(
                config,
                "full_gradient_probe_hybrid_support_relative_threshold",
                1e-7,
            )
        )
        self.clip_factor = float(_get(config, "full_gradient_probe_clip_factor", 2.5))
        self.seed = int(_get(config, "gradient_probe_seed", 42))
        self.eps = float(_get(config, "full_gradient_probe_eps", 1e-12))
        configured_device = str(
            _get(config, "full_gradient_probe_svd_device", "auto")
        ).lower()
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        if configured_device not in {"cpu", "cuda"}:
            raise ValueError("full_gradient_probe_svd_device must be auto, cpu or cuda")
        self.compute_device = torch.device(configured_device)
        output = str(_get(config, "gradient_probe_output_dir", ""))
        if not output:
            raise ValueError("gradient_probe_output_dir is required for Phase 0")
        self.output_dir = Path(output).expanduser().resolve()
        if self.rank <= 0 or self.sketch_width < self.rank:
            raise ValueError("Phase 0 requires sketch_width >= rank > 0")
        if min(self.discovery_target, self.calibration_target, self.audit_target) <= 0:
            raise ValueError("Phase 0 requires positive discovery/calibration/audit sizes")
        if self.crossfit_splits <= 0:
            raise ValueError("full_gradient_probe_crossfit_splits must be positive")
        if not self.hybrid_ranks or self.hybrid_ranks[0] < 0:
            raise ValueError("full_gradient_probe_hybrid_ranks must be non-negative")
        if self.hybrid_ranks[-1] > self.rank:
            raise ValueError("Phase-0 hybrid directions cannot exceed probe rank")
        if self.hybrid_support_relative_threshold < 0:
            raise ValueError("Phase-0 hybrid support threshold must be non-negative")
        probe_token_keep_count(
            0,
            keep_ratio=self.keep_ratio,
            min_keep=self.min_keep,
            final_tokens=self.keep_final,
        )

        families = _target_families(_get(config, "target_modules", "all-linear"))
        self.module_names: dict[int, str] = {}
        self.module_shapes: dict[str, tuple[int, int]] = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in families:
                stable_name = normalize_target_name(name)
                self.module_names[id(module)] = stable_name
                self.module_shapes[stable_name] = (
                    int(module.out_features),
                    int(module.in_features),
                )
        if not self.module_names:
            raise ValueError("Phase 0 found no target Linear modules")
        if any(min(shape) < self.rank for shape in self.module_shapes.values()):
            raise ValueError("full_gradient_probe_rank exceeds a target module dimension")

        self.distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.is_primary = not self.distributed or torch.distributed.get_rank() == 0
        self.omegas: dict[str, torch.Tensor] = {}
        for name, (_, in_features) in self.module_shapes.items():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(self.seed, name, "phase0_omega"))
            omega = torch.randn(
                (in_features, self.sketch_width),
                generator=generator,
                dtype=torch.float32,
            )
            self.omegas[name] = torch.linalg.qr(omega, mode="reduced").Q

        self.standard = (
            {
                selector: _Phase0MomentAccumulator(self.module_shapes, self.omegas)
                for selector in ("P0", "P1", "P2")
            }
            if self.is_primary
            else {}
        )
        self.standard_prompt_splits = (
            {
                selector: [
                    _Phase0MomentAccumulator(self.module_shapes, self.omegas)
                    for _ in range(2)
                ]
                for selector in ("P0", "P1", "P2")
            }
            if self.is_primary
            else {}
        )
        self.cross = (
            [
                _Phase0CrossAccumulator(self.module_shapes, self.omegas)
                for _ in range(self.crossfit_splits)
            ]
            if self.is_primary
            else []
        )
        self.cross_prompt_splits = (
            [
                [
                    _Phase0CrossAccumulator(self.module_shapes, self.omegas)
                    for _ in range(self.crossfit_splits)
                ]
                for _ in range(2)
            ]
            if self.is_primary
            else []
        )
        self.norm_references: dict[str, list[float]] = {
            selector: [] for selector in ("P0", "P1", "P2", "P3")
        }
        self.candidate_sets: dict[str, dict[str, torch.Tensor]] = {}
        self.spectra: dict[str, dict[str, torch.Tensor]] = {}
        self.module_diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
        self.held_out: dict[str, dict[str, dict[str, Any]]] = {}
        self.discovery_count = 0
        self.calibration_count = 0
        self.audit_count = 0
        self.prompt_ids: list[str] = []
        self.response_counts: list[int] = []
        self.response_tokens: list[int] = []
        self.advantage_rms: list[float] = []
        self.total_prompts = 0
        self.total_rollouts = 0
        self.positive_rollouts = 0
        self.rollout_cache: list[dict[str, Any]] = []
        self.ready = False
        self._cross_prompt_id: str | None = None
        self._cross_selector = "P3"
        self._cross_memberships: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._cross_sums: list[list[dict[str, torch.Tensor]]] = []
        self._cross_tokens: list[list[int]] = []

    @property
    def phase(self) -> str:
        if self.discovery_count < self.discovery_target:
            return "discovery"
        if self.calibration_count < self.calibration_target:
            return "calibration"
        if self.audit_count < self.audit_target:
            return "audit"
        return "complete"

    def record_rollout_batch(self, meta: dict[str, Any]) -> None:
        self.total_prompts += int(meta.get("full_gradient_total_prompts", 0))
        self.total_rollouts += int(meta.get("full_gradient_total_rollouts", 0))
        self.positive_rollouts += int(meta.get("full_gradient_positive_rollouts", 0))

    def cache_prompt_group(
        self, *, phase: str, prompt_id: str, batch: Any
    ) -> None:
        """Persist the exact shared rollout group needed to replay Phase 0."""

        if not self.is_primary:
            return
        from safetensors.torch import save_file

        cache_dir = self.output_dir / "rollout_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ordinal = self.discovery_count + self.calibration_count + self.audit_count
        prompt_hash = hashlib.sha256(str(prompt_id).encode()).hexdigest()[:12]
        path = cache_dir / f"{ordinal:04d}_{phase}_{prompt_hash}.safetensors"
        keys = (
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "advantages",
            "old_log_probs",
            "token_level_scores",
        )
        tensors = {
            key: batch[key].detach().cpu().contiguous().clone()
            for key in keys
            if key in batch
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
        self.rollout_cache.append(
            {
                "ordinal": ordinal,
                "phase": phase,
                "prompt_id": str(prompt_id),
                "path": str(path),
                "tensor_keys": sorted(tensors),
            }
        )

    def _units_and_contexts(self):
        if isinstance(self.model, FSDP):
            units = _leaf_fsdp_modules(self.model) or [self.model]
            return [
                (unit, FSDP.summon_full_params(unit, writeback=False, with_grads=True))
                for unit in units
            ]
        return [(self.model, nullcontext())]

    def _collect_gradients(
        self, *, with_second: bool
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
        gradients: dict[str, torch.Tensor] = {}
        second_updates: dict[str, torch.Tensor] = {}
        found: set[str] = set()
        norm_square = 0.0
        for unit, context in self._units_and_contexts():
            with context:
                for module in unit.modules():
                    if id(module) not in self.module_names:
                        continue
                    name = self.module_names[id(module)]
                    if name in found:
                        raise RuntimeError(f"Phase 0 encountered {name} more than once")
                    found.add(name)
                    gradient = module.weight.grad
                    if gradient is None or tuple(gradient.shape) != self.module_shapes[name]:
                        raise RuntimeError(f"Phase 0 is missing weight gradient for {name}")
                    value = gradient.detach().float()
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"Non-finite Phase-0 gradient for {name}")
                    norm_square += float(value.square().sum().item())
                    if self.is_primary:
                        gradients[name] = value.cpu().contiguous()
                    if with_second and self.is_primary:
                        omega = self.omegas[name].to(value.device, non_blocking=True)
                        second_updates[name] = (
                            value.T @ (value @ omega)
                        ).cpu().contiguous()
        if len(found) != len(self.module_names):
            missing = sorted(set(self.module_names.values()) - found)
            raise RuntimeError(
                f"Phase 0 captured {len(found)}/{len(self.module_names)} modules; "
                f"missing={missing[:5]}"
            )
        return gradients, second_updates, math.sqrt(norm_square)

    def _scale(self, selector: str, raw_norm: float) -> float:
        reference = self.norm_references[selector]
        threshold = max(
            statistics.median(reference[: self.discovery_target] or [raw_norm])
            * self.clip_factor,
            self.eps,
        )
        reference.append(raw_norm)
        return min(1.0, threshold / max(raw_norm, self.eps))

    def _held_out_scale(self, raw_norm: float) -> float:
        reference = self.norm_references["P0"][: self.discovery_target]
        threshold = max(
            statistics.median(reference or [raw_norm]) * self.clip_factor,
            self.eps,
        )
        return min(1.0, threshold / max(raw_norm, self.eps))

    @torch.no_grad()
    def capture_standard_discovery(self, selector: str) -> float:
        if selector not in {"P0", "P1", "P2"} or self.phase != "discovery":
            raise ValueError(f"Invalid Phase-0 discovery selector/state: {selector}/{self.phase}")
        gradients, second_updates, raw_norm = self._collect_gradients(with_second=True)
        scale = self._scale(selector, raw_norm)
        if self.is_primary:
            self.standard[selector].add(gradients, second_updates, scale)
            self.standard_prompt_splits[selector][self.discovery_count % 2].add(
                gradients, second_updates, scale
            )
        return raw_norm

    def begin_crossfit_prompt(self, prompt_id: str, response_count: int) -> None:
        if self._cross_prompt_id is not None:
            raise RuntimeError("Previous Phase-0 cross-fit prompt was not finished")
        self._cross_prompt_id = str(prompt_id)
        self._cross_memberships = deterministic_rollout_halves(
            self.seed, str(prompt_id), response_count, self.crossfit_splits
        )
        self._cross_sums = [[{}, {}] for _ in range(self.crossfit_splits)]
        self._cross_tokens = [[0, 0] for _ in range(self.crossfit_splits)]

    @torch.no_grad()
    def capture_crossfit_response(self, response_offset: int, selected_tokens: int) -> None:
        if self._cross_prompt_id is None:
            raise RuntimeError("begin_crossfit_prompt must be called first")
        gradients, _, _ = self._collect_gradients(with_second=False)
        if not self.is_primary:
            return
        for split_index, (first, _) in enumerate(self._cross_memberships):
            half = 0 if response_offset in first else 1
            self._cross_tokens[split_index][half] += int(selected_tokens)
            destination = self._cross_sums[split_index][half]
            for name, gradient in gradients.items():
                if name in destination:
                    destination[name].add_(gradient)
                else:
                    destination[name] = gradient.clone()

    @torch.no_grad()
    def finish_crossfit_prompt(self) -> list[float]:
        if self._cross_prompt_id is None:
            raise RuntimeError("No Phase-0 cross-fit prompt is active")
        if not self.is_primary:
            self._cross_prompt_id = None
            self._cross_memberships = []
            self._cross_sums = []
            self._cross_tokens = []
            self.norm_references[self._cross_selector].append(0.0)
            return [0.0] * self.crossfit_splits
        raw_norms: list[float] = []
        prompt_partition = self.discovery_count % 2
        for split_index in range(self.crossfit_splits):
            denominators = self._cross_tokens[split_index]
            if min(denominators) <= 0:
                raise RuntimeError("A Phase-0 rollout half selected no tokens")
            raw_norm = math.sqrt(
                sum(
                    float(
                        (
                            (
                                self._cross_sums[split_index][0][name]
                                / denominators[0]
                                + self._cross_sums[split_index][1][name]
                                / denominators[1]
                            )
                            * 0.5
                        )
                        .square()
                        .sum()
                        .item()
                    )
                    for name in self._cross_sums[split_index][0]
                )
            )
            raw_norms.append(raw_norm)
        shared_raw_norm = float(statistics.mean(raw_norms))
        scale = self._scale(self._cross_selector, shared_raw_norm)
        for split_index in range(self.crossfit_splits):
            denominators = self._cross_tokens[split_index]
            gradients_a = {
                name: value / denominators[0]
                for name, value in self._cross_sums[split_index][0].items()
            }
            gradients_b = {
                name: value / denominators[1]
                for name, value in self._cross_sums[split_index][1].items()
            }
            self.cross[split_index].add(
                gradients_a, gradients_b, scale, self.compute_device
            )
            self.cross_prompt_splits[prompt_partition][split_index].add(
                gradients_a, gradients_b, scale, self.compute_device
            )
        self._cross_prompt_id = None
        self._cross_memberships = []
        self._cross_sums = []
        self._cross_tokens = []
        return raw_norms

    def complete_discovery_prompt(
        self,
        *,
        prompt_id: str,
        response_count: int,
        response_tokens: int,
        advantage_rms: float,
    ) -> None:
        self.prompt_ids.append(str(prompt_id))
        self.response_counts.append(int(response_count))
        self.response_tokens.append(int(response_tokens))
        self.advantage_rms.append(float(advantage_rms))
        self.discovery_count += 1
        if self.discovery_count == self.discovery_target:
            if self.is_primary:
                self._finish_discovery()
            if self.distributed:
                torch.distributed.barrier()

    def _basis_from_moment(
        self,
        accumulator: _Phase0MomentAccumulator,
        name: str,
        estimator: str,
        purpose: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis, values, _ = _nystrom_covariance_basis(
            accumulator.sketch(name, estimator),
            self.omegas[name],
            self.rank,
            seed=_stable_seed(self.seed, name, purpose),
            eps=self.eps,
        )
        return basis, values

    def _basis_from_cross(
        self,
        accumulator: _Phase0CrossAccumulator,
        name: str,
        purpose: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis, values, _ = _nystrom_covariance_basis(
            accumulator.cross_sketch(name),
            self.omegas[name],
            self.rank,
            seed=_stable_seed(self.seed, name, purpose),
            eps=self.eps,
        )
        return basis, values

    @staticmethod
    def _average_overlap(items: list[dict[str, float]]) -> dict[str, float]:
        if not items:
            return {"mean_cosine": 0.0, "mean_squared_cosine": 0.0, "min_cosine": 0.0}
        return {
            key: float(statistics.mean(item[key] for item in items))
            for key in items[0]
        }

    def _crossfit_module_result(
        self, name: str, *, seed_label: str
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        list[tuple[torch.Tensor, torch.Tensor]],
    ]:
        response_bases: list[torch.Tensor] = []
        response_values: list[torch.Tensor] = []
        half_overlaps: list[dict[str, float]] = []
        for split_index, accumulator in enumerate(self.cross):
            basis, values = self._basis_from_cross(
                accumulator, name, f"{seed_label}_response_split_{split_index}"
            )
            response_bases.append(basis)
            response_values.append(values)
            half_a, _, _ = _nystrom_covariance_basis(
                accumulator.half_sketch(name, "a"),
                self.omegas[name],
                self.rank,
                seed=_stable_seed(
                    self.seed, name, f"{seed_label}_half_a_{split_index}"
                ),
                eps=self.eps,
            )
            half_b, _, _ = _nystrom_covariance_basis(
                accumulator.half_sketch(name, "b"),
                self.omegas[name],
                self.rank,
                seed=_stable_seed(
                    self.seed, name, f"{seed_label}_half_b_{split_index}"
                ),
                eps=self.eps,
            )
            half_overlaps.append(principal_subspace_overlap(half_a, half_b))
        basis, values = _consensus_basis(
            response_bases,
            response_values,
            self.rank,
            seed=_stable_seed(self.seed, name, f"{seed_label}_consensus"),
            eps=self.eps,
        )
        response_overlaps = [
            principal_subspace_overlap(response_bases[left], response_bases[right])
            for left in range(len(response_bases))
            for right in range(left + 1, len(response_bases))
        ]
        prompt_partition_results: list[tuple[torch.Tensor, torch.Tensor]] = []
        for partition in range(2):
            split_results = [
                self._basis_from_cross(
                    accumulator,
                    name,
                    f"{seed_label}_prompt_{partition}_response_{split_index}",
                )
                for split_index, accumulator in enumerate(
                    self.cross_prompt_splits[partition]
                )
            ]
            prompt_partition_results.append(
                _consensus_basis(
                    [item[0] for item in split_results],
                    [item[1] for item in split_results],
                    self.rank,
                    seed=_stable_seed(
                        self.seed,
                        name,
                        f"{seed_label}_prompt_{partition}_consensus",
                    ),
                    eps=self.eps,
                )
            )
        discovery_captures = []
        discovery_projected_energies = []
        discovery_total_energies = []
        for accumulator in self.cross:
            projected = _nystrom_projected_energy(
                accumulator.raw_full_second_moment(name),
                self.omegas[name],
                basis,
                self.eps,
            )
            denominator = accumulator.full_energy_sums[name] / accumulator.count
            discovery_captures.append(projected / max(denominator, self.eps))
            discovery_projected_energies.append(projected)
            discovery_total_energies.append(denominator)
        diagnostics = {
            "discovery_capture": float(statistics.mean(discovery_captures)),
            "discovery_projected_energy": float(
                statistics.mean(discovery_projected_energies)
            ),
            "discovery_total_energy": float(statistics.mean(discovery_total_energies)),
            "prompt_split_overlap": principal_subspace_overlap(
                prompt_partition_results[0][0], prompt_partition_results[1][0]
            ),
            "cross_half_overlap": self._average_overlap(half_overlaps),
            "response_split_overlap": self._average_overlap(response_overlaps),
            "effective_rank": self._average_overlap(
                [
                    covariance_effective_ranks(value, self.eps)
                    for value in response_values
                ]
            ),
        }
        return basis, values, diagnostics, prompt_partition_results

    @torch.no_grad()
    def _finish_discovery(self) -> None:
        hybrid_methods = tuple(f"H{count}" for count in self.hybrid_ranks)
        methods = (
            "P0",
            "P0_uncentered",
            "P1",
            "P1_uncentered",
            "P2",
            "P2_uncentered",
            "P3",
            *hybrid_methods,
        )
        self.candidate_sets = {method: {} for method in methods}
        self.spectra = {method: {} for method in methods}
        self.module_diagnostics = {method: {} for method in methods}
        for name in sorted(self.module_shapes):
            p2_basis: torch.Tensor | None = None
            p2_values: torch.Tensor | None = None
            p2_prompt_results: list[tuple[torch.Tensor, torch.Tensor]] | None = None
            for selector in ("P0", "P1", "P2"):
                for estimator, suffix in (
                    ("centered_population_covariance", ""),
                    ("uncentered_second_moment", "_uncentered"),
                ):
                    method = selector + suffix
                    basis, values = self._basis_from_moment(
                        self.standard[selector], name, estimator, f"{method}_main"
                    )
                    split_results = [
                        self._basis_from_moment(
                            split,
                            name,
                            estimator,
                            f"{method}_prompt_split_{index}",
                        )
                        for index, split in enumerate(self.standard_prompt_splits[selector])
                    ]
                    discovery_energy = _nystrom_projected_energy(
                        self.standard[selector].raw_second_moment(name),
                        self.omegas[name],
                        basis,
                        self.eps,
                    )
                    denominator = self.standard[selector].energy_sums[name] / self.standard[selector].count
                    self.candidate_sets[method][name] = basis
                    self.spectra[method][name] = values
                    self.module_diagnostics[method][name] = {
                        "discovery_capture": discovery_energy / max(denominator, self.eps),
                        "discovery_projected_energy": discovery_energy,
                        "discovery_total_energy": denominator,
                        "prompt_split_overlap": principal_subspace_overlap(
                            split_results[0][0], split_results[1][0]
                        ),
                        "effective_rank": covariance_effective_ranks(values, self.eps),
                    }
                    if method == "P2":
                        p2_basis = basis
                        p2_values = values
                        p2_prompt_results = split_results

            if p2_basis is None or p2_values is None or p2_prompt_results is None:
                raise RuntimeError("Phase-0 P2 basis construction did not complete")

            (
                p3_basis,
                p3_values,
                p3_diagnostics,
                prompt_partition_results,
            ) = self._crossfit_module_result(name, seed_label="P3")
            self.candidate_sets["P3"][name] = p3_basis
            self.spectra["P3"][name] = p3_values
            self.module_diagnostics["P3"][name] = p3_diagnostics

            p2_denominator = (
                self.standard["P2"].energy_sums[name] / self.standard["P2"].count
            )
            for stable_directions in self.hybrid_ranks:
                method = f"H{stable_directions}"
                if stable_directions == 0:
                    hybrid_basis = p2_basis.clone()
                    hybrid_values = p2_values.clone()
                    selected_stable = 0
                    hybrid_prompt_bases = [item[0] for item in p2_prompt_results]
                else:
                    hybrid_basis, hybrid_values, selected_stable = (
                        stability_supported_hybrid_basis(
                            p3_basis,
                            p3_values,
                            p2_basis,
                            p2_values,
                            self.rank,
                            stable_directions=stable_directions,
                            seed=_stable_seed(self.seed, name, f"{method}_main"),
                            relative_threshold=self.hybrid_support_relative_threshold,
                            eps=self.eps,
                        )
                    )
                    hybrid_prompt_bases = []
                    for partition in range(2):
                        prompt_basis, _, _ = stability_supported_hybrid_basis(
                            prompt_partition_results[partition][0],
                            prompt_partition_results[partition][1],
                            p2_prompt_results[partition][0],
                            p2_prompt_results[partition][1],
                            self.rank,
                            stable_directions=stable_directions,
                            seed=_stable_seed(
                                self.seed,
                                name,
                                f"{method}_prompt_split_{partition}",
                            ),
                            relative_threshold=self.hybrid_support_relative_threshold,
                            eps=self.eps,
                        )
                        hybrid_prompt_bases.append(prompt_basis)
                projected = _nystrom_projected_energy(
                    self.standard["P2"].raw_second_moment(name),
                    self.omegas[name],
                    hybrid_basis,
                    self.eps,
                )
                self.candidate_sets[method][name] = hybrid_basis
                self.spectra[method][name] = hybrid_values
                self.module_diagnostics[method][name] = {
                    "discovery_capture": projected
                    / max(p2_denominator, self.eps),
                    "discovery_projected_energy": projected,
                    "discovery_total_energy": p2_denominator,
                    "prompt_split_overlap": principal_subspace_overlap(
                        hybrid_prompt_bases[0], hybrid_prompt_bases[1]
                    ),
                    "effective_rank": covariance_effective_ranks(
                        hybrid_values, self.eps
                    ),
                    "p3_consensus_supported_rank": float(
                        positive_spectrum_rank(
                            p3_values,
                            relative_threshold=self.hybrid_support_relative_threshold,
                            eps=self.eps,
                        )
                    ),
                    "selected_p3_directions": float(selected_stable),
                    "fallback_p2_directions": float(self.rank - selected_stable),
                }

        self.held_out = {
            split: {
                method: {
                    name: {
                        "projected_energy": torch.zeros(self.rank, dtype=torch.float64),
                        "total_energy": 0.0,
                    }
                    for name in self.module_shapes
                }
                for method in methods
            }
            for split in ("calibration", "audit")
        }

    @torch.no_grad()
    def capture_held_out(
        self,
        *,
        split: str,
        prompt_id: str,
        response_count: int,
        response_tokens: int,
        advantage_rms: float,
    ) -> None:
        if split not in {"calibration", "audit"} or self.phase != split:
            raise ValueError(f"Invalid Phase-0 held-out split/state: {split}/{self.phase}")
        gradients, _, raw_norm = self._collect_gradients(with_second=False)
        scale = self._held_out_scale(raw_norm)
        if self.is_primary:
            for method, candidates in self.candidate_sets.items():
                for name, basis in candidates.items():
                    gradient = gradients[name] * scale
                    projected = gradient @ basis.T
                    stats = self.held_out[split][method][name]
                    stats["projected_energy"].add_(
                        projected.double().square().sum(dim=0)
                    )
                    stats["total_energy"] += float(gradient.square().sum().item())
        self.prompt_ids.append(str(prompt_id))
        self.response_counts.append(int(response_count))
        self.response_tokens.append(int(response_tokens))
        self.advantage_rms.append(float(advantage_rms))
        if split == "calibration":
            self.calibration_count += 1
        else:
            self.audit_count += 1
        if self.phase == "complete":
            self.ready = True
            if self.is_primary:
                self._export()
            if self.distributed:
                torch.distributed.barrier()

    def metrics(self) -> dict[str, float]:
        return {
            "full_gradient_probe/discovery_prompts": float(self.discovery_count),
            "full_gradient_probe/calibration_prompts": float(self.calibration_count),
            "full_gradient_probe/audit_prompts": float(self.audit_count),
            "full_gradient_probe/artifact_ready": float(self.ready),
        }

    def _global_capture(self, split: str, method: str) -> float:
        projected = sum(
            float(stats["projected_energy"].sum().item())
            for stats in self.held_out[split][method].values()
        )
        total = sum(
            float(stats["total_energy"])
            for stats in self.held_out[split][method].values()
        )
        return projected / max(total, self.eps)

    def _global_discovery_capture(self, method: str) -> float:
        projected = sum(
            float(item["discovery_projected_energy"])
            for item in self.module_diagnostics[method].values()
        )
        total = sum(
            float(item["discovery_total_energy"])
            for item in self.module_diagnostics[method].values()
        )
        return projected / max(total, self.eps)

    def _aggregate_module_diagnostic(
        self, method: str, field: str
    ) -> dict[str, float]:
        items = [item[field] for item in self.module_diagnostics[method].values()]
        return {
            key: float(statistics.mean(item[key] for item in items))
            for key in items[0]
        }

    def _export(self) -> None:
        from safetensors.torch import save_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, dict[str, str]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for method in self.candidate_sets:
            candidate_path = self.output_dir / f"candidates_{method}.safetensors"
            score_path = self.output_dir / f"atom_scores_{method}.safetensors"
            score_tensors: dict[str, torch.Tensor] = {}
            for name in self.module_shapes:
                calibration = self.held_out["calibration"][method][name]
                audit = self.held_out["audit"][method][name]
                calibration_f = (
                    calibration["projected_energy"] / self.calibration_count
                ).float()
                audit_f = (audit["projected_energy"] / self.audit_count).float()
                score_tensors[f"{name}.F"] = calibration_f
                score_tensors[f"{name}.P"] = calibration_f / (
                    calibration_f.sum() + self.eps
                )
                score_tensors[f"{name}.U"] = score_tensors[f"{name}.P"].clone()
                score_tensors[f"{name}.audit_F"] = audit_f
                score_tensors[f"{name}.audit_P"] = audit_f / (
                    audit_f.sum() + self.eps
                )
                score_tensors[f"{name}.audit_U"] = score_tensors[
                    f"{name}.audit_P"
                ].clone()
                score_tensors[f"{name}.spectrum"] = self.spectra[method][name]
            for path, tensors in (
                (candidate_path, self.candidate_sets[method]),
                (score_path, score_tensors),
            ):
                temporary = path.with_suffix(path.suffix + ".tmp")
                save_file(
                    {
                        key: value.float().cpu().contiguous()
                        for key, value in tensors.items()
                    },
                    str(temporary),
                )
                os.replace(temporary, path)
            artifacts[method] = {
                "candidates": str(candidate_path),
                "atom_scores": str(score_path),
            }
            discovery_capture = self._global_discovery_capture(method)
            calibration_capture = self._global_capture("calibration", method)
            audit_capture = self._global_capture("audit", method)
            diagnostics[method] = {
                "discovery_capture_nystrom_mean_over_modules": discovery_capture,
                "calibration_capture": calibration_capture,
                "audit_capture": audit_capture,
                "capture_gap": discovery_capture - audit_capture,
                "prompt_split_overlap": self._aggregate_module_diagnostic(
                    method, "prompt_split_overlap"
                ),
                "effective_rank": self._aggregate_module_diagnostic(
                    method, "effective_rank"
                ),
            }
            if method == "P3":
                diagnostics[method]["cross_half_overlap"] = self._aggregate_module_diagnostic(
                    method, "cross_half_overlap"
                )
                diagnostics[method]["response_split_overlap"] = self._aggregate_module_diagnostic(
                    method, "response_split_overlap"
                )
            if method.startswith("H"):
                diagnostics[method]["requested_p3_directions"] = int(method[1:])
                for field in (
                    "p3_consensus_supported_rank",
                    "selected_p3_directions",
                    "fallback_p2_directions",
                ):
                    diagnostics[method][f"mean_{field}"] = float(
                        statistics.mean(
                            item[field]
                            for item in self.module_diagnostics[method].values()
                        )
                    )

        discovery_end = self.discovery_count
        calibration_end = discovery_end + self.calibration_count
        summary = {
            "schema_version": 4,
            "method": "phase0_shared_rollout_gradient_diagnostics_v1",
            "candidate_methods": list(self.candidate_sets),
            "score_labels": ["F", "P", "U"],
            "constant_scaling": 2.0,
            "selection_uses_audit": False,
            "held_out_scoring_space": "unmasked_full_prompt_group_policy_gradient",
            "selectors": {
                "P0": "centered covariance, no mask",
                "P1": "centered covariance, deterministic density-matched random mask",
                "P2": "centered covariance, top surprisal plus final tokens",
                "P3": "symmetric centered cross-fit covariance, advantage-entropy stable band",
            },
            "uncentered_controls": ["P0_uncentered", "P1_uncentered", "P2_uncentered"],
            "stability_supported_hybrids": {
                "methods": [f"H{count}" for count in self.hybrid_ranks],
                "requested_p3_directions": list(self.hybrid_ranks),
                "fallback": "P2 centered top-surprisal basis, residualized in discovery order",
                "support_rule": "P3 consensus spectrum > max(max_spectrum * relative_threshold, eps)",
                "relative_threshold": self.hybrid_support_relative_threshold,
                "selection_uses_calibration": False,
                "selection_uses_audit": False,
            },
            "r_max": self.rank,
            "sketch_width": self.sketch_width,
            "crossfit_splits": self.crossfit_splits,
            "token_mask": {
                "keep_ratio": self.keep_ratio,
                "min_keep_per_response": self.min_keep,
                "top_surprisal_final_tokens": self.keep_final,
                "stable_surprisal_upper_quantile": self.stable_surprisal_quantile,
                "calibration_audit_mask": "none",
            },
            "discovery_prompts": self.discovery_count,
            "calibration_prompts": self.calibration_count,
            "audit_prompts": self.audit_count,
            "discovery_prompt_ids": self.prompt_ids[:discovery_end],
            "calibration_prompt_ids": self.prompt_ids[discovery_end:calibration_end],
            "audit_prompt_ids": self.prompt_ids[calibration_end:],
            "prompt_splits_disjoint": len(set(self.prompt_ids)) == len(self.prompt_ids),
            "shared_rollouts_across_selectors": True,
            "rollout_cache": self.rollout_cache,
            "total_prompts": self.total_prompts,
            "total_rollouts": self.total_rollouts,
            "positive_rollouts": self.positive_rollouts,
            "diagnostics": diagnostics,
            "artifacts": artifacts,
            "samples": {
                "response_counts": self.response_counts,
                "response_tokens": self.response_tokens,
                "advantage_rms": self.advantage_rms,
                "norm_references": self.norm_references,
            },
            "modules": {
                name: {
                    "shape": list(shape),
                    "candidate_rank": self.rank,
                    "diagnostics": {
                        method: self.module_diagnostics[method][name]
                        for method in self.candidate_sets
                    },
                }
                for name, shape in self.module_shapes.items()
            },
        }
        path = self.output_dir / "probe_summary.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)


class Phase06CrossfitReplayCollector(Phase0GradientDiagnosticsCollector):
    """Sequentially replay one Phase-0 cache through cross-fit token masks."""

    mode = "phase06_crossfit_replay"
    is_phase0 = False
    is_phase06_replay = True
    replay_method_masks = {
        "C0": "none",
        "C2": "top_surprisal",
        "C3": "advantage_entropy_stable_band",
    }

    def __init__(self, model: torch.nn.Module, config: Any) -> None:
        super().__init__(model, config)
        source = str(_get(config, "full_gradient_probe_replay_source_dir", ""))
        if not source:
            raise ValueError(
                "full_gradient_probe_replay_source_dir is required for Phase 0.6 replay"
            )
        self.source_dir = Path(source).expanduser().resolve()
        if self.source_dir == self.output_dir:
            raise ValueError("Phase 0.6 replay output must differ from its source artifact")
        self.source_summary = json.loads(
            (self.source_dir / "probe_summary.json").read_text(encoding="utf-8")
        )
        if int(self.source_summary.get("schema_version", -1)) != 4:
            raise ValueError("Phase 0.6 requires a validated schema-4 Phase-0 source")
        expected_counts = (
            self.discovery_target,
            self.calibration_target,
            self.audit_target,
        )
        source_counts = tuple(
            int(self.source_summary[field])
            for field in (
                "discovery_prompts",
                "calibration_prompts",
                "audit_prompts",
            )
        )
        if source_counts != expected_counts:
            raise ValueError(
                f"Phase 0.6 source split counts differ: {source_counts} != {expected_counts}"
            )
        if int(self.source_summary.get("r_max", -1)) != self.rank:
            raise ValueError("Phase 0.6 source candidate rank differs")
        if int(self.source_summary.get("sketch_width", -1)) != self.sketch_width:
            raise ValueError("Phase 0.6 source sketch width differs")
        source_shapes = {
            name: tuple(int(value) for value in item["shape"])
            for name, item in self.source_summary["modules"].items()
        }
        if source_shapes != self.module_shapes:
            raise ValueError("Phase 0.6 model modules differ from the source artifact")
        if "P3" not in self.source_summary.get("candidate_methods", []):
            raise ValueError("Phase 0.6 source artifact has no P3 candidate")

        self.source_prompt_ids = [
            str(value)
            for field in (
                "discovery_prompt_ids",
                "calibration_prompt_ids",
                "audit_prompt_ids",
            )
            for value in self.source_summary[field]
        ]
        self.source_rollout_cache = list(self.source_summary["rollout_cache"])
        if len(self.source_rollout_cache) != sum(expected_counts):
            raise ValueError("Phase 0.6 source cache manifest has the wrong length")
        for ordinal, entry in enumerate(self.source_rollout_cache):
            if int(entry["ordinal"]) != ordinal:
                raise ValueError("Phase 0.6 source cache ordinals are not contiguous")
            if str(entry["prompt_id"]) != self.source_prompt_ids[ordinal]:
                raise ValueError("Phase 0.6 source cache prompt IDs differ")
            path = Path(entry["path"]).expanduser().resolve()
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"Phase 0.6 source cache is missing: {path}")

        # The parent allocates the same peak-safe single cross-fit state used by P3.
        # Standard P0/P1/P2 accumulators are unnecessary for replay and are released.
        self.standard.clear()
        self.standard_prompt_splits.clear()
        gc.collect()
        source_p0_norms = self.source_summary["samples"]["norm_references"]["P0"]
        if len(source_p0_norms) < self.discovery_target:
            raise ValueError("Phase 0.6 source is missing P0 clipping references")
        self.norm_references = {
            "P0": [float(value) for value in source_p0_norms],
            **{method: [] for method in self.replay_method_masks},
        }
        self.candidate_sets = {method: {} for method in self.replay_method_masks}
        self.spectra = {method: {} for method in self.replay_method_masks}
        self.module_diagnostics = {
            method: {} for method in self.replay_method_masks
        }
        self.held_out = {}
        self.rollout_cache = self.source_rollout_cache
        self.total_prompts = int(self.source_summary.get("total_prompts", 0))
        self.total_rollouts = int(self.source_summary.get("total_rollouts", 0))
        self.positive_rollouts = int(self.source_summary.get("positive_rollouts", 0))
        self._active_replay_method: str | None = None

    @staticmethod
    def _prompt_ids_sha256(values: list[str]) -> str:
        payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def entries_for_phase(self, phase: str) -> list[dict[str, Any]]:
        entries = [entry for entry in self.source_rollout_cache if entry["phase"] == phase]
        expected = {
            "discovery": self.discovery_target,
            "calibration": self.calibration_target,
            "audit": self.audit_target,
        }
        if phase not in expected or len(entries) != expected[phase]:
            raise ValueError(f"Phase 0.6 source has an invalid {phase} cache split")
        return entries

    def start_replay_method(self, method: str) -> None:
        if method not in self.replay_method_masks:
            raise ValueError(f"Unknown Phase 0.6 replay method: {method}")
        if self._active_replay_method is not None:
            raise RuntimeError("A Phase 0.6 replay method is already active")
        if self.discovery_count not in {0, self.discovery_target}:
            raise RuntimeError("Previous Phase 0.6 discovery replay is incomplete")
        self.discovery_count = 0
        if self.is_primary and not self.cross:
            self.cross = [
                _Phase0CrossAccumulator(self.module_shapes, self.omegas)
                for _ in range(self.crossfit_splits)
            ]
            self.cross_prompt_splits = [
                [
                    _Phase0CrossAccumulator(self.module_shapes, self.omegas)
                    for _ in range(self.crossfit_splits)
                ]
                for _ in range(2)
            ]
        self._cross_selector = method
        self._active_replay_method = method

    def complete_replay_discovery_prompt(
        self,
        *,
        prompt_id: str,
        response_count: int,
        response_tokens: int,
        advantage_rms: float,
    ) -> None:
        if self._active_replay_method is None:
            raise RuntimeError("No Phase 0.6 replay method is active")
        if self._active_replay_method == "C0":
            self.prompt_ids.append(str(prompt_id))
            self.response_counts.append(int(response_count))
            self.response_tokens.append(int(response_tokens))
            self.advantage_rms.append(float(advantage_rms))
        self.discovery_count += 1
        if self.discovery_count > self.discovery_target:
            raise RuntimeError("Phase 0.6 replay exceeded the discovery target")

    @torch.no_grad()
    def finish_replay_method(self) -> None:
        method = self._active_replay_method
        if method is None or self.discovery_count != self.discovery_target:
            raise RuntimeError("Phase 0.6 replay method did not reach its target")
        if self.is_primary:
            seed_label = "P3" if method == "C3" else method
            for name in sorted(self.module_shapes):
                basis, values, diagnostics, _ = self._crossfit_module_result(
                    name, seed_label=seed_label
                )
                self.candidate_sets[method][name] = basis
                self.spectra[method][name] = values
                self.module_diagnostics[method][name] = diagnostics
        self.cross = []
        self.cross_prompt_splits = []
        self._active_replay_method = None
        self._cross_prompt_id = None
        self._cross_memberships = []
        self._cross_sums = []
        self._cross_tokens = []
        gc.collect()

    def prepare_held_out_replay(self) -> None:
        if self._active_replay_method is not None:
            raise RuntimeError("Cannot score held-out data during discovery replay")
        if self.is_primary and any(
            len(values) != len(self.module_shapes)
            for values in self.candidate_sets.values()
        ):
            raise RuntimeError("Phase 0.6 candidates are incomplete")
        self.discovery_count = self.discovery_target
        self.calibration_count = 0
        self.audit_count = 0
        self.held_out = {
            split: {
                method: {
                    name: {
                        "projected_energy": torch.zeros(self.rank, dtype=torch.float64),
                        "total_energy": 0.0,
                    }
                    for name in self.module_shapes
                }
                for method in self.replay_method_masks
            }
            for split in ("calibration", "audit")
        }

    def _export(self) -> None:
        from safetensors.torch import save_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, dict[str, str]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for method in self.replay_method_masks:
            candidate_path = self.output_dir / f"candidates_{method}.safetensors"
            score_path = self.output_dir / f"atom_scores_{method}.safetensors"
            score_tensors: dict[str, torch.Tensor] = {}
            for name in self.module_shapes:
                calibration = self.held_out["calibration"][method][name]
                audit = self.held_out["audit"][method][name]
                calibration_f = (
                    calibration["projected_energy"] / self.calibration_count
                ).float()
                audit_f = (audit["projected_energy"] / self.audit_count).float()
                score_tensors[f"{name}.F"] = calibration_f
                score_tensors[f"{name}.P"] = calibration_f / (
                    calibration_f.sum() + self.eps
                )
                score_tensors[f"{name}.U"] = score_tensors[f"{name}.P"].clone()
                score_tensors[f"{name}.audit_F"] = audit_f
                score_tensors[f"{name}.audit_P"] = audit_f / (
                    audit_f.sum() + self.eps
                )
                score_tensors[f"{name}.audit_U"] = score_tensors[
                    f"{name}.audit_P"
                ].clone()
                score_tensors[f"{name}.spectrum"] = self.spectra[method][name]
            for path, tensors in (
                (candidate_path, self.candidate_sets[method]),
                (score_path, score_tensors),
            ):
                temporary = path.with_suffix(path.suffix + ".tmp")
                save_file(
                    {
                        key: value.float().cpu().contiguous()
                        for key, value in tensors.items()
                    },
                    str(temporary),
                )
                os.replace(temporary, path)
            artifacts[method] = {
                "candidates": str(candidate_path),
                "atom_scores": str(score_path),
            }
            discovery_capture = self._global_discovery_capture(method)
            audit_capture = self._global_capture("audit", method)
            diagnostics[method] = {
                "discovery_capture_nystrom_mean_over_modules": discovery_capture,
                "calibration_capture": self._global_capture("calibration", method),
                "audit_capture": audit_capture,
                "capture_gap": discovery_capture - audit_capture,
                "prompt_split_overlap": self._aggregate_module_diagnostic(
                    method, "prompt_split_overlap"
                ),
                "cross_half_overlap": self._aggregate_module_diagnostic(
                    method, "cross_half_overlap"
                ),
                "response_split_overlap": self._aggregate_module_diagnostic(
                    method, "response_split_overlap"
                ),
                "effective_rank": self._aggregate_module_diagnostic(
                    method, "effective_rank"
                ),
            }

        source_hash = self._prompt_ids_sha256(self.source_prompt_ids)
        replay_hash = self._prompt_ids_sha256(self.prompt_ids)
        discovery_end = self.discovery_count
        calibration_end = discovery_end + self.calibration_count
        summary = {
            "schema_version": 1,
            "method": "phase06_sequential_crossfit_cache_replay_v1",
            "candidate_methods": list(self.replay_method_masks),
            "score_labels": ["F", "P", "U"],
            "constant_scaling": 2.0,
            "selection_uses_audit": False,
            "held_out_scoring_space": "replayed_unmasked_full_prompt_group_policy_gradient",
            "source_artifact": str(self.source_dir),
            "source_prompt_ids_sha256": source_hash,
            "replayed_prompt_ids_sha256": replay_hash,
            "replay_method_masks": self.replay_method_masks,
            "r_max": self.rank,
            "sketch_width": self.sketch_width,
            "crossfit_splits": self.crossfit_splits,
            "token_mask": {
                "keep_ratio": self.keep_ratio,
                "min_keep_per_response": self.min_keep,
                "top_surprisal_final_tokens": self.keep_final,
                "stable_surprisal_upper_quantile": self.stable_surprisal_quantile,
                "calibration_audit_mask": "none",
            },
            "discovery_prompts": self.discovery_count,
            "calibration_prompts": self.calibration_count,
            "audit_prompts": self.audit_count,
            "discovery_prompt_ids": self.prompt_ids[:discovery_end],
            "calibration_prompt_ids": self.prompt_ids[discovery_end:calibration_end],
            "audit_prompt_ids": self.prompt_ids[calibration_end:],
            "prompt_splits_disjoint": len(set(self.prompt_ids)) == len(self.prompt_ids),
            "shared_rollouts_across_selectors": True,
            "rollout_cache": self.source_rollout_cache,
            "total_prompts": self.total_prompts,
            "total_rollouts": self.total_rollouts,
            "positive_rollouts": self.positive_rollouts,
            "diagnostics": diagnostics,
            "artifacts": artifacts,
            "samples": {
                "response_counts": self.response_counts,
                "response_tokens": self.response_tokens,
                "advantage_rms": self.advantage_rms,
                "norm_references": self.norm_references,
            },
            "modules": {
                name: {
                    "shape": list(shape),
                    "candidate_rank": self.rank,
                    "diagnostics": {
                        method: self.module_diagnostics[method][name]
                        for method in self.replay_method_masks
                    },
                }
                for name, shape in self.module_shapes.items()
            },
        }
        if source_hash != replay_hash:
            raise RuntimeError("Phase 0.6 replay prompt IDs differ from the source")
        path = self.output_dir / "probe_summary.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)


class WindowedAdamConsensusCollector:
    """Discover recurrent right-gradient atoms from windowed policy gradients.

    Discovery maintains virtual full-weight Adam moments on CPU.  Candidate
    atoms are extracted independently from each window and retained only when
    their local right subspace recurs in other, non-adjacent windows.  Held-out
    calibration and audit always score candidates with raw policy gradients.
    """

    mode = "windowed_adam_consensus"
    is_windowed = True

    def __init__(self, model: torch.nn.Module, config: Any) -> None:
        self.model = model
        self.config = config
        self.rank = int(_get(config, "full_gradient_probe_rank", 32))
        self.window_prompts = int(
            _get(config, "full_gradient_probe_window_prompts", 16)
        )
        self.discovery_target = int(
            _get(config, "full_gradient_probe_discovery_windows", 8)
        )
        self.calibration_target = int(
            _get(config, "full_gradient_probe_calibration_windows", 2)
        )
        self.audit_target = int(
            _get(config, "full_gradient_probe_audit_windows", 2)
        )
        self.local_atoms = int(
            _get(config, "full_gradient_probe_local_atoms", 4)
        )
        self.beta1 = float(_get(config, "full_gradient_probe_adam_beta1", 0.9))
        self.beta2 = float(_get(config, "full_gradient_probe_adam_beta2", 0.999))
        self.adam_eps = float(_get(config, "full_gradient_probe_adam_eps", 1e-8))
        self.clip_factor = float(_get(config, "full_gradient_probe_clip_factor", 2.5))
        self.confidence_z = float(
            _get(config, "full_gradient_probe_future_lcb_z", 1.0)
        )
        self.support_floor = float(
            _get(config, "full_gradient_probe_support_floor", 1e-4)
        )
        self.oversample = int(_get(config, "full_gradient_probe_svd_oversample", 8))
        self.svd_niter = int(_get(config, "full_gradient_probe_svd_niter", 1))
        configured_svd_device = str(
            _get(config, "full_gradient_probe_svd_device", "auto")
        ).lower()
        if configured_svd_device == "auto":
            configured_svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        if configured_svd_device not in {"cpu", "cuda"}:
            raise ValueError("full_gradient_probe_svd_device must be auto, cpu or cuda")
        self.svd_device = torch.device(configured_svd_device)
        self.seed = int(_get(config, "gradient_probe_seed", 42))
        self.eps = float(_get(config, "full_gradient_probe_eps", 1e-12))
        output = str(_get(config, "gradient_probe_output_dir", ""))
        if not output:
            raise ValueError(
                "gradient_probe_output_dir is required for peft_type=full_gradient_probe"
            )
        self.output_dir = Path(output).expanduser().resolve()
        if self.rank <= 0 or self.local_atoms <= 0 or self.local_atoms > self.rank:
            raise ValueError("Windowed probe requires 0 < local_atoms <= rank")
        if self.window_prompts <= 0 or min(
            self.discovery_target, self.calibration_target, self.audit_target
        ) <= 0:
            raise ValueError("Window and discovery/calibration/audit counts must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Virtual Adam beta values must be in [0, 1)")

        families = _target_families(_get(config, "target_modules", "all-linear"))
        self.module_names: dict[int, str] = {}
        self.module_shapes: dict[str, tuple[int, int]] = {}
        for name, module in model.named_modules():
            if (
                not isinstance(module, torch.nn.Linear)
                or name.rsplit(".", 1)[-1] not in families
            ):
                continue
            stable_name = normalize_target_name(name)
            self.module_names[id(module)] = stable_name
            self.module_shapes[stable_name] = (
                int(module.out_features),
                int(module.in_features),
            )
        if not self.module_names:
            raise ValueError("Windowed full-gradient collector found no target Linear modules")
        if any(min(shape) < self.rank for shape in self.module_shapes.values()):
            raise ValueError("full_gradient_probe_rank exceeds a target module dimension")

        self.first_moments: dict[str, torch.Tensor] = {}
        self.second_moments: dict[str, torch.Tensor] = {}
        self.local_bases: dict[str, dict[str, list[torch.Tensor]]] = {
            source: {name: [] for name in self.module_shapes}
            for source in ("raw_momentum", "adam_update")
        }
        self.local_values: dict[str, dict[str, list[torch.Tensor]]] = {
            source: {name: [] for name in self.module_shapes}
            for source in ("raw_momentum", "adam_update")
        }
        self.candidate_sets: dict[str, dict[str, torch.Tensor]] = {}
        self.spectra: dict[str, dict[str, torch.Tensor]] = {}
        self.recurrence: dict[str, dict[str, torch.Tensor]] = {}
        self.split_stats: dict[str, dict[str, dict[str, dict[str, torch.Tensor]]]] = {}
        self.discovery_count = 0
        self.calibration_count = 0
        self.audit_count = 0
        self.prompt_ids: list[str] = []
        self.window_prompt_ids: list[list[str]] = []
        self.losses: list[float] = []
        self.response_counts: list[int] = []
        self.response_tokens: list[int] = []
        self.selected_response_offsets: list[int] = []
        self.advantage_rms: list[float] = []
        self.raw_norms: list[float] = []
        self.scales: list[float] = []
        self.total_prompts = 0
        self.total_rollouts = 0
        self.positive_rollouts = 0
        self.ready = False

    @property
    def phase(self) -> str:
        if self.discovery_count < self.discovery_target:
            return "discovery"
        if self.calibration_count < self.calibration_target:
            return "calibration"
        if self.audit_count < self.audit_target:
            return "audit"
        return "complete"

    @property
    def window_index(self) -> int:
        return self.discovery_count + self.calibration_count + self.audit_count

    def _units_and_contexts(self):
        if isinstance(self.model, FSDP):
            units = _leaf_fsdp_modules(self.model) or [self.model]
            return [
                (unit, FSDP.summon_full_params(unit, writeback=False, with_grads=True))
                for unit in units
            ]
        return [(self.model, nullcontext())]

    def record_rollout_batch(self, meta: dict[str, Any]) -> None:
        self.total_prompts += int(meta.get("full_gradient_total_prompts", 0))
        self.total_rollouts += int(meta.get("full_gradient_total_rollouts", 0))
        self.positive_rollouts += int(meta.get("full_gradient_positive_rollouts", 0))

    def _collect_gradients(self) -> tuple[dict[str, torch.Tensor], float]:
        gradients: dict[str, torch.Tensor] = {}
        found: set[str] = set()
        norm_square = 0.0
        for unit, context in self._units_and_contexts():
            with context:
                for module in unit.modules():
                    if id(module) not in self.module_names:
                        continue
                    name = self.module_names[id(module)]
                    if name in found:
                        raise RuntimeError(f"Windowed probe encountered {name} more than once")
                    found.add(name)
                    gradient = module.weight.grad
                    if gradient is None or tuple(gradient.shape) != self.module_shapes[name]:
                        raise RuntimeError(f"Windowed probe is missing weight gradient for {name}")
                    value = gradient.detach().float()
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"Non-finite full gradient for {name}")
                    norm_square += float(value.square().sum().item())
                    gradients[name] = value.cpu().contiguous()
        if len(found) != len(self.module_names):
            missing = sorted(set(self.module_names.values()) - found)
            raise RuntimeError(
                f"Windowed probe captured {len(found)}/{len(self.module_names)} modules; "
                f"missing={missing[:5]}"
            )
        return gradients, math.sqrt(norm_square)

    def _scale_for_norm(self, raw_norm: float) -> float:
        discovery_reference = self.raw_norms[: self.discovery_target] or [raw_norm]
        threshold = max(
            statistics.median(discovery_reference) * self.clip_factor, self.eps
        )
        scale = min(1.0, threshold / max(raw_norm, self.eps))
        self.raw_norms.append(raw_norm)
        self.scales.append(scale)
        return scale

    def _extract_local_atoms(
        self, name: str, matrix: torch.Tensor, purpose: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis, values, _ = _right_singular_basis(
            matrix,
            self.local_atoms,
            oversample=self.oversample,
            niter=self.svd_niter,
            seed=_stable_seed(self.seed, name, f"{purpose}_{self.discovery_count}"),
            device=self.svd_device,
        )
        return basis, values.square()

    @staticmethod
    def _comparison_windows(window: int, count: int) -> list[int]:
        non_adjacent = [other for other in range(count) if abs(other - window) > 1]
        return non_adjacent or [other for other in range(count) if other != window]

    def _aggregate_consensus(
        self,
        name: str,
        method: str,
        bases: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        supports: list[torch.Tensor] = []
        for window, basis in enumerate(bases):
            comparisons = self._comparison_windows(window, len(bases))
            if not comparisons:
                supports.append(torch.ones(basis.shape[0], dtype=torch.float32))
                continue
            per_other = []
            for other in comparisons:
                per_other.append((basis @ bases[other].T).square().amax(dim=1))
            supports.append(torch.stack(per_other).mean(dim=0))

        weighted_rows = []
        for basis, singular_energy, support in zip(bases, values, supports):
            relative = singular_energy / (singular_energy.sum() + self.eps)
            weight = relative * support.clamp_min(self.support_floor)
            weighted_rows.append(basis * weight.sqrt().unsqueeze(1))
        pooled = torch.cat(weighted_rows, dim=0)
        _, singular_values, vh = torch.linalg.svd(pooled, full_matrices=False)
        basis = _orthonormal_completion(
            vh[: self.rank],
            self.rank,
            seed=_stable_seed(self.seed, name, f"{method}_completion"),
        )
        spectrum = torch.zeros(self.rank, dtype=torch.float32)
        available = min(self.rank, singular_values.numel())
        spectrum[:available] = singular_values[:available].square().float()
        final_support = []
        for window_basis in bases:
            span = torch.linalg.qr(window_basis.T, mode="reduced").Q.T
            final_support.append((basis @ span.T).square().sum(dim=1).clamp_max(1.0))
        recurrence = torch.stack(final_support).mean(dim=0)
        return basis.contiguous(), spectrum.contiguous(), recurrence.contiguous()

    @torch.no_grad()
    def _finish_discovery(self) -> None:
        self.candidate_sets = {method: {} for method in _WINDOWED_CANDIDATE_METHODS}
        self.spectra = {method: {} for method in _WINDOWED_CANDIDATE_METHODS}
        self.recurrence = {method: {} for method in _WINDOWED_CANDIDATE_METHODS}
        for name in sorted(self.module_shapes):
            raw_bases = self.local_bases["raw_momentum"][name]
            raw_values = self.local_values["raw_momentum"][name]
            adam_bases = self.local_bases["adam_update"][name]
            adam_values = self.local_values["adam_update"][name]
            method_inputs = {
                "raw_momentum": (raw_bases, raw_values),
                "adam_update": (adam_bases, adam_values),
                "consensus_hybrid": (
                    [
                        torch.cat((raw, adam), dim=0)
                        for raw, adam in zip(raw_bases, adam_bases)
                    ],
                    [
                        torch.cat(
                            (
                                raw_value / (raw_value.sum() + self.eps),
                                adam_value / (adam_value.sum() + self.eps),
                            )
                        )
                        for raw_value, adam_value in zip(raw_values, adam_values)
                    ],
                ),
            }
            for method, (bases, values) in method_inputs.items():
                basis, spectrum, recurrence = self._aggregate_consensus(
                    name, method, bases, values
                )
                self.candidate_sets[method][name] = basis
                self.spectra[method][name] = spectrum
                self.recurrence[method][name] = recurrence

        self.first_moments.clear()
        self.second_moments.clear()
        for split in ("calibration", "audit"):
            self.split_stats[split] = {}
            for method in _WINDOWED_CANDIDATE_METHODS:
                self.split_stats[split][method] = {}
                for name, basis in self.candidate_sets[method].items():
                    out_features = self.module_shapes[name][0]
                    self.split_stats[split][method][name] = {
                        "sum": torch.zeros((out_features, self.rank), dtype=torch.float64),
                        "f": torch.zeros(self.rank, dtype=torch.float64),
                        "relative_sum": torch.zeros(self.rank, dtype=torch.float64),
                        "relative_square_sum": torch.zeros(self.rank, dtype=torch.float64),
                    }

    @torch.no_grad()
    def _add_held_out(
        self, split: str, gradients: dict[str, torch.Tensor], scale: float
    ) -> None:
        for method in _WINDOWED_CANDIDATE_METHODS:
            for name, raw_gradient in gradients.items():
                basis = self.candidate_sets[method][name]
                projected = (raw_gradient * scale) @ basis.T
                energy = projected.double().square().sum(dim=0)
                relative = energy / (energy.sum() + self.eps)
                stats = self.split_stats[split][method][name]
                stats["sum"].add_(projected.double())
                stats["f"].add_(energy)
                stats["relative_sum"].add_(relative)
                stats["relative_square_sum"].add_(relative.square())

    def capture_window(
        self,
        *,
        loss: float,
        prompt_ids: list[str],
        response_counts: list[int],
        response_tokens: list[int],
        selected_response_offsets: list[int],
        advantage_rms: list[float],
    ) -> dict[str, float]:
        if self.ready:
            return self.metrics()
        lengths = {
            len(prompt_ids),
            len(response_counts),
            len(response_tokens),
            len(selected_response_offsets),
            len(advantage_rms),
        }
        if lengths != {self.window_prompts}:
            raise ValueError(
                f"Expected exactly {self.window_prompts} prompt observations, got {sorted(lengths)}"
            )
        current_phase = self.phase
        gradients, raw_norm = self._collect_gradients()
        scale = self._scale_for_norm(raw_norm)
        self.prompt_ids.extend(str(item) for item in prompt_ids)
        self.window_prompt_ids.append([str(item) for item in prompt_ids])
        self.losses.append(float(loss))
        self.response_counts.extend(int(item) for item in response_counts)
        self.response_tokens.extend(int(item) for item in response_tokens)
        self.selected_response_offsets.extend(int(item) for item in selected_response_offsets)
        self.advantage_rms.extend(float(item) for item in advantage_rms)

        if current_phase == "discovery":
            adam_step = self.discovery_count + 1
            for name, raw_gradient in gradients.items():
                gradient = raw_gradient * scale
                first = self.first_moments.setdefault(name, torch.zeros_like(gradient))
                second = self.second_moments.setdefault(name, torch.zeros_like(gradient))
                adam_update = virtual_adam_update(
                    first,
                    second,
                    gradient,
                    step=adam_step,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.adam_eps,
                )
                momentum = first / (1.0 - self.beta1**adam_step)
                for source, matrix in (
                    ("raw_momentum", momentum),
                    ("adam_update", adam_update),
                ):
                    basis, values = self._extract_local_atoms(name, matrix, source)
                    self.local_bases[source][name].append(basis)
                    self.local_values[source][name].append(values)
            self.discovery_count += 1
            if self.discovery_count == self.discovery_target:
                self._finish_discovery()
        elif current_phase == "calibration":
            self._add_held_out("calibration", gradients, scale)
            self.calibration_count += 1
        elif current_phase == "audit":
            self._add_held_out("audit", gradients, scale)
            self.audit_count += 1

        if self.phase == "complete":
            self.ready = True
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if not distributed or torch.distributed.get_rank() == 0:
                self._export()
            if distributed:
                torch.distributed.barrier()
        return self.metrics()

    def metrics(self) -> dict[str, float]:
        return {
            "full_gradient_probe/discovery_windows": float(self.discovery_count),
            "full_gradient_probe/calibration_windows": float(self.calibration_count),
            "full_gradient_probe/audit_windows": float(self.audit_count),
            "full_gradient_probe/selected_prompts": float(len(self.prompt_ids)),
            "full_gradient_probe/total_prompts": float(self.total_prompts),
            "full_gradient_probe/total_rollouts": float(self.total_rollouts),
            "full_gradient_probe/positive_rate": float(
                self.positive_rollouts / max(self.total_rollouts, 1)
            ),
            "full_gradient_probe/artifact_ready": float(self.ready),
        }

    def _score_tensors(
        self, split: str, method: str, name: str, count: int
    ) -> dict[str, torch.Tensor]:
        stats = self.split_stats[split][method][name]
        f_score = stats["f"] / count
        mean_projected = stats["sum"] / count
        s_score = mean_projected.square().sum(dim=0)
        r_score = s_score / (f_score + self.eps)
        p_score = stats["relative_sum"] / count
        if count > 1:
            variance = (
                stats["relative_square_sum"]
                - stats["relative_sum"].square() / count
            ) / (count - 1)
            p_se = variance.clamp_min(0).sqrt() / math.sqrt(count)
        else:
            p_se = torch.zeros_like(p_score)
        p_lcb = p_score - self.confidence_z * p_se
        recurrence = self.recurrence[method][name].double()
        return {
            "F": f_score.float(),
            "S": s_score.float(),
            "R": r_score.float(),
            "P": p_score.float(),
            "P_se": p_se.float(),
            "P_lcb": p_lcb.float(),
            "recurrence": recurrence.float(),
            "U": (recurrence * p_lcb.clamp_min(0)).float(),
        }

    def _export(self) -> None:
        from safetensors.torch import save_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths: dict[str, dict[str, str]] = {}
        module_summary: dict[str, dict[str, Any]] = {}
        for method in _WINDOWED_CANDIDATE_METHODS:
            candidate_path = self.output_dir / f"candidates_{method}.safetensors"
            score_path = self.output_dir / f"atom_scores_{method}.safetensors"
            score_tensors: dict[str, torch.Tensor] = {}
            for name in sorted(self.candidate_sets[method]):
                calibration = self._score_tensors(
                    "calibration", method, name, self.calibration_count
                )
                audit = self._score_tensors("audit", method, name, self.audit_count)
                for label, value in calibration.items():
                    score_tensors[f"{name}.{label}"] = value.contiguous()
                for label, value in audit.items():
                    score_tensors[f"{name}.audit_{label}"] = value.contiguous()
                score_tensors[f"{name}.spectrum"] = self.spectra[method][name]
                if name not in module_summary:
                    out_features, in_features = self.module_shapes[name]
                    module_summary[name] = {
                        "shape": [out_features, in_features],
                        "candidate_rank": self.rank,
                    }
                basis = self.candidate_sets[method][name]
                module_summary[name][f"{method}_orthogonality_error"] = float(
                    (basis @ basis.T - torch.eye(self.rank)).abs().max().item()
                )
                module_summary[name][f"{method}_calibration_u_sum"] = float(
                    calibration["U"].sum().item()
                )
                module_summary[name][f"{method}_audit_u_sum"] = float(
                    audit["U"].sum().item()
                )
            for path, tensors in (
                (candidate_path, self.candidate_sets[method]),
                (score_path, score_tensors),
            ):
                temporary = path.with_suffix(path.suffix + ".tmp")
                save_file(
                    {
                        key: value.float().cpu().contiguous()
                        for key, value in tensors.items()
                    },
                    str(temporary),
                )
                os.replace(temporary, path)
            artifact_paths[method] = {
                "candidates": str(candidate_path),
                "atom_scores": str(score_path),
            }

        discovery_end = self.discovery_target
        calibration_end = discovery_end + self.calibration_target
        summary = {
            "schema_version": 2,
            "method": "windowed_adam_consensus_single_rollout_v1",
            "candidate_methods": list(_WINDOWED_CANDIDATE_METHODS),
            "score_labels": [
                "F", "S", "R", "P", "P_se", "P_lcb", "recurrence", "U"
            ],
            "r_max": self.rank,
            "constant_scaling": 2.0,
            "loss": "single-response unbiased signed-GRPO policy-gradient estimate",
            "aggregation": "prompt mean within fixed temporal windows",
            "estimator": {
                "rollouts_generated_per_prompt": "configured rollout.n",
                "responses_backward_per_prompt": 1,
                "inverse_probability_correction": "response_count/window_prompts",
                "selection": "deterministic uniform hash",
            },
            "virtual_adam": {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "eps": self.adam_eps,
                "bias_correction": True,
            },
            "window_prompts": self.window_prompts,
            "local_atoms": self.local_atoms,
            "discovery_windows": self.discovery_count,
            "calibration_windows": self.calibration_count,
            "audit_windows": self.audit_count,
            "discovery_prompt_ids": [
                item for window in self.window_prompt_ids[:discovery_end] for item in window
            ],
            "calibration_prompt_ids": [
                item
                for window in self.window_prompt_ids[discovery_end:calibration_end]
                for item in window
            ],
            "audit_prompt_ids": [
                item for window in self.window_prompt_ids[calibration_end:] for item in window
            ],
            "window_prompt_ids": self.window_prompt_ids,
            "prompt_splits_disjoint": len(set(self.prompt_ids)) == len(self.prompt_ids),
            "total_prompts": self.total_prompts,
            "total_rollouts": self.total_rollouts,
            "positive_rollouts": self.positive_rollouts,
            "positive_rate": self.positive_rollouts / max(self.total_rollouts, 1),
            "future_lcb_z": self.confidence_z,
            "sample_norm_control": {
                "method": "global-window-gradient discovery-median-clip",
                "clip_factor": self.clip_factor,
                "raw_norms": self.raw_norms,
                "scales": self.scales,
            },
            "samples": {
                "window_losses": self.losses,
                "response_counts": self.response_counts,
                "response_tokens": self.response_tokens,
                "selected_response_offsets": self.selected_response_offsets,
                "advantage_rms": self.advantage_rms,
            },
            "selection_uses_audit": False,
            "held_out_scoring_space": "raw_full_policy_gradient",
            "artifacts": artifact_paths,
            "modules": module_summary,
        }
        path = self.output_dir / "probe_summary.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)


def build_full_gradient_probe_collector(
    model: torch.nn.Module, config: Any
) -> (
    FullGradientRLProbeCollector
    | WindowedAdamConsensusCollector
    | Phase0GradientDiagnosticsCollector
    | Phase06CrossfitReplayCollector
):
    mode = str(_get(config, "full_gradient_probe_mode", "legacy")).lower()
    if mode == "legacy":
        return FullGradientRLProbeCollector(model, config)
    if mode == "windowed_adam_consensus":
        return WindowedAdamConsensusCollector(model, config)
    if mode == "phase0_diagnostics":
        return Phase0GradientDiagnosticsCollector(model, config)
    if mode == "phase06_crossfit_replay":
        return Phase06CrossfitReplayCollector(model, config)
    raise ValueError(f"Unknown full_gradient_probe_mode: {mode}")


def validate_full_gradient_probe_artifact(
    artifact_dir: str | Path,
    *,
    candidate_method: str = "mean",
    atol: float = 3e-4,
) -> dict[str, float]:
    """Validate one candidate family and all held-out scores entirely on CPU."""

    from safetensors import safe_open

    artifact_dir = Path(artifact_dir).expanduser().resolve()
    summary = json.loads(
        (artifact_dir / "probe_summary.json").read_text(encoding="utf-8")
    )
    candidate_methods = tuple(summary.get("candidate_methods", _CANDIDATE_METHODS))
    if candidate_method not in candidate_methods:
        raise ValueError(f"Unknown candidate method: {candidate_method}")
    if not summary.get("prompt_splits_disjoint", False):
        raise ValueError("Full-gradient prompt splits overlap")
    candidate_path = artifact_dir / f"candidates_{candidate_method}.safetensors"
    score_path = artifact_dir / f"atom_scores_{candidate_method}.safetensors"
    maximum_error = 0.0
    with safe_open(candidate_path, framework="pt", device="cpu") as candidates:
        with safe_open(score_path, framework="pt", device="cpu") as scores:
            for name, item in summary["modules"].items():
                basis = candidates.get_tensor(name).float()
                expected = (int(item["candidate_rank"]), int(item["shape"][1]))
                if tuple(basis.shape) != expected:
                    raise ValueError(
                        f"Invalid candidate shape for {name}: {tuple(basis.shape)} != {expected}"
                    )
                if not bool(torch.isfinite(basis).all()):
                    raise ValueError(
                        f"Non-finite {candidate_method} candidates for {name}"
                    )
                error = float(
                    (basis @ basis.T - torch.eye(basis.shape[0])).abs().max().item()
                )
                if error > atol:
                    raise ValueError(
                        f"Non-orthogonal {candidate_method} candidates for {name}: {error}"
                    )
                maximum_error = max(maximum_error, error)
                score_labels = tuple(
                    summary.get(
                        "score_labels",
                        (
                            "F",
                            "S",
                            "R",
                            "P",
                            "gain",
                            "gain_se",
                            "gain_lcb",
                            "adam_gain",
                            "U",
                        ),
                    )
                )
                for label in (*score_labels, *(f"audit_{key}" for key in score_labels), "spectrum"):
                    value = scores.get_tensor(f"{name}.{label}")
                    if tuple(value.shape) != (basis.shape[0],) or not bool(
                        torch.isfinite(value).all()
                    ):
                        raise ValueError(
                            f"Invalid {candidate_method} score {label} for {name}"
                        )
                diagnostic = (
                    "discovery_capture" if candidate_method == "mean"
                    else "nystrom_residual" if candidate_method == "covariance"
                    else None
                )
                if diagnostic is not None:
                    value = scores.get_tensor(f"{name}.{diagnostic}")
                    if tuple(value.shape) != (1,) or not bool(torch.isfinite(value).all()):
                        raise ValueError(
                            f"Invalid {candidate_method} diagnostic {diagnostic} for {name}"
                        )
    return {
        "module_count": float(len(summary["modules"])),
        "orthogonality_error_max": maximum_error,
    }
