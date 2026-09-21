"""Full-weight signed-GRPO probe for static LoRA subspace discovery."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
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
            covariance_basis, covariance_values, covariance_residual = (
                _nystrom_covariance_basis(
                    self.covariance_y[name] / self.discovery_count,
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
            "candidate_methods": list(_CANDIDATE_METHODS),
            "r_max": self.rank,
            "constant_scaling": 2.0,
            "loss": "prompt-group signed GRPO policy gradient at ratio=1",
            "aggregation": "token-mean within prompt group",
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


def validate_full_gradient_probe_artifact(
    artifact_dir: str | Path,
    *,
    candidate_method: str = "mean",
    atol: float = 3e-4,
) -> dict[str, float]:
    """Validate one candidate family and all held-out scores entirely on CPU."""

    from safetensors import safe_open

    if candidate_method not in _CANDIDATE_METHODS:
        raise ValueError(f"Unknown candidate method: {candidate_method}")
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    summary = json.loads(
        (artifact_dir / "probe_summary.json").read_text(encoding="utf-8")
    )
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
                for label in (
                    "F",
                    "S",
                    "R",
                    "P",
                    "gain",
                    "gain_se",
                    "gain_lcb",
                    "adam_gain",
                    "U",
                ):
                    value = scores.get_tensor(f"{name}.{label}")
                    if tuple(value.shape) != (basis.shape[0],) or not bool(
                        torch.isfinite(value).all()
                    ):
                        raise ValueError(
                            f"Invalid {candidate_method} score {label} for {name}"
                        )
    return {
        "module_count": float(len(summary["modules"])),
        "orthogonality_error_max": maximum_error,
    }
