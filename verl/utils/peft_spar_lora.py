"""Stable positive-rollout probe and static atom allocation for SPAR-LoRA."""

from __future__ import annotations

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
    GradientSubspaceAccumulator,
    _factor_basis_and_values,
    _fill_probe_factors,
    _get,
    _leaf_fsdp_modules,
    normalize_target_name,
)


def initialize_spar_probe(model, config: Any, adapter_name: str = "default") -> dict[str, float]:
    """Create a zero-function rank-r probe with deterministic random B factors."""

    from peft.tuners.lora.layer import Linear as LoraLinear

    rank = int(_get(config, "spar_r_max", 32))
    seed = int(_get(config, "gradient_probe_seed", 42))
    scaling = float(_get(config, "gradient_subspace_scaling", 2.0))
    if rank <= 0:
        raise ValueError("spar_r_max must be positive")
    if scaling <= 0:
        raise ValueError("gradient_subspace_scaling must be positive")

    count = 0
    b_rms = []
    for name, module in model.named_modules():
        if not isinstance(module, LoraLinear) or adapter_name not in module.lora_A:
            continue
        if module.lora_A[adapter_name].weight.shape[0] != rank:
            raise ValueError(f"SPAR probe rank mismatch for {normalize_target_name(name)}")
        observed_scaling = float(module.scaling[adapter_name])
        if not math.isclose(observed_scaling, scaling, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"SPAR probe scaling mismatch for {normalize_target_name(name)}: "
                f"{observed_scaling} != {scaling}"
            )
        _, rms = _fill_probe_factors(
            module,
            name=normalize_target_name(name),
            adapter_name=adapter_name,
            seed=seed,
            observation=0,
        )
        b_rms.append(rms)
        count += 1
    if not count:
        raise ValueError("SPAR probe found no PEFT LoRA Linear layers")
    return {
        "num_layers": float(count),
        "rank": float(rank),
        "scaling": scaling,
        "b_rms_mean": float(sum(b_rms) / len(b_rms)),
    }


class SparPositiveProbeCollector:
    """Collect per-sample CE gradients without retaining dense weight gradients."""

    def __init__(self, model: torch.nn.Module, config: Any, adapter_name: str = "default") -> None:
        from peft.tuners.lora.layer import Linear as LoraLinear

        self.model = model
        self.config = config
        self.adapter_name = adapter_name
        self.r_max = int(_get(config, "spar_r_max", 32))
        self.discovery_target = int(_get(config, "spar_discovery_samples", 32))
        self.calibration_target = int(_get(config, "spar_calibration_samples", 32))
        self.clip_factor = float(_get(config, "spar_sample_clip_factor", 2.5))
        self.eps = float(_get(config, "spar_score_eps", 1e-12))
        self.scaling = float(_get(config, "gradient_subspace_scaling", 2.0))
        output = str(_get(config, "gradient_probe_output_dir", ""))
        if not output:
            raise ValueError("gradient_probe_output_dir is required for peft_type=spar_probe")
        self.output_dir = Path(output).expanduser().resolve()
        if self.discovery_target <= 0 or self.calibration_target <= 0:
            raise ValueError("SPAR discovery and calibration sample targets must be positive")
        if self.clip_factor < 1:
            raise ValueError("spar_sample_clip_factor must be at least 1")

        self.module_names: dict[int, str] = {}
        self.module_shapes: dict[str, tuple[int, int]] = {}
        for name, candidate in model.named_modules():
            if not isinstance(candidate, LoraLinear) or adapter_name not in candidate.lora_A:
                continue
            stable_name = normalize_target_name(name)
            self.module_names[id(candidate)] = stable_name
            self.module_shapes[stable_name] = (int(candidate.out_features), int(candidate.in_features))
        if not self.module_names:
            raise ValueError("SPAR collector found no PEFT LoRA Linear layers")

        self.discovery = GradientSubspaceAccumulator(
            capacity=self.r_max,
            target_energy=1.0,
            rank_bins=[self.r_max],
            scaling_ratio=self.scaling,
        )
        self.candidates: dict[str, torch.Tensor] = {}
        self.singular_values: dict[str, torch.Tensor] = {}
        self.calibration_f: dict[str, torch.Tensor] = {}
        self.calibration_sum: dict[str, torch.Tensor] = {}
        self.discovery_count = 0
        self.calibration_count = 0
        self.sample_norms: list[float] = []
        self.sample_scales: list[float] = []
        self.sample_losses: list[float] = []
        self.selected_prompt_ids: list[str] = []
        self.selected_lengths: list[int] = []
        self.total_prompts = 0
        self.total_rollouts = 0
        self.positive_rollouts = 0
        self.ready = False

    def _units_and_contexts(self):
        if isinstance(self.model, FSDP):
            units = _leaf_fsdp_modules(self.model) or [self.model]
            return [(unit, FSDP.summon_full_params(unit, writeback=True, with_grads=True)) for unit in units]
        return [(self.model, nullcontext())]

    def record_rollout_batch(self, meta: dict[str, Any]) -> None:
        self.total_prompts += int(meta.get("spar_total_prompts", 0))
        self.total_rollouts += int(meta.get("spar_total_rollouts", 0))
        self.positive_rollouts += int(meta.get("spar_positive_rollouts", 0))

    def _collect_gradients(self, factor: str) -> dict[str, torch.Tensor]:
        from peft.tuners.lora.layer import Linear as LoraLinear

        values: dict[str, torch.Tensor] = {}
        found: set[str] = set()
        for unit, context in self._units_and_contexts():
            with context:
                for candidate in unit.modules():
                    if not isinstance(candidate, LoraLinear) or id(candidate) not in self.module_names:
                        continue
                    name = self.module_names[id(candidate)]
                    if name in found:
                        raise RuntimeError(f"SPAR probe encountered module {name} more than once")
                    found.add(name)
                    parameter = (
                        candidate.lora_A[self.adapter_name].weight
                        if factor == "A"
                        else candidate.lora_B[self.adapter_name].weight
                    )
                    if parameter.grad is None or parameter.grad.shape != parameter.shape:
                        raise RuntimeError(f"SPAR probe is missing {factor} gradient for {name}")
                    value = parameter.grad.detach().float()
                    if factor == "A":
                        value = value.T
                    value = value / float(candidate.scaling[self.adapter_name])
                    value = value.cpu().contiguous()
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"Non-finite SPAR {factor} gradient for {name}")
                    values[name] = value
        if len(found) != len(self.module_names):
            missing = sorted(set(self.module_names.values()) - found)
            raise RuntimeError(f"SPAR captured {len(found)}/{len(self.module_names)} modules; missing={missing[:5]}")
        return values

    @torch.no_grad()
    def _finish_discovery(self) -> None:
        self.discovery.compress()
        _, bases, _ = self.discovery.snapshot()
        for name, basis in bases.items():
            factor_basis, values = _factor_basis_and_values(self.discovery.factors[name])
            available = min(self.r_max, values.numel())
            padded = torch.zeros(self.r_max, dtype=torch.float32)
            padded[:available] = values[:available]
            self.singular_values[name] = padded
            self.candidates[name] = basis[:, : self.r_max].T.contiguous()
            if factor_basis.shape[1] < self.r_max:
                raise RuntimeError(f"SPAR candidate basis for {name} is incomplete")

        found: set[str] = set()
        for unit, context in self._units_and_contexts():
            with context:
                for candidate in unit.modules():
                    if id(candidate) not in self.module_names:
                        continue
                    name = self.module_names[id(candidate)]
                    found.add(name)
                    a = candidate.lora_A[self.adapter_name].weight
                    b = candidate.lora_B[self.adapter_name].weight
                    a.copy_(self.candidates[name].to(device=a.device, dtype=a.dtype))
                    b.zero_()
                    self.calibration_f[name] = torch.zeros(self.r_max, dtype=torch.float64)
                    self.calibration_sum[name] = torch.zeros(
                        (self.module_shapes[name][0], self.r_max), dtype=torch.float64
                    )
        if len(found) != len(self.module_names):
            raise RuntimeError("SPAR failed to install all discovery candidates")

    def capture_sample(self, *, loss: float, prompt_id: str, response_length: int) -> dict[str, float]:
        """Capture one synchronized sample gradient and advance the probe phase."""

        if self.ready:
            return self.metrics()
        self.sample_losses.append(float(loss))
        self.selected_prompt_ids.append(str(prompt_id))
        self.selected_lengths.append(int(response_length))

        if self.discovery_count < self.discovery_target:
            sketches = self._collect_gradients("A")
            sample_norm = math.sqrt(sum(float(value.square().sum().item()) for value in sketches.values()))
            prior = self.sample_norms or [sample_norm]
            threshold = max(statistics.median(prior) * self.clip_factor, self.eps)
            scale = min(1.0, threshold / max(sample_norm, self.eps))
            self.sample_norms.append(sample_norm)
            self.sample_scales.append(scale)
            for name, sketch in sketches.items():
                self.discovery.add(name, sketch * scale)
            self.discovery.compress()
            self.discovery_count += 1
            if self.discovery_count == self.discovery_target:
                self._finish_discovery()
        else:
            gradients = self._collect_gradients("B")
            for name, gradient in gradients.items():
                value = gradient.double()
                self.calibration_f[name].add_(value.square().sum(dim=0))
                self.calibration_sum[name].add_(value)
            self.calibration_count += 1
            if self.calibration_count == self.calibration_target:
                self.ready = True
                distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                if not distributed or torch.distributed.get_rank() == 0:
                    self._export()
                if distributed:
                    # Every actor rank processes the same selected sample.  Keep
                    # them aligned while rank zero atomically publishes the one
                    # shared CPU artifact.
                    torch.distributed.barrier()
        return self.metrics()

    def metrics(self) -> dict[str, float]:
        total_selected = self.discovery_count + self.calibration_count
        return {
            "spar_probe/discovery_samples": float(self.discovery_count),
            "spar_probe/calibration_samples": float(self.calibration_count),
            "spar_probe/selected_samples": float(total_selected),
            "spar_probe/total_prompts": float(self.total_prompts),
            "spar_probe/total_rollouts": float(self.total_rollouts),
            "spar_probe/positive_rollouts": float(self.positive_rollouts),
            "spar_probe/positive_rate": float(self.positive_rollouts / max(self.total_rollouts, 1)),
            "spar_probe/artifact_ready": float(self.ready),
        }

    def _export(self) -> None:
        from safetensors.torch import save_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = self.output_dir / "candidates.safetensors"
        score_path = self.output_dir / "atom_scores.safetensors"
        score_tensors: dict[str, torch.Tensor] = {}
        module_summary: dict[str, dict[str, Any]] = {}
        for name in sorted(self.candidates):
            f_score = self.calibration_f[name] / self.calibration_count
            mean_gradient = self.calibration_sum[name] / self.calibration_count
            s_score = mean_gradient.square().sum(dim=0)
            r_score = s_score / (f_score + self.eps)
            p_score = f_score / (f_score.sum() + self.eps)
            u_score = p_score * r_score
            for label, value in (
                ("F", f_score),
                ("S", s_score),
                ("R", r_score),
                ("P", p_score),
                ("U", u_score),
                ("singular_values", self.singular_values[name]),
            ):
                score_tensors[f"{name}.{label}"] = value.float().contiguous()
            out_features, in_features = self.module_shapes[name]
            module_summary[name] = {
                "shape": [out_features, in_features],
                "candidate_rank": self.r_max,
                "candidate_orthogonality_error": float(
                    (self.candidates[name].float() @ self.candidates[name].float().T - torch.eye(self.r_max))
                    .abs()
                    .max()
                    .item()
                ),
                "calibration_energy": float(f_score.sum().item()),
                "stable_signal": float(s_score.sum().item()),
                "u_total": float(u_score.sum().item()),
            }

        for path, tensors in ((candidate_path, self.candidates), (score_path, score_tensors)):
            temporary = path.with_suffix(path.suffix + ".tmp")
            save_file({key: value.float().cpu().contiguous() for key, value in tensors.items()}, str(temporary))
            os.replace(temporary, path)

        lengths = self.selected_lengths
        ordered_lengths = sorted(lengths)
        quantile = lambda q: float(ordered_lengths[round((len(ordered_lengths) - 1) * q)])
        summary = {
            "schema_version": 1,
            "method": "spar_lora_v0_positive_teacher_forced_probe",
            "r_max": self.r_max,
            "constant_scaling": self.scaling,
            "discovery_samples": self.discovery_count,
            "calibration_samples": self.calibration_count,
            "discovery_prompt_ids": self.selected_prompt_ids[: self.discovery_count],
            "calibration_prompt_ids": self.selected_prompt_ids[self.discovery_count :],
            "prompt_splits_disjoint": not bool(
                set(self.selected_prompt_ids[: self.discovery_count])
                & set(self.selected_prompt_ids[self.discovery_count :])
            ),
            "total_prompts": self.total_prompts,
            "total_rollouts": self.total_rollouts,
            "positive_rollouts": self.positive_rollouts,
            "positive_rate": self.positive_rollouts / max(self.total_rollouts, 1),
            "response_length": {
                "count": len(lengths),
                "mean": float(sum(lengths) / len(lengths)),
                "min": int(min(lengths)),
                "p50": quantile(0.50),
                "p90": quantile(0.90),
                "max": int(max(lengths)),
            },
            "teacher_forced_ce": {
                "aggregation": "token-mean-per-sample",
                "mean": float(sum(self.sample_losses) / len(self.sample_losses)),
            },
            "sample_norm_control": {
                "method": "global-projected-gradient-median-clip",
                "clip_factor": self.clip_factor,
                "norms": self.sample_norms,
                "scales": self.sample_scales,
            },
            "artifacts": {
                "candidates": str(candidate_path),
                "atom_scores": str(score_path),
            },
            "modules": module_summary,
        }
        path = self.output_dir / "probe_summary.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def validate_spar_artifact(artifact_dir: str | Path, *, atol: float = 2e-4) -> dict[str, float]:
    """Load and validate a completed SPAR artifact entirely on CPU."""

    from safetensors import safe_open

    artifact_dir = Path(artifact_dir).expanduser().resolve()
    summary = json.loads((artifact_dir / "probe_summary.json").read_text(encoding="utf-8"))
    if not summary.get("prompt_splits_disjoint", False):
        raise ValueError("SPAR discovery and calibration prompt sets overlap")
    maximum_error = 0.0
    module_count = 0
    with safe_open(artifact_dir / "candidates.safetensors", framework="pt", device="cpu") as candidates:
        with safe_open(artifact_dir / "atom_scores.safetensors", framework="pt", device="cpu") as scores:
            for name, item in summary["modules"].items():
                a = candidates.get_tensor(name).float()
                expected = (int(item["candidate_rank"]), int(item["shape"][1]))
                if tuple(a.shape) != expected:
                    raise ValueError(f"Invalid SPAR candidate shape for {name}: {tuple(a.shape)} != {expected}")
                error = float((a @ a.T - torch.eye(a.shape[0])).abs().max().item())
                if error > atol:
                    raise ValueError(f"Non-orthogonal SPAR candidates for {name}: error={error}")
                maximum_error = max(maximum_error, error)
                for label in ("F", "S", "R", "P", "U", "singular_values"):
                    value = scores.get_tensor(f"{name}.{label}")
                    if tuple(value.shape) != (a.shape[0],) or not bool(torch.isfinite(value).all()):
                        raise ValueError(f"Invalid SPAR score {label} for {name}")
                module_count += 1
    return {"module_count": float(module_count), "orthogonality_error_max": maximum_error}
