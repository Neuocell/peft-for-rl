"""Reward-gradient aligned LoRA (REAL), equations 6--9 and Algorithm 1.

Each observation is one complete prompt group. A zero-function LoRA probe
(``A=0, B=Psi``) obtains ``H=Psi.T @ G`` from ``A.grad / scaling`` without
materializing a dense weight gradient. Only ``Y=sum H.T @ (H @ Omega)`` and
the shared input sketch ``Omega`` persist between groups. The exported A
contains the leading Nystrom input directions; formal REAL freezes A and
starts training B at zero.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.peft_gradient_subspace import (
    GradientProbeInitStats,
    _fill_probe_factors,
    _get,
    _leaf_fsdp_modules,
    _stable_seed,
    normalize_target_name,
)


def real_probe_dimensions(config: Any) -> tuple[int, int, int]:
    """Return final rank, output width, and input width (paper default: 4r)."""

    rank = int(_get(config, "real_probe_rank", 32))
    output_width = int(_get(config, "real_probe_output_width", 0))
    input_width = int(_get(config, "real_probe_input_width", 0))
    if rank <= 0 or output_width < 0 or input_width < 0:
        raise ValueError("REAL rank must be positive and sketch widths non-negative")
    output_width = output_width or 4 * rank
    input_width = input_width or 4 * rank
    if input_width < rank:
        raise ValueError("real_probe_input_width must be at least real_probe_rank")
    return rank, output_width, input_width


def reward_contrast_coefficients(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    min_correct: int = 2,
    min_incorrect: int = 2,
) -> torch.Tensor:
    """Return ``advantage_i / (K * response_length_i)`` for paper equation 6.

    Rewards must be the binary correctness rewards of one complete group.
    Standardization uses the population standard deviation. Summing the
    masked log probabilities with these coefficients is the *sequence mean*
    objective, including when response lengths differ. A disallowed group
    raises ValueError so callers can explicitly count filtering decisions.
    """

    if rewards.ndim != 1 or response_mask.ndim != 2:
        raise ValueError("REAL expects a reward vector and a [responses, tokens] mask")
    if rewards.numel() != response_mask.shape[0] or rewards.numel() == 0:
        raise ValueError("REAL rewards and response mask must describe the same non-empty group")
    if min_correct < 1 or min_incorrect < 1:
        raise ValueError("REAL binary group minimum counts must be positive")
    values = rewards.detach().to(dtype=torch.float64 if rewards.dtype == torch.float64 else torch.float32)
    mask = response_mask.detach().to(device=values.device)
    if not bool(torch.isfinite(values).all()) or not bool(((values == 0) | (values == 1)).all()):
        raise ValueError("REAL correctness rewards must be finite and exactly binary (0 or 1)")
    if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("REAL response mask must be finite and binary")
    lengths = mask.to(dtype=values.dtype).sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("REAL responses must contain at least one unmasked token")
    correct = int((values == 1).sum().item())
    incorrect = values.numel() - correct
    if correct < min_correct or incorrect < min_incorrect:
        raise ValueError(
            f"REAL group needs >= {min_correct} correct and >= {min_incorrect} incorrect responses; "
            f"got {correct} and {incorrect}"
        )
    centered = values - values.mean()
    standard_deviation = centered.square().mean().sqrt()
    return centered / standard_deviation / values.numel() / lengths


@torch.no_grad()
def initialize_real_probe(model, config: Any, adapter_name: str = "default") -> GradientProbeInitStats:
    """Install independent output sketches; only zero-initialized A is trainable."""

    from peft.tuners.lora.layer import Linear as LoraLinear

    final_rank, width, _ = real_probe_dimensions(config)
    seed = int(_get(config, "real_probe_seed", 42))
    layers = [
        (normalize_target_name(name), module)
        for name, module in model.named_modules()
        if isinstance(module, LoraLinear) and adapter_name in module.lora_A
    ]
    if not layers:
        raise ValueError("REAL probe found no PEFT LoRA Linear layers")
    for name, module in layers:
        a_layer = module.lora_A[adapter_name]
        b_layer = module.lora_B[adapter_name]
        if a_layer.weight.shape[0] != width:
            raise ValueError(f"REAL probe {name} has width {a_layer.weight.shape[0]}, expected {width}")
        if final_rank > module.in_features:
            raise ValueError(f"REAL rank {final_rank} exceeds input dimension of {name}")
        if a_layer.bias is not None or b_layer.bias is not None or module.merged:
            raise ValueError("REAL requires unmerged, bias-free LoRA factors")
        if bool(getattr(module, "use_dora", {}).get(adapter_name, False)):
            raise ValueError("REAL gradient sketches require ordinary LoRA, without DoRA")
        scaling = float(module.scaling[adapter_name])
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError(f"REAL probe scaling must be finite and positive for {name}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    rms_values = []
    for name, module in layers:
        _, rms = _fill_probe_factors(
            module,
            name=f"real/output/{name}",
            adapter_name=adapter_name,
            seed=seed,
            observation=0,
        )
        module.lora_A[adapter_name].weight.requires_grad_(True)
        rms_values.append(rms)
    return GradientProbeInitStats(
        num_layers=len(layers), rank=width, a_abs_max=0.0, b_rms_mean=sum(rms_values) / len(rms_values)
    )


def nystrom_subspace(
    y: torch.Tensor,
    omega: torch.Tensor,
    rank: int,
    *,
    relative_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Recover rows of P.T from ``Y=C Omega`` using a PSD pseudoinverse.

    Factorization is done in double precision after rescaling Y, and every
    numerical threshold is relative to the observed spectrum. Thus changing
    the overall loss scale does not change which directions survive. Rank
    deficiency is an error; arbitrary/random completion would change REAL.
    """

    if y.ndim != 2 or y.shape != omega.shape or not y.numel():
        raise ValueError("Nystrom Y and Omega must be non-empty matrices of identical shape")
    if rank <= 0 or rank > min(y.shape):
        raise ValueError("Nystrom target rank must fit the input and sketch dimensions")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("Nystrom relative tolerance must lie in (0, 1)")
    if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(omega).all()):
        raise FloatingPointError("Nystrom sketches contain non-finite values")
    scale = float(y.abs().max().item())
    if scale == 0:
        raise ValueError("REAL sketch has zero energy; insufficient numerical rank")
    normalized = y.detach().to(device="cpu", dtype=torch.float64) / scale
    omega64 = omega.detach().to(device="cpu", dtype=torch.float64)
    core = omega64.T @ normalized
    core = (core + core.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(core)
    largest = float(eigenvalues.abs().max().item())
    tolerance = relative_tolerance * largest
    if largest == 0 or float(eigenvalues.max().item()) <= 0:
        raise ValueError("REAL Nystrom core has no positive energy; insufficient numerical rank")
    if float(eigenvalues.min().item()) < -tolerance:
        raise ValueError("REAL Nystrom core is not positive semidefinite")
    positive = eigenvalues > tolerance
    numerical_rank = int(positive.sum().item())
    if numerical_rank < rank:
        raise ValueError(f"REAL sketch has numerical rank {numerical_rank}, below requested rank {rank}")
    whitener = eigenvectors[:, positive] * eigenvalues[positive].rsqrt().unsqueeze(0)
    factor = normalized @ whitener
    left, values, _ = torch.linalg.svd(factor, full_matrices=False)
    spectrum = values.square() * scale
    factor_rank = int((spectrum > relative_tolerance * spectrum.max()).sum().item())
    if factor_rank < rank:
        raise ValueError(f"REAL recovered factor has numerical rank {factor_rank}, below requested rank {rank}")
    basis = left[:, :rank].T.to(dtype=torch.float32).contiguous()
    diagnostics = {
        "nystrom_core_rank": float(numerical_rank),
        "nystrom_factor_rank": float(factor_rank),
        "orthogonality_error": float((basis @ basis.T - torch.eye(rank)).abs().max().item()),
        "nystrom_energy": float(spectrum.sum().item()),
        "selected_nystrom_energy": float(spectrum[:rank].sum().item()),
    }
    return basis, spectrum, diagnostics


class RealSubspaceAccumulator:
    """Streaming, uncentered per-group second moment with O(d_in * s_in) buffers."""

    def __init__(self, *, rank: int, input_width: int, seed: int, relative_tolerance: float = 1e-6):
        if rank <= 0 or input_width < rank:
            raise ValueError("REAL accumulator requires 0 < rank <= input_width")
        self.rank = rank
        self.input_width = input_width
        self.seed = seed
        self.relative_tolerance = relative_tolerance
        self.omegas: dict[str, torch.Tensor] = {}
        self.sketches: dict[str, torch.Tensor] = {}
        self.energy_sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @torch.no_grad()
    def add(self, name: str, h: torch.Tensor) -> None:
        h = h.detach().to(device="cpu", dtype=torch.float32)
        if h.ndim != 2 or not h.numel() or h.shape[1] < self.rank:
            raise ValueError(f"Invalid REAL output sketch for {name}: {tuple(h.shape)}")
        if not bool(torch.isfinite(h).all()):
            raise FloatingPointError(f"Non-finite REAL output sketch for {name}")
        if name not in self.omegas:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(self.seed, f"real/input/{name}", 0))
            self.omegas[name] = torch.randn(h.shape[1], self.input_width, generator=generator)
            self.sketches[name] = torch.zeros_like(self.omegas[name])
            self.energy_sums[name] = 0.0
            self.counts[name] = 0
        if h.shape[1] != self.omegas[name].shape[0]:
            raise ValueError(f"REAL output sketch input dimension changed for {name}")
        update = h.T @ (h @ self.omegas[name])
        next_sketch = self.sketches[name] + update
        if not bool(torch.isfinite(next_sketch).all()):
            raise FloatingPointError(f"Non-finite REAL accumulated sketch for {name}")
        self.sketches[name].copy_(next_sketch)
        self.energy_sums[name] += float(h.double().square().sum().item())
        self.counts[name] += 1

    def snapshot(self) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
        if not self.counts:
            raise ValueError("REAL cannot export without an eligible prompt group")
        bases, diagnostics = {}, {}
        for name, count in self.counts.items():
            basis, spectrum, info = nystrom_subspace(
                self.sketches[name] / count,
                self.omegas[name],
                self.rank,
                relative_tolerance=self.relative_tolerance,
            )
            energy = self.energy_sums[name] / count
            bases[name] = basis
            diagnostics[name] = {
                **info,
                "selected_rank": self.rank,
                "prompt_groups": count,
                "input_dimension": basis.shape[1],
                "output_sketch_energy": energy,
                "selected_nystrom_energy_fraction": info["selected_nystrom_energy"] / energy if energy else 0.0,
                "spectrum": spectrum.tolist(),
            }
        return bases, diagnostics


class RealProbeCollector:
    """Capture one complete group's projected gradients before refreshing Psi.

    Distributed callers must replicate the same complete group on every data
    parallel rank. Otherwise FSDP averages distinct groups before squaring and
    estimates a different objective. The actor integration enforces this.
    """

    def __init__(self, model: torch.nn.Module, config: Any, adapter_name: str = "default"):
        from peft.tuners.lora.layer import Linear as LoraLinear

        self.model = model
        self.config = config
        self.adapter_name = adapter_name
        self.rank, self.output_width, self.input_width = real_probe_dimensions(config)
        self.seed = int(_get(config, "real_probe_seed", 42))
        self.target_prompts = int(_get(config, "real_probe_prompts", 0))
        self.target_steps = int(_get(config, "real_probe_steps", 64))
        self.scaling_ratio = float(_get(config, "gradient_subspace_scaling", 2.0))
        if self.target_prompts < 0 or self.target_steps <= 0:
            raise ValueError("REAL requires non-negative real_probe_prompts and positive real_probe_steps")
        if not math.isfinite(self.scaling_ratio) or self.scaling_ratio <= 0:
            raise ValueError("REAL exported LoRA scaling must be finite and positive")
        if not math.isclose(self.rank * self.scaling_ratio, round(self.rank * self.scaling_ratio), abs_tol=1e-12):
            raise ValueError("REAL exported rank times scaling must give an integral LoRA alpha")
        configured_output_dir = _get(config, "gradient_probe_output_dir", "")
        if configured_output_dir is None or not str(configured_output_dir).strip():
            raise ValueError("gradient_probe_output_dir is required for peft_type=real_probe")
        self.output_dir = Path(str(configured_output_dir)).expanduser().resolve()
        self.module_names = {
            id(module): normalize_target_name(name)
            for name, module in model.named_modules()
            if isinstance(module, LoraLinear) and adapter_name in module.lora_A
        }
        if not self.module_names or len(set(self.module_names.values())) != len(self.module_names):
            raise ValueError("REAL requires uniquely named PEFT LoRA Linear modules")
        self.module_shapes = {
            self.module_names[id(module)]: (int(module.out_features), int(module.in_features))
            for module in model.modules()
            if id(module) in self.module_names
        }
        self.accumulator = RealSubspaceAccumulator(
            rank=self.rank,
            input_width=self.input_width,
            seed=self.seed,
            relative_tolerance=float(_get(config, "real_probe_relative_tolerance", 1e-6)),
        )
        self.prompt_ids: list[str] = []
        self.observations = 0
        self.complete = False
        self.exported = False

    def _units_and_contexts(self):
        if isinstance(self.model, FSDP):
            units = _leaf_fsdp_modules(self.model) or [self.model]
            return [(unit, FSDP.summon_full_params(unit, writeback=True, with_grads=True)) for unit in units]
        return [(self.model, nullcontext())]

    @torch.no_grad()
    def capture_and_refresh(self, prompt_id: Any) -> dict[str, float]:
        if self.complete:
            raise RuntimeError("REAL observation budget is complete")
        process_rank = dist.get_rank() if dist.is_initialized() else 0
        found: set[str] = set()
        errors: list[str] = []
        for unit, context in self._units_and_contexts():
            with context:
                for module in unit.modules():
                    if id(module) not in self.module_names:
                        continue
                    name = self.module_names[id(module)]
                    if name in found:
                        errors.append(f"REAL encountered module {name} more than once")
                        continue
                    found.add(name)
                    a = module.lora_A[self.adapter_name].weight
                    b = module.lora_B[self.adapter_name].weight
                    if not a.requires_grad or b.requires_grad:
                        errors.append(f"REAL probe requires trainable A and frozen B for {name}")
                        continue
                    if a.grad is None or a.grad.shape != a.shape:
                        errors.append(f"REAL is missing a full LoRA A gradient for {name}")
                        continue
                    if bool(torch.count_nonzero(a).item()):
                        errors.append(f"REAL zero-function probe A was modified for {name}")
                        continue
                    scaling = float(module.scaling[self.adapter_name])
                    h = a.grad.detach().float() / scaling
                    if not bool(torch.isfinite(h).all()):
                        errors.append(f"REAL has non-finite output sketch for {name}")
                        continue
                    if process_rank == 0:
                        try:
                            # The loss gradient is -G; its second moment is identical.
                            self.accumulator.add(name, h)
                        except (ValueError, FloatingPointError) as exc:
                            errors.append(str(exc))
                    _fill_probe_factors(
                        module,
                        name=f"real/output/{name}",
                        adapter_name=self.adapter_name,
                        seed=self.seed,
                        observation=self.observations + 1,
                    )
        missing = set(self.module_names.values()) - found
        if missing:
            errors.append(f"REAL failed to capture modules: {sorted(missing)[:5]}")
        if dist.is_initialized():
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, errors)
            errors = [message for rank_errors in gathered for message in rank_errors]
        if errors:
            raise RuntimeError("; ".join(dict.fromkeys(errors)))
        self.observations += 1
        self.prompt_ids.append(str(prompt_id))
        self.complete = self.target_prompts > 0 and self.observations >= self.target_prompts
        return {
            "real_probe/prompt_groups": float(self.observations),
            "real_probe/module_coverage": float(len(found) / len(self.module_names)),
            "real_probe/nonfinite_modules": 0.0,
        }

    def finish_trainer_step(self, trainer_step: int) -> dict[str, float]:
        self.complete = self.complete or trainer_step >= self.target_steps
        process_rank = dist.get_rank() if dist.is_initialized() else 0
        metrics = {
            "real_probe/trainer_step": float(trainer_step),
            "real_probe/prompt_groups": float(self.observations),
            "real_probe/artifact_ready": float(self.exported),
            "real_probe/complete": float(self.complete),
        }
        error = None
        if process_rank == 0 and self.complete and not self.exported:
            try:
                if self.observations == 0:
                    raise ValueError("REAL observation budget ended without an eligible prompt group")
                bases, modules = self.accumulator.snapshot()
                if set(bases) != set(self.module_names.values()):
                    raise ValueError("REAL artifact has incomplete module coverage")
                if any(count != self.observations for count in self.accumulator.counts.values()):
                    raise ValueError("REAL artifact modules disagree on captured prompt counts")
                self._export(trainer_step, bases, modules)
                self.exported = True
                metrics["real_probe/artifact_ready"] = 1.0
                metrics["real_probe/orthogonality_error_max"] = max(
                    item["orthogonality_error"] for item in modules.values()
                )
            except Exception as exc:
                error = f"REAL export failed: {type(exc).__name__}: {exc}"
        if dist.is_initialized():
            payload = [metrics, error]
            dist.broadcast_object_list(payload, src=0)
            metrics, error = payload
        if error is not None:
            raise RuntimeError(error)
        self.complete = bool(metrics["real_probe/complete"])
        self.exported = bool(metrics["real_probe/artifact_ready"])
        return metrics

    def _export(self, trainer_step: int, bases: dict[str, torch.Tensor], modules: dict[str, dict[str, Any]]) -> None:
        from safetensors.torch import save_file

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = self.output_dir / "subspaces.safetensors"
        temporary = tensor_path.with_suffix(".safetensors.tmp")
        save_file(bases, str(temporary))
        os.replace(temporary, tensor_path)
        rank_map = {
            "schema_version": 1,
            "method": "real_reward_gradient_aligned_lora",
            "rank_pattern": {name: self.rank for name in bases},
            "alpha_pattern": {name: round(self.rank * self.scaling_ratio) for name in bases},
            "constant_scaling": self.scaling_ratio,
            "subspace_path": str(tensor_path),
            "freeze_a": True,
        }
        for name, item in modules.items():
            output_dim, input_dim = self.module_shapes[name]
            item["parameter_cost_per_rank"] = output_dim + input_dim
            item["b_only_parameter_cost_per_rank"] = output_dim
        summary = {
            **rank_map,
            "probe_method": "real",
            "trainer_step": trainer_step,
            "prompt_groups": self.observations,
            "prompt_ids": self.prompt_ids,
            "requested_prompt_groups": self.target_prompts,
            "requested_observation_steps": self.target_steps,
            "seed": self.seed,
            "source_model_path": _get(self.config, "path", None),
            "output_sketch_width": self.output_width,
            "input_sketch_width": self.input_width,
            "relative_spectral_tolerance": self.accumulator.relative_tolerance,
            "objective": "mean_response_length_normalized_binary_population_std_reward_contrast",
            "minimum_correct": 2,
            "minimum_incorrect": 2,
            "second_moment": "uncentered_per_prompt_group",
            "input_factor_scale": 1.0,
            "gradient_clipping": False,
            "optimizer_updates": 0,
            "distributed_observation": "complete_prompt_group_replicated_on_data_parallel_ranks",
            "persistent_accumulator_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in [*self.accumulator.omegas.values(), *self.accumulator.sketches.values()]
            ),
            "formal_trainable_parameters": sum(self.rank * dims[0] for dims in self.module_shapes.values()),
            "modules": modules,
        }
        for filename, payload in (("summary.json", summary), ("rank_map.json", rank_map)):
            path = self.output_dir / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(temporary, path)
