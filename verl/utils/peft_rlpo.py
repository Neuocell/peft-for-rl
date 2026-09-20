"""RLPO initialization for ordinary PEFT LoRA layers.

This follows the public RLPO implementation from
``geometry-preserving-orthonormal-init-rlvr``: initialize LoRA A with the
top-r right singular vectors of the frozen base weight and initialize LoRA B
to zero. After initialization the adapter remains an ordinary two-factor
LoRA adapter; no orthogonal loss or rank allocator is involved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RLPOInitStats:
    num_layers: int = 0
    num_parameters: int = 0
    total_s: float = 0.0
    mean_s: float = 0.0
    max_s: float = 0.0
    orthogonality_error_max: float = 0.0
    b_abs_max: float = 0.0
    rank_sum: int = 0
    rank_min: int = 0
    rank_max: int = 0

    @property
    def rank_mean(self) -> float:
        return self.rank_sum / self.num_layers if self.num_layers else 0.0


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _work_device(base_weight: torch.Tensor, configured_device: str) -> torch.device:
    if configured_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return base_weight.device
    return torch.device(configured_device)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def apply_rlpo_initialization(model, config, adapter_name: str = "default") -> RLPOInitStats:
    """Apply the paper-source RLPO initialization to PEFT LoRA Linear layers."""

    from peft.tuners.lora.layer import Linear as LoraLinear

    configured_rank = int(_get(config, "lora_rank", 0))
    if configured_rank <= 0:
        raise ValueError("RLPO requires model.lora_rank > 0")

    svd_device = str(_get(config, "rlpo_svd_device", "auto"))
    stats = RLPOInitStats()
    timings: list[float] = []

    for module in model.modules():
        if not isinstance(module, LoraLinear):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        base_weight = module.base_layer.weight
        if base_weight.ndim != 2:
            raise ValueError(f"RLPO requires 2D base weights, got shape={tuple(base_weight.shape)}")
        if base_weight.is_meta:
            raise ValueError("RLPO cannot initialize from a meta base weight")

        lora_a = module.lora_A[adapter_name].weight
        lora_b = module.lora_B[adapter_name].weight
        rank = lora_a.shape[0]
        max_rank = min(base_weight.shape)
        if rank > configured_rank:
            raise ValueError(
                f"RLPO layer rank exceeds configured maximum: config rank={configured_rank}, layer rank={rank}, "
                f"weight shape={tuple(base_weight.shape)}"
            )
        if rank > max_rank:
            raise ValueError(
                f"RLPO rank={rank} exceeds max rank {max_rank} for weight shape {tuple(base_weight.shape)}"
            )
        if lora_a.shape != (rank, base_weight.shape[1]) or lora_b.shape != (base_weight.shape[0], rank):
            raise ValueError(
                "RLPO encountered incompatible LoRA factor shapes: "
                f"base={tuple(base_weight.shape)}, A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
            )

        work_device = _work_device(base_weight, svd_device)
        work_weight = base_weight.detach().to(device=work_device, dtype=torch.float32)
        _synchronize(work_device)
        start = time.perf_counter()
        _, _, vh = torch.linalg.svd(work_weight, full_matrices=False)
        a_init = vh[:rank, :].contiguous()
        b_init = torch.zeros(base_weight.shape[0], rank, device=work_device, dtype=work_weight.dtype)
        _synchronize(work_device)
        elapsed = time.perf_counter() - start

        lora_a.copy_(a_init.to(device=lora_a.device, dtype=lora_a.dtype))
        lora_b.copy_(b_init.to(device=lora_b.device, dtype=lora_b.dtype))

        identity = torch.eye(rank, device=work_device, dtype=a_init.dtype)
        orthogonality_error = (a_init @ a_init.transpose(0, 1) - identity).abs().max().item()
        timings.append(elapsed)
        stats.num_layers += 1
        stats.num_parameters += base_weight.numel()
        stats.rank_sum += rank
        stats.rank_min = rank if stats.rank_min == 0 else min(stats.rank_min, rank)
        stats.rank_max = max(stats.rank_max, rank)
        stats.orthogonality_error_max = max(stats.orthogonality_error_max, orthogonality_error)
        stats.b_abs_max = max(stats.b_abs_max, float(b_init.abs().max().item()))

    if not stats.num_layers:
        raise ValueError("RLPO found no PEFT LoRA Linear layers to initialize")

    stats.total_s = float(sum(timings))
    stats.mean_s = stats.total_s / len(timings)
    stats.max_s = float(max(timings))
    return stats
