"""GeoRA initialization for PEFT LoRA layers.

This implements the initialization from "GeoRA: Geometry-Aware Low-Rank
Adaptation for RLVR" as a local PEFT-compatible reproduction. PEFT still owns
the LoRA modules; this helper overwrites their initial factors and turns the
frozen base layer into the residual anchor:

    W_res = W - scaling * B_geo @ A_geo
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GeoRAInitStats:
    num_layers: int = 0
    num_parameters: int = 0


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _low_rank_svd(matrix: torch.Tensor, rank: int, oversample: int, niter: int):
    min_dim = min(matrix.shape)
    q = min(min_dim, max(rank, rank + oversample))
    if q <= 0:
        raise ValueError(f"Invalid randomized SVD q={q} for shape={tuple(matrix.shape)}")
    if q >= min_dim:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u, s, vh.transpose(0, 1)
    return torch.svd_lowrank(matrix, q=q, niter=niter)


def _init_one_lora_linear(
    module,
    adapter_name: str,
    rank: int,
    sparsity_ratio: float,
    oversample: int,
    niter: int,
    svd_device: str,
    residual_anchor: bool,
    init_scale: float,
) -> int:
    base_weight = module.base_layer.weight
    if base_weight.ndim != 2:
        return 0

    dtype = base_weight.dtype
    device = base_weight.device
    if svd_device == "auto":
        work_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else device
    else:
        work_device = torch.device(svd_device)
    work_weight = base_weight.detach().to(device=work_device, dtype=torch.float32)

    u, s, v = _low_rank_svd(work_weight, rank=rank, oversample=oversample, niter=niter)
    effective_rank = min(rank, s.numel())
    u_r = u[:, :effective_rank]
    s_r = s[:effective_rank]
    v_r = v[:, :effective_rank]

    spectral_approx = (u_r * s_r.unsqueeze(0)) @ v_r.transpose(0, 1)
    if sparsity_ratio <= 0:
        mask = torch.zeros_like(work_weight, dtype=torch.bool)
    elif sparsity_ratio >= 1:
        mask = torch.ones_like(work_weight, dtype=torch.bool)
    else:
        tau_spec = torch.quantile(spectral_approx.abs().flatten(), sparsity_ratio)
        tau_euc = torch.quantile(work_weight.abs().flatten(), sparsity_ratio)
        mask = (spectral_approx.abs() <= tau_spec) | (work_weight.abs() <= tau_euc)

    geo_weight = work_weight * mask.to(work_weight.dtype)
    del spectral_approx, mask

    u_g, s_g, v_g = _low_rank_svd(geo_weight, rank=rank, oversample=oversample, niter=niter)
    effective_rank = min(rank, s_g.numel())
    u_g = u_g[:, :effective_rank]
    s_g = s_g[:effective_rank]
    v_g = v_g[:, :effective_rank]

    sqrt_s = torch.sqrt(torch.clamp(s_g, min=0.0))
    if init_scale != 1.0:
        sqrt_s = sqrt_s * (init_scale**0.5)

    b_init = u_g * sqrt_s.unsqueeze(0)
    a_init = sqrt_s.unsqueeze(1) * v_g.transpose(0, 1)

    lora_a = module.lora_A[adapter_name].weight
    lora_b = module.lora_B[adapter_name].weight
    lora_a.data.zero_()
    lora_b.data.zero_()
    lora_a.data[:effective_rank, :].copy_(a_init.to(device=lora_a.device, dtype=lora_a.dtype))
    lora_b.data[:, :effective_rank].copy_(b_init.to(device=lora_b.device, dtype=lora_b.dtype))

    if residual_anchor:
        scaling = module.scaling[adapter_name]
        delta = (b_init @ a_init) * scaling
        base_weight.data.sub_(delta.to(device=device, dtype=dtype))

    return base_weight.numel()


@torch.no_grad()
def apply_geora_initialization(model, config, adapter_name: str = "default") -> GeoRAInitStats:
    """Apply GeoRA initialization to all PEFT LoRA Linear layers in ``model``."""

    from peft.tuners.lora.layer import Linear as LoraLinear

    rank = int(config.get("lora_rank", 0))
    if rank <= 0:
        raise ValueError("GeoRA requires model.lora_rank > 0")

    sparsity_ratio = float(config.get("geora_sparsity_ratio", 0.2))
    if not 0.0 <= sparsity_ratio <= 1.0:
        raise ValueError(f"geora_sparsity_ratio must be in [0, 1], got {sparsity_ratio}")

    oversample = int(config.get("geora_oversample", 8))
    niter = int(config.get("geora_niter", 2))
    svd_device = str(config.get("geora_svd_device", "auto"))
    residual_anchor = _as_bool(config.get("geora_residual_anchor", True))
    init_scale = float(config.get("geora_init_scale", 1.0))
    seed = int(config.get("geora_seed", 42))

    stats = GeoRAInitStats()
    devices = {p.device for p in model.parameters() if p.device.type == "cuda"}
    with torch.random.fork_rng(devices=list(devices), enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for module in model.modules():
            if not isinstance(module, LoraLinear):
                continue
            if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                continue
            num_params = _init_one_lora_linear(
                module=module,
                adapter_name=adapter_name,
                rank=rank,
                sparsity_ratio=sparsity_ratio,
                oversample=oversample,
                niter=niter,
                svd_device=svd_device,
                residual_anchor=residual_anchor,
                init_scale=init_scale,
            )
            if num_params:
                stats.num_layers += 1
                stats.num_parameters += num_params
    return stats
