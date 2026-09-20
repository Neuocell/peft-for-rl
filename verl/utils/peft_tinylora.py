"""TinyLoRA adapters for extremely low-parameter RLVR finetuning.

TinyLoRA follows ``Learning to Reason in 13 Parameters`` (arXiv:2602.04118):

    W' = W + U Sigma (sum_i v_i P_i) V^T

``U``, ``Sigma``, and ``V`` are frozen truncated-SVD factors of the base
weight, ``P`` is a fixed random projection, and only the vector ``v`` is
trained. Vectors can be shared across nearby modules to reduce the entire
adapter to a handful of trainable scalars.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_LINEAR_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass
class TinyLoRAApplyStats:
    num_layers: int = 0
    num_vectors: int = 0
    num_parameters: int = 0
    rank: int = 0
    projection_dim: int = 0
    tie_factor: int = 0


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _as_target_set(target_modules) -> set[str]:
    if target_modules is None or target_modules == "all-linear":
        return set(DEFAULT_LINEAR_TARGETS)
    if isinstance(target_modules, str):
        return {item.strip() for item in target_modules.split(",") if item.strip()}
    return {str(item).strip() for item in target_modules if str(item).strip()}


def _matches_target(module_name: str, target_modules: set[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def _work_device(base_weight: torch.Tensor, configured_device: str) -> torch.device:
    if configured_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return base_weight.device
    return torch.device(configured_device)


def _truncated_svd(
    weight: torch.Tensor,
    rank: int,
    *,
    method: str,
    device: torch.device,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work_weight = weight.detach().to(device=device, dtype=torch.float32)
    min_dim = min(work_weight.shape)
    if rank > min_dim:
        raise ValueError(f"TinyLoRA rank={rank} exceeds max rank {min_dim} for weight shape {tuple(weight.shape)}")

    method = method.lower()
    if method == "exact":
        u, s, vh = torch.linalg.svd(work_weight, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank, :]
    if method != "lowrank":
        raise ValueError(f"Unsupported TinyLoRA SVD method {method!r}; expected 'lowrank' or 'exact'")

    q = min(min_dim, max(rank, rank + max(0, oversample)))
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        u, s, v = torch.svd_lowrank(work_weight, q=q, niter=max(0, niter))
    order = torch.argsort(s, descending=True)[:rank]
    return u[:, order], s[order], v[:, order].transpose(0, 1).contiguous()


class TinyLoRALinear(nn.Module):
    """Wrap a frozen linear layer with a projected LoRA-XS update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        vector: nn.Parameter,
        *,
        rank: int,
        projection_dim: int,
        seed: int,
        svd_device: str,
        svd_method: str,
        svd_oversample: int,
        svd_niter: int,
        projection_std: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"TinyLoRA rank must be positive, got {rank}")
        if projection_dim <= 0:
            raise ValueError(f"TinyLoRA projection_dim must be positive, got {projection_dim}")
        if vector.shape != (projection_dim,):
            raise ValueError(
                f"TinyLoRA shared vector must have shape {(projection_dim,)}, got {tuple(vector.shape)}"
            )
        if not isfinite(projection_std) or projection_std <= 0:
            raise ValueError(f"TinyLoRA projection_std must be finite and positive, got {projection_std}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)
        self.tinylora_vector = vector
        self.adapter_disabled = False

        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        work_device = _work_device(base_layer.weight, svd_device)
        u, s, vh = _truncated_svd(
            base_layer.weight,
            self.rank,
            method=svd_method,
            device=work_device,
            oversample=svd_oversample,
            niter=svd_niter,
            seed=self.seed,
        )
        u_sigma = u * s.unsqueeze(0)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 15485863)
        projection = torch.randn(
            self.projection_dim,
            self.rank,
            self.rank,
            generator=generator,
            dtype=torch.float32,
        )
        projection.mul_(float(projection_std))

        factory_device = base_layer.weight.device
        factory_dtype = base_layer.weight.dtype
        # These tensors are fully determined by the base model and config, so
        # checkpoints only need to store the tiny trainable vectors.
        self.register_buffer(
            "tinylora_u_sigma",
            u_sigma.to(device=factory_device, dtype=factory_dtype),
            persistent=False,
        )
        self.register_buffer(
            "tinylora_vh",
            vh.to(device=factory_device, dtype=factory_dtype),
            persistent=False,
        )
        self.register_buffer(
            "tinylora_projection",
            projection.to(device=factory_device, dtype=factory_dtype),
            persistent=False,
        )

    def _middle(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.einsum(
            "u,uij->ij",
            self.tinylora_vector.to(dtype=dtype),
            self.tinylora_projection.to(dtype=dtype),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(inputs)
        if self.adapter_disabled:
            return result

        projected = F.linear(inputs, self.tinylora_vh.to(dtype=inputs.dtype))
        mixed = F.linear(projected, self._middle(inputs.dtype))
        update = F.linear(mixed, self.tinylora_u_sigma.to(dtype=inputs.dtype))
        return result + update

    @torch.no_grad()
    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.base_layer.weight.detach()
        middle = self._middle(torch.float32)
        delta = self.tinylora_u_sigma.float() @ middle @ self.tinylora_vh.float()
        merged_weight = weight.float() + delta
        bias = self.base_layer.bias
        return merged_weight.to(weight.dtype), None if bias is None else bias.detach()


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def _group_keys(module_names: list[str], strategy: str, tie_factor: int) -> list[tuple[str, int]]:
    strategy = strategy.lower()
    if strategy not in {"tiled", "structured"}:
        raise ValueError(f"Unsupported TinyLoRA tie strategy {strategy!r}; expected 'tiled' or 'structured'")
    if tie_factor <= 0:
        raise ValueError(f"TinyLoRA tie_factor must be positive, got {tie_factor}")

    if strategy == "tiled":
        return [("tiled", index // tie_factor) for index in range(len(module_names))]

    type_counts: dict[str, int] = {}
    keys = []
    for name in module_names:
        module_type = name.rsplit(".", 1)[-1]
        index = type_counts.get(module_type, 0)
        keys.append((module_type, index // tie_factor))
        type_counts[module_type] = index + 1
    return keys


def apply_tinylora_adapters(model: nn.Module, config) -> TinyLoRAApplyStats:
    """Replace target linear modules with paper-style TinyLoRA adapters."""

    model.requires_grad_(False)
    rank = int(_get(config, "tinylora_rank", 2))
    projection_dim = int(_get(config, "tinylora_projection_dim", 1))
    tie_factor = int(_get(config, "tinylora_tie_factor", 16))
    tie_strategy = str(_get(config, "tinylora_tie_strategy", "tiled"))
    seed = int(_get(config, "tinylora_seed", 42))
    svd_device = str(_get(config, "tinylora_svd_device", "auto"))
    svd_method = str(_get(config, "tinylora_svd_method", "lowrank"))
    svd_oversample = int(_get(config, "tinylora_svd_oversample", 4))
    svd_niter = int(_get(config, "tinylora_svd_niter", 2))
    projection_std = float(_get(config, "tinylora_projection_std", 1.0))
    target_modules = _as_target_set(_get(config, "target_modules", "all-linear"))

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head") or name.endswith("embed_tokens"):
            continue
        if _matches_target(name, target_modules):
            replacements.append((name, module))
    if not replacements:
        raise ValueError(f"TinyLoRA found no target linear modules for targets={sorted(target_modules)}")

    group_keys = _group_keys([name for name, _ in replacements], tie_strategy, tie_factor)
    shared_vectors: dict[tuple[str, int], nn.Parameter] = {}
    for key, (_, module) in zip(group_keys, replacements, strict=True):
        if key not in shared_vectors:
            shared_vectors[key] = nn.Parameter(
                torch.zeros(projection_dim, device=module.weight.device, dtype=module.weight.dtype)
            )

    for index, ((name, module), key) in enumerate(zip(replacements, group_keys, strict=True)):
        wrapped = TinyLoRALinear(
            module,
            shared_vectors[key],
            rank=rank,
            projection_dim=projection_dim,
            seed=seed + 1000003 * index,
            svd_device=svd_device,
            svd_method=svd_method,
            svd_oversample=svd_oversample,
            svd_niter=svd_niter,
            projection_std=projection_std,
        )
        _set_module(model, name, wrapped)

    @contextmanager
    def disable_adapter(self):
        adapters = [module for module in self.modules() if isinstance(module, TinyLoRALinear)]
        old_values = [module.adapter_disabled for module in adapters]
        try:
            for module in adapters:
                module.adapter_disabled = True
            yield
        finally:
            for module, old_value in zip(adapters, old_values, strict=True):
                module.adapter_disabled = old_value

    model.disable_adapter = MethodType(disable_adapter, model)
    return TinyLoRAApplyStats(
        num_layers=len(replacements),
        num_vectors=len(shared_vectors),
        num_parameters=len(shared_vectors) * projection_dim,
        rank=rank,
        projection_dim=projection_dim,
        tie_factor=tie_factor,
    )
