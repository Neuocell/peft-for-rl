"""Low-rank skew-symmetric output rotations for PEFT-style RL training.

This adapter keeps the frozen base linear map and trains a first-order
isospectral tangent update on its output activation:

    y = W x + scale * (P (Q^T y) - Q (P^T y))

The exact Cayley variant can be added later. This minimal form has LoRA-like
O(d_out * r) parameters and initializes to the base model function.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from types import MethodType

import torch
from torch import nn


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
class SkewApplyStats:
    num_layers: int = 0
    num_parameters: int = 0


def _as_target_set(target_modules) -> set[str]:
    if target_modules is None or target_modules == "all-linear":
        return set(DEFAULT_LINEAR_TARGETS)
    if isinstance(target_modules, str):
        return {item.strip() for item in target_modules.split(",") if item.strip()}
    return {str(item).strip() for item in target_modules if str(item).strip()}


def _matches_target(module_name: str, target_modules: set[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


class SkewLinear(nn.Module):
    """Wrap ``nn.Linear`` with a trainable low-rank skew output adapter."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        init_std: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"SkewLinear requires rank > 0, got {rank}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        out_features = base_layer.out_features
        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.register_buffer("skew_scaling", torch.tensor(self.scaling, **factory_kwargs), persistent=True)
        self.skew_P = nn.Parameter(torch.empty(out_features, self.rank, **factory_kwargs))
        self.skew_Q = nn.Parameter(torch.empty(out_features, self.rank, **factory_kwargs))
        self.adapter_disabled = False
        self.reset_parameters(init_std)

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.skew_P, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.skew_Q)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(input)
        if self.adapter_disabled:
            return result
        adapter_input = self.dropout(result)
        dtype = adapter_input.dtype

        p = self.skew_P.to(dtype=dtype)
        q = self.skew_Q.to(dtype=dtype)
        q_proj = torch.matmul(adapter_input, q)
        p_proj = torch.matmul(adapter_input, p)
        update = torch.matmul(q_proj, p.transpose(0, 1)) - torch.matmul(p_proj, q.transpose(0, 1))
        return result + update * self.skew_scaling.to(dtype=dtype)

    @torch.no_grad()
    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the dense linear layer equivalent to this adapter."""

        weight = self.base_layer.weight.detach()
        dtype = weight.dtype
        p = self.skew_P.detach().to(device=weight.device, dtype=dtype)
        q = self.skew_Q.detach().to(device=weight.device, dtype=dtype)
        k = (p @ q.transpose(0, 1)) - (q @ p.transpose(0, 1))
        scaling = self.skew_scaling.detach().to(device=weight.device, dtype=dtype)
        rotation = torch.eye(weight.shape[0], device=weight.device, dtype=dtype) + scaling * k
        merged_weight = rotation @ weight
        bias = self.base_layer.bias
        merged_bias = None if bias is None else rotation @ bias.detach()
        return merged_weight, merged_bias


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_skew_adapters(model: nn.Module, config) -> SkewApplyStats:
    """Replace target ``nn.Linear`` modules with ``SkewLinear`` wrappers."""

    model.requires_grad_(False)

    rank = int(config.get("skew_rank", config.get("lora_rank", 0)))
    if rank <= 0:
        raise ValueError("Skew PEFT requires model.skew_rank or model.lora_rank > 0")

    alpha = float(config.get("skew_alpha", config.get("lora_alpha", rank)))
    dropout = float(config.get("skew_dropout", config.get("lora_dropout", 0.0)))
    init_std = float(config.get("skew_init_std", 0.01))
    target_modules = _as_target_set(config.get("target_modules", "all-linear"))

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head") or name.endswith("embed_tokens"):
            continue
        if _matches_target(name, target_modules):
            replacements.append((name, module))

    stats = SkewApplyStats()
    for name, module in replacements:
        wrapped = SkewLinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            init_std=init_std,
        )
        _set_module(model, name, wrapped)
        stats.num_layers += 1
        stats.num_parameters += wrapped.skew_P.numel() + wrapped.skew_Q.numel()

    @contextmanager
    def disable_adapter(self):
        skew_modules = [module for module in self.modules() if isinstance(module, SkewLinear)]
        old_values = [module.adapter_disabled for module in skew_modules]
        try:
            for module in skew_modules:
                module.adapter_disabled = True
            yield
        finally:
            for module, old_value in zip(skew_modules, old_values, strict=True):
                module.adapter_disabled = old_value

    model.disable_adapter = MethodType(disable_adapter, model)
    return stats
