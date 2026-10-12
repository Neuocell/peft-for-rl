"""NoRA-initialized additive low-rank adaptation with DoRA magnitude.

The learned direction is ``W + (alpha / rank) * B @ A``. ``A`` uses NoRA
column normalization, ``B`` starts at zero, and the trainable magnitude starts
at the base weight's row norm. This makes the adapter exactly function
preserving at initialization while separating direction and magnitude updates.
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
class NoRADoraApplyStats:
    num_layers: int = 0
    num_parameters: int = 0
    rank: int = 0
    normalization_error_max: float = 0.0


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


class NoRADoraLinear(nn.Module):
    """Wrap ``nn.Linear`` with a NoRA-initialized DoRA update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        normalization_mode: str,
        eps: float,
        seed: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"NoRA-DoRA rank must be positive, got {rank}")
        if rank > min(base_layer.in_features, base_layer.out_features):
            raise ValueError(
                "NoRA-DoRA rank exceeds the linear layer dimensions: "
                f"rank={rank}, in_features={base_layer.in_features}, out_features={base_layer.out_features}"
            )
        if not isfinite(alpha) or alpha <= 0:
            raise ValueError(f"NoRA-DoRA alpha must be finite and positive, got {alpha}")
        if not isfinite(eps) or eps <= 0:
            raise ValueError(f"NoRA-DoRA eps must be finite and positive, got {eps}")
        normalization_mode = normalization_mode.lower()
        if normalization_mode not in {"init", "continuous"}:
            raise ValueError(
                f"Unsupported NoRA-DoRA normalization mode {normalization_mode!r}; expected 'init' or 'continuous'"
            )

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.normalization_mode = normalization_mode
        self.eps = float(eps)
        self.seed = int(seed)
        self.adapter_disabled = False

        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.nora_dora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features, **factory_kwargs))
        self.nora_dora_B = nn.Parameter(torch.empty(base_layer.out_features, self.rank, **factory_kwargs))
        self.register_buffer(
            "nora_dora_scaling",
            torch.tensor(self.scaling, **factory_kwargs),
            persistent=True,
        )
        base_row_norm = torch.linalg.vector_norm(base_layer.weight.detach().float(), dim=1)
        self.nora_dora_magnitude = nn.Parameter(base_row_norm.to(**factory_kwargs))

        self.reset_parameters()
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        initial_a = torch.randn(
            self.nora_dora_A.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        initial_a = F.normalize(initial_a, p=2.0, dim=0, eps=self.eps)
        base_row_norm = torch.linalg.vector_norm(self.base_layer.weight.detach().float(), dim=1)
        with torch.no_grad():
            self.nora_dora_A.copy_(initial_a.to(self.nora_dora_A))
            self.nora_dora_B.zero_()
            self.nora_dora_magnitude.copy_(base_row_norm.to(self.nora_dora_magnitude))

    def normalized_a(self) -> torch.Tensor:
        if self.normalization_mode == "continuous":
            return F.normalize(self.nora_dora_A.float(), p=2.0, dim=0, eps=self.eps)
        return self.nora_dora_A.float()

    def delta_weight(self) -> torch.Tensor:
        return self.nora_dora_scaling.float() * (self.nora_dora_B.float() @ self.normalized_a())

    def _magnitude_scale(self) -> torch.Tensor:
        # Matching PEFT DoRA, the direction norm is a normalization statistic,
        # not a second gradient path through the additive low-rank update.
        direction = self.base_layer.weight.float() + self.delta_weight()
        direction_norm = torch.linalg.vector_norm(direction, dim=1).detach().clamp_min(self.eps)
        return self.nora_dora_magnitude.float() / direction_norm

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.adapter_disabled:
            return self.base_layer(inputs)

        a = self.normalized_a().to(dtype=inputs.dtype)
        b = self.nora_dora_B.to(dtype=inputs.dtype)
        base_result = F.linear(inputs, self.base_layer.weight, bias=None)
        low_rank_result = F.linear(F.linear(inputs, a), b) * self.nora_dora_scaling.to(dtype=inputs.dtype)
        result = (base_result + low_rank_result) * self._magnitude_scale().to(dtype=base_result.dtype)
        if self.base_layer.bias is not None:
            result = result + self.base_layer.bias
        return result

    @torch.no_grad()
    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.base_layer.weight.detach()
        direction = weight.float() + self.delta_weight()
        direction_norm = torch.linalg.vector_norm(direction, dim=1).clamp_min(self.eps)
        merged = direction * (self.nora_dora_magnitude.float() / direction_norm)[:, None]
        bias = self.base_layer.bias
        return merged.to(weight.dtype), None if bias is None else bias.detach()


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_nora_dora_adapters(model: nn.Module, config) -> NoRADoraApplyStats:
    """Replace target linear modules with NoRA-DoRA wrappers."""

    model.requires_grad_(False)
    rank = int(_get(config, "nora_dora_rank", 8))
    alpha = float(_get(config, "nora_dora_alpha", rank))
    normalization_mode = str(_get(config, "nora_dora_normalization", "init"))
    eps = float(_get(config, "nora_dora_eps", 1e-6))
    seed = int(_get(config, "nora_dora_seed", 42))
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
        raise ValueError(f"NoRA-DoRA found no target linear modules for targets={sorted(target_modules)}")

    stats = NoRADoraApplyStats(rank=rank)
    for index, (name, module) in enumerate(replacements):
        wrapped = NoRADoraLinear(
            module,
            rank=rank,
            alpha=alpha,
            normalization_mode=normalization_mode,
            eps=eps,
            seed=seed + 1000003 * index,
        )
        _set_module(model, name, wrapped)
        stats.num_layers += 1
        stats.num_parameters += (
            wrapped.nora_dora_A.numel()
            + wrapped.nora_dora_B.numel()
            + wrapped.nora_dora_magnitude.numel()
        )
        column_norms = torch.linalg.vector_norm(wrapped.nora_dora_A.detach().float(), dim=0)
        error = float((column_norms - 1.0).abs().max().item())
        stats.normalization_error_max = max(stats.normalization_error_max, error)

    @contextmanager
    def disable_adapter(self):
        adapters = [module for module in self.modules() if isinstance(module, NoRADoraLinear)]
        old_values = [module.adapter_disabled for module in adapters]
        try:
            for module in adapters:
                module.adapter_disabled = True
            yield
        finally:
            for module, old_value in zip(adapters, old_values, strict=True):
                module.adapter_disabled = old_value

    model.disable_adapter = MethodType(disable_adapter, model)
    return stats
