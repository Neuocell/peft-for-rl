"""NoRA-initialized, magnitude-decoupled orthogonal finetuning.

For each input block this adapter learns a low-rank skew generator

    K = (alpha / (2 * rank)) * (B A - A^T B^T)

and maps it to an exactly orthogonal rotation with a Cayley transform.  ``A``
uses NoRA column normalization and ``B`` starts at zero, so the adapter is the
identity at initialization.  A trainable row-magnitude vector adds the
DoRA/re-scaled-OFT degree of freedom without changing the initial function.
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
class NoRADoftApplyStats:
    num_layers: int = 0
    num_parameters: int = 0
    generator_rank: int = 0
    block_size: int = 0
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


class NoRADoftLinear(nn.Module):
    """Wrap ``nn.Linear`` with blockwise low-rank exact-Cayley rotations."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        generator_rank: int,
        block_size: int,
        alpha: float,
        normalization_mode: str,
        use_magnitude: bool,
        eps: float,
        seed: int,
    ) -> None:
        super().__init__()
        if generator_rank <= 0:
            raise ValueError(f"NoRA-DOFT generator_rank must be positive, got {generator_rank}")
        if block_size <= 1:
            raise ValueError(f"NoRA-DOFT block_size must be greater than one, got {block_size}")
        if base_layer.in_features % block_size != 0:
            raise ValueError(
                "NoRA-DOFT requires in_features divisible by block_size: "
                f"in_features={base_layer.in_features}, block_size={block_size}"
            )
        if generator_rank > block_size:
            raise ValueError(f"NoRA-DOFT generator_rank={generator_rank} exceeds block_size={block_size}")
        if not isfinite(alpha) or alpha <= 0:
            raise ValueError(f"NoRA-DOFT alpha must be finite and positive, got {alpha}")
        if not isfinite(eps) or eps <= 0:
            raise ValueError(f"NoRA-DOFT eps must be finite and positive, got {eps}")
        normalization_mode = normalization_mode.lower()
        if normalization_mode not in {"init", "continuous"}:
            raise ValueError(
                f"Unsupported NoRA-DOFT normalization mode {normalization_mode!r}; expected 'init' or 'continuous'"
            )

        self.base_layer = base_layer
        self.generator_rank = int(generator_rank)
        self.block_size = int(block_size)
        self.num_blocks = base_layer.in_features // self.block_size
        self.alpha = float(alpha)
        self.generator_scaling = self.alpha / self.generator_rank
        self.normalization_mode = normalization_mode
        self.use_magnitude = bool(use_magnitude)
        self.eps = float(eps)
        self.seed = int(seed)
        self.adapter_disabled = False

        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.nora_doft_A = nn.Parameter(
            torch.empty(self.num_blocks, self.generator_rank, self.block_size, **factory_kwargs)
        )
        self.nora_doft_B = nn.Parameter(
            torch.empty(self.num_blocks, self.block_size, self.generator_rank, **factory_kwargs)
        )
        self.register_buffer(
            "nora_doft_generator_scaling",
            torch.tensor(self.generator_scaling, **factory_kwargs),
            persistent=True,
        )
        base_row_norm = torch.linalg.vector_norm(base_layer.weight.detach().float(), dim=1)
        self.register_buffer(
            "nora_doft_base_row_norm",
            base_row_norm.to(**factory_kwargs),
            persistent=True,
        )
        if self.use_magnitude:
            self.nora_doft_magnitude = nn.Parameter(base_row_norm.to(**factory_kwargs))
        else:
            self.register_parameter("nora_doft_magnitude", None)

        self.reset_parameters()
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    def reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        initial_a = torch.randn(
            self.nora_doft_A.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        initial_a = F.normalize(initial_a, p=2.0, dim=-2, eps=self.eps)
        with torch.no_grad():
            self.nora_doft_A.copy_(initial_a.to(self.nora_doft_A))
            self.nora_doft_B.zero_()
            if self.nora_doft_magnitude is not None:
                self.nora_doft_magnitude.copy_(self.nora_doft_base_row_norm)

    def normalized_a(self) -> torch.Tensor:
        if self.normalization_mode == "continuous":
            return F.normalize(self.nora_doft_A.float(), p=2.0, dim=-2, eps=self.eps)
        return self.nora_doft_A.float()

    def rotations(self) -> torch.Tensor:
        a = self.normalized_a()
        b = self.nora_doft_B.float()
        product = torch.matmul(b, a)
        skew = product - product.transpose(-1, -2)
        # Cayley has derivative 2K at the identity; the half factor makes the
        # first-order adapter scale equal to alpha / rank.
        skew = 0.5 * self.nora_doft_generator_scaling.float() * skew
        eye = torch.eye(self.block_size, device=skew.device, dtype=torch.float32).expand_as(skew)
        return torch.linalg.solve(eye + skew, eye - skew)

    def _rotate_input(self, inputs: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        original_shape = inputs.shape
        blocks = inputs.reshape(*original_shape[:-1], self.num_blocks, self.block_size)
        rotated = torch.einsum("...nb,nbc->...nc", blocks, rotations.to(dtype=inputs.dtype))
        return rotated.reshape(original_shape)

    def _magnitude_scale(self, dtype: torch.dtype) -> torch.Tensor:
        if self.nora_doft_magnitude is None:
            return torch.ones(
                self.base_layer.out_features,
                device=self.base_layer.weight.device,
                dtype=dtype,
            )
        base_norm = self.nora_doft_base_row_norm.float()
        scale = torch.where(
            base_norm > self.eps,
            self.nora_doft_magnitude.float() / base_norm.clamp_min(self.eps),
            torch.ones_like(base_norm),
        )
        return scale.to(dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.adapter_disabled:
            return self.base_layer(inputs)

        rotated_inputs = self._rotate_input(inputs, self.rotations())
        result = F.linear(rotated_inputs, self.base_layer.weight, bias=None)
        result = result * self._magnitude_scale(result.dtype)
        if self.base_layer.bias is not None:
            result = result + self.base_layer.bias
        return result

    @torch.no_grad()
    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.base_layer.weight.detach()
        rotations = self.rotations()
        blocks = weight.float().reshape(weight.shape[0], self.num_blocks, self.block_size)
        merged = torch.einsum("onb,nbc->onc", blocks, rotations.transpose(-1, -2))
        merged = merged.reshape_as(weight.float())
        merged = merged * self._magnitude_scale(torch.float32)[:, None]
        bias = self.base_layer.bias
        return merged.to(weight.dtype), None if bias is None else bias.detach()


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_nora_doft_adapters(model: nn.Module, config) -> NoRADoftApplyStats:
    """Replace target linear modules with NoRA-DOFT wrappers."""

    model.requires_grad_(False)
    generator_rank = int(_get(config, "nora_doft_rank", 8))
    block_size = int(_get(config, "nora_doft_block_size", 32))
    alpha = float(_get(config, "nora_doft_alpha", generator_rank))
    normalization_mode = str(_get(config, "nora_doft_normalization", "init"))
    use_magnitude = bool(_get(config, "nora_doft_use_magnitude", True))
    eps = float(_get(config, "nora_doft_eps", 1e-6))
    seed = int(_get(config, "nora_doft_seed", 42))
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
        raise ValueError(f"NoRA-DOFT found no target linear modules for targets={sorted(target_modules)}")

    stats = NoRADoftApplyStats(generator_rank=generator_rank, block_size=block_size)
    for index, (name, module) in enumerate(replacements):
        wrapped = NoRADoftLinear(
            module,
            generator_rank=generator_rank,
            block_size=block_size,
            alpha=alpha,
            normalization_mode=normalization_mode,
            use_magnitude=use_magnitude,
            eps=eps,
            seed=seed + 1000003 * index,
        )
        _set_module(model, name, wrapped)
        stats.num_layers += 1
        stats.num_parameters += wrapped.nora_doft_A.numel() + wrapped.nora_doft_B.numel()
        if wrapped.nora_doft_magnitude is not None:
            stats.num_parameters += wrapped.nora_doft_magnitude.numel()
        column_norms = torch.linalg.vector_norm(wrapped.nora_doft_A.detach().float(), dim=-2)
        error = float((column_norms - 1.0).abs().max().item())
        stats.normalization_error_max = max(stats.normalization_error_max, error)

    @contextmanager
    def disable_adapter(self):
        adapters = [module for module in self.modules() if isinstance(module, NoRADoftLinear)]
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
