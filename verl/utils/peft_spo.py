"""POET-style structured primitive orthogonal endpoint transforms.

This adapter keeps a frozen linear map and trains a product of small
block-orthogonal primitive factors on both endpoints:

    y = R_out W R_in x + b

Each primitive factor is block diagonal in a permuted channel order.  The
default ``depth=1`` matches the POET-BS-style primitive endpoint transform;
larger depths compose multiple independently permuted factors as an
expression-enhanced ablation while keeping training compute close to the
existing BISO block-CN path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from verl.utils.peft_biso import _cayley_blocks, _primitive_flat_indices, _primitive_size


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
class SPOApplyStats:
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


def _num_blocks(features: int, block_size: int) -> int:
    return ceil(features / block_size)


def _make_permutations(features: int, depth: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    perms = []
    inv_perms = []
    generator = torch.Generator(device="cpu")
    for factor_idx in range(depth):
        generator.manual_seed(int(seed) + 104729 * (factor_idx + 1) + 1009 * features)
        perm = torch.randperm(features, generator=generator, dtype=torch.long)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(features, dtype=torch.long)
        perms.append(perm.to(device=device))
        inv_perms.append(inv_perm.to(device=device))
    return torch.stack(perms, dim=0), torch.stack(inv_perms, dim=0)


def _apply_row_blocks(x: torch.Tensor, rotations: torch.Tensor, features: int, block_size: int) -> torch.Tensor:
    padded_features = rotations.shape[0] * block_size
    if padded_features != features:
        x = F.pad(x, (0, padded_features - features))
    original_shape = x.shape
    x_blocks = x.reshape(*original_shape[:-1], rotations.shape[0], block_size)
    rotated = torch.einsum("...nb,nbc->...nc", x_blocks, rotations)
    rotated = rotated.reshape(*original_shape[:-1], padded_features)
    return rotated[..., :features]


def _apply_spo_factor(
    x: torch.Tensor,
    rotations: torch.Tensor,
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    features: int,
    block_size: int,
    transpose: bool = False,
) -> torch.Tensor:
    y = x.index_select(-1, perm)
    if transpose:
        rotations = rotations.transpose(-1, -2)
    y = _apply_row_blocks(y, rotations, features, block_size)
    return y.index_select(-1, inv_perm)


def _apply_spo_row(
    x: torch.Tensor,
    rotations: torch.Tensor,
    perms: torch.Tensor,
    inv_perms: torch.Tensor,
    features: int,
    block_size: int,
) -> torch.Tensor:
    for factor_idx in range(rotations.shape[0]):
        x = _apply_spo_factor(
            x,
            rotations[factor_idx],
            perms[factor_idx],
            inv_perms[factor_idx],
            features,
            block_size,
            transpose=False,
        )
    return x


def _apply_spo_row_transpose(
    x: torch.Tensor,
    rotations: torch.Tensor,
    perms: torch.Tensor,
    inv_perms: torch.Tensor,
    features: int,
    block_size: int,
) -> torch.Tensor:
    for factor_idx in range(rotations.shape[0] - 1, -1, -1):
        x = _apply_spo_factor(
            x,
            rotations[factor_idx],
            perms[factor_idx],
            inv_perms[factor_idx],
            features,
            block_size,
            transpose=True,
        )
    return x


class SPOLinear(nn.Module):
    """Wrap ``nn.Linear`` with bilateral structured primitive orthogonal products."""

    def __init__(
        self,
        base_layer: nn.Linear,
        block_size: int,
        depth: int,
        alpha: float,
        init_std: float,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        cayley_neumann_eps: float = 0.9,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if block_size <= 1:
            raise ValueError(f"SPOLinear requires block_size > 1, got {block_size}")
        if depth <= 0:
            raise ValueError(f"SPOLinear requires depth > 0, got {depth}")
        if base_layer.in_features % block_size != 0 or base_layer.out_features % block_size != 0:
            raise ValueError(
                "SPOLinear requires in/out features divisible by block_size: "
                f"in_features={base_layer.in_features}, out_features={base_layer.out_features}, "
                f"block_size={block_size}"
            )
        if num_cayley_neumann_terms <= 0:
            raise ValueError(f"num_cayley_neumann_terms must be positive, got {num_cayley_neumann_terms}")

        self.base_layer = base_layer
        self.block_size = int(block_size)
        self.depth = int(depth)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.block_size
        self.use_cayley_neumann = bool(use_cayley_neumann)
        self.num_cayley_neumann_terms = int(num_cayley_neumann_terms)
        self.cayley_neumann_eps = float(cayley_neumann_eps)
        self.seed = int(seed)

        in_blocks = _num_blocks(base_layer.in_features, self.block_size)
        out_blocks = _num_blocks(base_layer.out_features, self.block_size)
        primitive_width = _primitive_size(self.block_size)
        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}

        self.register_buffer("spo_scaling", torch.tensor(self.scaling, **factory_kwargs), persistent=True)
        self.register_buffer("spo_block_size", torch.tensor(self.block_size, device=base_layer.weight.device), persistent=True)
        self.register_buffer("spo_depth", torch.tensor(self.depth, device=base_layer.weight.device), persistent=True)
        self.register_buffer(
            "spo_use_cayley_neumann",
            torch.tensor(self.use_cayley_neumann, device=base_layer.weight.device),
            persistent=True,
        )
        self.register_buffer(
            "spo_num_cayley_neumann_terms",
            torch.tensor(self.num_cayley_neumann_terms, device=base_layer.weight.device),
            persistent=True,
        )
        self.register_buffer(
            "spo_cayley_neumann_eps",
            torch.tensor(self.cayley_neumann_eps, **factory_kwargs),
            persistent=True,
        )
        self.register_buffer(
            "spo_primitive_flat_indices",
            _primitive_flat_indices(self.block_size, base_layer.weight.device),
            persistent=False,
        )

        in_perms, in_inv_perms = _make_permutations(base_layer.in_features, self.depth, self.seed, base_layer.weight.device)
        out_perms, out_inv_perms = _make_permutations(
            base_layer.out_features, self.depth, self.seed + 7919, base_layer.weight.device
        )
        self.register_buffer("spo_in_perm", in_perms, persistent=True)
        self.register_buffer("spo_in_inv_perm", in_inv_perms, persistent=True)
        self.register_buffer("spo_out_perm", out_perms, persistent=True)
        self.register_buffer("spo_out_inv_perm", out_inv_perms, persistent=True)

        self.spo_in_primitive = nn.Parameter(torch.empty(self.depth, in_blocks, primitive_width, **factory_kwargs))
        self.spo_out_primitive = nn.Parameter(torch.empty(self.depth, out_blocks, primitive_width, **factory_kwargs))
        self.adapter_disabled = False
        self.reset_parameters(init_std)

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

    def reset_parameters(self, init_std: float) -> None:
        if init_std and init_std > 0:
            nn.init.normal_(self.spo_in_primitive, mean=0.0, std=float(init_std))
            nn.init.normal_(self.spo_out_primitive, mean=0.0, std=float(init_std))
        else:
            nn.init.zeros_(self.spo_in_primitive)
            nn.init.zeros_(self.spo_out_primitive)

    def _rotations(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        scaling = self.spo_scaling.to(dtype=dtype)
        return (
            _cayley_blocks(
                self.spo_in_primitive.to(dtype=dtype),
                scaling,
                self.block_size,
                self.use_cayley_neumann,
                self.num_cayley_neumann_terms,
                self.cayley_neumann_eps,
                self.spo_primitive_flat_indices,
            ),
            _cayley_blocks(
                self.spo_out_primitive.to(dtype=dtype),
                scaling,
                self.block_size,
                self.use_cayley_neumann,
                self.num_cayley_neumann_terms,
                self.cayley_neumann_eps,
                self.spo_primitive_flat_indices,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.adapter_disabled:
            return self.base_layer(input)

        in_rot, out_rot = self._rotations(input.dtype)
        transformed_input = _apply_spo_row(
            input,
            in_rot,
            self.spo_in_perm,
            self.spo_in_inv_perm,
            self.base_layer.in_features,
            self.block_size,
        )
        result = self.base_layer(transformed_input)
        return _apply_spo_row(
            result,
            out_rot,
            self.spo_out_perm,
            self.spo_out_inv_perm,
            self.base_layer.out_features,
            self.block_size,
        )

    @torch.no_grad()
    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.base_layer.weight.detach()
        dtype = weight.dtype
        in_rot, out_rot = self._rotations(torch.float32)

        merged_weight = _apply_spo_row_transpose(
            weight.float(),
            in_rot,
            self.spo_in_perm,
            self.spo_in_inv_perm,
            self.base_layer.in_features,
            self.block_size,
        )
        merged_weight = _apply_spo_row(
            merged_weight.transpose(0, 1),
            out_rot,
            self.spo_out_perm,
            self.spo_out_inv_perm,
            self.base_layer.out_features,
            self.block_size,
        ).transpose(0, 1)

        bias = self.base_layer.bias
        if bias is None:
            merged_bias = None
        else:
            merged_bias = _apply_spo_row(
                bias.detach().float(),
                out_rot,
                self.spo_out_perm,
                self.spo_out_inv_perm,
                self.base_layer.out_features,
                self.block_size,
            )
        return merged_weight.to(dtype), None if merged_bias is None else merged_bias.to(bias.dtype)


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_spo_adapters(model: nn.Module, config) -> SPOApplyStats:
    """Replace target ``nn.Linear`` modules with ``SPOLinear`` wrappers."""

    model.requires_grad_(False)

    block_size = int(config.get("spo_block_size", 16))
    depth = int(config.get("spo_depth", 1))
    alpha = float(config.get("spo_alpha", 8))
    init_std = float(config.get("spo_init_std", 0.0))
    use_cayley_neumann = bool(config.get("spo_use_cayley_neumann", True))
    num_cayley_neumann_terms = int(config.get("spo_num_cayley_neumann_terms", 5))
    cayley_neumann_eps = float(config.get("spo_cayley_neumann_eps", 0.9))
    seed = int(config.get("spo_seed", 42))
    target_modules = _as_target_set(config.get("target_modules", "all-linear"))

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head") or name.endswith("embed_tokens"):
            continue
        if _matches_target(name, target_modules):
            replacements.append((name, module))

    stats = SPOApplyStats()
    for idx, (name, module) in enumerate(replacements):
        wrapped = SPOLinear(
            base_layer=module,
            block_size=block_size,
            depth=depth,
            alpha=alpha,
            init_std=init_std,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
            cayley_neumann_eps=cayley_neumann_eps,
            seed=seed + 1000003 * idx,
        )
        _set_module(model, name, wrapped)
        stats.num_layers += 1
        stats.num_parameters += wrapped.spo_in_primitive.numel() + wrapped.spo_out_primitive.numel()

    @contextmanager
    def disable_adapter(self):
        spo_modules = [module for module in self.modules() if isinstance(module, SPOLinear)]
        old_values = [module.adapter_disabled for module in spo_modules]
        try:
            for module in spo_modules:
                module.adapter_disabled = True
            yield
        finally:
            for module, old_value in zip(spo_modules, old_values, strict=True):
                module.adapter_disabled = old_value

    model.disable_adapter = MethodType(disable_adapter, model)
    return stats
