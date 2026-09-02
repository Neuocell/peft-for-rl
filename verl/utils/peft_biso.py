"""Bilateral block-orthogonal PEFT with non-redundant primitive coordinates.

The default parameterization stores only the independent upper-triangular
coordinates of each skew-symmetric block.  A block is mapped to an
orthogonal matrix with either an exact Cayley transform or its truncated
Cayley-Neumann approximation.

``legacy_raw`` remains available for historical checkpoints.  It keeps the
old full square raw matrices and old exact Cayley convention, but is not the
default because its symmetric degrees of freedom never affect the model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil
from types import MethodType

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
class BISOApplyStats:
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


def _primitive_size(block_size: int) -> int:
    return block_size * (block_size - 1) // 2


def _primitive_indices(block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(block_size, block_size, offset=1, device=device)


def _primitive_flat_indices(block_size: int, device: torch.device) -> torch.Tensor:
    rows, cols = _primitive_indices(block_size, device)
    return rows * block_size + cols


def _primitive_to_skew(
    primitive: torch.Tensor,
    block_size: int,
    flat_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build skew blocks from packed upper-triangular coordinates."""

    expected = _primitive_size(block_size)
    if primitive.shape[-1] != expected:
        raise ValueError(
            f"Invalid BISO primitive width: expected {expected} for block_size={block_size}, "
            f"got {primitive.shape[-1]}"
        )
    if flat_indices is None:
        flat_indices = _primitive_flat_indices(block_size, primitive.device)
    else:
        flat_indices = flat_indices.to(device=primitive.device)
    flat = primitive.new_zeros(*primitive.shape[:-1], block_size * block_size)
    indices = flat_indices.reshape((1,) * (primitive.ndim - 1) + (-1,))
    indices = indices.expand(*primitive.shape[:-1], -1)
    flat = flat.scatter(-1, indices, primitive)
    matrix = flat.reshape(*primitive.shape[:-1], block_size, block_size)
    return matrix - matrix.transpose(-1, -2)


def _project_skew(skew: torch.Tensor, frobenius_bound: float) -> torch.Tensor:
    """Project each generator block into the useful CN convergence regime."""

    if frobenius_bound <= 0:
        return skew
    norm = torch.linalg.vector_norm(skew.float().reshape(*skew.shape[:-2], -1), dim=-1)
    factor = (float(frobenius_bound) / (norm + 1e-12)).clamp(max=1.0)
    return skew * factor.to(dtype=skew.dtype)[..., None, None]


def _cayley_neumann_from_skew(
    skew: torch.Tensor,
    num_terms: int,
    frobenius_bound: float = 0.0,
) -> torch.Tensor:
    """Approximate ``(I-Q)^(-1)(I+Q)`` with a truncated CN expansion."""

    if num_terms <= 0:
        raise ValueError(f"num_terms must be positive, got {num_terms}")
    skew = _project_skew(skew, frobenius_bound)
    q = skew.float()
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype).expand_as(q)
    rotation = eye
    if num_terms == 1:
        return rotation.to(dtype=skew.dtype)

    power = q
    for degree in range(1, num_terms):
        coefficient = 1.0 if degree == num_terms - 1 else 2.0
        rotation = rotation + power * coefficient
        if degree + 1 < num_terms:
            power = power @ q
    return rotation.to(dtype=skew.dtype)


def _cayley_exact_from_skew(skew: torch.Tensor) -> torch.Tensor:
    q = skew.float()
    eye = torch.eye(q.shape[-1], device=q.device, dtype=q.dtype).expand_as(q)
    return torch.linalg.solve(eye - q, eye + q).to(dtype=skew.dtype)


def _legacy_cayley_blocks(raw: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
    """Historical BISO transform: solve(I + K, I - K) on full raw blocks."""

    raw_dtype = raw.dtype
    raw = raw.float()
    scaling = scaling.to(dtype=torch.float32, device=raw.device)
    skew = (raw - raw.transpose(-1, -2)) * scaling
    eye = torch.eye(raw.shape[-1], device=raw.device, dtype=torch.float32).expand_as(skew)
    return torch.linalg.solve(eye + skew, eye - skew).to(raw_dtype)


def _cayley_blocks(
    primitive: torch.Tensor,
    scaling: torch.Tensor,
    block_size: int,
    use_cayley_neumann: bool = True,
    num_cayley_neumann_terms: int = 5,
    cayley_neumann_eps: float = 0.9,
    primitive_flat_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map packed primitive coordinates to block rotations."""

    skew = _primitive_to_skew(primitive.float(), block_size, primitive_flat_indices)
    skew = skew * scaling.float().to(device=skew.device)
    if use_cayley_neumann:
        # Match OFT's per-layer projection convention: the budget is divided
        # by sqrt(number_of_blocks), keeping the whole layer's generator norm
        # controlled as the hidden size grows.
        layer_eps = cayley_neumann_eps / max(1.0, float(primitive.shape[-2]) ** 0.5)
        return _cayley_neumann_from_skew(skew, num_cayley_neumann_terms, layer_eps)
    return _cayley_exact_from_skew(skew)


def _apply_row_blocks(x: torch.Tensor, rotations: torch.Tensor, features: int, block_size: int) -> torch.Tensor:
    """Right-multiply the last dimension by block-diagonal rotations."""

    padded_features = rotations.shape[0] * block_size
    if padded_features != features:
        x = F.pad(x, (0, padded_features - features))
    original_shape = x.shape
    x_blocks = x.reshape(*original_shape[:-1], rotations.shape[0], block_size)
    rotated = torch.einsum("...nb,nbc->...nc", x_blocks, rotations)
    rotated = rotated.reshape(*original_shape[:-1], padded_features)
    return rotated[..., :features]


def _right_multiply_blocks(matrix: torch.Tensor, rotations_t: torch.Tensor, features: int, block_size: int) -> torch.Tensor:
    """Compute ``matrix @ R^T`` for block-diagonal R without materializing R."""

    padded_features = rotations_t.shape[0] * block_size
    if padded_features != features:
        matrix = F.pad(matrix, (0, padded_features - features))
    blocks = matrix.reshape(matrix.shape[0], rotations_t.shape[0], block_size)
    rotated = torch.einsum("onb,nbc->onc", blocks, rotations_t)
    rotated = rotated.reshape(matrix.shape[0], padded_features)
    return rotated[:, :features]


def _left_multiply_blocks(rotations_t: torch.Tensor, matrix: torch.Tensor, features: int, block_size: int) -> torch.Tensor:
    """Compute ``R^T @ matrix`` for block-diagonal R without materializing R."""

    padded_features = rotations_t.shape[0] * block_size
    if padded_features != features:
        matrix = F.pad(matrix, (0, 0, 0, padded_features - features))
    blocks = matrix.reshape(rotations_t.shape[0], block_size, matrix.shape[1])
    rotated = torch.einsum("nab,nbc->nac", rotations_t, blocks)
    rotated = rotated.reshape(padded_features, matrix.shape[1])
    return rotated[:features, :]


def _topk_mask(scores: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    if keep_ratio >= 1.0:
        return torch.ones_like(scores, dtype=torch.bool)
    if keep_ratio <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)
    flat = scores.flatten()
    keep_k = max(1, min(flat.numel(), int(round(flat.numel() * keep_ratio))))
    threshold = torch.topk(flat, keep_k, largest=True).values.min()
    return scores >= threshold


def _principal_block_masks(base_weight: torch.Tensor, block_size: int, keep_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build row/column block masks from base-weight block saliency."""

    out_blocks = _num_blocks(base_weight.shape[0], block_size)
    in_blocks = _num_blocks(base_weight.shape[1], block_size)
    padded = F.pad(
        base_weight.abs(),
        (0, in_blocks * block_size - base_weight.shape[1], 0, out_blocks * block_size - base_weight.shape[0]),
    )
    blocks = padded.view(out_blocks, block_size, in_blocks, block_size)
    block_saliency = blocks.mean(dim=(1, 3))
    out_scores = block_saliency.mean(dim=1)
    in_scores = block_saliency.mean(dim=0)
    return _topk_mask(out_scores, keep_ratio), _topk_mask(in_scores, keep_ratio)


class BISOLinear(nn.Module):
    """Wrap ``nn.Linear`` with bilateral block-orthogonal transforms."""

    def __init__(
        self,
        base_layer: nn.Linear,
        block_size: int,
        alpha: float,
        init_std: float,
        selective_mode: str = "none",
        selective_keep_ratio: float = 0.3,
        parameterization: str = "primitive_cn",
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        cayley_neumann_eps: float = 0.9,
    ) -> None:
        super().__init__()
        if block_size <= 1:
            raise ValueError(f"BISOLinear requires block_size > 1, got {block_size}")
        if base_layer.in_features % block_size != 0 or base_layer.out_features % block_size != 0:
            raise ValueError(
                "BISOLinear requires in/out features divisible by block_size for strict spectrum preservation: "
                f"in_features={base_layer.in_features}, out_features={base_layer.out_features}, "
                f"block_size={block_size}"
            )
        parameterization = str(parameterization).lower()
        if parameterization not in {"primitive_cn", "legacy_raw"}:
            raise ValueError(
                f"Unsupported BISO parameterization {parameterization!r}; "
                "expected 'primitive_cn' or 'legacy_raw'"
            )
        if num_cayley_neumann_terms <= 0:
            raise ValueError(f"num_cayley_neumann_terms must be positive, got {num_cayley_neumann_terms}")

        self.base_layer = base_layer
        self.block_size = int(block_size)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.block_size
        self.selective_mode = str(selective_mode)
        self.selective_keep_ratio = float(selective_keep_ratio)
        self.parameterization = parameterization
        self._selective_masks_ready = False

        in_blocks = _num_blocks(base_layer.in_features, self.block_size)
        out_blocks = _num_blocks(base_layer.out_features, self.block_size)
        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.register_buffer("biso_scaling", torch.tensor(self.scaling, **factory_kwargs), persistent=True)
        self.register_buffer("biso_block_size", torch.tensor(self.block_size, device=base_layer.weight.device), persistent=True)
        self.register_buffer(
            "biso_use_cayley_neumann",
            torch.tensor(bool(use_cayley_neumann), device=base_layer.weight.device),
            persistent=True,
        )
        self.register_buffer(
            "biso_num_cayley_neumann_terms",
            torch.tensor(int(num_cayley_neumann_terms), device=base_layer.weight.device),
            persistent=True,
        )
        self.register_buffer(
            "biso_cayley_neumann_eps",
            torch.tensor(float(cayley_neumann_eps), **factory_kwargs),
            persistent=True,
        )
        self.register_buffer(
            "biso_primitive_flat_indices",
            _primitive_flat_indices(self.block_size, base_layer.weight.device),
            persistent=False,
        )

        if self.parameterization == "primitive_cn":
            primitive_width = _primitive_size(self.block_size)
            self.biso_in_primitive = nn.Parameter(torch.empty(in_blocks, primitive_width, **factory_kwargs))
            self.biso_out_primitive = nn.Parameter(torch.empty(out_blocks, primitive_width, **factory_kwargs))
        else:
            self.biso_in_raw = nn.Parameter(
                torch.empty(in_blocks, self.block_size, self.block_size, **factory_kwargs)
            )
            self.biso_out_raw = nn.Parameter(
                torch.empty(out_blocks, self.block_size, self.block_size, **factory_kwargs)
            )

        self.biso_out_selective_mask = None
        self.biso_in_selective_mask = None
        if self.selective_mode == "geora_block_mask":
            self._maybe_init_selective_masks()
        self.adapter_disabled = False
        self.reset_parameters(init_std)

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

    def reset_parameters(self, init_std: float) -> None:
        parameters = (
            (self.biso_in_primitive, self.biso_out_primitive)
            if self.parameterization == "primitive_cn"
            else (self.biso_in_raw, self.biso_out_raw)
        )
        if init_std and init_std > 0:
            for parameter in parameters:
                nn.init.normal_(parameter, mean=0.0, std=float(init_std))
        else:
            for parameter in parameters:
                nn.init.zeros_(parameter)

    def _maybe_init_selective_masks(self) -> None:
        if self.selective_mode != "geora_block_mask" or self._selective_masks_ready:
            return
        base_weight = self.base_layer.weight
        if getattr(base_weight, "is_meta", False):
            return
        out_mask, in_mask = _principal_block_masks(
            base_weight.detach().to(dtype=torch.float32),
            self.block_size,
            self.selective_keep_ratio,
        )
        self.biso_out_selective_mask = out_mask.detach()
        self.biso_in_selective_mask = in_mask.detach()
        self._selective_masks_ready = True

    def _rotations(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        self._maybe_init_selective_masks()
        if self.parameterization == "primitive_cn":
            in_primitive = self.biso_in_primitive
            out_primitive = self.biso_out_primitive
            if self.biso_in_selective_mask is not None:
                in_mask = self.biso_in_selective_mask.to(device=in_primitive.device, dtype=in_primitive.dtype)
                in_primitive = in_primitive * in_mask[:, None]
            if self.biso_out_selective_mask is not None:
                out_mask = self.biso_out_selective_mask.to(device=out_primitive.device, dtype=out_primitive.dtype)
                out_primitive = out_primitive * out_mask[:, None]
            return (
                _cayley_blocks(
                    in_primitive.to(dtype=dtype),
                    self.biso_scaling.to(dtype=dtype),
                    self.block_size,
                    bool(self.biso_use_cayley_neumann.item()),
                    int(self.biso_num_cayley_neumann_terms.item()),
                    float(self.biso_cayley_neumann_eps.item()),
                    self.biso_primitive_flat_indices,
                ),
                _cayley_blocks(
                    out_primitive.to(dtype=dtype),
                    self.biso_scaling.to(dtype=dtype),
                    self.block_size,
                    bool(self.biso_use_cayley_neumann.item()),
                    int(self.biso_num_cayley_neumann_terms.item()),
                    float(self.biso_cayley_neumann_eps.item()),
                    self.biso_primitive_flat_indices,
                ),
            )

        in_raw = self.biso_in_raw
        out_raw = self.biso_out_raw
        if self.biso_in_selective_mask is not None:
            in_mask = self.biso_in_selective_mask.to(device=in_raw.device, dtype=in_raw.dtype)
            in_raw = in_raw * in_mask[:, None, None]
        if self.biso_out_selective_mask is not None:
            out_mask = self.biso_out_selective_mask.to(device=out_raw.device, dtype=out_raw.dtype)
            out_raw = out_raw * out_mask[:, None, None]
        scaling = self.biso_scaling.to(dtype=dtype)
        return _legacy_cayley_blocks(in_raw.to(dtype=dtype), scaling), _legacy_cayley_blocks(
            out_raw.to(dtype=dtype), scaling
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.adapter_disabled:
            return self.base_layer(input)
        in_rot, out_rot = self._rotations(input.dtype)
        transformed_input = _apply_row_blocks(input, in_rot, self.base_layer.in_features, self.block_size)
        result = self.base_layer(transformed_input)
        return _apply_row_blocks(result, out_rot, self.base_layer.out_features, self.block_size)

    @torch.no_grad()
    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.base_layer.weight.detach()
        dtype = weight.dtype
        in_rot, out_rot = self._rotations(torch.float32)
        merged_weight = _right_multiply_blocks(
            weight.float(), in_rot.transpose(-1, -2), self.base_layer.in_features, self.block_size
        )
        merged_weight = _left_multiply_blocks(
            out_rot.transpose(-1, -2), merged_weight, self.base_layer.out_features, self.block_size
        )

        bias = self.base_layer.bias
        if bias is None:
            merged_bias = None
        else:
            merged_bias = _apply_row_blocks(
                bias.detach().float(), out_rot, self.base_layer.out_features, self.block_size
            )
        return merged_weight.to(dtype), None if merged_bias is None else merged_bias.to(bias.dtype)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate historical full-square raw states to primitive coordinates."""

        new_in_key = f"{prefix}biso_in_primitive"
        new_out_key = f"{prefix}biso_out_primitive"
        old_in_key = f"{prefix}biso_in_raw"
        old_out_key = f"{prefix}biso_out_raw"
        if self.parameterization == "primitive_cn" and new_in_key not in state_dict and old_in_key in state_dict:
            rows, cols = _primitive_indices(self.block_size, state_dict[old_in_key].device)
            old_in = state_dict.pop(old_in_key)
            state_dict[new_in_key] = -(old_in[..., rows, cols] - old_in[..., cols, rows])
            if old_out_key in state_dict:
                old_out = state_dict.pop(old_out_key)
                state_dict[new_out_key] = -(old_out[..., rows, cols] - old_out[..., cols, rows])

        defaults = {
            f"{prefix}biso_block_size": self.biso_block_size,
            f"{prefix}biso_use_cayley_neumann": self.biso_use_cayley_neumann,
            f"{prefix}biso_num_cayley_neumann_terms": self.biso_num_cayley_neumann_terms,
            f"{prefix}biso_cayley_neumann_eps": self.biso_cayley_neumann_eps,
        }
        for key, value in defaults.items():
            state_dict.setdefault(key, value.detach().clone())
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_biso_adapters(model: nn.Module, config) -> BISOApplyStats:
    """Replace target ``nn.Linear`` modules with BISO wrappers."""

    model.requires_grad_(False)
    block_size = int(config.get("biso_block_size", config.get("oft_block_size", 32)))
    alpha = float(config.get("biso_alpha", block_size / 2))
    init_std = float(config.get("biso_init_std", 0.0))
    parameterization = str(config.get("biso_parameterization", "primitive_cn"))
    use_cayley_neumann = bool(config.get("biso_use_cayley_neumann", True))
    num_cayley_neumann_terms = int(config.get("biso_num_cayley_neumann_terms", 5))
    cayley_neumann_eps = float(config.get("biso_cayley_neumann_eps", 0.9))
    target_modules = _as_target_set(config.get("target_modules", "all-linear"))
    selective_mode = str(config.get("biso_selective_mode", "none"))
    selective_keep_ratio = float(config.get("biso_selective_keep_ratio", 0.3))

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head") or name.endswith("embed_tokens"):
            continue
        if _matches_target(name, target_modules):
            replacements.append((name, module))

    stats = BISOApplyStats()
    for name, module in replacements:
        wrapped = BISOLinear(
            base_layer=module,
            block_size=block_size,
            alpha=alpha,
            init_std=init_std,
            selective_mode=selective_mode,
            selective_keep_ratio=selective_keep_ratio,
            parameterization=parameterization,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
            cayley_neumann_eps=cayley_neumann_eps,
        )
        _set_module(model, name, wrapped)
        stats.num_layers += 1
        if parameterization == "primitive_cn":
            stats.num_parameters += wrapped.biso_in_primitive.numel() + wrapped.biso_out_primitive.numel()
        else:
            stats.num_parameters += wrapped.biso_in_raw.numel() + wrapped.biso_out_raw.numel()

    @contextmanager
    def disable_adapter(self):
        biso_modules = [module for module in self.modules() if isinstance(module, BISOLinear)]
        old_values = [module.adapter_disabled for module in biso_modules]
        try:
            for module in biso_modules:
                module.adapter_disabled = True
            yield
        finally:
            for module, old_value in zip(biso_modules, old_values, strict=True):
                module.adapter_disabled = old_value

    model.disable_adapter = MethodType(disable_adapter, model)
    return stats
