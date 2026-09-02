"""Bi-orthogonal endpoint transforms for PEFT-style RL training.

This adapter keeps the base linear map frozen and trains low-rank
skew-symmetric transforms on both endpoints:

    y = R_out W R_in x + b

where ``K = P Q^T - Q P^T``. This is the first-order form of applying
orthogonal transforms on the input and output spaces of a linear map. When
``use_cayley_neumann`` is enabled, ``R`` is a truncated Cayley-Neumann
approximation built from the full low-rank skew generator. The generator is
global over the whole input/output channel dimension of each Linear layer; it
is not block diagonal.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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
class BOETApplyStats:
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


class BOETLinear(nn.Module):
    """Wrap ``nn.Linear`` with trainable input/output skew endpoint transforms."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        init_std: float,
        use_cayley_neumann: bool = False,
        num_cayley_neumann_terms: int = 5,
        cayley_neumann_eps: float = 0.9,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"BOETLinear requires rank > 0, got {rank}")
        if num_cayley_neumann_terms <= 0:
            raise ValueError(
                f"num_cayley_neumann_terms must be positive, got {num_cayley_neumann_terms}"
            )
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.use_cayley_neumann = bool(use_cayley_neumann)
        self.num_cayley_neumann_terms = int(num_cayley_neumann_terms)
        self.cayley_neumann_eps = float(cayley_neumann_eps)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.register_buffer("boet_scaling", torch.tensor(self.scaling, **factory_kwargs), persistent=True)
        self.register_buffer(
            "boet_use_cayley_neumann",
            torch.tensor(self.use_cayley_neumann, device=base_layer.weight.device),
            persistent=True,
        )
        self.register_buffer(
            "boet_num_cayley_neumann_terms",
            torch.tensor(self.num_cayley_neumann_terms, device=base_layer.weight.device),
            persistent=True,
        )
        self.register_buffer(
            "boet_cayley_neumann_eps",
            torch.tensor(self.cayley_neumann_eps, **factory_kwargs),
            persistent=True,
        )
        self.boet_in_P = nn.Parameter(torch.empty(in_features, self.rank, **factory_kwargs))
        self.boet_in_Q = nn.Parameter(torch.empty(in_features, self.rank, **factory_kwargs))
        self.boet_out_P = nn.Parameter(torch.empty(out_features, self.rank, **factory_kwargs))
        self.boet_out_Q = nn.Parameter(torch.empty(out_features, self.rank, **factory_kwargs))
        self.adapter_disabled = False
        self.reset_parameters(init_std)

        for param in self.base_layer.parameters():
            param.requires_grad_(False)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.boet_in_P, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.boet_in_Q)
        nn.init.normal_(self.boet_out_P, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.boet_out_Q)

    @staticmethod
    def _right_multiply_skew(x: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Return ``x @ K`` for ``K = P Q^T - Q P^T`` without building ``K``."""

        p_proj = torch.matmul(x, p)
        q_proj = torch.matmul(x, q)
        return torch.matmul(p_proj, q.transpose(0, 1)) - torch.matmul(q_proj, p.transpose(0, 1))

    @staticmethod
    def _skew_frobenius_norm(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute ``||P Q^T - Q P^T||_F`` using only rank-sized matrices."""

        p32 = p.float()
        q32 = q.float()
        ptp = p32.transpose(0, 1) @ p32
        qtq = q32.transpose(0, 1) @ q32
        ptq = p32.transpose(0, 1) @ q32
        norm_sq = 2.0 * torch.sum(ptp * qtq.transpose(0, 1)) - 2.0 * torch.trace(ptq @ ptq)
        return torch.sqrt(torch.clamp(norm_sq, min=0.0))

    def _effective_scaling(self, p: torch.Tensor, q: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        scaling = self.boet_scaling.to(device=p.device, dtype=torch.float32)
        if not self.use_cayley_neumann:
            return scaling.to(dtype=dtype)

        eps = float(self.boet_cayley_neumann_eps.item())
        if eps <= 0.0:
            return scaling.to(dtype=dtype)
        norm = self._skew_frobenius_norm(p, q)
        factor = (eps / (scaling.abs() * norm + 1e-12)).clamp(max=1.0)
        return (scaling * factor).to(dtype=dtype)

    def _right_multiply_rotation_t(self, x: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Return ``x @ R^T``.

        For the first-order adapter this is ``x @ (I - sK)``. For CN this
        applies the same truncated series as BISO's Cayley-Neumann path, but
        directly to row vectors and without materializing the full dense
        rotation during training.
        """

        dtype = x.dtype
        p = p.to(dtype=dtype)
        q = q.to(dtype=dtype)
        scaling = self._effective_scaling(p, q, dtype)

        if not self.use_cayley_neumann:
            return x - self._right_multiply_skew(x, p, q) * scaling

        result = x
        if self.num_cayley_neumann_terms == 1:
            return result

        power = x
        for degree in range(1, self.num_cayley_neumann_terms):
            power = -self._right_multiply_skew(power, p, q) * scaling
            coefficient = 1.0 if degree == self.num_cayley_neumann_terms - 1 else 2.0
            result = result + power * coefficient
        return result

    @staticmethod
    def _dense_rotation(
        p: torch.Tensor,
        q: torch.Tensor,
        scaling: torch.Tensor,
        use_cayley_neumann: bool,
        num_terms: int,
        eps: float,
    ) -> torch.Tensor:
        p = p.float()
        q = q.float()
        k = (p @ q.transpose(0, 1)) - (q @ p.transpose(0, 1))
        effective_scaling = scaling.float()
        if use_cayley_neumann and eps > 0.0:
            norm = torch.linalg.vector_norm(k.reshape(-1))
            factor = (float(eps) / (effective_scaling.abs() * norm + 1e-12)).clamp(max=1.0)
            effective_scaling = effective_scaling * factor
        skew = k * effective_scaling
        eye = torch.eye(skew.shape[0], device=skew.device, dtype=torch.float32)
        if not use_cayley_neumann:
            return eye + skew
        rotation = eye
        if num_terms == 1:
            return rotation
        power = skew
        for degree in range(1, num_terms):
            coefficient = 1.0 if degree == num_terms - 1 else 2.0
            rotation = rotation + power * coefficient
            if degree + 1 < num_terms:
                power = power @ skew
        return rotation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.adapter_disabled:
            return self.base_layer(input)

        dtype = input.dtype

        adapter_input = self.dropout(input)
        transformed_input = self._right_multiply_rotation_t(adapter_input, self.boet_in_P, self.boet_in_Q)

        result = self.base_layer(transformed_input)
        out_input = self.dropout(result)
        return self._right_multiply_rotation_t(out_input, self.boet_out_P, self.boet_out_Q).to(dtype=dtype)

    @torch.no_grad()
    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the dense linear layer equivalent to this adapter."""

        weight = self.base_layer.weight.detach()
        dtype = weight.dtype
        device = weight.device
        scaling = self.boet_scaling.detach().to(device=device, dtype=torch.float32)

        in_p = self.boet_in_P.detach().to(device=device, dtype=torch.float32)
        in_q = self.boet_in_Q.detach().to(device=device, dtype=torch.float32)
        out_p = self.boet_out_P.detach().to(device=device, dtype=torch.float32)
        out_q = self.boet_out_Q.detach().to(device=device, dtype=torch.float32)

        use_cn = bool(self.boet_use_cayley_neumann.item())
        terms = int(self.boet_num_cayley_neumann_terms.item())
        eps = float(self.boet_cayley_neumann_eps.item())
        r_in = self._dense_rotation(in_p, in_q, scaling, use_cn, terms, eps)
        r_out = self._dense_rotation(out_p, out_q, scaling, use_cn, terms, eps)

        merged_weight = r_out @ weight.float() @ r_in
        bias = self.base_layer.bias
        merged_bias = None if bias is None else r_out @ bias.detach().float()
        return merged_weight.to(dtype), None if merged_bias is None else merged_bias.to(bias.dtype)


def _set_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, new_module)


def apply_boet_adapters(model: nn.Module, config) -> BOETApplyStats:
    """Replace target ``nn.Linear`` modules with ``BOETLinear`` wrappers."""

    model.requires_grad_(False)

    rank = int(config.get("boet_rank", config.get("skew_rank", config.get("lora_rank", 0))))
    if rank <= 0:
        raise ValueError("BOET PEFT requires model.boet_rank, model.skew_rank, or model.lora_rank > 0")

    alpha = float(config.get("boet_alpha", config.get("skew_alpha", config.get("lora_alpha", rank))))
    dropout = float(config.get("boet_dropout", config.get("skew_dropout", config.get("lora_dropout", 0.0))))
    init_std = float(config.get("boet_init_std", config.get("skew_init_std", 0.01)))
    use_cayley_neumann = bool(config.get("boet_use_cayley_neumann", False))
    num_cayley_neumann_terms = int(config.get("boet_num_cayley_neumann_terms", 5))
    cayley_neumann_eps = float(config.get("boet_cayley_neumann_eps", 0.9))
    target_modules = _as_target_set(config.get("target_modules", "all-linear"))

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head") or name.endswith("embed_tokens"):
            continue
        if _matches_target(name, target_modules):
            replacements.append((name, module))

    stats = BOETApplyStats()
    for name, module in replacements:
        wrapped = BOETLinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            init_std=init_std,
            use_cayley_neumann=use_cayley_neumann,
            num_cayley_neumann_terms=num_cayley_neumann_terms,
            cayley_neumann_eps=cayley_neumann_eps,
        )
        _set_module(model, name, wrapped)
        stats.num_layers += 1
        stats.num_parameters += (
            wrapped.boet_in_P.numel()
            + wrapped.boet_in_Q.numel()
            + wrapped.boet_out_P.numel()
            + wrapped.boet_out_Q.numel()
        )

    @contextmanager
    def disable_adapter(self):
        boet_modules = [module for module in self.modules() if isinstance(module, BOETLinear)]
        old_values = [module.adapter_disabled for module in boet_modules]
        try:
            for module in boet_modules:
                module.adapter_disabled = True
            yield
        finally:
            for module, old_value in zip(boet_modules, old_values, strict=True):
                module.adapter_disabled = old_value

    model.disable_adapter = MethodType(disable_adapter, model)
    return stats
