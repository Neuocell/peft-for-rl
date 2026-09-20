"""Utilities that connect verl training loops to PEFT's official AdaLoRA."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _regular_module_names(value: Any) -> Any:
    if value is None or isinstance(value, (str, set)):
        return value
    return list(value)


def build_adalora_config(model_config: Any, total_training_steps: int | None = None):
    """Build the installed PEFT release's official ``AdaLoraConfig``.

    ``r`` is intentionally omitted: PEFT ignores it for AdaLoRA. The initial
    decomposition rank is controlled exclusively by ``init_r``.
    """

    try:
        from peft import AdaLoraConfig, TaskType
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise ImportError("AdaLoRA requires a PEFT release that exports AdaLoraConfig") from exc

    init_r = int(_get(model_config, "adalora_init_r", 0))
    target_r = int(_get(model_config, "adalora_target_r", 0))
    configured_total_step = int(_get(model_config, "adalora_total_step", 0) or 0)
    total_step = configured_total_step or int(total_training_steps or 0)

    if init_r <= 0:
        raise ValueError("adalora_init_r must be greater than zero")
    if target_r <= 0 or target_r > init_r:
        raise ValueError("adalora_target_r must be greater than zero and no larger than adalora_init_r")
    if total_step <= 0:
        raise ValueError("adalora_total_step must be the number of optimizer updates and greater than zero")

    return AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=_regular_module_names(_get(model_config, "target_modules", "all-linear")),
        exclude_modules=_regular_module_names(_get(model_config, "exclude_modules", None)),
        bias="none",
        init_r=init_r,
        target_r=target_r,
        lora_alpha=int(_get(model_config, "lora_alpha", 16)),
        lora_dropout=float(_get(model_config, "lora_dropout", 0.0)),
        tinit=int(_get(model_config, "adalora_tinit", 0)),
        tfinal=int(_get(model_config, "adalora_tfinal", 0)),
        deltaT=int(_get(model_config, "adalora_delta_t", 1)),
        beta1=float(_get(model_config, "adalora_beta1", 0.85)),
        beta2=float(_get(model_config, "adalora_beta2", 0.85)),
        orth_reg_weight=float(_get(model_config, "adalora_orth_reg_weight", 1e-3)),
        total_step=total_step,
    )


def find_adalora_model(module: torch.nn.Module):
    """Return the real PEFT ``AdaLoraModel``, never a delegating wrapper."""

    try:
        from peft.tuners.adalora.model import AdaLoraModel
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise ImportError("AdaLoRA requires peft.tuners.adalora.model.AdaLoraModel") from exc

    for candidate in module.modules():
        if isinstance(candidate, AdaLoraModel):
            return candidate
    return None


def adalora_config_for_vllm(adalora_config):
    """Return a LoRA-compatible config view with rank matching exported A/B."""

    rollout_config = copy.deepcopy(adalora_config)
    rollout_config.r = rollout_config.init_r
    return rollout_config


def update_and_allocate(
    module: torch.nn.Module,
    optimizer_step: int,
    *,
    adalora_model=None,
) -> dict[str, float]:
    """Run PEFT's allocator after one successful optimizer update.

    PEFT's allocator reads the complete A/B/E values and their gradients. For
    FSDP1 the full tensors are materialized on every rank and written back so
    every worker applies the same mask. The caller must invoke this before
    gradients are cleared.
    """

    if optimizer_step <= 0:
        raise ValueError("optimizer_step must start at one")

    adalora_model = adalora_model or find_adalora_model(module)
    if adalora_model is None:
        raise RuntimeError("PEFT AdaLoraModel was not found in the wrapped actor")

    rankallocator = adalora_model.rankallocator
    budget, mask_due = rankallocator.budget_schedule(optimizer_step)
    module_count = len(rankallocator.name_set)

    summon_ctx = (
        FSDP.summon_full_params(module, writeback=True, with_grads=True) if isinstance(module, FSDP) else nullcontext()
    )
    with summon_ctx:
        adalora_model.update_and_allocate(optimizer_step)

        current_ranks = []
        for candidate in adalora_model.modules():
            lora_e = getattr(candidate, "lora_E", None)
            if lora_e is None:
                continue
            for value in lora_e.values():
                current_ranks.append(float((value.detach().abs().view(-1) > 1e-8).sum().item()))

        final_phase = optimizer_step >= rankallocator.peft_config.total_step - rankallocator.peft_config.tfinal
        mask_metrics: dict[str, float] = {}
        if current_ranks and (mask_due or final_phase):
            uncertainty_sum = None
            importance_sum = None
            score_sum = None
            state_numel = 0
            for name, uncertainty in rankallocator.exp_avg_unc.items():
                importance = rankallocator.exp_avg_ipt[name]
                unc_sum = uncertainty.float().sum()
                ipt_sum = importance.float().sum()
                pair_sum = (uncertainty.float() * importance.float()).sum()
                uncertainty_sum = unc_sum if uncertainty_sum is None else uncertainty_sum + unc_sum
                importance_sum = ipt_sum if importance_sum is None else importance_sum + ipt_sum
                score_sum = pair_sum if score_sum is None else score_sum + pair_sum
                state_numel += uncertainty.numel()

            mask_metrics = {
                "adalora/mask_applied": 1.0,
                "adalora/masked_at_optimizer_step": float(optimizer_step),
                "adalora/masked_rank_mean": sum(current_ranks) / len(current_ranks),
                "adalora/masked_rank_min": min(current_ranks),
                "adalora/masked_rank_max": max(current_ranks),
                "adalora/masked_zero_module_ratio": sum(rank == 0 for rank in current_ranks) / len(current_ranks),
            }
            if state_numel:
                mask_metrics.update(
                    {
                        "adalora/importance_ema_mean": (importance_sum / state_numel).item(),
                        "adalora/uncertainty_ema_mean": (uncertainty_sum / state_numel).item(),
                        "adalora/importance_uncertainty_score_mean": (score_sum / state_numel).item(),
                    }
                )

    metrics = {
        "adalora/optimizer_step": float(optimizer_step),
        "adalora/scheduled_rank_mean": float(budget / module_count) if module_count else 0.0,
    }
    metrics.update(mask_metrics)
    if current_ranks:
        metrics.update(
            {
                "adalora/current_nonzero_rank_mean": sum(current_ranks) / len(current_ranks),
                "adalora/current_nonzero_rank_min": min(current_ranks),
                "adalora/current_nonzero_rank_max": max(current_ranks),
            }
        )
    return metrics
