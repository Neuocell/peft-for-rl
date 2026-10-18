"""Explicit, microbatch-invariant objectives for frozen-policy GradTop probes.

Prompt signs multiply the already clipped PPO loss. Flipping the advantages
before clipping would select a different clipping branch and is not equivalent.
These objectives are deliberately opt-in and are never used for policy updates.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist


GRADTOP_VARIANTS = frozenset({"legacy", "token_mean", "prompt_mean", "prompt_sign", "masked"})


def gradtop_probe_variant(model_config) -> str:
    """Keep all ordinary training and historical probes on the existing path."""
    if model_config is None or model_config.get("peft_type", "lora") != "grad_probe":
        return "legacy"
    variant = str(model_config.get("gradient_probe_variant", "legacy")).lower()
    if variant not in GRADTOP_VARIANTS:
        raise ValueError(f"Unknown gradient_probe_variant: {variant}")
    return variant


def validate_gradtop_actor_config(config) -> None:
    if config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
        raise ValueError("GradTop objective variants require vanilla clipped PPO loss")
    if int(config.ppo_epochs) != 1:
        raise ValueError("GradTop objective variants require ppo_epochs=1")
    if float(config.entropy_coeff) != 0 or bool(config.use_kl_loss):
        raise ValueError("GradTop objective variants require entropy_coeff=0 and use_kl_loss=false")
    if int(config.get("ulysses_sequence_parallel_size", 1)) != 1:
        raise ValueError("GradTop objective variants require sequence parallel size 1")
    if float(config.get("biso_raw_l2_coef", 0.0)) != 0:
        raise ValueError("GradTop objective variants require biso_raw_l2_coef=0")


@dataclass(frozen=True)
class GradTopNormalization:
    """Global minibatch counts, fixed before token packing or microbatching."""

    prompt_groups: int
    responses: int
    valid_tokens: int
    dp_size: int
    rollouts_per_prompt: int

    def metadata(self) -> dict[str, int]:
        return {
            "prompt_groups": self.prompt_groups,
            "responses": self.responses,
            "valid_tokens": self.valid_tokens,
        }


def build_gradtop_normalization(
    prompt_ids: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    expected_rollouts: int,
    collective_device=None,
    process_group=None,
) -> GradTopNormalization:
    """Validate complete local groups and sum counts across the actor DP world.

    The trainer preserves contiguous prompt groups and assigns each to exactly one
    DP rank. SP=1 is required. The objective compensates for FSDP gradient averaging
    with ``dp_size``; no additional microbatch loss scaling may be applied.
    """
    if prompt_ids.ndim != 1 or response_mask.ndim != 2 or len(prompt_ids) != len(response_mask):
        raise ValueError("GradTop expects one integer prompt ID per response")
    if prompt_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("GradTop prompt IDs must be integer tensors")
    if expected_rollouts < 1:
        raise ValueError("GradTop expected_rollouts must be positive")
    unique_ids, counts = torch.unique(prompt_ids, return_counts=True)
    invalid = not len(unique_ids) or bool((counts != expected_rollouts).any())
    invalid = invalid or bool((response_mask.sum(-1) <= 0).any())
    totals = torch.tensor(
        [len(unique_ids), len(prompt_ids), int(response_mask.sum().item()), int(invalid)],
        dtype=torch.int64,
        device=collective_device if collective_device is not None else response_mask.device,
    )
    dp_size = 1
    if dist.is_available() and dist.is_initialized():
        dp_size = dist.get_world_size(process_group)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=process_group)
    groups, responses, tokens, invalid_ranks = totals.tolist()
    if invalid_ranks:
        raise ValueError("GradTop requires complete prompt groups with equal rollout counts and nonempty responses")
    return GradTopNormalization(groups, responses, tokens, dp_size, expected_rollouts)


def gradtop_prompt_signs(
    prompt_ids: torch.Tensor,
    *,
    seed: int = 42,
    trainer_step: int,
    minibatch_index: int,
) -> torch.Tensor:
    """Stable Rademacher signs shared by every rollout of the same prompt.

    No Python hash, RNG state, UUID, rank, or microbatch position enters the key.
    """
    ids = prompt_ids.detach().cpu().tolist()
    values = {}
    for prompt_id in set(ids):
        key = f"gradtop-prompt-sign-v1:{seed}:{trainer_step}:{minibatch_index}:{prompt_id}"
        bit = hashlib.blake2b(key.encode("ascii"), digest_size=8).digest()[0] & 1
        values[prompt_id] = float(2 * bit - 1)
    return torch.tensor([values[prompt_id] for prompt_id in ids], device=prompt_ids.device)


def gradtop_policy_loss(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    variant: str,
    normalization: GradTopNormalization,
    config,
    signs: torch.Tensor | None = None,
    token_keep_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute exact vanilla dual-clipped loss with explicit probe aggregation.

    Token/masked objectives divide by the original global valid-token count.
    Prompt objectives average token means over each prompt's responses. The signed
    sum divides by sqrt(P), so its expected outer product is the mean uncentered
    prompt-gradient second moment, including zero-advantage prompt groups.
    """
    if variant not in GRADTOP_VARIANTS - {"legacy"}:
        raise ValueError(f"Unsupported explicit GradTop objective: {variant}")
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clip_c = float(config.get("clip_ratio_c", 3.0))
    if clip_c <= 1:
        raise ValueError("clip_ratio_c must be greater than 1")
    delta = (log_prob - old_log_prob).clamp(-20.0, 20.0)
    ratio = delta.exp()
    raw = -advantages * ratio
    clipped = -advantages * ratio.clamp(1 - clip_low, 1 + clip_high)
    upper = torch.maximum(raw, clipped)
    dual_bound = -advantages * clip_c
    per_token = torch.where(advantages < 0, torch.minimum(dual_bound, upper), upper)
    keep_mask = response_mask
    if variant == "masked":
        if token_keep_mask is None or token_keep_mask.shape != response_mask.shape:
            raise ValueError("masked GradTop requires a token_keep_mask matching response_mask")
        keep_mask = token_keep_mask.detach() * response_mask
    if variant in {"token_mean", "masked"}:
        loss = (per_token * keep_mask).sum() / normalization.valid_tokens
    else:
        response_means = (per_token * response_mask).sum(-1) / response_mask.sum(-1)
        denominator = normalization.prompt_groups * normalization.rollouts_per_prompt
        if variant == "prompt_sign":
            if signs is None or signs.shape != response_means.shape:
                raise ValueError("prompt_sign GradTop requires one sign per response")
            response_means = response_means * signs.detach()
            denominator = math.sqrt(normalization.prompt_groups) * normalization.rollouts_per_prompt
        loss = response_means.sum() / denominator
    loss = loss * normalization.dp_size
    local_tokens = response_mask.sum().clamp_min(1)
    metrics = {
        "actor/pg_clipfrac": float((((clipped > raw).float() * response_mask).sum() / local_tokens).detach()),
        "actor/ppo_kl": float(((-delta * response_mask).sum() / local_tokens).detach()),
        "actor/pg_clipfrac_lower": float(
            (((upper > dual_bound) * (advantages < 0) * response_mask).sum() / local_tokens).detach()
        ),
    }
    if variant == "masked":
        metrics["gradient_probe/token_keep_fraction"] = float((keep_mask.sum() / local_tokens).detach())
    return loss, metrics
