# Copyright 2026 Individual contributors.
#
# PeRL-style math outcome reward for local LoRA RLVR baselines.
# This intentionally keeps the existing DAPO/Minerva answer parser used by
# our data pipeline, but changes the reward scale from +1/-1 to 1/0.

from typing import Optional

from verl.utils.reward_score.math_dapo import verify


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict:
    """Return PeRL-style binary accuracy reward.

    PeRL's OpenR1/DAPO LoRA recipe uses accuracy-only rewards with no format
    reward and no cosine or overlong shaping. For our local DAPO parquet, the
    prompt asks for a final `Answer: ...`, so we reuse the existing
    `math_dapo.verify` parser and only alter the incorrect reward value.
    """

    del data_source, extra_info

    solution_tail = solution_str[-300:]
    correct, pred = verify(
        solution_tail,
        ground_truth,
        strict_box_verify=strict_box_verify,
        pause_tokens_index=pause_tokens_index,
    )

    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "pred": pred,
    }
