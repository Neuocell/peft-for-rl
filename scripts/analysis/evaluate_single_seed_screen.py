#!/usr/bin/env python3
"""Apply the preregistered seed-42 full-benchmark screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def evaluate(
    comparison: dict[str, Any],
    *,
    minimum_delta: float = 0.005,
    minimum_probability: float = 0.90,
) -> dict[str, Any]:
    if comparison.get("record_keys_exactly_matched") is not True:
        raise ValueError("Paired evaluation record keys were not exactly matched")
    if comparison.get("interpretation") != "single_seed_screen_only":
        raise ValueError("Input is not a single-seed paired screen")
    delta = float(comparison["macro_delta_avg_at_k"])
    probability = float(comparison["probability_delta_gt_zero"])
    passed = delta >= minimum_delta and probability >= minimum_probability
    return {
        "schema_version": 1,
        "method": "preregistered_single_seed_training_screen_v1",
        "thresholds": {
            "minimum_macro_delta_avg_at_k": minimum_delta,
            "minimum_probability_delta_gt_zero": minimum_probability,
        },
        "observed": {
            "macro_delta_avg_at_k": delta,
            "probability_delta_gt_zero": probability,
        },
        "decision": "advance_multiseed" if passed else "stop_single_seed",
        "reason": (
            "seed-42 effect and paired problem-bootstrap probability passed"
            if passed
            else "seed-42 effect did not pass the preregistered multi-seed screen"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delta", type=float, default=0.005)
    parser.add_argument("--minimum-probability", type=float, default=0.90)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    result = evaluate(
        comparison,
        minimum_delta=args.minimum_delta,
        minimum_probability=args.minimum_probability,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
