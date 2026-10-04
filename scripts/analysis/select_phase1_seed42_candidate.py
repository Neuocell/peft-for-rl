#!/usr/bin/env python3
"""Select the preregistered Phase-1 mixed initialization using seed 42 only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.analysis.compare_paired_full_bench import EXPECTED_BENCHMARKS
from scripts.analysis.evaluate_single_seed_screen import evaluate


MIXED_METHODS = ("I8", "I16")
CONTROL_METHOD = "I32"


def select_phase1_seed42_candidate(
    comparisons: dict[str, dict[str, Any]],
    *,
    minimum_delta: float = 0.005,
    minimum_probability: float = 0.90,
) -> dict[str, Any]:
    expected = {*MIXED_METHODS, CONTROL_METHOD}
    if set(comparisons) != expected:
        raise ValueError(f"Phase-1 comparisons must be exactly {sorted(expected)}")

    candidates: dict[str, Any] = {}
    for method in (*MIXED_METHODS, CONTROL_METHOD):
        comparison = comparisons[method]
        if int(comparison.get("num_samples", -1)) != 7248:
            raise ValueError(
                f"Phase-1 {method} comparison does not contain 7248 samples"
            )
        if int(comparison.get("bootstrap_samples", -1)) != 10_000:
            raise ValueError(
                f"Phase-1 {method} comparison does not use 10000 bootstrap samples"
            )
        if int(comparison.get("bootstrap_seed", -1)) != 42:
            raise ValueError(f"Phase-1 {method} comparison does not use seed 42")
        observed_benchmarks = {
            str(row.get("benchmark")) for row in comparison.get("benchmarks", [])
        }
        if observed_benchmarks != set(EXPECTED_BENCHMARKS):
            raise ValueError(f"Phase-1 {method} comparison benchmark set differs")
        screen = evaluate(
            comparison,
            minimum_delta=minimum_delta,
            minimum_probability=minimum_probability,
        )
        candidates[method] = {
            "eligible_for_selection": method in MIXED_METHODS,
            "macro_delta_avg_at_k": float(comparison["macro_delta_avg_at_k"]),
            "probability_delta_gt_zero": float(comparison["probability_delta_gt_zero"]),
            "screen_decision": screen["decision"],
            "screen_reason": screen["reason"],
        }

    passing = [
        method
        for method in MIXED_METHODS
        if candidates[method]["screen_decision"] == "advance_multiseed"
    ]
    # The numeric suffix is the deterministic tie-break: less signal and more
    # random coverage wins an exact macro-delta tie.
    selected = (
        min(
            passing,
            key=lambda method: (
                -candidates[method]["macro_delta_avg_at_k"],
                int(method[1:]),
            ),
        )
        if passing
        else None
    )
    return {
        "schema_version": 1,
        "method": "phase1_seed42_mixed_candidate_selection_v1",
        "baseline_method": "I0",
        "selection_seed": 42,
        "selection_metric": "macro_delta_avg_at_k",
        "eligible_methods": list(MIXED_METHODS),
        "signal_only_control": CONTROL_METHOD,
        "thresholds": {
            "minimum_macro_delta_avg_at_k": minimum_delta,
            "minimum_probability_delta_gt_zero": minimum_probability,
        },
        "tie_break": "smaller signal direction count",
        "candidates": candidates,
        "selected_method": selected,
        "decision": "advance_multiseed" if selected else "stop_single_seed",
        "reason": (
            "best seed-42 mixed candidate passed the preregistered screen"
            if selected
            else "neither mixed candidate passed the preregistered seed-42 screen"
        ),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i8-comparison", type=Path, required=True)
    parser.add_argument("--i16-comparison", type=Path, required=True)
    parser.add_argument("--i32-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delta", type=float, default=0.005)
    parser.add_argument("--minimum-probability", type=float, default=0.90)
    args = parser.parse_args()
    comparisons = {
        "I8": json.loads(args.i8_comparison.read_text(encoding="utf-8")),
        "I16": json.loads(args.i16_comparison.read_text(encoding="utf-8")),
        "I32": json.loads(args.i32_comparison.read_text(encoding="utf-8")),
    }
    result = select_phase1_seed42_candidate(
        comparisons,
        minimum_delta=args.minimum_delta,
        minimum_probability=args.minimum_probability,
    )
    _write_json_atomic(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
