#!/usr/bin/env python3
"""Compare two six-task full-bench summaries and record the next experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SAMPLES = 7248
EXPECTED_BENCHMARKS = {"aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--match-margin", type=float, default=0.02)
    return parser.parse_args()


def _load_checked(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {row["benchmark"] for row in payload.get("benchmarks", [])}
    if payload.get("num_samples") != EXPECTED_SAMPLES:
        raise ValueError(f"{path} has {payload.get('num_samples')} samples, expected {EXPECTED_SAMPLES}")
    if names != EXPECTED_BENCHMARKS:
        raise ValueError(f"{path} benchmark set is {sorted(names)}, expected {sorted(EXPECTED_BENCHMARKS)}")
    if payload.get("num_benchmarks") != len(EXPECTED_BENCHMARKS):
        raise ValueError(f"{path} does not contain six benchmark summaries")
    return payload


def _rows_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["benchmark"]: row for row in payload["benchmarks"]}


def compare(candidate: dict[str, Any], baseline: dict[str, Any], match_margin: float) -> dict[str, Any]:
    candidate_rows = _rows_by_name(candidate)
    baseline_rows = _rows_by_name(baseline)
    comparisons = []
    for name in sorted(EXPECTED_BENCHMARKS):
        current = candidate_rows[name]
        reference = baseline_rows[name]
        comparisons.append(
            {
                "benchmark": name,
                "candidate_avg_at_k": current["macro_avg_at_k"],
                "baseline_avg_at_k": reference["macro_avg_at_k"],
                "delta_avg_at_k": current["macro_avg_at_k"] - reference["macro_avg_at_k"],
                "candidate_pass_at_k": current["pass_at_k"],
                "delta_pass_at_k": current["pass_at_k"] - reference["pass_at_k"],
                "candidate_parse_rate": current["parse_rate"],
                "delta_parse_rate": current["parse_rate"] - reference["parse_rate"],
                "candidate_hit_max_rate": current["hit_max_rate"],
                "delta_hit_max_rate": current["hit_max_rate"] - reference["hit_max_rate"],
                "candidate_length_mean": current["length_mean"],
                "delta_length_mean": current["length_mean"] - reference["length_mean"],
            }
        )

    candidate_macro = float(candidate["macro_avg_at_k_over_benchmarks"])
    baseline_macro = float(baseline["macro_avg_at_k_over_benchmarks"])
    macro_delta = candidate_macro - baseline_macro
    heldout_gate = "matches_wide_step50" if macro_delta >= -match_margin else "regresses_vs_wide_step50"
    rationale = (
        "The equivalent-rank-8 budget remains viable; isolate whether heterogeneous rank allocation adds value."
        if heldout_gate == "matches_wide_step50"
        else "Separate a poor heterogeneous allocation from insufficient total rank-8 capacity."
    )
    return {
        "candidate": candidate["checkpoint"],
        "baseline": baseline["checkpoint"],
        "candidate_num_samples": candidate["num_samples"],
        "baseline_num_samples": baseline["num_samples"],
        "match_margin": match_margin,
        "heldout_gate": heldout_gate,
        "candidate_macro_avg_at_k": candidate_macro,
        "baseline_macro_avg_at_k": baseline_macro,
        "delta_macro_avg_at_k": macro_delta,
        "candidate_macro_pass_at_k": candidate["macro_pass_at_k_over_benchmarks"],
        "delta_macro_pass_at_k": (
            candidate["macro_pass_at_k_over_benchmarks"] - baseline["macro_pass_at_k_over_benchmarks"]
        ),
        "candidate_parse_rate": candidate["parse_rate_overall"],
        "delta_parse_rate": candidate["parse_rate_overall"] - baseline["parse_rate_overall"],
        "candidate_hit_max_rate": candidate["hit_max_rate_overall"],
        "delta_hit_max_rate": candidate["hit_max_rate_overall"] - baseline["hit_max_rate_overall"],
        "candidate_length_mean": candidate["length_mean_overall"],
        "delta_length_mean": candidate["length_mean_overall"] - baseline["length_mean_overall"],
        "benchmarks": comparisons,
        "next_experiment": "dominant_atoms_uniform_r8_trainableA_a2_b64m16n8_50_v1",
        "next_experiment_rationale": rationale,
        "controlled_difference": "heterogeneous rank map {4,8,12,16} -> uniform rank 8",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dominant-Atoms Step-50 Full-Bench Decision",
        "",
        f"- Gate: `{report['heldout_gate']}`",
        f"- Candidate macro Avg@k: `{report['candidate_macro_avg_at_k']:.4%}`",
        f"- GradTop step-50 macro Avg@k: `{report['baseline_macro_avg_at_k']:.4%}`",
        f"- Difference: `{report['delta_macro_avg_at_k']:+.4%}`",
        f"- Next experiment: `{report['next_experiment']}`",
        f"- Reason: {report['next_experiment_rationale']}",
        "",
        "| Benchmark | Candidate Avg@k | Baseline Avg@k | Delta | Pass@k delta | Parse delta | Hit-max delta | Length delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["benchmarks"]:
        lines.append(
            f"| {row['benchmark']} | {row['candidate_avg_at_k']:.4%} | {row['baseline_avg_at_k']:.4%} | "
            f"{row['delta_avg_at_k']:+.4%} | {row['delta_pass_at_k']:+.4%} | "
            f"{row['delta_parse_rate']:+.4%} | {row['delta_hit_max_rate']:+.4%} | "
            f"{row['delta_length_mean']:+.1f} |"
        )
    lines.extend(
        [
            "",
            "The next run keeps the same probe observations, dominant consensus basis construction, equivalent parameter budget, scaling, and optimizer configuration. Only the per-module rank allocation changes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    candidate = _load_checked(args.candidate.resolve())
    baseline = _load_checked(args.baseline.resolve())
    report = compare(candidate, baseline, args.match_margin)
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_prefix.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
