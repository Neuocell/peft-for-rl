#!/usr/bin/env python3
"""Compare two matched full-benchmark runs with problem-level paired bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_BENCHMARKS = ("aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva")


def _read_records(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["benchmark"]),
                int(row["problem_index"]),
                int(row["sample_index"]),
            )
            if key in records:
                raise ValueError(f"Duplicate record key at {path}:{line_number}: {key}")
            if not isinstance(row.get("correct"), bool):
                raise ValueError(f"Non-boolean correctness at {path}:{line_number}")
            records[key] = row
    return records


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate a quantile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compare_paired_records(
    baseline_path: Path,
    candidate_path: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    baseline = _read_records(baseline_path.expanduser().resolve())
    candidate = _read_records(candidate_path.expanduser().resolve())
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))[:3]
        extra = sorted(set(candidate) - set(baseline))[:3]
        raise ValueError(f"Evaluation record keys differ: missing={missing}, extra={extra}")
    observed_benchmarks = tuple(sorted({key[0] for key in baseline}))
    if observed_benchmarks != tuple(sorted(EXPECTED_BENCHMARKS)):
        raise ValueError(f"Benchmark set differs: {observed_benchmarks}")

    problem_deltas: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    baseline_scores: dict[str, list[float]] = defaultdict(list)
    candidate_scores: dict[str, list[float]] = defaultdict(list)
    invariant_fields = ("id", "problem", "gold_answer", "samples_per_problem")
    for key in sorted(baseline):
        first = baseline[key]
        second = candidate[key]
        for field in invariant_fields:
            if first.get(field) != second.get(field):
                raise ValueError(f"Matched record invariant {field} differs for {key}")
        first_score = float(first["correct"])
        second_score = float(second["correct"])
        baseline_scores[key[0]].append(first_score)
        candidate_scores[key[0]].append(second_score)
        problem_deltas[key[0]][key[1]].append(second_score - first_score)

    benchmark_rows: dict[str, dict[str, Any]] = {}
    problem_means: dict[str, list[float]] = {}
    for benchmark in EXPECTED_BENCHMARKS:
        by_problem = problem_deltas[benchmark]
        problem_means[benchmark] = [
            statistics.fmean(by_problem[index]) for index in sorted(by_problem)
        ]
        baseline_mean = statistics.fmean(baseline_scores[benchmark])
        candidate_mean = statistics.fmean(candidate_scores[benchmark])
        benchmark_rows[benchmark] = {
            "benchmark": benchmark,
            "num_problems": len(by_problem),
            "num_samples": len(baseline_scores[benchmark]),
            "baseline_avg_at_k": baseline_mean,
            "candidate_avg_at_k": candidate_mean,
            "delta_avg_at_k": candidate_mean - baseline_mean,
        }

    observed_macro_delta = statistics.fmean(
        benchmark_rows[name]["delta_avg_at_k"] for name in EXPECTED_BENCHMARKS
    )
    rng = random.Random(seed)
    macro_draws: list[float] = []
    benchmark_draws: dict[str, list[float]] = {name: [] for name in EXPECTED_BENCHMARKS}
    for _ in range(bootstrap_samples):
        current = []
        for benchmark in EXPECTED_BENCHMARKS:
            values = problem_means[benchmark]
            draw = statistics.fmean(rng.choice(values) for _ in range(len(values)))
            benchmark_draws[benchmark].append(draw)
            current.append(draw)
        macro_draws.append(statistics.fmean(current))

    def interval(draws: list[float]) -> dict[str, float]:
        ordered = sorted(draws)
        positive = sum(value > 0 for value in draws)
        ties = sum(value == 0 for value in draws)
        return {
            "ci95_low": _quantile(ordered, 0.025),
            "ci95_high": _quantile(ordered, 0.975),
            "probability_delta_gt_zero": (positive + 0.5 * ties) / len(draws),
        }

    for benchmark in EXPECTED_BENCHMARKS:
        benchmark_rows[benchmark].update(interval(benchmark_draws[benchmark]))
    baseline_checkpoint = next(iter(baseline.values())).get("checkpoint")
    candidate_checkpoint = next(iter(candidate.values())).get("checkpoint")
    return {
        "schema_version": 1,
        "method": "matched_problem_cluster_bootstrap_v1",
        "baseline_checkpoint": baseline_checkpoint,
        "candidate_checkpoint": candidate_checkpoint,
        "record_keys_exactly_matched": True,
        "num_samples": len(baseline),
        "num_problems": sum(len(problem_deltas[name]) for name in EXPECTED_BENCHMARKS),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "macro_delta_avg_at_k": observed_macro_delta,
        **interval(macro_draws),
        "benchmarks": [benchmark_rows[name] for name in EXPECTED_BENCHMARKS],
        "interpretation": "single_seed_screen_only",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired Full-Benchmark Comparison",
        "",
        f"- Baseline: `{report['baseline_checkpoint']}`",
        f"- Candidate: `{report['candidate_checkpoint']}`",
        f"- Macro Avg@k delta: `{report['macro_delta_avg_at_k']:+.4%}`",
        f"- Problem-bootstrap 95% CI: `[{report['ci95_low']:+.4%}, {report['ci95_high']:+.4%}]`",
        f"- P(delta > 0): `{report['probability_delta_gt_zero']:.4f}`",
        "- Scope: single-seed screening evidence; multi-seed confirmation is still required.",
        "",
        "| Benchmark | Baseline | Candidate | Delta | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["benchmarks"]:
        lines.append(
            f"| {row['benchmark']} | {row['baseline_avg_at_k']:.4%} | "
            f"{row['candidate_avg_at_k']:.4%} | {row['delta_avg_at_k']:+.4%} | "
            f"[{row['ci95_low']:+.4%}, {row['ci95_high']:+.4%}] |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--candidate-records", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = compare_paired_records(
        args.baseline_records,
        args.candidate_records,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, json_path)
    temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary.write_text(_markdown(report), encoding="utf-8")
    os.replace(temporary, markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
