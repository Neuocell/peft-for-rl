#!/usr/bin/env python3
"""Aggregate matched full-benchmark pairs across fixed training seeds."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.analysis.compare_paired_full_bench import EXPECTED_BENCHMARKS, _read_records


EXPECTED_TRAINING_SEEDS = (42, 43, 44)
EXPECTED_SAMPLES = 7_248


def _load_pair(
    baseline_path: Path,
    candidate_path: Path,
    *,
    expected_samples: int,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, int, int], tuple[Any, ...]]]:
    baseline = _read_records(baseline_path.expanduser().resolve())
    candidate = _read_records(candidate_path.expanduser().resolve())
    if set(baseline) != set(candidate):
        raise ValueError("Baseline/candidate record keys differ within a training seed")
    if len(baseline) != expected_samples:
        raise ValueError(
            f"Each training seed must contain exactly {expected_samples} samples; "
            f"found {len(baseline)}"
        )
    if tuple(sorted({key[0] for key in baseline})) != tuple(sorted(EXPECTED_BENCHMARKS)):
        raise ValueError("A training seed does not contain the six expected benchmarks")
    invariants: dict[tuple[str, int, int], tuple[Any, ...]] = {}
    by_problem: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in sorted(baseline):
        first = baseline[key]
        second = candidate[key]
        invariant = (
            first.get("id"),
            first.get("problem"),
            first.get("gold_answer"),
            first.get("samples_per_problem"),
        )
        if invariant != (
            second.get("id"),
            second.get("problem"),
            second.get("gold_answer"),
            second.get("samples_per_problem"),
        ):
            raise ValueError(f"Matched record invariant differs for {key}")
        invariants[key] = invariant
        by_problem[key[0]][key[1]].append(
            float(second["correct"]) - float(first["correct"])
        )
    problem_means = {
        benchmark: np.asarray(
            [
                np.mean(by_problem[benchmark][index], dtype=np.float64)
                for index in sorted(by_problem[benchmark])
            ],
            dtype=np.float64,
        )
        for benchmark in EXPECTED_BENCHMARKS
    }
    return problem_means, invariants


def aggregate(
    baseline_paths: list[Path],
    candidate_paths: list[Path],
    training_seeds: list[int],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
    expected_samples: int = EXPECTED_SAMPLES,
) -> dict[str, Any]:
    if tuple(training_seeds) != EXPECTED_TRAINING_SEEDS:
        raise ValueError(f"Training seeds must be exactly {EXPECTED_TRAINING_SEEDS}")
    if len(baseline_paths) != 3 or len(candidate_paths) != 3:
        raise ValueError("Exactly three baseline/candidate record pairs are required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")

    per_seed: list[dict[str, np.ndarray]] = []
    reference_invariants = None
    for baseline_path, candidate_path in zip(baseline_paths, candidate_paths, strict=True):
        values, invariants = _load_pair(
            baseline_path, candidate_path, expected_samples=expected_samples
        )
        if reference_invariants is None:
            reference_invariants = invariants
        elif invariants != reference_invariants:
            raise ValueError("Benchmark snapshot or sample keys differ across training seeds")
        per_seed.append(values)

    observed_by_seed = []
    for seed, values in zip(training_seeds, per_seed, strict=True):
        benchmark_deltas = {
            benchmark: float(values[benchmark].mean())
            for benchmark in EXPECTED_BENCHMARKS
        }
        observed_by_seed.append(
            {
                "training_seed": seed,
                "macro_delta_avg_at_k": float(np.mean(list(benchmark_deltas.values()))),
                "benchmark_deltas": benchmark_deltas,
            }
        )
    seed_deltas = [item["macro_delta_avg_at_k"] for item in observed_by_seed]
    mean_benchmark_deltas = {
        benchmark: float(
            np.mean([values[benchmark].mean() for values in per_seed])
        )
        for benchmark in EXPECTED_BENCHMARKS
    }
    observed_mean = float(np.mean(seed_deltas))

    rng = np.random.default_rng(bootstrap_seed)
    seed_draws = rng.integers(0, 3, size=(bootstrap_samples, 3))
    benchmark_draws: dict[str, np.ndarray] = {}
    for benchmark in EXPECTED_BENCHMARKS:
        draws = np.zeros((bootstrap_samples, 3), dtype=np.float64)
        for slot in range(3):
            selected = seed_draws[:, slot]
            for seed_index in range(3):
                rows = np.flatnonzero(selected == seed_index)
                if rows.size == 0:
                    continue
                values = per_seed[seed_index][benchmark]
                indices = rng.integers(0, len(values), size=(rows.size, len(values)))
                draws[rows, slot] = values[indices].mean(axis=1)
        benchmark_draws[benchmark] = draws.mean(axis=1)
    macro_draws = np.mean(
        np.stack([benchmark_draws[name] for name in EXPECTED_BENCHMARKS], axis=1),
        axis=1,
    )
    ci_low, ci_high = (float(value) for value in np.quantile(macro_draws, [0.025, 0.975]))
    probability = float(
        ((macro_draws > 0).sum() + 0.5 * (macro_draws == 0).sum())
        / bootstrap_samples
    )
    positive_seed_count = sum(value > 0 for value in seed_deltas)
    positive_benchmark_count = sum(value > 0 for value in mean_benchmark_deltas.values())
    checks = {
        "mean_delta_at_least_0.005": observed_mean >= 0.005,
        "at_least_two_positive_seeds": positive_seed_count >= 2,
        "worst_seed_at_least_minus_0.002": min(seed_deltas) >= -0.002,
        "hierarchical_ci_low_above_zero": ci_low > 0,
        "at_least_four_positive_benchmarks": positive_benchmark_count >= 4,
        "worst_benchmark_above_minus_0.02": min(mean_benchmark_deltas.values()) >= -0.02,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "method": "training_seed_and_problem_hierarchical_bootstrap_v1",
        "training_seeds": training_seeds,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "samples_per_training_seed": expected_samples,
        "record_keys_and_snapshot_exactly_matched": True,
        "per_seed": observed_by_seed,
        "mean_macro_delta_avg_at_k": observed_mean,
        "minimum_seed_delta_avg_at_k": min(seed_deltas),
        "positive_seed_count": positive_seed_count,
        "hierarchical_ci95_low": ci_low,
        "hierarchical_ci95_high": ci_high,
        "probability_delta_gt_zero": probability,
        "mean_benchmark_deltas": mean_benchmark_deltas,
        "positive_benchmark_count": positive_benchmark_count,
        "minimum_benchmark_delta_avg_at_k": min(mean_benchmark_deltas.values()),
        "preregistered_checks": checks,
        "decision": "stable_generalizable" if passed else "not_stable_generalizable",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-records", type=Path, nargs="+", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLES)
    args = parser.parse_args()
    result = aggregate(
        args.baseline_records,
        args.candidate_records,
        args.training_seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_samples=args.expected_samples,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
