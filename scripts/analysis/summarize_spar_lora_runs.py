#!/usr/bin/env python3
"""Parse verl console logs and build the SPAR-LoRA 50-step comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")


def parse_step_metrics(path: Path) -> dict[int, dict[str, float]]:
    """Return the last complete metric record for every logged step."""

    records: dict[int, dict[str, float]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = STEP_RE.search(line)
        if match is None:
            continue
        values: dict[str, float] = {}
        for field in match.group(2).split(" - "):
            if ":" not in field:
                continue
            key, raw_value = field.rsplit(":", 1)
            try:
                value = float(raw_value)
            except ValueError:
                continue
            values[key.strip()] = value
        records[int(match.group(1))] = values
    if not records:
        raise ValueError(f"No verl step metrics found in {path}")
    return records


def _series(records: dict[int, dict[str, float]], key: str, last_step: int) -> list[float]:
    missing = [step for step in range(1, last_step + 1) if step not in records or key not in records[step]]
    if missing:
        raise ValueError(f"Missing {key} at steps {missing} (required through step {last_step})")
    values = [records[step][key] for step in range(1, last_step + 1)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite {key} through step {last_step}")
    return values


def _optional_mean(records: dict[int, dict[str, float]], key: str, last_step: int) -> float | None:
    values = [records[step][key] for step in range(1, last_step + 1) if step in records and key in records[step]]
    return sum(values) / len(values) if values else None


def _optional_max(records: dict[int, dict[str, float]], key: str, last_step: int) -> float | None:
    values = [records[step][key] for step in range(1, last_step + 1) if step in records and key in records[step]]
    return max(values) if values else None


def summarize_gpu_memory(path: Path) -> dict[str, object]:
    peaks: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as output:
        for row in csv.DictReader(output):
            index = int(row["gpu_index"])
            peaks[index] = max(peaks.get(index, 0), int(row["used_mib"]))
    if not peaks:
        raise ValueError(f"No GPU memory samples found in {path}")
    return {
        "observed_full_gpu_peak_gib": max(peaks.values()) / 1024,
        "observed_gpu_peak_by_index_gib": {
            str(index): memory / 1024 for index, memory in sorted(peaks.items())
        },
        "gpu_memory_csv": str(path.resolve()),
    }


def summarize_run(
    path: Path,
    *,
    expected_steps: int = 50,
    world_size: int = 4,
    gpu_memory_path: Path | None = None,
) -> dict[str, object]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    records = parse_step_metrics(path)
    missing_steps = [step for step in range(1, expected_steps + 1) if step not in records]
    if missing_steps:
        raise ValueError(f"Incomplete run {path}: missing steps {missing_steps}")

    rewards = _series(records, "critic/score/mean", expected_steps)
    peak_allocated_aggregate = _optional_max(records, "perf/max_memory_allocated_gb", expected_steps)
    summary: dict[str, object] = {
        "log_path": str(path.resolve()),
        "steps": expected_steps,
        "reward_at_10": rewards[9] if expected_steps >= 10 else None,
        "reward_at_20": rewards[19] if expected_steps >= 20 else None,
        "reward_at_50": rewards[49] if expected_steps >= 50 else None,
        "reward_auc_1_10": sum(rewards[:10]) / 10 if expected_steps >= 10 else None,
        "reward_auc_1_20": sum(rewards[:20]) / 20 if expected_steps >= 20 else None,
        "reward_auc_1_50": sum(rewards[:50]) / 50 if expected_steps >= 50 else None,
        "reward_mean_2_10": sum(rewards[1:10]) / 9 if expected_steps >= 10 else None,
        "reward_mean_2_20": sum(rewards[1:20]) / 19 if expected_steps >= 20 else None,
        "reward_mean_2_50": sum(rewards[1:50]) / 49 if expected_steps >= 50 else None,
        "boxed_accuracy_at_50": records[50].get("reward_extra/acc/mean") if expected_steps >= 50 else None,
        "response_length_mean_1_50": _optional_mean(records, "response_length/mean", expected_steps),
        "entropy_mean_1_50": _optional_mean(records, "actor/entropy", expected_steps),
        "kl_mean_1_50": _optional_mean(records, "actor/ppo_kl", expected_steps),
        "policy_loss_mean_1_50": _optional_mean(records, "actor/pg_loss", expected_steps),
        "grad_norm_mean_1_50": _optional_mean(records, "actor/grad_norm", expected_steps),
        "clip_fraction_mean_1_50": _optional_mean(records, "actor/pg_clipfrac", expected_steps),
        "step_time_mean_s_1_50": _optional_mean(records, "timing_s/step", expected_steps),
        # Ray dispatch aggregates this worker scalar across data-parallel ranks.
        "peak_allocated_memory_gb": (
            peak_allocated_aggregate / world_size if peak_allocated_aggregate is not None else None
        ),
        "peak_allocated_memory_gb_aggregate": peak_allocated_aggregate,
        "memory_metric_world_size": world_size,
        "active_parameter_count": records[expected_steps].get("actor/active_parameter_count"),
        "active_rank_mean": records[expected_steps].get("spar_structure/rank_mean"),
        "active_rank_min": records[expected_steps].get("spar_structure/rank_min"),
        "active_rank_max": records[expected_steps].get("spar_structure/rank_max"),
        "calibration_energy_capture": records[expected_steps].get(
            "spar_structure/calibration_energy_capture"
        ),
        "u_score_capture": records[expected_steps].get("spar_structure/u_score_capture"),
        "has_non_finite_metric": any(
            not math.isfinite(value)
            for step in range(1, expected_steps + 1)
            for value in records[step].values()
        ),
        "per_step": {str(step): records[step] for step in range(1, expected_steps + 1)},
    }
    if gpu_memory_path is not None:
        summary.update(summarize_gpu_memory(gpu_memory_path))
    return summary


def comparison_markdown(uniform: dict[str, object], adaptive: dict[str, object]) -> str:
    columns = (
        ("reward@20", "reward_at_20"),
        ("reward@50", "reward_at_50"),
        ("reward AUC1:20", "reward_auc_1_20"),
        ("reward AUC1:50", "reward_auc_1_50"),
        ("accuracy@50", "boxed_accuracy_at_50"),
        ("mean length", "response_length_mean_1_50"),
        ("entropy", "entropy_mean_1_50"),
        ("KL", "kl_mean_1_50"),
        ("trainable params", "active_parameter_count"),
        ("mean rank", "active_rank_mean"),
        ("step time (s)", "step_time_mean_s_1_50"),
        ("peak actor GPU/rank (GiB)", "peak_allocated_memory_gb"),
        ("observed full GPU peak (GiB)", "observed_full_gpu_peak_gib"),
    )

    def render(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            if value.is_integer() and abs(value) >= 1000:
                return str(int(value))
            return f"{value:.6g}"
        return str(value)

    header = "| run | " + " | ".join(label for label, _ in columns) + " |"
    divider = "|---|" + "|".join("---:" for _ in columns) + "|"
    rows = []
    for label, summary in (("uniform-r8", uniform), ("adaptive-eqr8", adaptive)):
        rows.append("| " + label + " | " + " | ".join(render(summary.get(key)) for _, key in columns) + " |")
    return "\n".join([header, divider, *rows]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform-log", type=Path, required=True)
    parser.add_argument("--adaptive-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--uniform-gpu-csv", type=Path)
    parser.add_argument("--adaptive-gpu-csv", type=Path)
    args = parser.parse_args()

    uniform = summarize_run(
        args.uniform_log.resolve(),
        world_size=args.world_size,
        gpu_memory_path=args.uniform_gpu_csv.resolve() if args.uniform_gpu_csv else None,
    )
    adaptive = summarize_run(
        args.adaptive_log.resolve(),
        world_size=args.world_size,
        gpu_memory_path=args.adaptive_gpu_csv.resolve() if args.adaptive_gpu_csv else None,
    )
    result = {
        "auc_definition": "arithmetic mean of per-step reward (normalized discrete AUC)",
        "uniform": uniform,
        "adaptive": adaptive,
        "adaptive_minus_uniform_reward_auc_1_50": adaptive["reward_auc_1_50"] - uniform["reward_auc_1_50"],
        "markdown": comparison_markdown(uniform, adaptive),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["markdown"], end="")


if __name__ == "__main__":
    main()
