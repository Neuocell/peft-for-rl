#!/usr/bin/env python3
"""Compare aligned TinyLoRA and main-experiment training dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


METRICS = {
    "entropy": "actor/entropy",
    "pg_loss": "actor/pg_loss",
    "pg_clipfrac": "actor/pg_clipfrac",
    "ppo_kl": "actor/ppo_kl",
    "grad_norm": "actor/grad_norm",
    "reward": "critic/rewards/mean",
    "response_length": "response_length/mean",
    "response_clip_ratio": "response_length/clip_ratio",
    "parse_success": "reward_extra/parse_success/mean",
    "gen_s": "timing_s/gen",
    "old_log_prob_s": "timing_s/old_log_prob",
    "update_actor_s": "timing_s/update_actor",
    "step_s": "timing_s/step",
    "throughput_tokens_s": "perf/throughput",
    "total_tokens": "perf/total_num_tokens",
    "max_memory_allocated_gb": "perf/max_memory_allocated_gb",
    "max_memory_reserved_gb": "perf/max_memory_reserved_gb",
}

STEP_RE = re.compile(r"step:(\d+) - actor/")
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_steps(path: Path, max_step: int) -> list[dict[str, float | int]]:
    rows: dict[int, dict[str, float | int]] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(raw_line)
        if not match:
            continue
        step = int(match.group(1))
        if step > max_step or step in rows:
            continue
        payload = raw_line[match.start() :]
        values: dict[str, float | int] = {"step": step}
        for output_name, log_name in METRICS.items():
            value_match = re.search(
                rf"(?:^| - ){re.escape(log_name)}:({FLOAT_PATTERN})", payload
            )
            if value_match is None:
                raise ValueError(f"Missing {log_name} at step {step} in {path}")
            values[output_name] = float(value_match.group(1))
        rows[step] = values
    expected = list(range(1, max_step + 1))
    if sorted(rows) != expected:
        raise ValueError(f"Expected steps {expected}, found {sorted(rows)} in {path}")
    return [rows[step] for step in expected]


def slope(rows: list[dict[str, float | int]], key: str) -> float:
    xs = [float(row["step"]) for row in rows]
    ys = [float(row[key]) for row in rows]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def summarize(rows: list[dict[str, float | int]], trajectories_per_step: int) -> dict[str, float | int]:
    half = len(rows) // 2
    summary: dict[str, float | int] = {
        "steps": len(rows),
        "trajectories": len(rows) * trajectories_per_step,
        "correct": round(sum(float(row["reward"]) for row in rows) * trajectories_per_step),
        "reward_mean": mean(float(row["reward"]) for row in rows),
        "reward_first_half": mean(float(row["reward"]) for row in rows[:half]),
        "reward_last_half": mean(float(row["reward"]) for row in rows[half:]),
        "reward_ols_slope_per_step": slope(rows, "reward"),
        "parse_success_mean": mean(float(row["parse_success"]) for row in rows),
        "response_length_mean": mean(float(row["response_length"]) for row in rows),
        "response_clip_ratio_mean": mean(float(row["response_clip_ratio"]) for row in rows),
        "grad_norm_mean": mean(float(row["grad_norm"]) for row in rows),
        "grad_norm_min": min(float(row["grad_norm"]) for row in rows),
        "grad_norm_max": max(float(row["grad_norm"]) for row in rows),
        "pg_loss_mean": mean(float(row["pg_loss"]) for row in rows),
        "entropy_mean": mean(float(row["entropy"]) for row in rows),
        "pg_clipfrac_mean": mean(float(row["pg_clipfrac"]) for row in rows),
        "step_s_mean": mean(float(row["step_s"]) for row in rows),
        "gen_s_mean": mean(float(row["gen_s"]) for row in rows),
        "old_log_prob_s_mean": mean(float(row["old_log_prob_s"]) for row in rows),
        "update_actor_s_mean": mean(float(row["update_actor_s"]) for row in rows),
        "throughput_tokens_s_mean": mean(float(row["throughput_tokens_s"]) for row in rows),
        "max_memory_allocated_gb": max(float(row["max_memory_allocated_gb"]) for row in rows),
        "max_memory_reserved_gb": max(float(row["max_memory_reserved_gb"]) for row in rows),
    }
    return summary


def relative_delta(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--tinylora-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--trajectories-per-step", type=int, default=512)
    args = parser.parse_args()

    baseline = parse_steps(args.baseline_log, args.steps)
    tinylora = parse_steps(args.tinylora_log, args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dynamics_path = args.output_dir / "dynamics.csv"
    fieldnames = ["method", *baseline[0].keys()]
    with dynamics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in (("main", baseline), ("tinylora", tinylora)):
            for row in rows:
                writer.writerow({"method": method, **row})

    paired_path = args.output_dir / "paired-steps.csv"
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "main_reward",
                "tinylora_reward",
                "reward_delta",
                "main_step_s",
                "tinylora_step_s",
                "step_time_delta_s",
                "step_time_delta_ratio",
            ],
        )
        writer.writeheader()
        for base_row, tiny_row in zip(baseline, tinylora):
            writer.writerow(
                {
                    "step": base_row["step"],
                    "main_reward": base_row["reward"],
                    "tinylora_reward": tiny_row["reward"],
                    "reward_delta": float(tiny_row["reward"]) - float(base_row["reward"]),
                    "main_step_s": base_row["step_s"],
                    "tinylora_step_s": tiny_row["step_s"],
                    "step_time_delta_s": float(tiny_row["step_s"]) - float(base_row["step_s"]),
                    "step_time_delta_ratio": relative_delta(
                        float(tiny_row["step_s"]), float(base_row["step_s"])
                    ),
                }
            )

    baseline_summary = summarize(baseline, args.trajectories_per_step)
    tinylora_summary = summarize(tinylora, args.trajectories_per_step)
    comparison = {
        "scope": {
            "steps": args.steps,
            "trajectories_per_step": args.trajectories_per_step,
            "fixed_validation_run": False,
            "tinylora_checkpoint_saved": False,
            "stop_reason": "external process/session termination after step 8",
        },
        "main": baseline_summary,
        "tinylora": tinylora_summary,
        "delta": {
            "reward_absolute": tinylora_summary["reward_mean"] - baseline_summary["reward_mean"],
            "reward_relative": relative_delta(
                float(tinylora_summary["reward_mean"]), float(baseline_summary["reward_mean"])
            ),
            "correct": tinylora_summary["correct"] - baseline_summary["correct"],
            "parse_success_absolute": (
                tinylora_summary["parse_success_mean"] - baseline_summary["parse_success_mean"]
            ),
            "step_s": tinylora_summary["step_s_mean"] - baseline_summary["step_s_mean"],
            "step_time_relative": relative_delta(
                float(tinylora_summary["step_s_mean"]), float(baseline_summary["step_s_mean"])
            ),
            "throughput_relative": relative_delta(
                float(tinylora_summary["throughput_tokens_s_mean"]),
                float(baseline_summary["throughput_tokens_s_mean"]),
            ),
            "gen_time_relative": relative_delta(
                float(tinylora_summary["gen_s_mean"]), float(baseline_summary["gen_s_mean"])
            ),
            "old_log_prob_time_relative": relative_delta(
                float(tinylora_summary["old_log_prob_s_mean"]),
                float(baseline_summary["old_log_prob_s_mean"]),
            ),
            "update_actor_time_relative": relative_delta(
                float(tinylora_summary["update_actor_s_mean"]),
                float(baseline_summary["update_actor_s_mean"]),
            ),
        },
    }
    for value in comparison["delta"].values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite comparison result")
    (args.output_dir / "summary.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
