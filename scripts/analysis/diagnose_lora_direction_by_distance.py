#!/usr/bin/env python3
"""Diagnose radial growth versus direction drift across LoRA checkpoints.

Every comparison is performed in functional adapter space,

    DeltaW_t = module_scale * B_t @ A_t,

so LoRA factor reparameterizations do not affect the result.  The script
computes checkpoint-to-checkpoint radial fits and directions of every chord
without materializing dense DeltaW matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from diagnose_lora_linear_extrapolation import (
    FactoredMatrix,
    adapter_module_key,
    discover_adapters,
    layer_band,
    layer_index,
    linear_combination,
    matrix_inner,
    module_name,
    module_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--module-filter", default="")
    parser.add_argument("--max-modules", type=int, default=0)
    return parser.parse_args()


def safe_cosine(inner: float, left_sq: float, right_sq: float) -> float:
    denominator = math.sqrt(max(left_sq, 0.0) * max(right_sq, 0.0))
    return inner / denominator if denominator > 0 else float("nan")


def vector_inner(gram: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left @ gram @ right).item())


def chord_vector(size: int, start: int, end: int) -> torch.Tensor:
    vector = torch.zeros(size, dtype=torch.float64)
    vector[start] = -1.0
    vector[end] = 1.0
    return vector


def quantiles(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"p10": float("nan"), "median": float("nan"), "p90": float("nan")}
    tensor = torch.tensor(finite, dtype=torch.float64)
    result = torch.quantile(tensor, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {"p10": float(result[0]), "median": float(result[1]), "p90": float(result[2])}


def radial_metrics(gram: torch.Tensor, early: int, late: int) -> dict[str, float]:
    early_sq = float(gram[early, early])
    late_sq = float(gram[late, late])
    cross = float(gram[early, late])
    gamma = cross / early_sq if early_sq > 0 else float("nan")
    residual_sq = max(late_sq - cross * cross / early_sq, 0.0) if early_sq > 0 else float("nan")
    unchanged_sq = max(early_sq + late_sq - 2.0 * cross, 0.0)
    return {
        "cosine": safe_cosine(cross, early_sq, late_sq),
        "early_norm": math.sqrt(max(early_sq, 0.0)),
        "late_norm": math.sqrt(max(late_sq, 0.0)),
        "norm_ratio": math.sqrt(late_sq / early_sq) if early_sq > 0 else float("nan"),
        "oracle_radial_factor": gamma,
        "oracle_radial_residual_over_late": (
            math.sqrt(residual_sq / late_sq) if late_sq > 0 else float("nan")
        ),
        "unchanged_error_over_late": (
            math.sqrt(unchanged_sq / late_sq) if late_sq > 0 else float("nan")
        ),
    }


def chord_metrics(
    gram: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> dict[str, float]:
    left_sq = vector_inner(gram, left, left)
    right_sq = vector_inner(gram, right, right)
    cross = vector_inner(gram, left, right)
    fitted = cross / left_sq if left_sq > 0 else float("nan")
    residual_sq = max(right_sq - cross * cross / left_sq, 0.0) if left_sq > 0 else float("nan")
    return {
        "left_norm": math.sqrt(max(left_sq, 0.0)),
        "right_norm": math.sqrt(max(right_sq, 0.0)),
        "cosine": safe_cosine(cross, left_sq, right_sq),
        "best_scalar_left_to_right": fitted,
        "best_scalar_residual_over_right": (
            math.sqrt(residual_sq / right_sq) if right_sq > 0 else float("nan")
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def render_report(
    checkpoint_root: Path,
    steps: list[int],
    norm_rows: list[dict[str, float]],
    cumulative_rows: list[dict[str, Any]],
    from_first_rows: list[dict[str, Any]],
    to_last_rows: list[dict[str, Any]],
    nonoverlap_rows: list[dict[str, Any]],
    cosine_matrix: list[list[float]],
) -> str:
    lines = [
        "# Functional DeltaW Direction by Checkpoint Distance",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"Checkpoint root: `{checkpoint_root}`",
        "",
        "All comparisons use `DeltaW_t = module_scale * B_t @ A_t`.",
        "`Oracle radial residual` is the error of the best possible scalar-only fit "
        "`DeltaW_late ~= gamma * DeltaW_early`; it is the direct test of pure magnitude growth.",
        "",
        "## Norm growth",
        "",
        "| Step | Global Frobenius norm | Norm / sqrt(step) | Norm / step |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in norm_rows:
        lines.append(
            f"| {int(row['step'])} | {row['norm']:.9f} | "
            f"{row['norm_over_sqrt_step']:.9f} | {row['norm_over_step']:.9f} |"
        )
    lines.extend(
        [
        "",
        "## Cumulative DeltaW radial test",
        "",
        "| Early | Late | Distance | Cosine | Norm ratio | Oracle gamma | Radial residual | Unchanged error | Module cosine p10 / median / p90 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in cumulative_rows:
        lines.append(
            f"| {row['early_step']} | {row['late_step']} | {row['distance']} | "
            f"{row['cosine']:.6f} | {row['norm_ratio']:.4f} | "
            f"{row['oracle_radial_factor']:.4f} | "
            f"{pct(row['oracle_radial_residual_over_late'])} | "
            f"{pct(row['unchanged_error_over_late'])} | "
            f"{row['module_cosine_p10']:.4f} / {row['module_cosine_median']:.4f} / "
            f"{row['module_cosine_p90']:.4f} |"
        )

    lines.extend(["", "## Cumulative DeltaW cosine matrix", ""])
    lines.append("| Step | " + " | ".join(str(step) for step in steps) + " |")
    lines.append("| ---: | " + " | ".join("---:" for _ in steps) + " |")
    for step, row in zip(steps, cosine_matrix, strict=True):
        lines.append(f"| {step} | " + " | ".join(f"{value:.4f}" for value in row) + " |")

    lines.extend(
        [
            "",
            f"## Chords from step {steps[0]} versus the full {steps[0]}->{steps[-1]} direction",
            "",
            "| Chord | Span | Cosine to full | Best-scalar residual | Module cosine p10 / median / p90 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in from_first_rows:
        lines.append(
            f"| {row['start_step']}->{row['end_step']} | {row['span']} | "
            f"{row['cosine']:.6f} | {pct(row['best_scalar_residual_over_right'])} | "
            f"{row['module_cosine_p10']:.4f} / {row['module_cosine_median']:.4f} / "
            f"{row['module_cosine_p90']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## Chords ending at step {steps[-1]} versus the full {steps[0]}->{steps[-1]} direction",
            "",
            "| Chord | Span | Cosine to full | Best-scalar residual |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in to_last_rows:
        lines.append(
            f"| {row['start_step']}->{row['end_step']} | {row['span']} | "
            f"{row['cosine']:.6f} | {pct(row['best_scalar_residual_over_right'])} |"
        )

    lines.extend(
        [
            "",
            "## Non-overlapping adjacent chords",
            "",
            "| Left chord | Right chord | Spans | Cosine | Best-scalar residual |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in nonoverlap_rows:
        lines.append(
            f"| {row['left_start']}->{row['left_end']} | "
            f"{row['right_start']}->{row['right_end']} | "
            f"{row['left_span']} / {row['right_span']} | {row['cosine']:.6f} | "
            f"{pct(row['best_scalar_residual_over_right'])} |"
        )

    lines.extend(
        [
            "",
            "## Reading the metrics",
            "",
            "- A cosine near one is necessary but not sufficient for scalar-only growth.",
            "- The oracle radial residual is `sqrt(1-cosine^2)` and cannot be repaired by choosing a different scalar.",
            "- Fixed-direction accumulation requires near-zero radial residual. Approximately constant `norm / sqrt(step)` instead supports noise-like or mutually orthogonal increments, not a single ray.",
            "- `Unchanged error` measures how close the earlier checkpoint already is to the later one; it is not a direction metric.",
            "- Overlapping chords share updates and can have inflated cosine, so the final table only uses chords that meet at one boundary and do not overlap.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(args.threads, 1))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    adapters = discover_adapters(args.checkpoint_root.resolve())
    steps = [step for step, _ in adapters]
    configs = {
        step: json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        for step, path in adapters
    }
    filters = [value.strip() for value in args.module_filter.split(",") if value.strip()]
    module_grams: list[tuple[dict[str, Any], torch.Tensor]] = []
    global_gram = torch.zeros((len(steps), len(steps)), dtype=torch.float64)

    with ExitStack() as stack:
        handles = {
            step: stack.enter_context(
                safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu")
            )
            for step, path in adapters
        }
        keys = sorted(key for key in handles[steps[0]].keys() if key.endswith(".lora_A.weight"))
        if filters:
            keys = [key for key in keys if any(value in key for value in filters)]
        if args.max_modules > 0:
            keys = keys[: args.max_modules]
        layers = [layer_index(key) for key in keys]
        layer_count = max((layer for layer in layers if layer is not None), default=-1) + 1

        for module_index, a_key in enumerate(keys, 1):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            scales = {module_scale(configs[step], a_key) for step in steps}
            if len(scales) != 1:
                raise ValueError(f"LoRA scale changes across checkpoints for {a_key}: {scales}")
            scale = scales.pop()
            updates: list[FactoredMatrix] = []
            shapes: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
            for step in steps:
                a = handles[step].get_tensor(a_key).to(dtype=dtype)
                b = handles[step].get_tensor(b_key).to(dtype=dtype)
                shapes.add((tuple(a.shape), tuple(b.shape)))
                updates.append(linear_combination([(1.0, a, b)], scale))
            if len(shapes) != 1:
                raise ValueError(f"Factor shapes change across checkpoints for {a_key}: {shapes}")

            gram = torch.empty((len(steps), len(steps)), dtype=torch.float64)
            for i, left in enumerate(updates):
                for j in range(i, len(updates)):
                    value = matrix_inner(left, updates[j])
                    gram[i, j] = value
                    gram[j, i] = value
            global_gram += gram
            layer = layer_index(a_key)
            module_grams.append(
                (
                    {
                        "adapter_key": adapter_module_key(a_key),
                        "layer": layer,
                        "layer_band": layer_band(layer, layer_count),
                        "module": module_name(a_key),
                    },
                    gram,
                )
            )
            print(f"[{module_index:03d}/{len(keys):03d}] {adapter_module_key(a_key)}", flush=True)

    cumulative_rows: list[dict[str, Any]] = []
    per_module_rows: list[dict[str, Any]] = []
    for i, early_step in enumerate(steps):
        for j in range(i + 1, len(steps)):
            late_step = steps[j]
            row: dict[str, Any] = {
                "early_step": early_step,
                "late_step": late_step,
                "distance": late_step - early_step,
                **radial_metrics(global_gram, i, j),
            }
            module_cosines = []
            for metadata, gram in module_grams:
                metrics = radial_metrics(gram, i, j)
                module_cosines.append(metrics["cosine"])
                per_module_rows.append(
                    {
                        **metadata,
                        "early_step": early_step,
                        "late_step": late_step,
                        "distance": late_step - early_step,
                        **metrics,
                    }
                )
            distribution = quantiles(module_cosines)
            row.update({f"module_cosine_{key}": value for key, value in distribution.items()})
            cumulative_rows.append(row)

    full = chord_vector(len(steps), 0, len(steps) - 1)
    from_first_rows: list[dict[str, Any]] = []
    to_last_rows: list[dict[str, Any]] = []
    for end in range(1, len(steps)):
        chord = chord_vector(len(steps), 0, end)
        row = {
            "start_step": steps[0],
            "end_step": steps[end],
            "span": steps[end] - steps[0],
            **chord_metrics(global_gram, chord, full),
        }
        module_cosines = [chord_metrics(gram, chord, full)["cosine"] for _, gram in module_grams]
        row.update({f"module_cosine_{key}": value for key, value in quantiles(module_cosines).items()})
        from_first_rows.append(row)
    for start in range(len(steps) - 1):
        chord = chord_vector(len(steps), start, len(steps) - 1)
        to_last_rows.append(
            {
                "start_step": steps[start],
                "end_step": steps[-1],
                "span": steps[-1] - steps[start],
                **chord_metrics(global_gram, chord, full),
            }
        )

    nonoverlap_rows: list[dict[str, Any]] = []
    for boundary in range(1, len(steps) - 1):
        for left_start in range(boundary):
            for right_end in range(boundary + 1, len(steps)):
                left = chord_vector(len(steps), left_start, boundary)
                right = chord_vector(len(steps), boundary, right_end)
                nonoverlap_rows.append(
                    {
                        "left_start": steps[left_start],
                        "left_end": steps[boundary],
                        "right_start": steps[boundary],
                        "right_end": steps[right_end],
                        "left_span": steps[boundary] - steps[left_start],
                        "right_span": steps[right_end] - steps[boundary],
                        **chord_metrics(global_gram, left, right),
                    }
                )

    cosine_matrix = []
    for i in range(len(steps)):
        cosine_matrix.append(
            [
                safe_cosine(
                    float(global_gram[i, j]),
                    float(global_gram[i, i]),
                    float(global_gram[j, j]),
                )
                for j in range(len(steps))
            ]
        )

    norm_rows = []
    for index, step in enumerate(steps):
        norm = math.sqrt(max(float(global_gram[index, index]), 0.0))
        norm_rows.append(
            {
                "step": step,
                "norm": norm,
                "norm_over_sqrt_step": norm / math.sqrt(step),
                "norm_over_step": norm / step,
            }
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "cumulative_radial.csv", cumulative_rows)
    write_csv(out_dir / "per_module_cumulative.csv", per_module_rows)
    write_csv(out_dir / "chords_from_first.csv", from_first_rows)
    write_csv(out_dir / "chords_to_last.csv", to_last_rows)
    write_csv(out_dir / "nonoverlap_adjacent_chords.csv", nonoverlap_rows)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "steps": steps,
        "modules": len(module_grams),
        "definition": {
            "functional_update": "DeltaW_t = module_scale * B_t @ A_t",
            "radial_fit": "DeltaW_late ~= gamma * DeltaW_early",
            "chord": "DeltaW_end - DeltaW_start",
        },
        "norm_growth": norm_rows,
        "cumulative_radial": cumulative_rows,
        "cumulative_cosine_matrix": cosine_matrix,
        "chords_from_first": from_first_rows,
        "chords_to_last": to_last_rows,
        "nonoverlap_adjacent_chords": nonoverlap_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(
        render_report(
            args.checkpoint_root.resolve(),
            steps,
            norm_rows,
            cumulative_rows,
            from_first_rows,
            to_last_rows,
            nonoverlap_rows,
            cosine_matrix,
        ),
        encoding="utf-8",
    )
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
