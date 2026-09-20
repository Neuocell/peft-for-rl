#!/usr/bin/env python3
"""Measure linearity and checkpoint extrapolation in functional LoRA space.

LoRA factors are not identifiable: ``B @ A`` is unchanged by invertible
changes of basis between B and A.  This diagnostic therefore performs every
comparison on the functional update

    Delta W_t = (alpha / rank) * B_t @ A_t

without materializing the dense weight matrices.  For three consecutive
checkpoints ``a < b < c``, it predicts the third checkpoint with constant
velocity:

    Delta W_hat_c = Delta W_b
                    + (c - b) / (b - a) * (Delta W_b - Delta W_a).

The reported error relative to the next move is the most direct test of
trajectory linearity.  A small direction residual but a large constant-
velocity error means that the path is approximately straight but its speed
changes, so a fitted or decayed extrapolation coefficient may still work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


STEP_RE = re.compile(r"global_step_(\d+)")


@dataclass
class FactoredMatrix:
    left: torch.Tensor
    right: torch.Tensor


@dataclass
class CompactSVD:
    left: torch.Tensor
    singular: torch.Tensor
    right: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", default="4,8,16")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--module-filter", default="")
    parser.add_argument("--max-modules", type=int, default=0)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = STEP_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"Not a checkpoint directory: {path}")
    return int(match.group(1))


def discover_adapters(root: Path) -> list[tuple[int, Path]]:
    adapters = []
    for checkpoint in root.glob("global_step_*"):
        for relative in ("actor/peft_adapter", "actor/lora_adapter"):
            adapter = checkpoint / relative
            if (adapter / "adapter_config.json").is_file() and (
                adapter / "adapter_model.safetensors"
            ).is_file():
                adapters.append((checkpoint_step(checkpoint), adapter))
                break
    if len(adapters) < 3:
        raise FileNotFoundError(f"Need at least three complete adapters below {root}")
    return sorted(adapters)


def adapter_module_key(a_key: str) -> str:
    return a_key.removesuffix(".lora_A.weight").removeprefix("base_model.model.")


def module_pattern_value(
    config: dict[str, Any], field: str, a_key: str, default: float
) -> float:
    pattern = config.get(field) or {}
    module = adapter_module_key(a_key)
    if module in pattern:
        return float(pattern[module])
    matches = [float(value) for key, value in pattern.items() if module.endswith(key)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {field} entries for {module}")
    return matches[0] if matches else float(default)


def module_scale(config: dict[str, Any], a_key: str) -> float:
    rank = module_pattern_value(config, "rank_pattern", a_key, config["r"])
    alpha = module_pattern_value(config, "alpha_pattern", a_key, config["lora_alpha"])
    if config.get("use_rslora", False):
        return alpha / math.sqrt(rank)
    return alpha / rank


def layer_index(key: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def layer_band(layer: int | None, layer_count: int) -> str:
    if layer is None:
        return "other"
    boundary = max(1, math.ceil(layer_count / 3))
    if layer < boundary:
        return "early"
    if layer < 2 * boundary:
        return "middle"
    return "late"


def module_name(key: str) -> str:
    return key.removesuffix(".lora_A.weight").split(".")[-1]


def linear_combination(
    terms: list[tuple[float, torch.Tensor, torch.Tensor]], scale: float
) -> FactoredMatrix:
    """Return scale * sum_i coefficient_i * B_i @ A_i in factored form."""

    return FactoredMatrix(
        left=torch.cat([b for _, _, b in terms], dim=1),
        right=torch.cat([coefficient * scale * a for coefficient, a, _ in terms], dim=0),
    )


def matrix_inner(left: FactoredMatrix, right: FactoredMatrix) -> float:
    value = torch.trace(
        (left.left.mT @ right.left) @ (right.right @ left.right.mT)
    )
    return float(value.item())


def squared_norm(matrix: FactoredMatrix) -> float:
    return max(matrix_inner(matrix, matrix), 0.0)


def compact_svd(matrix: FactoredMatrix) -> CompactSVD:
    q_left, r_left = torch.linalg.qr(matrix.left, mode="reduced")
    q_right, r_right = torch.linalg.qr(matrix.right.mT, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(
        r_left @ r_right.mT, full_matrices=False
    )
    return CompactSVD(
        left=q_left @ u_core,
        singular=singular,
        right=q_right @ vh_core.mT,
    )


def principal_overlap(left: CompactSVD, right: CompactSVD, k: int) -> tuple[float, float]:
    actual_k = min(
        k,
        left.left.shape[1],
        right.left.shape[1],
        left.right.shape[1],
        right.right.shape[1],
    )
    if actual_k <= 0:
        return float("nan"), float("nan")

    def one_side(x: torch.Tensor, y: torch.Tensor) -> float:
        cosines = torch.linalg.svdvals(x[:, :actual_k].mT @ y[:, :actual_k]).clamp(0, 1)
        return float(cosines.square().mean().item())

    return one_side(left.left, right.left), one_side(left.right, right.right)


def safe_cosine(inner: float, left_sq: float, right_sq: float) -> float:
    denominator = math.sqrt(max(left_sq, 0.0) * max(right_sq, 0.0))
    return inner / denominator if denominator > 0 else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_group(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    from_sq = sum(float(row["from_sq"]) for row in rows)
    anchor_sq = sum(float(row["anchor_sq"]) for row in rows)
    prev_sq = sum(float(row["previous_move_sq"]) for row in rows)
    next_sq = sum(float(row["next_move_sq"]) for row in rows)
    cross = sum(float(row["move_inner"]) for row in rows)
    error_sq = sum(float(row["prediction_error_sq"]) for row in rows)
    target_sq = sum(float(row["target_sq"]) for row in rows)
    predicted_sq = sum(float(row["predicted_target_sq"]) for row in rows)
    predicted_target_inner = sum(float(row["predicted_target_inner"]) for row in rows)
    anchor_target_inner = sum(float(row["anchor_target_inner"]) for row in rows)
    schedule_ratio = float(rows[0]["schedule_ratio"])
    fitted_ratio = cross / prev_sq if prev_sq > 0 else float("nan")
    fitted_error_sq = max(next_sq - cross * cross / prev_sq, 0.0) if prev_sq > 0 else float("nan")
    no_extrapolation_error = math.sqrt(next_sq / target_sq) if target_sq > 0 else float("nan")
    origin_schedule_factor = float(rows[0]["target_step"]) / float(rows[0]["anchor_step"])
    origin_schedule_error_sq = max(
        target_sq
        + origin_schedule_factor * origin_schedule_factor * anchor_sq
        - 2.0 * origin_schedule_factor * anchor_target_inner,
        0.0,
    )
    from_norm = math.sqrt(max(from_sq, 0.0))
    anchor_norm = math.sqrt(max(anchor_sq, 0.0))
    norm_trend_factor = (
        1.0 + schedule_ratio * (anchor_norm - from_norm) / anchor_norm
        if anchor_norm > 0
        else float("nan")
    )
    norm_trend_error_sq = max(
        target_sq
        + norm_trend_factor * norm_trend_factor * anchor_sq
        - 2.0 * norm_trend_factor * anchor_target_inner,
        0.0,
    )
    oracle_radial_factor = anchor_target_inner / anchor_sq if anchor_sq > 0 else float("nan")
    oracle_radial_error_sq = (
        max(target_sq - anchor_target_inner * anchor_target_inner / anchor_sq, 0.0)
        if anchor_sq > 0
        else float("nan")
    )
    output: dict[str, Any] = {
        "group": label,
        "modules": len(rows),
        "from_step": int(rows[0]["from_step"]),
        "anchor_step": int(rows[0]["anchor_step"]),
        "target_step": int(rows[0]["target_step"]),
        "schedule_ratio": schedule_ratio,
        "fitted_move_ratio": fitted_ratio,
        "fitted_vs_schedule": fitted_ratio / schedule_ratio if schedule_ratio else float("nan"),
        "previous_move_global_fro": math.sqrt(prev_sq),
        "next_move_global_fro": math.sqrt(next_sq),
        "move_cosine": safe_cosine(cross, prev_sq, next_sq),
        "constant_velocity_error_over_next_move": math.sqrt(error_sq / next_sq) if next_sq > 0 else float("nan"),
        "best_scalar_error_over_next_move": math.sqrt(fitted_error_sq / next_sq) if next_sq > 0 else float("nan"),
        "predicted_target_error_over_target": math.sqrt(error_sq / target_sq) if target_sq > 0 else float("nan"),
        "predicted_target_cosine": safe_cosine(predicted_target_inner, predicted_sq, target_sq),
        "anchor_target_cosine": safe_cosine(anchor_target_inner, anchor_sq, target_sq),
        "no_extrapolation_error_over_target": no_extrapolation_error,
        "origin_schedule_factor": origin_schedule_factor,
        "origin_schedule_error_over_target": (
            math.sqrt(origin_schedule_error_sq / target_sq) if target_sq > 0 else float("nan")
        ),
        "norm_trend_factor": norm_trend_factor,
        "norm_trend_error_over_target": (
            math.sqrt(norm_trend_error_sq / target_sq) if target_sq > 0 else float("nan")
        ),
        "oracle_radial_factor": oracle_radial_factor,
        "oracle_radial_error_over_target": (
            math.sqrt(oracle_radial_error_sq / target_sq) if target_sq > 0 else float("nan")
        ),
    }
    weights = [float(row["next_move_sq"]) for row in rows]
    denominator = max(sum(weights), 1e-300)
    for key in rows[0]:
        if key.startswith("k") and key.endswith(("_left_overlap", "_right_overlap")):
            output[key] = sum(float(row[key]) * weight for row, weight in zip(rows, weights, strict=True)) / denominator
    return output


def markdown_percent(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def render_report(
    checkpoint_root: Path,
    triplet_summary: list[dict[str, Any]],
    breakdown: list[dict[str, Any]],
    top_ks: list[int],
) -> str:
    lines = [
        "# Functional LoRA Linear Extrapolation",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"Checkpoint root: `{checkpoint_root}`",
        "",
        "All calculations use `DeltaW = scaling * B @ A`; raw LoRA factors are never interpolated.",
        "The constant-velocity prediction rescales the previous move by the ratio of checkpoint intervals.",
        "",
        "## Global trajectory",
        "",
        "| From | Anchor | Target | Interval ratio | Fitted ratio | Move cosine | CV error / next move | Best-scalar error | Target rel. error | Target cosine |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in triplet_summary:
        lines.append(
            "| {from_step} | {anchor_step} | {target_step} | {schedule_ratio:.3f} | "
            "{fitted_move_ratio:.3f} | {move_cosine:.4f} | {cv} | {best} | {target} | {predicted_target_cosine:.6f} |".format(
                **row,
                cv=markdown_percent(float(row["constant_velocity_error_over_next_move"])),
                best=markdown_percent(float(row["best_scalar_error_over_next_move"])),
                target=markdown_percent(float(row["predicted_target_error_over_target"])),
            )
        )

    if top_ks:
        lines.extend(["", "## Consecutive move subspaces", ""])
        header = "| From | Anchor | Target | " + " | ".join(
            f"L@{k} / R@{k}" for k in top_ks
        ) + " |"
        lines.extend([header, "| ---: | ---: | ---: | " + " | ".join("---:" for _ in top_ks) + " |"])
        for row in triplet_summary:
            cells = [
                f"{markdown_percent(float(row[f'k{k}_left_overlap']))} / "
                f"{markdown_percent(float(row[f'k{k}_right_overlap']))}"
                for k in top_ks
            ]
            lines.append(
                f"| {row['from_step']} | {row['anchor_step']} | {row['target_step']} | "
                + " | ".join(cells)
                + " |"
            )

    lines.extend(["", "## Breakdown", ""])
    for group_field in ("layer_band", "module"):
        lines.extend(
            [
                f"### By {group_field.replace('_', ' ')}",
                "",
                "| From | Anchor | Target | Group | Move cosine | CV error / next move | Fitted ratio |",
                "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in breakdown:
            if row["breakdown"] != group_field:
                continue
            lines.append(
                f"| {row['from_step']} | {row['anchor_step']} | {row['target_step']} | "
                f"{row['group']} | {row['move_cosine']:.4f} | "
                f"{markdown_percent(float(row['constant_velocity_error_over_next_move']))} | "
                f"{row['fitted_move_ratio']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Radial extrapolation from the base model",
            "",
            "| From | Anchor | Target | Anchor/target cosine | No extrap. error | Step-ratio factor / error | Norm-trend factor / error | Oracle radial factor / error |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in triplet_summary:
        lines.append(
            "| {from_step} | {anchor_step} | {target_step} | {anchor_target_cosine:.6f} | "
            "{no_extrap} | {origin_schedule_factor:.3f} / {origin_error} | "
            "{norm_trend_factor:.3f} / {norm_error} | "
            "{oracle_radial_factor:.3f} / {oracle_error} |".format(
                **row,
                no_extrap=markdown_percent(float(row["no_extrapolation_error_over_target"])),
                origin_error=markdown_percent(float(row["origin_schedule_error_over_target"])),
                norm_error=markdown_percent(float(row["norm_trend_error_over_target"])),
                oracle_error=markdown_percent(float(row["oracle_radial_error_over_target"])),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Move cosine` tests whether consecutive checkpoint increments point in the same functional direction.",
            "- `CV error / next move` tests strict constant-velocity extrapolation. Values above 100% mean copying the previous velocity is worse than predicting no next move.",
            "- `Best-scalar error` removes speed mismatch and tests whether a single rescaled previous move can explain the next move.",
            "- `Target rel. error` can be tiny even when the next move is predicted poorly because the accumulated adapter update is much larger than one late interval. Do not use it alone to claim linearity.",
            "- Subspace overlap can remain high while signed matrix cosine is modest; that indicates recombination or sign changes inside a stable span rather than a genuinely linear trajectory.",
            "- The radial table tests extrapolating `base + gamma * DeltaW_anchor`. `No extrap. error` is the mandatory baseline; a proposed factor is useful as a weight predictor only when its error is smaller.",
            "- `Oracle radial factor` is fitted with access to the future checkpoint. Its residual is a lower bound for every single-scalar radial extrapolation of that anchor.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(args.threads, 1))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    top_ks = sorted({int(value) for value in args.top_k.split(",") if value.strip()})
    adapters = discover_adapters(args.checkpoint_root.resolve())
    steps = [step for step, _ in adapters]
    configs = {
        step: json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        for step, path in adapters
    }
    filters = [value.strip() for value in args.module_filter.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []

    with ExitStack() as stack:
        handles = {
            step: stack.enter_context(
                safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu")
            )
            for step, path in adapters
        }
        first_handle = handles[steps[0]]
        keys = sorted(key for key in first_handle.keys() if key.endswith(".lora_A.weight"))
        if filters:
            keys = [key for key in keys if any(value in key for value in filters)]
        if args.max_modules > 0:
            keys = keys[: args.max_modules]
        layers = [layer_index(key) for key in keys]
        layer_count = max((layer for layer in layers if layer is not None), default=-1) + 1

        for module_index, a_key in enumerate(keys, 1):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            factors = {
                step: (
                    handles[step].get_tensor(a_key).to(dtype=dtype),
                    handles[step].get_tensor(b_key).to(dtype=dtype),
                )
                for step in steps
            }
            shapes = {(tuple(a.shape), tuple(b.shape)) for a, b in factors.values()}
            if len(shapes) != 1:
                raise ValueError(f"Factor shapes change across checkpoints for {a_key}: {shapes}")
            scales = {module_scale(configs[step], a_key) for step in steps}
            if len(scales) != 1:
                raise ValueError(f"LoRA scale changes across checkpoints for {a_key}: {scales}")
            scale = scales.pop()
            layer = layer_index(a_key)
            common = {
                "adapter_key": a_key.removesuffix(".lora_A.weight"),
                "layer": layer,
                "layer_band": layer_band(layer, layer_count),
                "module": module_name(a_key),
                "rank": factors[steps[0]][0].shape[0],
                "scale": scale,
            }

            for from_step, anchor_step, target_step in zip(steps, steps[1:], steps[2:]):
                from_a, from_b = factors[from_step]
                anchor_a, anchor_b = factors[anchor_step]
                target_a, target_b = factors[target_step]
                schedule_ratio = (target_step - anchor_step) / (anchor_step - from_step)
                previous_move = linear_combination(
                    [(-1.0, from_a, from_b), (1.0, anchor_a, anchor_b)], scale
                )
                next_move = linear_combination(
                    [(-1.0, anchor_a, anchor_b), (1.0, target_a, target_b)], scale
                )
                prediction_error = linear_combination(
                    [
                        (schedule_ratio, from_a, from_b),
                        (-(1.0 + schedule_ratio), anchor_a, anchor_b),
                        (1.0, target_a, target_b),
                    ],
                    scale,
                )
                predicted_target = linear_combination(
                    [
                        (-schedule_ratio, from_a, from_b),
                        (1.0 + schedule_ratio, anchor_a, anchor_b),
                    ],
                    scale,
                )
                target = linear_combination([(1.0, target_a, target_b)], scale)
                from_target = linear_combination([(1.0, from_a, from_b)], scale)
                anchor_target = linear_combination([(1.0, anchor_a, anchor_b)], scale)

                previous_sq = squared_norm(previous_move)
                next_sq = squared_norm(next_move)
                error_sq = squared_norm(prediction_error)
                target_sq = squared_norm(target)
                predicted_sq = squared_norm(predicted_target)
                row: dict[str, Any] = {
                    **common,
                    "from_step": from_step,
                    "anchor_step": anchor_step,
                    "target_step": target_step,
                    "schedule_ratio": schedule_ratio,
                    "from_sq": squared_norm(from_target),
                    "anchor_sq": squared_norm(anchor_target),
                    "previous_move_sq": previous_sq,
                    "next_move_sq": next_sq,
                    "move_inner": matrix_inner(previous_move, next_move),
                    "prediction_error_sq": error_sq,
                    "target_sq": target_sq,
                    "predicted_target_sq": predicted_sq,
                    "predicted_target_inner": matrix_inner(predicted_target, target),
                    "anchor_target_inner": matrix_inner(anchor_target, target),
                }
                previous_svd = compact_svd(previous_move)
                next_svd = compact_svd(next_move)
                for k in top_ks:
                    left_overlap, right_overlap = principal_overlap(previous_svd, next_svd, k)
                    row[f"k{k}_left_overlap"] = left_overlap
                    row[f"k{k}_right_overlap"] = right_overlap
                rows.append(row)

            print(f"[{module_index:03d}/{len(keys):03d}] {common['adapter_key']}", flush=True)

    triplets = sorted(
        {(int(row["from_step"]), int(row["anchor_step"]), int(row["target_step"])) for row in rows}
    )
    summary = []
    breakdown = []
    for triplet in triplets:
        group = [
            row
            for row in rows
            if (int(row["from_step"]), int(row["anchor_step"]), int(row["target_step"])) == triplet
        ]
        summary.append(aggregate_group(group, "all"))
        for field in ("layer_band", "module"):
            for value in sorted({str(row[field]) for row in group}):
                item = aggregate_group([row for row in group if str(row[field]) == value], value)
                item["breakdown"] = field
                breakdown.append(item)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_module.csv", rows)
    write_csv(out_dir / "triplet_summary.csv", summary)
    write_csv(out_dir / "breakdown.csv", breakdown)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "steps": steps,
        "top_k": top_ks,
        "modules": len({str(row["adapter_key"]) for row in rows}),
        "definition": {
            "functional_update": "DeltaW_t = module_scale * B_t @ A_t",
            "prediction": "DeltaW_hat_c = DeltaW_b + ((c-b)/(b-a)) * (DeltaW_b-DeltaW_a)",
            "best_scalar": "least-squares scalar multiplying the previous functional move",
            "radial_prediction": "DeltaW_hat_c = gamma * DeltaW_b, always relative to the frozen base model",
        },
        "triplet_summary": summary,
        "breakdown": breakdown,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(
        render_report(args.checkpoint_root.resolve(), summary, breakdown, top_ks),
        encoding="utf-8",
    )
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
