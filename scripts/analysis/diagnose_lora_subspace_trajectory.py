#!/usr/bin/env python3
"""Diagnose LoRA Delta-W spectra and subspace motion across checkpoints.

The implementation keeps every matrix in factored form.  For X = L @ R,
reduced QR decompositions turn the SVD of a potentially large weight update
into the SVD of a matrix no larger than 2r x 2r.  This is especially useful
for cumulative or adjacent differences such as

    B_t A_t - B_ref A_ref.

For non-zero initializations (for example residual-anchor GeoRA), pass the
earliest saved checkpoint as --reference-step.  The resulting cumulative
increment is then explicitly relative to that checkpoint, not to step zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


STEP_RE = re.compile(r"global_step_(\d+)")


@dataclass
class FactoredMatrix:
    left: torch.Tensor
    right: torch.Tensor
    scale: float


@dataclass
class CompactSVD:
    u: torch.Tensor
    s: torch.Tensor
    v: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-step", type=int)
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
    result = []
    for checkpoint in root.glob("global_step_*"):
        for relative in ("actor/lora_adapter", "actor/peft_adapter"):
            adapter = checkpoint / relative
            if (adapter / "adapter_config.json").is_file() and (
                adapter / "adapter_model.safetensors"
            ).is_file():
                result.append((checkpoint_step(checkpoint), adapter))
                break
    if not result:
        raise FileNotFoundError(f"No complete adapters below {root}")
    return sorted(result)


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


def raw_matrix(a: torch.Tensor, b: torch.Tensor, scale: float) -> FactoredMatrix:
    return FactoredMatrix(left=b, right=a, scale=scale)


def difference_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    a_reference: torch.Tensor,
    b_reference: torch.Tensor,
    scale: float,
) -> FactoredMatrix:
    return FactoredMatrix(
        left=torch.cat((b, b_reference), dim=1),
        right=torch.cat((a, -a_reference), dim=0),
        scale=scale,
    )


def compact_svd(matrix: FactoredMatrix) -> CompactSVD:
    q_left, r_left = torch.linalg.qr(matrix.left, mode="reduced")
    q_right, r_right = torch.linalg.qr(matrix.right.mT, mode="reduced")
    u_core, singular_values, vh_core = torch.linalg.svd(
        r_left @ r_right.mT, full_matrices=False
    )
    singular_values = singular_values * abs(matrix.scale)
    return CompactSVD(
        u=q_left @ u_core,
        s=singular_values,
        v=q_right @ vh_core.mT,
    )


def spectrum_metrics(svd: CompactSVD) -> dict[str, float | int]:
    squared = svd.s.square()
    total = float(squared.sum().item())
    if total <= 0:
        return {
            "fro": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
            "top1_energy": 0.0,
            "top4_energy": 0.0,
            "top8_energy": 0.0,
        }
    energy = squared / total
    positive = energy[energy > 0]
    cumulative = torch.cumsum(energy, dim=0)
    return {
        "fro": math.sqrt(total),
        "stable_rank": total / float(squared[0].item()),
        "effective_rank": float(torch.exp(-(positive * positive.log()).sum()).item()),
        "rank_90": int(torch.searchsorted(cumulative, 0.90).item()) + 1,
        "rank_95": int(torch.searchsorted(cumulative, 0.95).item()) + 1,
        "rank_99": int(torch.searchsorted(cumulative, 0.99).item()) + 1,
        "top1_energy": float(energy[:1].sum().item()),
        "top4_energy": float(energy[:4].sum().item()),
        "top8_energy": float(energy[:8].sum().item()),
    }


def matrix_inner(left: FactoredMatrix, right: FactoredMatrix) -> float:
    value = torch.trace(
        (left.left.mT @ right.left) @ (right.right @ left.right.mT)
    )
    return float(value.item()) * left.scale * right.scale


def matrix_cosine(left: FactoredMatrix, right: FactoredMatrix) -> float:
    denominator = math.sqrt(
        max(matrix_inner(left, left), 0.0) * max(matrix_inner(right, right), 0.0)
    )
    return matrix_inner(left, right) / denominator if denominator > 0 else float("nan")


def principal_metrics(left: CompactSVD, right: CompactSVD, k: int) -> dict[str, float]:
    actual_k = min(k, left.u.shape[1], right.u.shape[1], left.v.shape[1], right.v.shape[1])
    if actual_k <= 0:
        return {
            "left_overlap": float("nan"),
            "right_overlap": float("nan"),
            "left_mean_angle_deg": float("nan"),
            "right_mean_angle_deg": float("nan"),
            "left_max_angle_deg": float("nan"),
            "right_max_angle_deg": float("nan"),
        }

    def one_side(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
        cosines = torch.linalg.svdvals(x[:, :actual_k].mT @ y[:, :actual_k]).clamp(0, 1)
        angles = torch.rad2deg(torch.acos(cosines))
        return (
            float(cosines.square().mean().item()),
            float(angles.mean().item()),
            float(angles.max().item()),
        )

    left_overlap, left_mean, left_max = one_side(left.u, right.u)
    right_overlap, right_mean, right_max = one_side(left.v, right.v)
    return {
        "left_overlap": left_overlap,
        "right_overlap": right_overlap,
        "left_mean_angle_deg": left_mean,
        "right_mean_angle_deg": right_mean,
        "left_max_angle_deg": left_max,
        "right_max_angle_deg": right_max,
    }


def projection_energy(source: CompactSVD, target: FactoredMatrix, k: int) -> dict[str, float]:
    actual_k = min(k, source.u.shape[1], source.v.shape[1])
    if actual_k <= 0:
        return {"left_capture": float("nan"), "right_capture": float("nan"), "both_capture": float("nan")}
    u = source.u[:, :actual_k]
    v = source.v[:, :actual_k]
    left_coordinates = u.mT @ target.left
    right_coordinates = target.right @ v
    left_gram = target.left.mT @ target.left
    right_gram = target.right @ target.right.mT
    total = max(matrix_inner(target, target), 0.0)
    scale_sq = target.scale * target.scale
    left_energy = float(torch.trace((left_coordinates.mT @ left_coordinates) @ right_gram).item()) * scale_sq
    right_energy = float(torch.trace(left_gram @ (right_coordinates @ right_coordinates.mT)).item()) * scale_sq
    both_energy = float(torch.sum((left_coordinates @ right_coordinates).square()).item()) * scale_sq
    denominator = max(total, 1e-300)
    return {
        "left_capture": min(max(left_energy / denominator, 0.0), 1.0),
        "right_capture": min(max(right_energy / denominator, 0.0), 1.0),
        "both_capture": min(max(both_energy / denominator, 0.0), 1.0),
    }


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def weighted_mean(values: list[float], weights: list[float]) -> float:
    pairs = [(value, weight) for value, weight in zip(values, weights) if math.isfinite(value) and weight > 0]
    denominator = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / denominator if denominator else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_checkpoint(rows: list[dict], top_ks: list[int]) -> list[dict]:
    result = []
    for step in sorted({int(row["step"]) for row in rows}):
        group = [row for row in rows if row["step"] == step]
        if not group:
            continue
        weights = [float(row["increment_fro"]) ** 2 for row in group]
        output = {
            "step": step,
            "modules": len(group),
            "increment_global_fro": math.sqrt(sum(weights)),
            "increment_stable_rank_mean": mean([float(row["increment_stable_rank"]) for row in group]),
            "increment_effective_rank_mean": mean([float(row["increment_effective_rank"]) for row in group]),
            "increment_rank_95_mean": mean([float(row["increment_rank_95"]) for row in group]),
            "increment_rank_99_mean": mean([float(row["increment_rank_99"]) for row in group]),
            "increment_top8_energy_weighted": weighted_mean(
                [float(row["increment_top8_energy"]) for row in group], weights
            ),
            "increment_cosine_previous_weighted": weighted_mean(
                [float(row["increment_cosine_previous"]) for row in group], weights
            ),
        }
        for k in top_ks:
            for metric in ("left_overlap", "right_overlap", "left_mean_angle_deg", "right_mean_angle_deg"):
                key = f"increment_previous_k{k}_{metric}"
                output[key] = mean([float(row[key]) for row in group])
        result.append(output)
    return result


def aggregate_intervals(rows: list[dict], top_ks: list[int]) -> list[dict]:
    result = []
    pairs = sorted({(int(row["from_step"]), int(row["to_step"])) for row in rows})
    for from_step, to_step in pairs:
        group = [row for row in rows if row["from_step"] == from_step and row["to_step"] == to_step]
        weights = [float(row["move_fro"]) ** 2 for row in group]
        output = {
            "from_step": from_step,
            "to_step": to_step,
            "modules": len(group),
            "move_global_fro": math.sqrt(sum(weights)),
            "move_stable_rank_mean": mean([float(row["move_stable_rank"]) for row in group]),
            "move_effective_rank_mean": mean([float(row["move_effective_rank"]) for row in group]),
            "move_rank_95_mean": mean([float(row["move_rank_95"]) for row in group]),
            "move_rank_99_mean": mean([float(row["move_rank_99"]) for row in group]),
            "move_top8_energy_weighted": weighted_mean(
                [float(row["move_top8_energy"]) for row in group], weights
            ),
            "move_cosine_previous_weighted": weighted_mean(
                [float(row["move_cosine_previous"]) for row in group], weights
            ),
        }
        for k in top_ks:
            for metric in (
                "left_overlap",
                "right_overlap",
                "left_mean_angle_deg",
                "right_mean_angle_deg",
                "left_capture",
                "right_capture",
                "both_capture",
            ):
                key = f"previous_move_k{k}_{metric}"
                output[key] = weighted_mean([float(row[key]) for row in group], weights)
        for module in sorted({str(row["module"]) for row in group}):
            module_energy = sum(
                float(row["move_fro"]) ** 2 for row in group if row["module"] == module
            )
            output[f"energy_share_{module}"] = module_energy / max(sum(weights), 1e-300)
        result.append(output)
    return result


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    top_ks = sorted({int(value) for value in args.top_k.split(",") if value.strip()})
    adapters = discover_adapters(args.checkpoint_root.resolve())
    steps = [step for step, _ in adapters]
    reference_step = args.reference_step if args.reference_step is not None else steps[0]
    if reference_step not in steps:
        raise ValueError(f"Reference step {reference_step} is not available: {steps}")

    configs = [json.loads((path / "adapter_config.json").read_text()) for _, path in adapters]
    ranks = {int(config["r"]) for config in configs}
    scales = {float(config["lora_alpha"]) / int(config["r"]) for config in configs}
    if len(ranks) != 1 or len(scales) != 1:
        raise ValueError(f"Rank or scaling changes across checkpoints: ranks={ranks}, scales={scales}")
    rank = ranks.pop()
    scale = scales.pop()
    filters = [value.strip() for value in args.module_filter.split(",") if value.strip()]

    checkpoint_rows: list[dict] = []
    interval_rows: list[dict] = []
    with ExitStack() as stack:
        handles = {
            step: stack.enter_context(
                safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu")
            )
            for step, path in adapters
        }
        reference_handle = handles[reference_step]
        keys = sorted(key for key in reference_handle.keys() if key.endswith(".lora_A.weight"))
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
            raw = {step: raw_matrix(*factors[step], scale) for step in steps}
            raw_svd = {step: compact_svd(raw[step]) for step in steps}
            a_reference, b_reference = factors[reference_step]
            increments = {
                step: difference_matrix(*factors[step], a_reference, b_reference, scale)
                for step in steps
                if step != reference_step
            }
            increment_svd = {step: compact_svd(value) for step, value in increments.items()}
            moves = {
                right_step: difference_matrix(
                    *factors[right_step], *factors[left_step], scale
                )
                for left_step, right_step in zip(steps, steps[1:])
            }
            move_svd = {step: compact_svd(value) for step, value in moves.items()}
            layer = layer_index(a_key)
            common = {
                "adapter_key": a_key.removesuffix(".lora_A.weight"),
                "layer": layer,
                "layer_band": layer_band(layer, layer_count),
                "module": module_name(a_key),
            }

            previous_increment_step = None
            for step in steps:
                if step == reference_step:
                    continue
                metrics = spectrum_metrics(increment_svd[step])
                row = {
                    **common,
                    "step": step,
                    "reference_step": reference_step,
                    **{f"increment_{key}": value for key, value in metrics.items()},
                    "raw_cosine_reference": matrix_cosine(raw[reference_step], raw[step]),
                    "increment_cosine_previous": (
                        matrix_cosine(increments[previous_increment_step], increments[step])
                        if previous_increment_step is not None
                        else float("nan")
                    ),
                }
                for k in top_ks:
                    raw_principal = principal_metrics(raw_svd[reference_step], raw_svd[step], k)
                    row.update({f"raw_reference_k{k}_{key}": value for key, value in raw_principal.items()})
                    if previous_increment_step is None:
                        increment_principal = {key: float("nan") for key in principal_metrics(increment_svd[step], increment_svd[step], k)}
                    else:
                        increment_principal = principal_metrics(
                            increment_svd[previous_increment_step], increment_svd[step], k
                        )
                    row.update(
                        {f"increment_previous_k{k}_{key}": value for key, value in increment_principal.items()}
                    )
                checkpoint_rows.append(row)
                previous_increment_step = step

            previous_move_step = None
            for left_step, right_step in zip(steps, steps[1:]):
                metrics = spectrum_metrics(move_svd[right_step])
                row = {
                    **common,
                    "from_step": left_step,
                    "to_step": right_step,
                    **{f"move_{key}": value for key, value in metrics.items()},
                    "move_cosine_previous": (
                        matrix_cosine(moves[previous_move_step], moves[right_step])
                        if previous_move_step is not None
                        else float("nan")
                    ),
                }
                for k in top_ks:
                    if previous_move_step is None:
                        principal = {key: float("nan") for key in principal_metrics(move_svd[right_step], move_svd[right_step], k)}
                        capture = {key: float("nan") for key in projection_energy(move_svd[right_step], moves[right_step], k)}
                    else:
                        principal = principal_metrics(move_svd[previous_move_step], move_svd[right_step], k)
                        capture = projection_energy(move_svd[previous_move_step], moves[right_step], k)
                    row.update({f"previous_move_k{k}_{key}": value for key, value in principal.items()})
                    row.update({f"previous_move_k{k}_{key}": value for key, value in capture.items()})
                interval_rows.append(row)
                previous_move_step = right_step

            print(f"[{module_index:03d}/{len(keys):03d}] {common['adapter_key']}", flush=True)

    checkpoint_summary = aggregate_checkpoint(checkpoint_rows, top_ks)
    interval_summary = aggregate_intervals(interval_rows, top_ks)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "checkpoint_per_module.csv", checkpoint_rows)
    write_csv(out_dir / "checkpoint_summary.csv", checkpoint_summary)
    write_csv(out_dir / "interval_per_module.csv", interval_rows)
    write_csv(out_dir / "interval_summary.csv", interval_summary)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "steps": steps,
        "reference_step": reference_step,
        "configured_rank": rank,
        "scale": scale,
        "top_k": top_ks,
        "modules": len({row["adapter_key"] for row in checkpoint_rows}),
        "definitions": {
            "cumulative_increment": "scale * (B_t A_t - B_reference A_reference)",
            "adjacent_move": "scale * (B_t A_t - B_previous A_previous)",
            "subspace_overlap": "mean squared cosine of the top-k principal angles; 1 means identical",
            "left_or_right_capture": "fraction of next adjacent-move energy retained by the previous move's left or right top-k subspace",
            "both_capture": "fraction retained after simultaneous left and right projection",
        },
        "checkpoint_summary": checkpoint_summary,
        "interval_summary": interval_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
