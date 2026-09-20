#!/usr/bin/env python3
"""Diagnose B/Delta-W spectra and rank-map stability across LoRA checkpoints.

The script never forms the large ``Delta W = scale * B @ A`` matrix.  Reduced
QR factorizations turn its non-zero spectrum into an exact rank-sized problem:

    B = Q_b R_b, A.T = Q_a R_a
    sv(Delta W) = abs(scale) * sv(R_b @ R_a.T)

When A has orthonormal rows, the scaled B and Delta-W spectra are identical.
Reporting both therefore also tests the fixed-A pruning assumption directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


TARGETS = (0.90, 0.95, 0.98, 0.99)
RANK_BINS = (8, 12, 16, 20, 24, 28, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"global_step_(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot infer checkpoint step from {path}")
    return int(match.group(1))


def discover_adapters(root: Path) -> list[Path]:
    paths = [
        path
        for path in root.glob("global_step_*/actor/peft_adapter")
        if (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
    ]
    paths.sort(key=checkpoint_step)
    if not paths:
        raise FileNotFoundError(f"No complete adapters found below {root}")
    return paths


def suffix_lookup(pattern: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not pattern:
        return default
    matches = [(name, value) for name, value in pattern.items() if key.endswith(name)]
    if not matches:
        return default
    return max(matches, key=lambda item: len(item[0]))[1]


def layer_index(key: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def module_type(key: str) -> str:
    return key.split(".")[-1]


def layer_band(layer: int | None, layer_count: int) -> str:
    if layer is None:
        return "other"
    width = max(1, math.ceil(layer_count / 3))
    if layer < width:
        return "early"
    if layer < 2 * width:
        return "middle"
    return "late"


def energy_rank(energy: torch.Tensor, target: float) -> int:
    if float(energy.sum()) <= 0:
        return int(energy.numel())
    normalized = energy / energy.sum()
    return int(torch.searchsorted(normalized.cumsum(0), target).item()) + 1


def quantize_rank(rank: int, maximum: int) -> int:
    bins = sorted({value for value in RANK_BINS if value <= maximum} | {maximum})
    return next((value for value in bins if value >= rank), maximum)


def spectrum_metrics(values: torch.Tensor, prefix: str) -> dict[str, Any]:
    values = values.double().clamp_min(0)
    squared = values.square()
    total = float(squared.sum().item())
    if total <= 0:
        energy = torch.zeros_like(values)
        stable_rank = 0.0
        effective_rank = 0.0
    else:
        energy = squared / total
        stable_rank = total / float(squared[0].item())
        positive = energy[energy > 0]
        effective_rank = float(torch.exp(-(positive * positive.log()).sum()).item())
    result: dict[str, Any] = {
        f"{prefix}_frobenius": math.sqrt(total),
        f"{prefix}_stable_rank": stable_rank,
        f"{prefix}_effective_rank": effective_rank,
        f"{prefix}_top1_energy": float(energy[:1].sum().item()),
        f"{prefix}_top4_energy": float(energy[:4].sum().item()),
        f"{prefix}_energy_at_8": float(energy[:8].sum().item()),
        f"{prefix}_energy_at_16": float(energy[:16].sum().item()),
        f"{prefix}_energy_at_24": float(energy[:24].sum().item()),
        f"{prefix}_singular_values": [float(value) for value in values.tolist()],
        f"{prefix}_energy_distribution": [float(value) for value in energy.tolist()],
    }
    for target in TARGETS:
        suffix = int(round(target * 100))
        raw = energy_rank(energy, target)
        result[f"{prefix}_rank_{suffix}"] = raw
        result[f"{prefix}_qrank_{suffix}"] = quantize_rank(raw, values.numel())
    return result


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        value = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = value
        start = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 1.0


def distribution_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def checkpoint_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "a_orthogonality_error",
        "b_delta_energy_l1",
        "b_delta_energy_cosine",
        "b_stable_rank",
        "b_effective_rank",
        "delta_stable_rank",
        "delta_effective_rank",
    ]
    for prefix in ("b", "delta"):
        for target in (90, 95, 98, 99):
            metrics.extend((f"{prefix}_rank_{target}", f"{prefix}_qrank_{target}"))
    summary = {metric: distribution_summary([float(row[metric]) for row in rows]) for metric in metrics}

    current_parameters = sum(
        int(row["configured_rank"]) * (int(row["d_in"]) + int(row["d_out"])) for row in rows
    )
    current_b_parameters = sum(int(row["configured_rank"]) * int(row["d_out"]) for row in rows)
    maps = {}
    for target in (90, 95, 98, 99):
        field = f"delta_qrank_{target}"
        total = sum(int(row[field]) * (int(row["d_in"]) + int(row["d_out"])) for row in rows)
        trainable_b = sum(int(row[field]) * int(row["d_out"]) for row in rows)
        maps[str(target)] = {
            "rank": distribution_summary([float(row[field]) for row in rows]),
            "adapter_parameters": total,
            "adapter_parameter_ratio": total / current_parameters,
            "trainable_b_parameters": trainable_b,
            "trainable_b_ratio": trainable_b / current_b_parameters,
        }
    summary["maps"] = maps
    summary["current_adapter_parameters"] = current_parameters
    summary["current_trainable_b_parameters"] = current_b_parameters
    return summary


def map_comparison(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], left_step: int, right_step: int
) -> dict[str, Any]:
    left = {row["adapter_key"]: row for row in left_rows}
    right = {row["adapter_key"]: row for row in right_rows}
    keys = sorted(left.keys() & right.keys())
    result: dict[str, Any] = {
        "left_step": left_step,
        "right_step": right_step,
        "modules": len(keys),
        "targets": {},
    }
    for target in (90, 95, 98, 99):
        field = f"delta_qrank_{target}"
        x = [float(left[key][field]) for key in keys]
        y = [float(right[key][field]) for key in keys]
        diffs = [abs(a - b) for a, b in zip(x, y)]
        result["targets"][str(target)] = {
            "mae": statistics.fmean(diffs),
            "exact_ratio": sum(value == 0 for value in diffs) / len(diffs),
            "within_4_ratio": sum(value <= 4 for value in diffs) / len(diffs),
            "spearman": correlation(rankdata(x), rankdata(y)),
            "mean_rank_left": statistics.fmean(x),
            "mean_rank_right": statistics.fmean(y),
        }
    for prefix in ("b", "delta"):
        l1_values = []
        cosine_values = []
        for key in keys:
            p = torch.tensor(left[key][f"{prefix}_energy_distribution"], dtype=torch.float64)
            q = torch.tensor(right[key][f"{prefix}_energy_distribution"], dtype=torch.float64)
            l1_values.append(float(torch.linalg.vector_norm(p - q, ord=1).item()))
            denominator = float(torch.linalg.vector_norm(p) * torch.linalg.vector_norm(q))
            cosine_values.append(float(torch.dot(p, q)) / denominator if denominator else 1.0)
        result[f"{prefix}_energy_l1_mean"] = statistics.fmean(l1_values)
        result[f"{prefix}_energy_cosine_mean"] = statistics.fmean(cosine_values)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {"b_singular_values", "b_energy_distribution", "delta_singular_values", "delta_energy_distribution"}
    fields = sorted({key for row in rows for key in row if key not in excluded})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fields})


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# B / Delta-W Spectrum And Rank-Map Trajectory",
        "",
        f"Generated: `{result['created_at_utc']}`",
        "",
        "All energy metrics use squared singular values. Delta-W spectra are exact and are computed from the rank-sized QR core without materializing `B @ A`.",
        "",
        "## Checkpoint spectra",
        "",
        "| Step | A orth err | B/Delta energy L1 | Delta eff. rank | Rank@90 | Rank@95 | Rank@98 | Rank@99 | E@8 | E@16 | E@24 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows_by_step = {int(row["step"]): row for row in result["checkpoint_summaries"]}
    for step in result["steps"]:
        item = rows_by_step[step]
        summary = item["summary"]
        raw_rows = result["rows_by_step"][str(step)]
        lines.append(
            f"| {step} | {summary['a_orthogonality_error']['mean']:.6f} | "
            f"{summary['b_delta_energy_l1']['mean']:.6f} | {summary['delta_effective_rank']['mean']:.2f} | "
            f"{summary['delta_rank_90']['mean']:.2f} | {summary['delta_rank_95']['mean']:.2f} | "
            f"{summary['delta_rank_98']['mean']:.2f} | {summary['delta_rank_99']['mean']:.2f} | "
            f"{statistics.fmean(float(row['delta_energy_at_8']) for row in raw_rows):.3%} | "
            f"{statistics.fmean(float(row['delta_energy_at_16']) for row in raw_rows):.3%} | "
            f"{statistics.fmean(float(row['delta_energy_at_24']) for row in raw_rows):.3%} |"
        )
    lines.extend(
        [
            "",
            "## Quantized rank maps",
            "",
            "Ranks are rounded up to `{8,12,16,20,24,28,32}` subject to each module's available rank.",
            "",
            "| Step | Target | Mean rank | Adapter params | Adapter ratio | Trainable B params | B ratio |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for step in result["steps"]:
        summary = rows_by_step[step]["summary"]
        for target in (90, 95, 98, 99):
            item = summary["maps"][str(target)]
            lines.append(
                f"| {step} | {target}% | {item['rank']['mean']:.2f} | {item['adapter_parameters']:,} | "
                f"{item['adapter_parameter_ratio']:.2%} | {item['trainable_b_parameters']:,} | "
                f"{item['trainable_b_ratio']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Temporal stability versus final checkpoint",
            "",
            "| From | To | Target | Rank MAE | Exact | Within 4 | Spearman |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    final_step = result["steps"][-1]
    for item in result["comparisons"]:
        if item["right_step"] != final_step:
            continue
        for target in (90, 95, 98, 99):
            metric = item["targets"][str(target)]
            lines.append(
                f"| {item['left_step']} | {item['right_step']} | {target}% | {metric['mae']:.2f} | "
                f"{metric['exact_ratio']:.2%} | {metric['within_4_ratio']:.2%} | {metric['spearman']:.4f} |"
            )
    lines.extend(
        [
            "",
            "`B/Delta energy L1` tests whether A is sufficiently row-orthonormal for the scaled B spectrum to stand in for the Delta-W spectrum. Rank maps here are spectral diagnostics only; policy-level truncation effects are evaluated separately.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    adapters = discover_adapters(args.checkpoint_root.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    all_rows = []
    summaries = []
    for adapter in adapters:
        step = checkpoint_step(adapter)
        config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
        rank_pattern = config.get("rank_pattern") or {}
        alpha_pattern = config.get("alpha_pattern") or {}
        rows = []
        with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
            layers = [layer_index(key) for key in a_keys]
            layer_count = max((value for value in layers if value is not None), default=-1) + 1
            for a_key in a_keys:
                b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
                if b_key not in keys:
                    raise KeyError(f"Missing B tensor for {a_key}")
                adapter_key = a_key.removesuffix(".lora_A.weight")
                a = handle.get_tensor(a_key).to(dtype=dtype)
                b = handle.get_tensor(b_key).to(dtype=dtype)
                actual_rank = int(a.shape[0])
                expected_rank = int(suffix_lookup(rank_pattern, adapter_key, config["r"]))
                if actual_rank != expected_rank:
                    raise ValueError(
                        f"Rank mismatch for {adapter_key}: tensor={actual_rank}, config={expected_rank}"
                    )
                alpha = float(suffix_lookup(alpha_pattern, adapter_key, config["lora_alpha"]))
                scale = alpha / actual_rank
                _qb, rb = torch.linalg.qr(b, mode="reduced")
                _qa, ra = torch.linalg.qr(a.mT, mode="reduced")
                b_values = torch.linalg.svdvals(rb).mul(abs(scale))
                delta_values = torch.linalg.svdvals(rb @ ra.mT).mul(abs(scale))
                b_metrics = spectrum_metrics(b_values, "b")
                delta_metrics = spectrum_metrics(delta_values, "delta")
                p = torch.tensor(b_metrics["b_energy_distribution"], dtype=torch.float64)
                q = torch.tensor(delta_metrics["delta_energy_distribution"], dtype=torch.float64)
                denominator = float(torch.linalg.vector_norm(p) * torch.linalg.vector_norm(q))
                gram = a @ a.mT
                orth_error = float(
                    torch.linalg.matrix_norm(gram - torch.eye(actual_rank, dtype=dtype), ord="fro").item()
                    / math.sqrt(actual_rank)
                )
                layer = layer_index(adapter_key)
                row = {
                    "step": step,
                    "adapter_key": adapter_key,
                    "layer": layer,
                    "layer_band": layer_band(layer, layer_count),
                    "module_type": module_type(adapter_key),
                    "d_in": int(a.shape[1]),
                    "d_out": int(b.shape[0]),
                    "configured_rank": actual_rank,
                    "alpha": alpha,
                    "scale": scale,
                    "a_orthogonality_error": orth_error,
                    "b_delta_energy_l1": float(torch.linalg.vector_norm(p - q, ord=1).item()),
                    "b_delta_energy_cosine": float(torch.dot(p, q).item()) / denominator if denominator else 1.0,
                    **b_metrics,
                    **delta_metrics,
                }
                rows.append(row)
        rows_by_step[step] = rows
        all_rows.extend(rows)
        summaries.append({"step": step, "summary": checkpoint_summary(rows)})
        print(f"step={step}: modules={len(rows)}", flush=True)

    steps = sorted(rows_by_step)
    comparisons = []
    pairs = {(steps[index], steps[index + 1]) for index in range(len(steps) - 1)}
    pairs.update((step, steps[-1]) for step in steps[:-1])
    for left, right in sorted(pairs):
        comparisons.append(map_comparison(rows_by_step[left], rows_by_step[right], left, right))

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "steps": steps,
        "targets": list(TARGETS),
        "rank_bins": list(RANK_BINS),
        "checkpoint_summaries": summaries,
        "comparisons": comparisons,
        "rows_by_step": {str(step): rows for step, rows in rows_by_step.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out_dir / "per_module.csv", all_rows)
    with (out_dir / "rank_map_stability.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "left_step", "right_step", "target", "mae", "exact_ratio", "within_4_ratio",
            "spearman", "mean_rank_left", "mean_rank_right",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            for target, values in comparison["targets"].items():
                writer.writerow(
                    {
                        "left_step": comparison["left_step"],
                        "right_step": comparison["right_step"],
                        "target": target,
                        **values,
                    }
                )
    write_report(out_dir / "report.md", result)
    print(f"Wrote trajectory diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
