#!/usr/bin/env python3
"""Analyze the intrinsic spectrum of LoRA updates without forming ``B @ A``.

For ``B in R^(m x r)`` and ``A in R^(r x n)``, reduced QR decompositions give

    B = Q_b R_b,  A.T = Q_a R_a,
    B A = Q_b (R_b R_a.T) Q_a.T.

The nonzero singular values of the large update are therefore exactly those
of the small ``r x r`` core. This keeps the analysis cheap for wide MLP
matrices and avoids loading the frozen base model.
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

import torch
from safetensors import safe_open


ENERGY_TARGETS = (0.90, 0.95, 0.99, 0.999)
SUMMARY_METRICS = (
    "stable_rank",
    "energy_effective_rank",
    "energy_rank_90",
    "energy_rank_95",
    "energy_rank_99",
    "energy_rank_999",
    "energy_at_rank_8",
    "energy_at_rank_16",
    "energy_at_rank_24",
    "top1_energy_ratio",
    "top4_energy_ratio",
    "tail8_energy_ratio",
    "sigma_last_over_first",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="PEFT adapter directory; may be passed more than once",
    )
    parser.add_argument(
        "--checkpoint-root",
        help="Discover global_step_*/actor/peft_adapter directories below this run",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--module-filter", default="", help="Comma-separated key substrings")
    parser.add_argument("--max-modules", type=int, default=0, help="Debug limit; 0 means all")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int | None:
    for parent in (path, *path.parents):
        match = re.fullmatch(r"global_step_(\d+)", parent.name)
        if match:
            return int(match.group(1))
    return None


def discover_adapters(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value).resolve() for value in args.adapter]
    if args.checkpoint_root:
        root = Path(args.checkpoint_root).resolve()
        paths.extend(root.glob("global_step_*/actor/peft_adapter"))

    valid = []
    for path in paths:
        if not (path / "adapter_config.json").is_file():
            continue
        if not (path / "adapter_model.safetensors").is_file():
            continue
        valid.append(path)
    valid = sorted(set(valid), key=lambda path: (checkpoint_step(path) or -1, str(path)))
    if not valid:
        raise FileNotFoundError("No complete PEFT adapter directories were found")
    return valid


def load_config(path: Path) -> dict:
    with open(path / "adapter_config.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def module_name_from_a_key(key: str) -> str:
    return key.removesuffix(".lora_A.weight").split(".")[-1]


def layer_index_from_key(key: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def layer_band(layer: int | None, layer_count: int) -> str:
    if layer is None:
        return "other"
    third = max(1, math.ceil(layer_count / 3))
    if layer < third:
        return "early"
    if layer < 2 * third:
        return "middle"
    return "late"


def energy_rank(energy: torch.Tensor, target: float) -> int:
    return int(torch.searchsorted(torch.cumsum(energy, dim=0), target).item()) + 1


def spectrum_metrics(singular_values: torch.Tensor) -> dict[str, float | int | list[float]]:
    singular_values = singular_values.double().clamp_min(0)
    squared = singular_values.square()
    total = float(squared.sum().item())
    if total == 0:
        zeros = [0.0] * singular_values.numel()
        return {
            "numerical_rank": 0,
            "stable_rank": 0.0,
            "energy_effective_rank": 0.0,
            **{f"energy_rank_{int(target * 1000) if target == 0.999 else int(target * 100)}": 0 for target in ENERGY_TARGETS},
            "top1_energy_ratio": 0.0,
            "top4_energy_ratio": 0.0,
            "tail8_energy_ratio": 0.0,
            "energy_at_rank_8": 0.0,
            "energy_at_rank_16": 0.0,
            "energy_at_rank_24": 0.0,
            "sigma_last_over_first": 0.0,
            "singular_values": zeros,
            "energy_ratios": zeros,
        }

    energy = squared / total
    positive = energy[energy > 0]
    effective_rank = float(torch.exp(-(positive * positive.log()).sum()).item())
    sigma0 = float(singular_values[0].item())
    tolerance = max(1, singular_values.numel()) * torch.finfo(singular_values.dtype).eps * sigma0
    result: dict[str, float | int | list[float]] = {
        "numerical_rank": int((singular_values > tolerance).sum().item()),
        "stable_rank": total / (sigma0 * sigma0),
        "energy_effective_rank": effective_rank,
        "top1_energy_ratio": float(energy[0].item()),
        "top4_energy_ratio": float(energy[:4].sum().item()),
        "tail8_energy_ratio": float(energy[-8:].sum().item()),
        "energy_at_rank_8": float(energy[:8].sum().item()),
        "energy_at_rank_16": float(energy[:16].sum().item()),
        "energy_at_rank_24": float(energy[:24].sum().item()),
        "sigma_last_over_first": float(singular_values[-1].item()) / sigma0,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "energy_ratios": [float(value) for value in energy.tolist()],
    }
    for target in ENERGY_TARGETS:
        suffix = int(target * 1000) if target == 0.999 else int(target * 100)
        result[f"energy_rank_{suffix}"] = energy_rank(energy, target)
    return result


def compact_singular_values(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(f"Incompatible LoRA factors: A={tuple(a.shape)}, B={tuple(b.shape)}")
    _, r_b = torch.linalg.qr(b, mode="reduced")
    _, r_a = torch.linalg.qr(a.mT, mode="reduced")
    return torch.linalg.svdvals(r_b @ r_a.mT).mul_(abs(scale))


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, object] = {"modules": len(rows)}
    for metric in SUMMARY_METRICS:
        values = sorted(float(row[metric]) for row in rows)
        summary[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": values[0],
            "p10": values[int(0.10 * (len(values) - 1))],
            "p90": values[int(0.90 * (len(values) - 1))],
            "max": values[-1],
        }
    return summary


def group_summaries(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def trend_summary(all_rows: list[dict]) -> dict | None:
    steps = sorted({row["step"] for row in all_rows if row["step"] is not None})
    if len(steps) < 2:
        return None
    first_step, last_step = steps[0], steps[-1]
    first = {row["adapter_key"]: row for row in all_rows if row["step"] == first_step}
    last = {row["adapter_key"]: row for row in all_rows if row["step"] == last_step}
    pairs = [(first[key], last[key]) for key in sorted(first.keys() & last.keys())]
    if not pairs:
        return None

    deltas = {}
    for metric in ("stable_rank", "energy_effective_rank", "energy_rank_95", "energy_rank_99"):
        values = [float(right[metric]) - float(left[metric]) for left, right in pairs]
        deltas[metric] = {
            "mean_delta": statistics.fmean(values),
            "median_delta": statistics.median(values),
            "increased_ratio": sum(value > 0 for value in values) / len(values),
            "decreased_ratio": sum(value < 0 for value in values) / len(values),
        }

    l1_distances = []
    cosine_similarities = []
    for left, right in pairs:
        p = torch.tensor(left["energy_ratios"], dtype=torch.float64)
        q = torch.tensor(right["energy_ratios"], dtype=torch.float64)
        l1_distances.append(float(torch.linalg.vector_norm(p - q, ord=1).item()))
        denominator = float(torch.linalg.vector_norm(p).item() * torch.linalg.vector_norm(q).item())
        cosine_similarities.append(float(torch.dot(p, q).item()) / denominator if denominator else 1.0)
    return {
        "first_step": first_step,
        "last_step": last_step,
        "matched_modules": len(pairs),
        "metric_deltas": deltas,
        "energy_distribution_l1_mean": statistics.fmean(l1_distances),
        "energy_distribution_l1_p90": sorted(l1_distances)[int(0.9 * (len(l1_distances) - 1))],
        "energy_distribution_cosine_mean": statistics.fmean(cosine_similarities),
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# LoRA Delta-W Rank Analysis",
        "",
        f"Generated: `{result['created_at_utc']}`",
        "",
        "Metrics use squared singular values as update energy. `energy_rank_99` is the smallest rank retaining 99% of `||Delta W||_F^2`; `stable_rank = ||Delta W||_F^2 / sigma_1^2`.",
        "",
    ]
    for checkpoint in result["checkpoints"]:
        overall = checkpoint["overall"]
        lines.extend(
            [
                f"## Step {checkpoint['step']}",
                "",
                f"- Modules: `{overall['modules']}`",
                f"- Stable rank: mean `{overall['stable_rank']['mean']:.3f}`, median `{overall['stable_rank']['median']:.3f}`",
                f"- Energy effective rank: mean `{overall['energy_effective_rank']['mean']:.3f}`, median `{overall['energy_effective_rank']['median']:.3f}`",
                f"- Rank@95% energy: mean `{overall['energy_rank_95']['mean']:.3f}`, median `{overall['energy_rank_95']['median']:.1f}`",
                f"- Rank@99% energy: mean `{overall['energy_rank_99']['mean']:.3f}`, median `{overall['energy_rank_99']['median']:.1f}`",
                f"- Energy retained at rank 8/16/24: `{overall['energy_at_rank_8']['mean']:.4f}` / `{overall['energy_at_rank_16']['mean']:.4f}` / `{overall['energy_at_rank_24']['mean']:.4f}`",
                f"- Top-1 energy: mean `{overall['top1_energy_ratio']['mean']:.4f}`",
                f"- Top-4 energy: mean `{overall['top4_energy_ratio']['mean']:.4f}`",
                "",
            ]
        )
        lines.extend(
            [
                "### By module type",
                "",
                "| Module | Stable rank | Effective rank | Energy@8 | Energy@16 | Energy@24 | Rank@95 | Rank@99 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, group in checkpoint["by_module"].items():
            lines.append(
                f"| {name} | {group['stable_rank']['mean']:.2f} | "
                f"{group['energy_effective_rank']['mean']:.2f} | {group['energy_at_rank_8']['mean']:.3f} | "
                f"{group['energy_at_rank_16']['mean']:.3f} | {group['energy_at_rank_24']['mean']:.3f} | "
                f"{group['energy_rank_95']['mean']:.2f} | {group['energy_rank_99']['mean']:.2f} |"
            )
        lines.extend(
            [
                "",
                "### By layer band",
                "",
                "| Band | Stable rank | Effective rank | Energy@8 | Energy@16 | Energy@24 | Rank@95 | Rank@99 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, group in checkpoint["by_layer_band"].items():
            lines.append(
                f"| {name} | {group['stable_rank']['mean']:.2f} | "
                f"{group['energy_effective_rank']['mean']:.2f} | {group['energy_at_rank_8']['mean']:.3f} | "
                f"{group['energy_at_rank_16']['mean']:.3f} | {group['energy_at_rank_24']['mean']:.3f} | "
                f"{group['energy_rank_95']['mean']:.2f} | {group['energy_rank_99']['mean']:.2f} |"
            )
        lines.append("")
    trend = result.get("trend")
    if trend:
        lines.extend(
            [
                "## Checkpoint Trend",
                "",
                f"Compared step `{trend['first_step']}` to `{trend['last_step']}` over `{trend['matched_modules']}` modules.",
                "",
                f"- Mean stable-rank delta: `{trend['metric_deltas']['stable_rank']['mean_delta']:+.4f}`",
                f"- Mean rank@99 delta: `{trend['metric_deltas']['energy_rank_99']['mean_delta']:+.4f}`",
                f"- Mean energy-distribution L1 distance: `{trend['energy_distribution_l1_mean']:.5f}`",
                f"- Mean energy-distribution cosine similarity: `{trend['energy_distribution_cosine_mean']:.6f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    adapter_dirs = discover_adapters(args)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    filters = [value.strip() for value in args.module_filter.split(",") if value.strip()]

    all_rows = []
    checkpoints = []
    for adapter_dir in adapter_dirs:
        config = load_config(adapter_dir)
        rank = int(config["r"])
        scale = float(config["lora_alpha"]) / rank
        step = checkpoint_step(adapter_dir)
        adapter_path = adapter_dir / "adapter_model.safetensors"
        rows = []
        with safe_open(adapter_path, framework="pt", device="cpu") as handle:
            keys = sorted(key for key in handle.keys() if key.endswith(".lora_A.weight"))
            if filters:
                keys = [key for key in keys if any(value in key for value in filters)]
            if args.max_modules > 0:
                keys = keys[: args.max_modules]
            layer_indices = [layer_index_from_key(key) for key in keys]
            layer_count = max((value for value in layer_indices if value is not None), default=-1) + 1
            for index, a_key in enumerate(keys, 1):
                b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
                if b_key not in handle.keys():
                    raise KeyError(f"Missing paired LoRA B tensor for {a_key}")
                a = handle.get_tensor(a_key).to(dtype=dtype)
                b = handle.get_tensor(b_key).to(dtype=dtype)
                singular_values = compact_singular_values(a, b, scale)
                metrics = spectrum_metrics(singular_values)
                layer = layer_index_from_key(a_key)
                row = {
                    "step": step,
                    "adapter_key": a_key.removesuffix(".lora_A.weight"),
                    "layer": layer,
                    "layer_band": layer_band(layer, layer_count),
                    "module": module_name_from_a_key(a_key),
                    "d_out": b.shape[0],
                    "d_in": a.shape[1],
                    "configured_rank": rank,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"[{step}:{index:03d}/{len(keys):03d}] {row['adapter_key']} "
                    f"stable={row['stable_rank']:.3f} erank={row['energy_effective_rank']:.3f} "
                    f"r95={row['energy_rank_95']} r99={row['energy_rank_99']}",
                    flush=True,
                )

        checkpoint = {
            "step": step,
            "adapter": str(adapter_dir),
            "scale": scale,
            "overall": summarize(rows),
            "by_module": group_summaries(rows, "module"),
            "by_layer_band": group_summaries(rows, "layer_band"),
            "by_layer": group_summaries(rows, "layer"),
        }
        checkpoints.append(checkpoint)
        all_rows.extend(rows)

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "stable_rank": "sum_i sigma_i^2 / sigma_1^2",
            "energy_effective_rank": "exp(-sum_i p_i log p_i), p_i=sigma_i^2/sum_j sigma_j^2",
            "energy_rank_x": "smallest k retaining x percent of squared-singular-value energy",
            "spectrum_algorithm": "exact singular values through reduced QR of B and A.T plus an r-by-r core SVD",
        },
        "checkpoints": checkpoints,
        "trend": trend_summary(all_rows),
    }

    csv_path = out_dir / "delta_rank_modules.csv"
    json_path = out_dir / "delta_rank_summary.json"
    report_path = out_dir / "delta_rank_report.md"
    csv_rows = [{key: value for key, value in row.items() if key not in ("singular_values", "energy_ratios")} for row in all_rows]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    write_markdown(report_path, result)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
