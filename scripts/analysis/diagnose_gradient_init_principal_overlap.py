#!/usr/bin/env python3
"""Compare gradient-initialized LoRA subspaces across checkpoints.

The diagnostic uses the functional adapter update

    DeltaW_t = scale * B_t @ A_t

rather than treating A_t itself as DeltaW.  It compares DeltaW principal
subspaces across every checkpoint pair and against two fixed references:

* the ordered gradient-sketch basis used to initialize A at step zero;
* the top singular subspaces of the frozen base-model weight.

All large matrices stay factored except for one base weight at a time.  The
LoRA SVD is obtained from a rank-by-rank core after reduced QR factorizations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


STEP_RE = re.compile(r"global_step_(\d+)")


@dataclass
class CompactSVD:
    left: torch.Tensor
    singular: torch.Tensor
    right: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--gradient-subspaces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", default="4,8,16,32")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--module-filter", default="")
    parser.add_argument("--max-modules", type=int, default=0)
    return parser.parse_args()


def discover_adapters(root: Path) -> list[tuple[int, Path]]:
    adapters = []
    for checkpoint in root.glob("global_step_*"):
        match = STEP_RE.fullmatch(checkpoint.name)
        if not match:
            continue
        for relative in ("actor/peft_adapter", "actor/lora_adapter"):
            adapter = checkpoint / relative
            if (adapter / "adapter_config.json").is_file() and (
                adapter / "adapter_model.safetensors"
            ).is_file():
                adapters.append((int(match.group(1)), adapter))
                break
    if not adapters:
        raise FileNotFoundError(f"No complete adapters below {root}")
    return sorted(adapters)


def base_shard_map(base_model: Path) -> dict[str, Path]:
    indexes = list(base_model.glob("*.safetensors.index.json"))
    if indexes:
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
        return {key: base_model / shard for key, shard in index["weight_map"].items()}
    files = sorted(base_model.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"Expected one safetensors file or index below {base_model}")
    with safe_open(files[0], framework="pt", device="cpu") as handle:
        return {key: files[0] for key in handle.keys()}


def adapter_module_key(a_key: str) -> str:
    return a_key.removesuffix(".lora_A.weight").removeprefix("base_model.model.")


def base_weight_key(a_key: str) -> str:
    return f"{adapter_module_key(a_key)}.weight"


def gradient_key(a_key: str) -> str:
    return adapter_module_key(a_key)


def layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def module_type(name: str) -> str:
    return name.removesuffix(".lora_A.weight").split(".")[-1]


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


def orthonormal_basis(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(matrix, mode="reduced").Q


def compact_delta_svd(a: torch.Tensor, b: torch.Tensor, scale: float) -> CompactSVD:
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.mT, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(
        r_b @ r_a.mT, full_matrices=False
    )
    return CompactSVD(
        left=q_b @ u_core,
        singular=singular * abs(scale),
        right=q_a @ vh_core.mT,
    )


def subspace_metrics(reference: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Principal-angle metrics, normalized by the target dimension."""

    if reference.ndim != 2 or target.ndim != 2 or reference.shape[0] != target.shape[0]:
        raise ValueError(f"Incompatible bases: {reference.shape} and {target.shape}")
    cosines = torch.linalg.svdvals(reference.mT @ target).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(cosines))
    return {
        "overlap": float(cosines.square().sum().item()) / target.shape[1],
        "mean_angle_deg": float(angles.mean().item()),
        "max_angle_deg": float(angles.max().item()),
    }


def energy_capture(reference: torch.Tensor, delta: CompactSVD) -> float:
    energy = delta.singular.square()
    total = float(energy.sum().item())
    if total <= 0:
        return float("nan")
    component_capture = (reference.mT @ delta.right).square().sum(dim=0).clamp(0, 1)
    return float((energy * component_capture).sum().item()) / total


def a_drift_path_ratio(
    a: torch.Tensor, b: torch.Tensor, a_initial: torch.Tensor
) -> float:
    """Return ||B(A-A0)||_F / ||BA||_F without materializing either product."""

    gram_b = b.mT @ b

    def squared_norm(right: torch.Tensor) -> float:
        return float(torch.trace(gram_b @ right @ right.mT).clamp_min(0).item())

    full = squared_norm(a)
    drift = squared_norm(a - a_initial)
    return math.sqrt(drift / full) if full > 0 else float("nan")


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else float("nan")


def weighted_mean(values: list[float], weights: list[float]) -> float:
    pairs = [
        (value, weight)
        for value, weight in zip(values, weights, strict=True)
        if math.isfinite(value) and math.isfinite(weight) and weight > 0
    ]
    denominator = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / denominator if denominator else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_checkpoint(
    rows: list[dict[str, Any]], top_ks: list[int]
) -> list[dict[str, Any]]:
    output = []
    for step in sorted({int(row["step"]) for row in rows}):
        group = [row for row in rows if int(row["step"]) == step]
        weights = [float(row["delta_fro"]) ** 2 for row in group]
        item: dict[str, Any] = {
            "step": step,
            "modules": len(group),
            "delta_global_fro": math.sqrt(sum(weights)),
            "a_vs_gradient_full_overlap_mean": mean(
                [float(row["a_vs_gradient_full_overlap"]) for row in group]
            ),
            "a_vs_gradient_full_overlap_weighted": weighted_mean(
                [float(row["a_vs_gradient_full_overlap"]) for row in group], weights
            ),
            "delta_energy_in_gradient_full": weighted_mean(
                [float(row["delta_energy_in_gradient_full"]) for row in group], weights
            ),
            "delta_energy_in_base_right_full": weighted_mean(
                [float(row["delta_energy_in_base_right_full"]) for row in group], weights
            ),
            "a_drift_path_global_ratio": math.sqrt(
                sum(float(row["a_drift_path_ratio"]) ** 2 * weight for row, weight in zip(group, weights, strict=True))
                / max(sum(weights), 1e-300)
            ),
        }
        for k in top_ks:
            eligible = [row for row in group if int(row["rank"]) >= k]
            eligible_weights = [float(row["delta_fro"]) ** 2 for row in eligible]
            item[f"k{k}_eligible_modules"] = len(eligible)
            for field in (
                "delta_right_vs_gradient_overlap",
                "delta_right_vs_base_overlap",
                "delta_left_vs_base_overlap",
            ):
                key = f"k{k}_{field}"
                item[f"{key}_mean"] = mean([float(row[key]) for row in eligible])
                item[f"{key}_weighted"] = weighted_mean(
                    [float(row[key]) for row in eligible], eligible_weights
                )
        output.append(item)
    return output


def aggregate_pairs(
    rows: list[dict[str, Any]], top_ks: list[int]
) -> list[dict[str, Any]]:
    output = []
    pairs = sorted({(int(row["from_step"]), int(row["to_step"])) for row in rows})
    for from_step, to_step in pairs:
        group = [
            row
            for row in rows
            if int(row["from_step"]) == from_step and int(row["to_step"]) == to_step
        ]
        weights = [float(row["to_delta_fro"]) ** 2 for row in group]
        item: dict[str, Any] = {
            "from_step": from_step,
            "to_step": to_step,
            "modules": len(group),
            "a_full_overlap_mean": mean([float(row["a_full_overlap"]) for row in group]),
            "a_full_overlap_weighted": weighted_mean(
                [float(row["a_full_overlap"]) for row in group], weights
            ),
        }
        for k in top_ks:
            eligible = [row for row in group if int(row["rank"]) >= k]
            eligible_weights = [float(row["to_delta_fro"]) ** 2 for row in eligible]
            item[f"k{k}_eligible_modules"] = len(eligible)
            for side in ("right", "left"):
                key = f"k{k}_delta_{side}_overlap"
                item[f"{key}_mean"] = mean([float(row[key]) for row in eligible])
                item[f"{key}_weighted"] = weighted_mean(
                    [float(row[key]) for row in eligible], eligible_weights
                )
        output.append(item)
    return output


def aggregate_reference(rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    final_step = max(int(row["step"]) for row in rows)
    final_rows = [row for row in rows if int(row["step"]) == final_step]
    weights = [float(row["delta_fro"]) ** 2 for row in final_rows]
    output: dict[str, Any] = {
        "weighting_step": final_step,
        "modules": len(final_rows),
        "gradient_vs_base_full_overlap_mean": mean(
            [float(row["gradient_vs_base_full_overlap"]) for row in final_rows]
        ),
        "gradient_vs_base_full_overlap_weighted": weighted_mean(
            [float(row["gradient_vs_base_full_overlap"]) for row in final_rows], weights
        ),
    }
    for k in top_ks:
        eligible = [row for row in final_rows if int(row["rank"]) >= k]
        eligible_weights = [float(row["delta_fro"]) ** 2 for row in eligible]
        key = f"k{k}_gradient_vs_base_overlap"
        output[f"k{k}_eligible_modules"] = len(eligible)
        output[f"{key}_mean"] = mean([float(row[key]) for row in eligible])
        output[f"{key}_weighted"] = weighted_mean(
            [float(row[key]) for row in eligible], eligible_weights
        )
    return output


def matrix_table(
    steps: list[int], pair_summary: list[dict[str, Any]], key: str
) -> list[str]:
    lookup = {
        (int(row["from_step"]), int(row["to_step"])): float(row[key])
        for row in pair_summary
    }
    lines = [
        "| From / To | " + " | ".join(str(step) for step in steps) + " |",
        "| --- | " + " | ".join("---:" for _ in steps) + " |",
    ]
    for left in steps:
        values = []
        for right in steps:
            if left == right:
                values.append("100.00%")
            else:
                pair = (min(left, right), max(left, right))
                value = lookup.get(pair, float("nan"))
                values.append(f"{value:.2%}" if math.isfinite(value) else "-")
        lines.append(f"| {left} | " + " | ".join(values) + " |")
    return lines


def write_report(path: Path, summary: dict[str, Any]) -> None:
    trajectory = summary["checkpoint_summary"]
    pair_summary = summary["pairwise_summary"]
    reference = summary["reference_summary"]
    top_ks = summary["top_k"]
    report_k = max(k for k in top_ks if k <= 8) if any(k <= 8 for k in top_ks) else top_ks[0]
    lines = [
        "# Gradient-Initialized LoRA Subspace Comparison",
        "",
        "## Scope",
        "",
        f"- Checkpoints: `{summary['steps']}`",
        f"- Modules: `{summary['modules']}`",
        f"- Per-module rank range/mean: `{summary['rank_min']}`-`{summary['rank_max']}` / "
        f"`{summary['rank_mean']:.4f}`",
        "- Functional update: `DeltaW_t = scale * B_t @ A_t`; step-zero `B_0=0`, so `DeltaW_0=0`.",
        "- Gradient reference: the ordered sketch-SVD basis that initialized `A_0`.",
        "- Principal reference: exact top singular vectors of each frozen base-model weight.",
        "- Overlap: mean squared principal cosine; 100% means identical subspaces.",
        "",
        "## Checkpoint Alignment",
        "",
        "| Step | Delta Fro | A span in gradient init | Delta energy in gradient init | "
        "Delta energy in base principal-r | B(A-A0) / BA |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in trajectory:
        lines.append(
            f"| {row['step']} | {row['delta_global_fro']:.4f} | "
            f"{row['a_vs_gradient_full_overlap_weighted']:.2%} | "
            f"{row['delta_energy_in_gradient_full']:.2%} | "
            f"{row['delta_energy_in_base_right_full']:.2%} | "
            f"{row['a_drift_path_global_ratio']:.3%} |"
        )
    lines.extend(
        [
            "",
            f"## Pairwise DeltaW Right-Subspace Overlap, Top-{report_k}",
            "",
            *matrix_table(
                summary["steps"], pair_summary, f"k{report_k}_delta_right_overlap_weighted"
            ),
            "",
            f"## Pairwise DeltaW Left-Subspace Overlap, Top-{report_k}",
            "",
            *matrix_table(
                summary["steps"], pair_summary, f"k{report_k}_delta_left_overlap_weighted"
            ),
            "",
            "Rows are weighted across modules by the later checkpoint's `DeltaW` energy. "
            "The pairwise metric compares cumulative functional adapters, not parameter `A` alone.",
            "",
            "## Alignment With Fixed Principal References",
            "",
            "| Step | k | Delta right vs gradient basis | Delta right vs base principal | "
            "Delta left vs base principal |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in trajectory:
        for k in top_ks:
            if row[f"k{k}_eligible_modules"] == 0:
                continue
            lines.append(
                f"| {row['step']} | {k} ({row[f'k{k}_eligible_modules']} modules) | "
                f"{row[f'k{k}_delta_right_vs_gradient_overlap_weighted']:.2%} | "
                f"{row[f'k{k}_delta_right_vs_base_overlap_weighted']:.2%} | "
                f"{row[f'k{k}_delta_left_vs_base_overlap_weighted']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Initialization Basis Versus Base Principal Basis",
            "",
            "| k | Gradient basis vs base right principal | Eligible modules |",
            "| ---: | ---: | ---: |",
        ]
    )
    for k in top_ks:
        if reference[f"k{k}_eligible_modules"] == 0:
            continue
        lines.append(
            f"| {k} | {reference[f'k{k}_gradient_vs_base_overlap_weighted']:.2%} | "
            f"{reference[f'k{k}_eligible_modules']} |"
        )
    lines.extend(
        [
            "",
            f"Full heterogeneous-rank gradient/base overlap is "
            f"`{reference['gradient_vs_base_full_overlap_weighted']:.2%}` when weighted by "
            f"step-{reference['weighting_step']} adapter energy "
            f"(`{reference['gradient_vs_base_full_overlap_mean']:.2%}` unweighted).",
            "",
            "## Files",
            "",
            "- `checkpoint_per_module.csv`: checkpoint/reference metrics for every module.",
            "- `checkpoint_summary.csv`: checkpoint-level aggregates.",
            "- `pairwise_per_module.csv`: every checkpoint pair for every module.",
            "- `pairwise_summary.csv`: pairwise aggregates.",
            "- `summary.json`: definitions and all aggregate values.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    top_ks = sorted({int(value) for value in args.top_k.split(",") if value.strip()})
    adapters = discover_adapters(args.checkpoint_root.resolve())
    steps = [step for step, _ in adapters]
    configs = [json.loads((adapter / "adapter_config.json").read_text()) for _, adapter in adapters]
    structural_fields = ("r", "lora_alpha", "rank_pattern", "alpha_pattern", "use_rslora")
    signatures = [tuple(config.get(field) for field in structural_fields) for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Adapter rank/scaling configuration changed across checkpoints")
    config = configs[0]
    configured_rank = int(config["r"])
    filters = [value.strip() for value in args.module_filter.split(",") if value.strip()]
    shard_map = base_shard_map(args.base_model.resolve())

    checkpoint_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        adapter_handles = {
            step: stack.enter_context(
                safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu")
            )
            for step, adapter in adapters
        }
        gradient_handle = stack.enter_context(
            safe_open(args.gradient_subspaces.resolve(), framework="pt", device="cpu")
        )
        base_handles = {
            shard: stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
            for shard in sorted(set(shard_map.values()))
        }
        first_handle = adapter_handles[steps[0]]
        keys = sorted(key for key in first_handle.keys() if key.endswith(".lora_A.weight"))
        if filters:
            keys = [key for key in keys if any(value in key for value in filters)]
        if args.max_modules > 0:
            keys = keys[: args.max_modules]
        if not keys:
            raise ValueError("No adapter modules selected")

        for module_number, a_key in enumerate(keys, 1):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            weight_key = base_weight_key(a_key)
            probe_key = gradient_key(a_key)
            if weight_key not in shard_map:
                raise KeyError(f"Base weight missing for {a_key}: {weight_key}")
            if probe_key not in gradient_handle.keys():
                raise KeyError(f"Gradient basis missing for {a_key}: {probe_key}")

            rank = int(first_handle.get_tensor(a_key).shape[0])
            expected_rank = int(module_pattern_value(config, "rank_pattern", a_key, configured_rank))
            if rank != expected_rank:
                raise ValueError(f"Rank mismatch for {a_key}: tensor={rank}, config={expected_rank}")
            alpha = module_pattern_value(config, "alpha_pattern", a_key, config["lora_alpha"])
            scale = alpha / rank

            a_initial = gradient_handle.get_tensor(probe_key).to(device=device, dtype=torch.float32)
            if tuple(a_initial.shape) != tuple(first_handle.get_tensor(a_key).shape):
                raise ValueError(
                    f"Gradient/adapter shape mismatch for {a_key}: "
                    f"{tuple(a_initial.shape)} vs {tuple(first_handle.get_tensor(a_key).shape)}"
                )
            gradient_basis = orthonormal_basis(a_initial.mT)

            base_weight = base_handles[shard_map[weight_key]].get_tensor(weight_key).to(
                device=device, dtype=torch.float32
            )
            base_u, _base_s, base_vh = torch.linalg.svd(base_weight, full_matrices=False)
            base_left = base_u[:, :rank].contiguous()
            base_right = base_vh[:rank].mT.contiguous()
            del base_weight, base_u, _base_s, base_vh

            factors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            deltas: dict[int, CompactSVD] = {}
            a_bases: dict[int, torch.Tensor] = {}
            common = {
                "adapter_key": a_key.removesuffix(".lora_A.weight"),
                "base_weight_key": weight_key,
                "layer": layer_index(a_key),
                "module_type": module_type(a_key),
                "rank": rank,
                "alpha": alpha,
                "scale": scale,
            }
            gradient_base_full = subspace_metrics(base_right, gradient_basis)

            for step in steps:
                a = adapter_handles[step].get_tensor(a_key).to(device=device, dtype=torch.float32)
                b = adapter_handles[step].get_tensor(b_key).to(device=device, dtype=torch.float32)
                delta = compact_delta_svd(a, b, scale)
                a_basis = orthonormal_basis(a.mT)
                factors[step] = (a, b)
                deltas[step] = delta
                a_bases[step] = a_basis
                delta_energy = delta.singular.square()
                row: dict[str, Any] = {
                    **common,
                    "step": step,
                    "delta_fro": math.sqrt(float(delta_energy.sum().item())),
                    "a_vs_gradient_full_overlap": subspace_metrics(
                        gradient_basis, a_basis
                    )["overlap"],
                    "delta_energy_in_gradient_full": energy_capture(gradient_basis, delta),
                    "delta_energy_in_base_right_full": energy_capture(base_right, delta),
                    "a_drift_path_ratio": a_drift_path_ratio(a, b, a_initial),
                    "gradient_vs_base_full_overlap": gradient_base_full["overlap"],
                    "gradient_vs_base_full_mean_angle_deg": gradient_base_full["mean_angle_deg"],
                }
                for k in top_ks:
                    if k <= rank:
                        row[f"k{k}_delta_right_vs_gradient_overlap"] = subspace_metrics(
                            gradient_basis[:, :k], delta.right[:, :k]
                        )["overlap"]
                        row[f"k{k}_delta_right_vs_base_overlap"] = subspace_metrics(
                            base_right[:, :k], delta.right[:, :k]
                        )["overlap"]
                        row[f"k{k}_delta_left_vs_base_overlap"] = subspace_metrics(
                            base_left[:, :k], delta.left[:, :k]
                        )["overlap"]
                        row[f"k{k}_gradient_vs_base_overlap"] = subspace_metrics(
                            base_right[:, :k], gradient_basis[:, :k]
                        )["overlap"]
                    else:
                        for field in (
                            "delta_right_vs_gradient_overlap",
                            "delta_right_vs_base_overlap",
                            "delta_left_vs_base_overlap",
                            "gradient_vs_base_overlap",
                        ):
                            row[f"k{k}_{field}"] = float("nan")
                checkpoint_rows.append(row)

            for left_index, from_step in enumerate(steps):
                for to_step in steps[left_index + 1 :]:
                    row = {
                        **common,
                        "from_step": from_step,
                        "to_step": to_step,
                        "to_delta_fro": math.sqrt(
                            float(deltas[to_step].singular.square().sum().item())
                        ),
                        "a_full_overlap": subspace_metrics(
                            a_bases[from_step], a_bases[to_step]
                        )["overlap"],
                    }
                    for k in top_ks:
                        if k <= rank:
                            row[f"k{k}_delta_right_overlap"] = subspace_metrics(
                                deltas[from_step].right[:, :k], deltas[to_step].right[:, :k]
                            )["overlap"]
                            row[f"k{k}_delta_left_overlap"] = subspace_metrics(
                                deltas[from_step].left[:, :k], deltas[to_step].left[:, :k]
                            )["overlap"]
                        else:
                            row[f"k{k}_delta_right_overlap"] = float("nan")
                            row[f"k{k}_delta_left_overlap"] = float("nan")
                    pairwise_rows.append(row)

            del factors, deltas, a_bases, a_initial, gradient_basis, base_left, base_right
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{module_number:03d}/{len(keys):03d}] {common['adapter_key']}", flush=True)

    checkpoint_summary = aggregate_checkpoint(checkpoint_rows, top_ks)
    pairwise_summary = aggregate_pairs(pairwise_rows, top_ks)
    reference_summary = aggregate_reference(checkpoint_rows, top_ks)
    ranks = [int(row["rank"]) for row in checkpoint_rows if int(row["step"]) == steps[0]]
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "base_model": str(args.base_model.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "gradient_subspaces": str(args.gradient_subspaces.resolve()),
        "steps": steps,
        "modules": len(ranks),
        "rank_min": min(ranks),
        "rank_max": max(ranks),
        "rank_mean": statistics.fmean(ranks),
        "top_k": top_ks,
        "definitions": {
            "functional_delta": "scale * B_t @ A_t; B_0=0, hence DeltaW_0=0",
            "gradient_reference": "ordered sketch-SVD basis exported by the gradient probe and used as A_0",
            "base_principal_reference": "exact top-r_m left/right singular vectors of each frozen base weight",
            "overlap": "sum squared principal cosines divided by target dimension; 1 means identical",
            "checkpoint_weighting": "later checkpoint DeltaW Frobenius energy across modules",
        },
        "checkpoint_summary": checkpoint_summary,
        "pairwise_summary": pairwise_summary,
        "reference_summary": reference_summary,
    }
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "checkpoint_per_module.csv", checkpoint_rows)
    write_csv(out_dir / "checkpoint_summary.csv", checkpoint_summary)
    write_csv(out_dir / "pairwise_per_module.csv", pairwise_rows)
    write_csv(out_dir / "pairwise_summary.csv", pairwise_summary)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", result)
    print(f"Wrote gradient/principal subspace diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
