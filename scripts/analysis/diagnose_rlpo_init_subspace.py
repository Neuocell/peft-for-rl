#!/usr/bin/env python3
"""Measure adapter span motion and low-rank recomposition across checkpoints.

For RLPO, the anchor is reconstructed exactly from each frozen base weight's
top-r_m right singular vectors.  A saved gradient-probe subspace can also be
used as the exact ordered step-0 anchor.  For ordinary LoRA controls without
a saved step-0 adapter, the earliest checkpoint can be used as an explicitly
labeled post-initialization span anchor.
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
class DeltaSVD:
    singular: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--anchor-mode",
        choices=("rlpo-base", "gradient-subspace", "first-checkpoint"),
        default="rlpo-base",
        help="Use exact RLPO, saved gradient-probe, or earliest-checkpoint A as the span anchor.",
    )
    parser.add_argument(
        "--anchor-subspaces",
        type=Path,
        help="subspaces.safetensors used to initialize A; required for --anchor-mode gradient-subspace.",
    )
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
            if (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file():
                adapters.append((int(match.group(1)), adapter))
                break
    if not adapters:
        raise FileNotFoundError(f"No complete PEFT adapters below {root}")
    return sorted(adapters)


def base_weight_key(adapter_a_key: str) -> str:
    module = adapter_a_key.removesuffix(".lora_A.weight")
    module = module.removeprefix("base_model.model.")
    return f"{module}.weight"


def layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
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


def module_type(name: str) -> str:
    return name.removesuffix(".lora_A.weight").split(".")[-1]


def compact_delta_svd(a: torch.Tensor, b: torch.Tensor, scale: float) -> DeltaSVD:
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.mT, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(r_b @ r_a.mT, full_matrices=False)
    return DeltaSVD(
        singular=singular * abs(scale),
        left=q_b @ u_core,
        right=q_a @ vh_core.mT,
    )


def adapter_module_key(adapter_a_key: str) -> str:
    module = adapter_a_key.removesuffix(".lora_A.weight")
    return module.removeprefix("base_model.model.")


def module_pattern_value(config: dict[str, Any], field: str, adapter_a_key: str, default: float) -> float:
    pattern = config.get(field) or {}
    module = adapter_module_key(adapter_a_key)
    if module in pattern:
        return float(pattern[module])
    matches = [float(value) for key, value in pattern.items() if module.endswith(key)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {field} entries for {module}")
    return matches[0] if matches else float(default)


def distribution_metrics(energy: torch.Tensor) -> dict[str, float | int]:
    total = float(energy.sum().item())
    if total <= 0:
        return {
            "effective_atoms": 0.0,
            "participation_atoms": 0.0,
            "rank95_atoms": 0,
            "top_quarter_energy": 0.0,
            "top_half_energy": 0.0,
            "prefix_quarter_energy": 0.0,
            "prefix_half_energy": 0.0,
        }
    probability = energy / total
    positive = probability[probability > 0]
    sorted_probability = probability.sort(descending=True).values
    rank = probability.numel()
    quarter = max(1, math.ceil(rank / 4))
    half = max(1, math.ceil(rank / 2))
    cumulative = sorted_probability.cumsum(dim=0)
    return {
        "effective_atoms": float(torch.exp(-(positive * positive.log()).sum()).item()),
        "participation_atoms": float(1.0 / probability.square().sum().item()),
        "rank95_atoms": int(torch.searchsorted(cumulative, 0.95).item()) + 1,
        "top_quarter_energy": float(sorted_probability[:quarter].sum().item()),
        "top_half_energy": float(sorted_probability[:half].sum().item()),
        "prefix_quarter_energy": float(probability[:quarter].sum().item()),
        "prefix_half_energy": float(probability[:half].sum().item()),
    }


def direction_mixing_metrics(anchor_basis: torch.Tensor, delta: DeltaSVD) -> dict[str, float]:
    """Measure how densely learned right singular vectors mix anchor atoms."""

    singular_energy = delta.singular.square()
    total = float(singular_energy.sum().item())
    if total <= 0:
        return {
            "direction_effective_atoms": 0.0,
            "direction_participation_atoms": 0.0,
            "direction_largest_atom_share": 0.0,
        }
    coordinates = anchor_basis.mT @ delta.right
    coordinate_energy = coordinates.square()
    contained = coordinate_energy.sum(dim=0).clamp_min(1e-30)
    probability = coordinate_energy / contained
    entropy = -(probability.clamp_min(1e-30) * probability.clamp_min(1e-30).log()).sum(dim=0)
    effective = entropy.exp()
    participation = 1.0 / probability.square().sum(dim=0).clamp_min(1e-30)
    largest = probability.max(dim=0).values
    weights = singular_energy / singular_energy.sum()
    return {
        "direction_effective_atoms": float((weights * effective).sum().item()),
        "direction_participation_atoms": float((weights * participation).sum().item()),
        "direction_largest_atom_share": float((weights * largest).sum().item()),
    }


def same_dimensional_subspace_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"Subspace dimensions differ: {left.shape} vs {right.shape}")
    return subspace_metrics(left, right)


def subspace_metrics(reference: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return target-space containment in an orthonormal reference space."""

    if reference.ndim != 2 or target.ndim != 2 or reference.shape[0] != target.shape[0]:
        raise ValueError("Subspace bases must be 2D with the same ambient dimension")
    if not reference.shape[1] or not target.shape[1]:
        raise ValueError("Subspace bases must be non-empty")
    cosines = torch.linalg.svdvals(reference.mT @ target).clamp(0, 1)
    captured = float(cosines.square().sum().item()) / target.shape[1]
    angles = torch.rad2deg(torch.acos(cosines))
    return {
        "overlap": captured,
        "mean_angle_deg": float(angles.mean().item()),
        "max_angle_deg": float(angles.max().item()),
    }


def energy_capture(reference: torch.Tensor, delta: DeltaSVD) -> float:
    squared = delta.singular.square()
    total = float(squared.sum().item())
    if total <= 0:
        return float("nan")
    component_capture = (reference.mT @ delta.right).square().sum(dim=0).clamp(0, 1)
    return float((squared * component_capture).sum().item()) / total


def a_orthogonality_error(a: torch.Tensor) -> float:
    identity = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    return float(torch.linalg.matrix_norm(a @ a.mT - identity).item()) / math.sqrt(a.shape[0])


def delta_path_decomposition(
    a: torch.Tensor, b: torch.Tensor, a_initial: torch.Tensor, scale: float
) -> dict[str, float]:
    """Decompose B A_t exactly as B A_0 + B (A_t - A_0), without densifying."""

    gram_b = b.mT @ b

    def squared_norm(right: torch.Tensor) -> torch.Tensor:
        return torch.trace(gram_b @ right @ right.mT).clamp_min(0)

    a_drift = a - a_initial
    full_sq = squared_norm(a)
    initial_path_sq = squared_norm(a_initial)
    drift_path_sq = squared_norm(a_drift)
    inner = torch.trace(gram_b @ a_initial @ a.mT)
    full = math.sqrt(float(full_sq.item())) * abs(scale)
    initial_path = math.sqrt(float(initial_path_sq.item())) * abs(scale)
    drift_path = math.sqrt(float(drift_path_sq.item())) * abs(scale)
    denominator = math.sqrt(float((full_sq * initial_path_sq).item()))
    return {
        "delta_from_initial_a_fro": initial_path,
        "delta_from_a_drift_fro": drift_path,
        "delta_from_a_drift_relative_fro": drift_path / full if full > 0 else float("nan"),
        "delta_cosine_initial_a_path": float(inner.item()) / denominator if denominator > 0 else float("nan"),
        "a_relative_fro_change": float(torch.linalg.matrix_norm(a_drift).item())
        / max(float(torch.linalg.matrix_norm(a_initial).item()), 1e-300),
        "delta_inner_initial_a_path": float(inner.item()) * scale * scale,
    }


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


def aggregate(rows: list[dict[str, Any]], top_ks: list[int]) -> list[dict[str, Any]]:
    output = []
    for step in sorted({int(row["step"]) for row in rows}):
        group = [row for row in rows if int(row["step"]) == step]
        weights = [float(row["delta_fro"]) ** 2 for row in group]
        item: dict[str, Any] = {
            "step": step,
            "modules": len(group),
            "rank_mean": mean([float(row["rank"]) for row in group]),
            "rank_min": min(int(row["rank"]) for row in group),
            "rank_max": max(int(row["rank"]) for row in group),
            "delta_global_fro": math.sqrt(sum(weights)),
            "a_in_init32_overlap_mean": mean([float(row["a_in_init32_overlap"]) for row in group]),
            "a_in_init32_overlap_energy_weighted": weighted_mean(
                [float(row["a_in_init32_overlap"]) for row in group], weights
            ),
            "delta_in_init32_overlap_mean": mean([float(row["delta_in_init32_overlap"]) for row in group]),
            "delta_in_init32_overlap_energy_weighted": weighted_mean(
                [float(row["delta_in_init32_overlap"]) for row in group], weights
            ),
            "delta_energy_in_init32": weighted_mean([float(row["delta_energy_in_init32"]) for row in group], weights),
            "a_orthogonality_error_mean": mean([float(row["a_orthogonality_error"]) for row in group]),
            "a_relative_fro_change_mean": mean([float(row["a_relative_fro_change"]) for row in group]),
            "delta_from_a_drift_global_relative_fro": math.sqrt(
                sum(float(row["delta_from_a_drift_fro"]) ** 2 for row in group) / max(sum(weights), 1e-300)
            ),
            "delta_from_initial_a_global_relative_fro": math.sqrt(
                sum(float(row["delta_from_initial_a_fro"]) ** 2 for row in group) / max(sum(weights), 1e-300)
            ),
            "delta_cosine_initial_a_path_global": sum(float(row["delta_inner_initial_a_path"]) for row in group)
            / max(
                math.sqrt(sum(weights) * sum(float(row["delta_from_initial_a_fro"]) ** 2 for row in group)),
                1e-300,
            ),
            "anchor_atom_effective_count_weighted": weighted_mean(
                [float(row["anchor_atom_effective_count"]) for row in group], weights
            ),
            "anchor_atom_participation_count_weighted": weighted_mean(
                [float(row["anchor_atom_participation_count"]) for row in group], weights
            ),
            "anchor_atom_rank95_mean": weighted_mean([float(row["anchor_atom_rank95"]) for row in group], weights),
            "anchor_atom_prefix_half_energy_weighted": weighted_mean(
                [float(row["anchor_atom_prefix_half_energy"]) for row in group], weights
            ),
            "anchor_atom_top_half_energy_weighted": weighted_mean(
                [float(row["anchor_atom_top_half_energy"]) for row in group], weights
            ),
            "direction_effective_atoms_weighted": weighted_mean(
                [float(row["direction_effective_atoms"]) for row in group], weights
            ),
            "direction_participation_atoms_weighted": weighted_mean(
                [float(row["direction_participation_atoms"]) for row in group], weights
            ),
            "direction_largest_atom_share_weighted": weighted_mean(
                [float(row["direction_largest_atom_share"]) for row in group], weights
            ),
            "a_previous_full_overlap_weighted": weighted_mean(
                [float(row["a_previous_full_overlap"]) for row in group], weights
            ),
            "delta_previous_full_right_overlap_weighted": weighted_mean(
                [float(row["delta_previous_full_right_overlap"]) for row in group], weights
            ),
            "delta_previous_half_right_overlap_weighted": weighted_mean(
                [float(row["delta_previous_half_right_overlap"]) for row in group], weights
            ),
            "delta_previous_full_left_overlap_weighted": weighted_mean(
                [float(row["delta_previous_full_left_overlap"]) for row in group], weights
            ),
            "delta_previous_half_left_overlap_weighted": weighted_mean(
                [float(row["delta_previous_half_left_overlap"]) for row in group], weights
            ),
        }
        for k in top_ks:
            item[f"k{k}_eligible_modules"] = sum(int(row["rank"]) >= k for row in group)
            for field in (
                "delta_energy_in_init_topk",
                "delta_topk_in_init32_overlap",
                "delta_topk_vs_init_topk_overlap",
                "delta_topk_vs_init_topk_mean_angle_deg",
                "delta_topk_vs_init_topk_max_angle_deg",
            ):
                key = f"k{k}_{field}"
                item[f"{key}_mean"] = mean([float(row[key]) for row in group])
                item[f"{key}_energy_weighted"] = weighted_mean([float(row[key]) for row in group], weights)
        output.append(item)
    return output


def grouped_final(rows: list[dict[str, Any]], field: str, top_ks: list[int]) -> dict[str, Any]:
    final_step = max(int(row["step"]) for row in rows)
    final_rows = [row for row in rows if int(row["step"]) == final_step]
    result = {}
    for value in sorted({str(row[field]) for row in final_rows}):
        group = [row for row in final_rows if str(row[field]) == value]
        weights = [float(row["delta_fro"]) ** 2 for row in group]
        item = {
            "modules": len(group),
            "rank_mean": mean([float(row["rank"]) for row in group]),
            "energy_share": sum(weights) / sum(float(row["delta_fro"]) ** 2 for row in final_rows),
            "delta_energy_in_init32": weighted_mean([float(row["delta_energy_in_init32"]) for row in group], weights),
            "delta_in_init32_overlap": weighted_mean([float(row["delta_in_init32_overlap"]) for row in group], weights),
            "anchor_atom_effective_count": weighted_mean(
                [float(row["anchor_atom_effective_count"]) for row in group], weights
            ),
            "anchor_atom_rank95": weighted_mean([float(row["anchor_atom_rank95"]) for row in group], weights),
            "anchor_atom_prefix_half_energy": weighted_mean(
                [float(row["anchor_atom_prefix_half_energy"]) for row in group], weights
            ),
            "anchor_atom_top_half_energy": weighted_mean(
                [float(row["anchor_atom_top_half_energy"]) for row in group], weights
            ),
            "direction_effective_atoms": weighted_mean(
                [float(row["direction_effective_atoms"]) for row in group], weights
            ),
        }
        for k in top_ks:
            energy_key = f"k{k}_delta_energy_in_init_topk"
            item[energy_key] = weighted_mean([float(row[energy_key]) for row in group], weights)
            key = f"k{k}_delta_topk_vs_init_topk_overlap"
            item[f"k{k}_topk_exact_overlap"] = weighted_mean([float(row[key]) for row in group], weights)
        result[value] = item
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    top_ks = summary["top_k"]
    final = summary["trajectory"][-1]
    if summary["anchor_mode"] == "rlpo-base":
        anchor_order = "base-weight singular-value"
    elif summary["anchor_mode"] == "gradient-subspace":
        anchor_order = "gradient-probe sketch-energy"
    else:
        anchor_order = "anchor-A singular-value"
    lines = [
        "# Adapter Span And Recomposition Diagnostic",
        "",
        "## Scope",
        "",
        f"- Checkpoints: `{summary['steps']}`",
        f"- Modules: `{summary['modules']}`",
        f"- Configured maximum rank/base scaling: `{summary['configured_rank']}` / `{summary['scale']}`",
        f"- Actual per-module rank range/mean: `{summary['rank_min']}`-`{summary['rank_max']}` / "
        f"`{summary['rank_mean']:.4f}`",
        f"- Span anchor: {summary['anchor_description']}",
        "",
        "## Trajectory",
        "",
        "| Step | Delta Fro | Delta span in anchor | Delta energy in anchor | "
        "A span in anchor | B(A-A_anchor) / BA | A orth. error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["trajectory"]:
        lines.append(
            f"| {row['step']} | {row['delta_global_fro']:.4f} | "
            f"{row['delta_in_init32_overlap_energy_weighted']:.2%} | "
            f"{row['delta_energy_in_init32']:.2%} | "
            f"{row['a_in_init32_overlap_energy_weighted']:.2%} | "
            f"{row['delta_from_a_drift_global_relative_fro']:.3%} | "
            f"{row['a_orthogonality_error_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Temporal Span Overlap",
            "",
            "| Step | A full vs previous | DeltaW right full vs previous | "
            "DeltaW right half vs previous | DeltaW left full vs previous | "
            "DeltaW left half vs previous |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["trajectory"]:
        if not math.isfinite(row["a_previous_full_overlap_weighted"]):
            continue
        lines.append(
            f"| {row['step']} | {row['a_previous_full_overlap_weighted']:.2%} | "
            f"{row['delta_previous_full_right_overlap_weighted']:.2%} | "
            f"{row['delta_previous_half_right_overlap_weighted']:.2%} | "
            f"{row['delta_previous_full_left_overlap_weighted']:.2%} | "
            f"{row['delta_previous_half_left_overlap_weighted']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Recomposition Inside The Initial Span",
            "",
            "| Step | Effective active atoms | Atoms for 95% energy | Prefix-half energy | "
            "Best-half energy | Atoms mixed per DeltaW direction | Largest-atom share |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["trajectory"]:
        lines.append(
            f"| {row['step']} | {row['anchor_atom_effective_count_weighted']:.2f} | "
            f"{row['anchor_atom_rank95_mean']:.2f} | "
            f"{row['anchor_atom_prefix_half_energy_weighted']:.2%} | "
            f"{row['anchor_atom_top_half_energy_weighted']:.2%} | "
            f"{row['direction_effective_atoms_weighted']:.2f} | "
            f"{row['direction_largest_atom_share_weighted']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"The anchor atoms use {anchor_order} order. `Prefix-half` keeps the "
            "first half in that order; `best-half` selects the half with the largest learned "
            "left-factor energy in hindsight. Their gap measures reordering inside an almost fixed span.",
            "",
            "## Final Main-Direction Alignment",
            "",
            "| k | Total Delta energy in anchor prefix-k | Delta top-k contained in anchor | "
            "Delta top-k vs anchor prefix-k | Mean angle | Max angle |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in top_ks:
        if final[f"k{k}_eligible_modules"] == 0:
            continue
        lines.append(
            f"| {k} ({final[f'k{k}_eligible_modules']} modules) | "
            f"{final[f'k{k}_delta_energy_in_init_topk_energy_weighted']:.2%} | "
            f"{final[f'k{k}_delta_topk_in_init32_overlap_energy_weighted']:.2%} | "
            f"{final[f'k{k}_delta_topk_vs_init_topk_overlap_energy_weighted']:.2%} | "
            f"{final[f'k{k}_delta_topk_vs_init_topk_mean_angle_deg_energy_weighted']:.2f} deg | "
            f"{final[f'k{k}_delta_topk_vs_init_topk_max_angle_deg_energy_weighted']:.2f} deg |"
        )
    lines.extend(
        [
            "",
            "Rows at k only aggregate modules whose actual rank is at least k. `Contained in anchor` asks "
            "whether training left each module's reference span. `Vs anchor prefix-k` is stricter and asks "
            "whether the strongest current directions retain the anchor ordering. Overlap is mean squared "
            "principal cosine; 1 is identical.",
            "",
            "## Interpretation",
            "",
            f"At step {final['step']}, `{final['delta_energy_in_init32']:.4%}` of DeltaW energy remains inside "
            "each module's anchor rank-r_m right subspace. The full anchor "
            f"span therefore barely rotates. Directly, `B_t(A_t-A_anchor)` is only "
            f"`{final['delta_from_a_drift_global_relative_fro']:.3%}` of `B_tA_t` in global Frobenius norm. "
            f"The recomposition table separately tests whether the learned update preserves the {anchor_order} "
            "ordering inside that span. A fixed full span can coexist with strong atom mixing and reordering, so "
            "span containment alone is not evidence that a smaller base top-r prefix is sufficient.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def base_shard_map(base_model: Path) -> dict[str, Path]:
    index_files = list(base_model.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        return {key: base_model / shard for key, shard in index["weight_map"].items()}
    files = sorted(base_model.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"Expected one safetensors file or an index under {base_model}")
    with safe_open(files[0], framework="pt", device="cpu") as handle:
        return {key: files[0] for key in handle.keys()}


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
    base_scale = float(config["lora_alpha"]) / configured_rank
    if max(top_ks) > configured_rank:
        raise ValueError(f"top-k cannot exceed configured rank {configured_rank}: {top_ks}")
    filters = [value.strip() for value in args.module_filter.split(",") if value.strip()]
    shard_map = base_shard_map(args.base_model.resolve()) if args.anchor_mode == "rlpo-base" else {}
    if args.anchor_mode == "gradient-subspace" and args.anchor_subspaces is None:
        raise ValueError("--anchor-subspaces is required for --anchor-mode gradient-subspace")

    rows: list[dict[str, Any]] = []
    with ExitStack() as stack:
        adapter_handles = {
            step: stack.enter_context(safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu"))
            for step, adapter in adapters
        }
        first_handle = adapter_handles[steps[0]]
        keys = sorted(key for key in first_handle.keys() if key.endswith(".lora_A.weight"))
        if filters:
            keys = [key for key in keys if any(value in key for value in filters)]
        if args.max_modules > 0:
            keys = keys[: args.max_modules]
        if not keys:
            raise ValueError("No adapter modules selected")
        layers = [layer_index(key) for key in keys]
        layer_count = max((layer for layer in layers if layer is not None), default=-1) + 1
        base_handles = (
            {
                shard: stack.enter_context(safe_open(shard, framework="pt", device="cpu"))
                for shard in sorted(set(shard_map.values()))
            }
            if args.anchor_mode == "rlpo-base"
            else {}
        )
        gradient_handle = (
            stack.enter_context(
                safe_open(args.anchor_subspaces.resolve(), framework="pt", device="cpu")
            )
            if args.anchor_mode == "gradient-subspace"
            else None
        )

        for module_number, a_key in enumerate(keys, 1):
            weight_key = base_weight_key(a_key)
            rank = int(first_handle.get_tensor(a_key).shape[0])
            configured_module_rank = int(module_pattern_value(config, "rank_pattern", a_key, configured_rank))
            if configured_module_rank != rank:
                raise ValueError(f"Configured/tensor rank mismatch for {a_key}: {configured_module_rank} vs {rank}")
            alpha = module_pattern_value(config, "alpha_pattern", a_key, config["lora_alpha"])
            scale = alpha / rank
            if not math.isclose(scale, base_scale, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"Per-module scaling changed for {a_key}: alpha={alpha}, rank={rank}, "
                    f"scale={scale}, expected={base_scale}"
                )
            if args.anchor_mode == "rlpo-base":
                if weight_key not in shard_map:
                    raise KeyError(f"Base weight not found for {a_key}: {weight_key}")
                base_weight = (
                    base_handles[shard_map[weight_key]].get_tensor(weight_key).to(device=device, dtype=torch.float32)
                )
                _u, _s, vh = torch.linalg.svd(base_weight, full_matrices=False)
                del base_weight, _u, _s
                init_basis = vh[:rank].mT.contiguous()
                a_initial = init_basis.mT.contiguous()
                a_earliest = first_handle.get_tensor(a_key).to(device=device, dtype=torch.float32)
                signs = torch.sign((a_earliest * a_initial).sum(dim=1))
                signs[signs == 0] = 1
                a_initial = a_initial * signs[:, None]
                init_basis = init_basis * signs[None, :]
                anchor_transform = torch.eye(rank, dtype=torch.float32, device=device)
                del vh, a_earliest, signs
            elif args.anchor_mode == "gradient-subspace":
                gradient_key = adapter_module_key(a_key)
                if gradient_handle is None or gradient_key not in gradient_handle.keys():
                    raise KeyError(f"Gradient subspace not found for {a_key}: {gradient_key}")
                a_initial = gradient_handle.get_tensor(gradient_key).to(
                    device=device, dtype=torch.float32
                )
                expected_shape = tuple(first_handle.get_tensor(a_key).shape)
                if tuple(a_initial.shape) != expected_shape:
                    raise ValueError(
                        f"Gradient/checkpoint A shape mismatch for {a_key}: "
                        f"{tuple(a_initial.shape)} vs {expected_shape}"
                    )
                orthogonality_error = a_orthogonality_error(a_initial)
                if orthogonality_error > 1e-3:
                    raise ValueError(
                        f"Gradient anchor is not row-orthonormal for {a_key}: "
                        f"error={orthogonality_error}"
                    )
                init_basis = a_initial.mT.contiguous()
                anchor_transform = torch.eye(rank, dtype=torch.float32, device=device)
            else:
                a_initial = first_handle.get_tensor(a_key).to(device=device, dtype=torch.float32)
                anchor_u, _anchor_s, anchor_vh = torch.linalg.svd(a_initial, full_matrices=False)
                init_basis = anchor_vh.mT.contiguous()
                anchor_transform = (anchor_u * _anchor_s[None, :]).contiguous()
                del anchor_u, _anchor_s, anchor_vh
            layer = layer_index(a_key)
            common = {
                "adapter_key": a_key.removesuffix(".lora_A.weight"),
                "base_weight_key": weight_key,
                "layer": layer,
                "layer_band": layer_band(layer, layer_count),
                "module_type": module_type(a_key),
                "rank": rank,
                "alpha": alpha,
                "scale": scale,
            }
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            previous_a_basis = None
            previous_delta = None
            for step in steps:
                a = adapter_handles[step].get_tensor(a_key).to(device=device, dtype=torch.float32)
                b = adapter_handles[step].get_tensor(b_key).to(device=device, dtype=torch.float32)
                delta = compact_delta_svd(a, b, scale)
                a_basis = torch.linalg.qr(a.mT, mode="reduced").Q
                weights = delta.singular.square()
                atom_metrics = distribution_metrics((b @ anchor_transform).square().sum(dim=0))
                mixing_metrics = direction_mixing_metrics(init_basis, delta)
                row: dict[str, Any] = {
                    **common,
                    "step": step,
                    "delta_fro": math.sqrt(float(weights.sum().item())),
                    "delta_stable_rank": (float(weights.sum().item() / weights[0].item()) if weights[0] > 0 else 0.0),
                    "a_orthogonality_error": a_orthogonality_error(a),
                    "delta_energy_in_init32": energy_capture(init_basis, delta),
                    "anchor_atom_effective_count": atom_metrics["effective_atoms"],
                    "anchor_atom_participation_count": atom_metrics["participation_atoms"],
                    "anchor_atom_rank95": atom_metrics["rank95_atoms"],
                    "anchor_atom_top_quarter_energy": atom_metrics["top_quarter_energy"],
                    "anchor_atom_top_half_energy": atom_metrics["top_half_energy"],
                    "anchor_atom_prefix_quarter_energy": atom_metrics["prefix_quarter_energy"],
                    "anchor_atom_prefix_half_energy": atom_metrics["prefix_half_energy"],
                    **mixing_metrics,
                    "a_previous_full_overlap": (
                        same_dimensional_subspace_metrics(previous_a_basis, a_basis)["overlap"]
                        if previous_a_basis is not None
                        else float("nan")
                    ),
                    "delta_previous_full_right_overlap": (
                        same_dimensional_subspace_metrics(previous_delta.right, delta.right)["overlap"]
                        if previous_delta is not None
                        else float("nan")
                    ),
                    "delta_previous_half_right_overlap": (
                        subspace_metrics(
                            previous_delta.right[:, : max(1, math.ceil(rank / 2))],
                            delta.right[:, : max(1, math.ceil(rank / 2))],
                        )["overlap"]
                        if previous_delta is not None
                        else float("nan")
                    ),
                    "delta_previous_full_left_overlap": (
                        same_dimensional_subspace_metrics(previous_delta.left, delta.left)["overlap"]
                        if previous_delta is not None
                        else float("nan")
                    ),
                    "delta_previous_half_left_overlap": (
                        subspace_metrics(
                            previous_delta.left[:, : max(1, math.ceil(rank / 2))],
                            delta.left[:, : max(1, math.ceil(rank / 2))],
                        )["overlap"]
                        if previous_delta is not None
                        else float("nan")
                    ),
                    **delta_path_decomposition(a, b, a_initial, scale),
                }
                for prefix, metrics in (
                    ("a_in_init32", subspace_metrics(init_basis, a_basis)),
                    ("delta_in_init32", subspace_metrics(init_basis, delta.right)),
                ):
                    row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
                for k in top_ks:
                    if k <= rank:
                        contained = subspace_metrics(init_basis, delta.right[:, :k])
                        exact = subspace_metrics(init_basis[:, :k], delta.right[:, :k])
                        row[f"k{k}_delta_energy_in_init_topk"] = energy_capture(init_basis[:, :k], delta)
                        row[f"k{k}_delta_topk_in_init32_overlap"] = contained["overlap"]
                        row[f"k{k}_delta_topk_vs_init_topk_overlap"] = exact["overlap"]
                        row[f"k{k}_delta_topk_vs_init_topk_mean_angle_deg"] = exact["mean_angle_deg"]
                        row[f"k{k}_delta_topk_vs_init_topk_max_angle_deg"] = exact["max_angle_deg"]
                    else:
                        for field in (
                            "delta_energy_in_init_topk",
                            "delta_topk_in_init32_overlap",
                            "delta_topk_vs_init_topk_overlap",
                            "delta_topk_vs_init_topk_mean_angle_deg",
                            "delta_topk_vs_init_topk_max_angle_deg",
                        ):
                            row[f"k{k}_{field}"] = float("nan")
                rows.append(row)
                previous_a_basis = a_basis
                previous_delta = delta
                del a, b
            del init_basis, a_initial, anchor_transform
            print(f"[{module_number:03d}/{len(keys):03d}] {common['adapter_key']}", flush=True)

    trajectory = aggregate(rows, top_ks)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "base_model": str(args.base_model.resolve()),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "anchor_mode": args.anchor_mode,
        "anchor_description": (
            "exact top-r_m right singular vectors of each frozen base weight (true RLPO step-0 A_0)"
            if args.anchor_mode == "rlpo-base"
            else (
                "exact ordered gradient-probe subspace used as step-0 A_0"
                if args.anchor_mode == "gradient-subspace"
                else f"orthonormalized row span of the earliest available adapter A at step {steps[0]} (not step 0)"
            )
        ),
        "steps": steps,
        "configured_rank": configured_rank,
        "scale": base_scale,
        "rank_mean": mean([float(row["rank"]) for row in rows if int(row["step"]) == steps[0]]),
        "rank_min": min(int(row["rank"]) for row in rows),
        "rank_max": max(int(row["rank"]) for row in rows),
        "top_k": top_ks,
        "modules": len(keys),
        "definitions": {
            "init_subspace": (
                "top-r_m right singular vectors of each frozen base weight, matching heterogeneous RLPO initialization"
                if args.anchor_mode == "rlpo-base"
                else (
                    "ordered randomized-gradient-sketch basis that initialized A at step 0"
                    if args.anchor_mode == "gradient-subspace"
                    else f"row span of A at earliest saved step {steps[0]}; no claim about initialization"
                )
            ),
            "overlap": "target-space mean squared principal cosine in the reference space; 1 means fully contained",
            "delta_energy_in_init32": (
                "fraction of final DeltaW Frobenius energy whose right singular directions "
                "project into the initial rank-r_m span"
            ),
            "a_orthogonality_error": "||A A^T - I||_F / sqrt(r)",
            "delta_path_decomposition": (
                "DeltaW_t = scale * B_t A_t = scale * B_t A_anchor + scale * B_t (A_t - A_anchor)"
            ),
            "anchor_atom_energy": (
                "column energy after expressing A_anchor in an orthonormal row basis; "
                "additive Frobenius energy for B_t A_anchor"
            ),
            "direction_mixing": (
                "effective number of initial A_0 atoms contributing to learned DeltaW "
                "right singular directions, singular-energy weighted"
            ),
        },
        "trajectory": trajectory,
        "final_by_module_type": grouped_final(rows, "module_type", top_ks),
        "final_by_layer_band": grouped_final(rows, "layer_band", top_ks),
    }
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_module_trajectory.csv", rows)
    write_csv(out_dir / "trajectory.csv", trajectory)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", result)
    print(f"Wrote RLPO init-subspace diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
