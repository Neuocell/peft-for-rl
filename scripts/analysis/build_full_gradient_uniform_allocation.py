#!/usr/bin/env python3
"""Build equal-rank or cost-aware LoRA initialization from a full-gradient probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.analysis.build_spar_lora_allocations import _allocate_adaptive, _write_allocation
from verl.utils.full_gradient_rl_probe import validate_full_gradient_probe_artifact


def build_uniform_allocation(
    artifact_dir: Path,
    output_dir: Path,
    *,
    candidate_method: str,
    uniform_rank: int,
    selection_utility: str = "gain_lcb",
) -> dict:
    validation = validate_full_gradient_probe_artifact(
        artifact_dir, candidate_method=candidate_method
    )
    summary = json.loads(
        (artifact_dir / "probe_summary.json").read_text(encoding="utf-8")
    )
    r_max = int(summary["r_max"])
    if not 0 < uniform_rank <= r_max:
        raise ValueError(f"uniform_rank must be in [1, {r_max}]")
    scaling = float(summary["constant_scaling"])
    if scaling != 2.0:
        raise ValueError(f"Full-gradient v1 requires constant scaling=2, got {scaling}")

    candidates: dict[str, torch.Tensor] = {}
    scores: dict[str, dict[str, torch.Tensor]] = {}
    shapes = {name: tuple(item["shape"]) for name, item in summary["modules"].items()}
    candidate_path = artifact_dir / f"candidates_{candidate_method}.safetensors"
    score_path = artifact_dir / f"atom_scores_{candidate_method}.safetensors"
    with safe_open(candidate_path, framework="pt", device="cpu") as candidate_file:
        with safe_open(score_path, framework="pt", device="cpu") as score_file:
            for name in sorted(summary["modules"]):
                candidates[name] = candidate_file.get_tensor(name).float()
                scores[name] = {
                    label: score_file.get_tensor(f"{name}.{label}").float()
                    for label in ("F", "S", "R", "P", "U")
                }

    allocation_scores = _selection_scores(scores, selection_utility)
    selected = {
        name: torch.argsort(allocation_scores[name], descending=True, stable=True)[
            :uniform_rank
        ].tolist()
        for name in scores
    }
    budget = sum(
        (out_features + in_features) * uniform_rank
        for out_features, in_features in shapes.values()
    )
    allocation = _write_allocation(
        output_dir,
        method=(
            f"full_gradient_signed_grpo_{candidate_method}_uniform_r{uniform_rank}"
            + (f"_{selection_utility}" if selection_utility != "gain_lcb" else "")
        ),
        candidates=candidates,
        scores=scores,
        selected=selected,
        shapes=shapes,
        uniform_budget=budget,
        scaling=scaling,
    )
    allocation["probe_validation"] = validation
    allocation["candidate_method"] = candidate_method
    allocation["selection_utility"] = selection_utility
    (output_dir / "allocation_summary.json").write_text(
        json.dumps(allocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return allocation


def _selection_scores(
    scores: dict[str, dict[str, torch.Tensor]], utility: str
) -> dict[str, torch.Tensor]:
    if utility == "gain_lcb":
        return {name: module_scores["U"] for name, module_scores in scores.items()}
    if utility == "future_lcb":
        return {name: module_scores["U"] for name, module_scores in scores.items()}
    if utility == "stable_energy":
        return {
            name: module_scores["P"] * module_scores["R"]
            for name, module_scores in scores.items()
        }
    raise ValueError(f"Unknown allocation utility: {utility}")


def build_adaptive_allocation(
    artifact_dir: Path,
    output_dir: Path,
    *,
    candidate_method: str,
    uniform_rank: int,
    r_min: int,
    adaptive_utility: str = "gain_lcb",
) -> dict:
    """Allocate complete atoms globally under the matching uniform-rank budget."""

    validation = validate_full_gradient_probe_artifact(
        artifact_dir, candidate_method=candidate_method
    )
    summary = json.loads(
        (artifact_dir / "probe_summary.json").read_text(encoding="utf-8")
    )
    r_max = int(summary["r_max"])
    if not 0 < r_min <= uniform_rank <= r_max:
        raise ValueError(f"Expected 0 < r_min <= uniform_rank <= {r_max}")
    scaling = float(summary["constant_scaling"])
    if scaling != 2.0:
        raise ValueError(f"Full-gradient v1 requires constant scaling=2, got {scaling}")

    candidates: dict[str, torch.Tensor] = {}
    scores: dict[str, dict[str, torch.Tensor]] = {}
    shapes = {name: tuple(item["shape"]) for name, item in summary["modules"].items()}
    candidate_path = artifact_dir / f"candidates_{candidate_method}.safetensors"
    score_path = artifact_dir / f"atom_scores_{candidate_method}.safetensors"
    with safe_open(candidate_path, framework="pt", device="cpu") as candidate_file:
        with safe_open(score_path, framework="pt", device="cpu") as score_file:
            for name in sorted(summary["modules"]):
                candidates[name] = candidate_file.get_tensor(name).float()
                scores[name] = {
                    label: score_file.get_tensor(f"{name}.{label}").float()
                    for label in ("F", "S", "R", "P", "U")
                }

    costs = {name: sum(shapes[name]) for name in shapes}
    budget = sum(costs[name] * uniform_rank for name in costs)
    allocation_scores = _selection_scores(scores, adaptive_utility)
    selected = _allocate_adaptive(
        allocation_scores,
        costs,
        r_min=r_min,
        r_max=r_max,
        budget=budget,
    )
    allocation = _write_allocation(
        output_dir,
        method=(
            f"full_gradient_signed_grpo_{candidate_method}_adaptive_"
            f"eqr{uniform_rank}_rmin{r_min}_{adaptive_utility}"
        ),
        candidates=candidates,
        scores=scores,
        selected=selected,
        shapes=shapes,
        uniform_budget=budget,
        scaling=scaling,
    )
    allocation["probe_validation"] = validation
    allocation["candidate_method"] = candidate_method
    allocation["allocation_mode"] = "adaptive"
    allocation["adaptive_utility"] = adaptive_utility
    allocation["r_min"] = r_min
    (output_dir / "allocation_summary.json").write_text(
        json.dumps(allocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-method", default="mean")
    parser.add_argument(
        "--allocation-mode", choices=("uniform", "adaptive"), default="uniform"
    )
    parser.add_argument("--uniform-rank", type=int, default=8)
    parser.add_argument("--r-min", type=int, default=2)
    parser.add_argument(
        "--adaptive-utility",
        choices=("gain_lcb", "stable_energy", "future_lcb"),
        default="gain_lcb",
    )
    parser.add_argument(
        "--uniform-utility",
        choices=("gain_lcb", "stable_energy", "future_lcb"),
        default="gain_lcb",
    )
    args = parser.parse_args()
    if args.allocation_mode == "uniform":
        result = build_uniform_allocation(
            args.artifact_dir.resolve(),
            args.output_dir.resolve(),
            candidate_method=args.candidate_method,
            uniform_rank=args.uniform_rank,
            selection_utility=args.uniform_utility,
        )
    else:
        result = build_adaptive_allocation(
            args.artifact_dir.resolve(),
            args.output_dir.resolve(),
            candidate_method=args.candidate_method,
            uniform_rank=args.uniform_rank,
            r_min=args.r_min,
            adaptive_utility=args.adaptive_utility,
        )
    print(
        json.dumps(
            {
                "candidate_method": result["candidate_method"],
                "allocation_mode": args.allocation_mode,
                "adaptive_utility": result.get("adaptive_utility"),
                "selection_utility": result.get("selection_utility"),
                "trainable_parameters": result["trainable_parameters"],
                "rank_mean": result["structure"]["active_rank_mean"],
                "rank_min": result["structure"]["active_rank_min"],
                "rank_max": result["structure"]["active_rank_max"],
                "budget_residual": result["budget_residual"],
                "subspace_path": result["subspace_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
