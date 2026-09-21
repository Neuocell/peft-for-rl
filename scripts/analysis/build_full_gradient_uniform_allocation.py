#!/usr/bin/env python3
"""Build an equal-rank LoRA initialization from a full-gradient probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.analysis.build_spar_lora_allocations import _write_allocation
from verl.utils.full_gradient_rl_probe import validate_full_gradient_probe_artifact


def build_uniform_allocation(
    artifact_dir: Path,
    output_dir: Path,
    *,
    candidate_method: str,
    uniform_rank: int,
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

    selected = {
        name: torch.argsort(module_scores["U"], descending=True, stable=True)[
            :uniform_rank
        ].tolist()
        for name, module_scores in scores.items()
    }
    budget = sum(
        (out_features + in_features) * uniform_rank
        for out_features, in_features in shapes.values()
    )
    allocation = _write_allocation(
        output_dir,
        method=f"full_gradient_signed_grpo_{candidate_method}_uniform_r{uniform_rank}",
        candidates=candidates,
        scores=scores,
        selected=selected,
        shapes=shapes,
        uniform_budget=budget,
        scaling=scaling,
    )
    allocation["probe_validation"] = validation
    allocation["candidate_method"] = candidate_method
    (output_dir / "allocation_summary.json").write_text(
        json.dumps(allocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-method", choices=("mean", "covariance", "hybrid"), default="mean"
    )
    parser.add_argument("--uniform-rank", type=int, default=8)
    args = parser.parse_args()
    result = build_uniform_allocation(
        args.artifact_dir.resolve(),
        args.output_dir.resolve(),
        candidate_method=args.candidate_method,
        uniform_rank=args.uniform_rank,
    )
    print(
        json.dumps(
            {
                "candidate_method": result["candidate_method"],
                "trainable_parameters": result["trainable_parameters"],
                "rank_mean": result["structure"]["active_rank_mean"],
                "subspace_path": result["subspace_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
