#!/usr/bin/env python3
"""Truncate a legacy randomized-B gradient-probe artifact to equal rank."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _module_family(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def _layer_segment(index: int, maximum: int) -> str:
    third = (maximum + 1) / 3
    if index < third:
        return "low"
    if index < 2 * third:
        return "middle"
    return "top"


def build_gradient_probe_uniform_allocation(
    artifact_dir: Path,
    output_dir: Path,
    *,
    uniform_rank: int = 8,
) -> dict:
    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("probe_method") != "energy":
        raise ValueError("This baseline builder requires probe_method=energy")
    scaling = float(summary["constant_scaling"])
    if not math.isclose(scaling, 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Expected constant scaling=2, got {scaling}")
    if uniform_rank <= 0:
        raise ValueError("uniform_rank must be positive")

    source_path = artifact_dir / "subspaces.safetensors"
    subspaces: dict[str, torch.Tensor] = {}
    ranks: dict[str, int] = {}
    alphas: dict[str, int] = {}
    module_records: dict[str, dict[str, float | int]] = {}
    trainable_parameters = 0
    maximum_orthogonality_error = 0.0
    with safe_open(source_path, framework="pt", device="cpu") as source:
        source_names = set(source.keys())
        expected_names = set(summary["modules"])
        if source_names != expected_names:
            missing = sorted(expected_names - source_names)
            extra = sorted(source_names - expected_names)
            raise ValueError(f"Artifact module mismatch: missing={missing[:3]}, extra={extra[:3]}")
        for name in sorted(expected_names):
            basis = source.get_tensor(name).float()
            if basis.ndim != 2 or basis.shape[0] < uniform_rank:
                raise ValueError(
                    f"Module {name} only has shape {tuple(basis.shape)}; cannot select rank {uniform_rank}"
                )
            selected = basis[:uniform_rank].contiguous()
            if not bool(torch.isfinite(selected).all()):
                raise ValueError(f"Non-finite subspace for {name}")
            error = float(
                (selected @ selected.T - torch.eye(uniform_rank)).abs().max().item()
            )
            maximum_orthogonality_error = max(maximum_orthogonality_error, error)
            if error > 3e-4:
                raise ValueError(f"Non-orthogonal subspace for {name}: {error}")
            cost = summary["modules"][name].get("parameter_cost_per_rank")
            if cost is None:
                raise ValueError(
                    "Probe artifact does not contain parameter_cost_per_rank; rebuild it with the current exporter"
                )
            cost = int(cost)
            subspaces[name] = selected
            ranks[name] = uniform_rank
            alpha = uniform_rank * scaling
            if not float(alpha).is_integer():
                raise ValueError(f"Non-integral alpha for {name}: {alpha}")
            alphas[name] = int(alpha)
            trainable_parameters += uniform_rank * cost
            module_records[name] = {
                "source_rank": int(basis.shape[0]),
                "rank": uniform_rank,
                "atom_parameter_cost": cost,
                "trainable_parameters": uniform_rank * cost,
                "orthogonality_error": error,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    subspace_path = (output_dir / "subspaces.safetensors").resolve()
    save_file(subspaces, subspace_path)
    rank_map = {
        "schema_version": 4,
        "method": f"legacy_random_b_gradient_probe_energy_uniform_r{uniform_rank}",
        "rank_pattern": ranks,
        "alpha_pattern": alphas,
        "constant_scaling": scaling,
        "subspace_path": str(subspace_path),
    }
    family_ranks: dict[str, list[int]] = defaultdict(list)
    segment_ranks: dict[str, list[int]] = defaultdict(list)
    maximum_layer = max(_layer_index(name) for name in ranks)
    for name, rank in ranks.items():
        family_ranks[_module_family(name)].append(rank)
        segment_ranks[_layer_segment(_layer_index(name), maximum_layer)].append(rank)
    structure = {
        "active_rank_mean": sum(ranks.values()) / len(ranks),
        "active_rank_min": min(ranks.values()),
        "active_rank_max": max(ranks.values()),
        "rank_by_family": {
            key: sum(values) / len(values)
            for key, values in sorted(family_ranks.items())
        },
        "rank_by_layer_segment": {
            key: sum(values) / len(values)
            for key, values in sorted(segment_ranks.items())
        },
        # Legacy energy-probe artifacts do not retain per-atom F/U scores, so
        # capture cannot be reconstructed after truncating the saved basis.
        "calibration_energy_capture": 0.0,
        "u_score_capture": 0.0,
        "capture_metrics_available": False,
    }
    allocation = {
        **rank_map,
        "source_artifact": str(artifact_dir.resolve()),
        "source_probe_method": summary["probe_method"],
        "trainable_parameters": trainable_parameters,
        "active_rank_mean": float(uniform_rank),
        "active_rank_min": uniform_rank,
        "active_rank_max": uniform_rank,
        "orthogonality_error_max": maximum_orthogonality_error,
        "structure": structure,
        "modules": module_records,
    }
    (output_dir / "rank_map.json").write_text(
        json.dumps(rank_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "allocation_summary.json").write_text(
        json.dumps(allocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uniform-rank", type=int, default=8)
    args = parser.parse_args()
    result = build_gradient_probe_uniform_allocation(
        args.artifact_dir.resolve(),
        args.output_dir.resolve(),
        uniform_rank=args.uniform_rank,
    )
    print(
        json.dumps(
            {
                "trainable_parameters": result["trainable_parameters"],
                "rank": result["active_rank_mean"],
                "orthogonality_error_max": result["orthogonality_error_max"],
                "subspace_path": result["subspace_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
