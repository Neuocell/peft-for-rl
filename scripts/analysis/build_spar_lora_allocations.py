#!/usr/bin/env python3
"""Build uniform and cost-aware static SPAR-LoRA allocations from one probe."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from verl.utils.peft_spar_lora import validate_spar_artifact


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


def _allocate_adaptive(
    utilities: dict[str, torch.Tensor],
    costs: dict[str, int],
    *,
    r_min: int,
    r_max: int,
    budget: int,
) -> dict[str, list[int]]:
    selected = {
        name: torch.argsort(score, descending=True, stable=True)[:r_min].tolist()
        for name, score in utilities.items()
    }
    used = sum(costs[name] * r_min for name in selected)
    candidates: list[tuple[float, str, int]] = []
    for name, score in utilities.items():
        already = set(selected[name])
        for atom in range(r_max):
            if atom not in already:
                candidates.append((float(score[atom].item()) / costs[name], name, atom))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, name, atom in candidates:
        if used + costs[name] > budget or len(selected[name]) >= r_max:
            continue
        selected[name].append(atom)
        used += costs[name]
    return selected


def _write_allocation(
    output_dir: Path,
    *,
    method: str,
    candidates: dict[str, torch.Tensor],
    scores: dict[str, dict[str, torch.Tensor]],
    selected: dict[str, list[int]],
    shapes: dict[str, tuple[int, int]],
    uniform_budget: int,
    scaling: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    subspaces = {}
    rank_pattern = {}
    alpha_pattern = {}
    module_records = {}
    total_parameters = 0
    total_f = 0.0
    selected_f = 0.0
    total_u = 0.0
    selected_u = 0.0
    for name in sorted(candidates):
        indices = sorted(selected[name], key=lambda atom: (-float(scores[name]["U"][atom]), atom))
        index_tensor = torch.tensor(indices, dtype=torch.long)
        subspaces[name] = candidates[name].index_select(0, index_tensor).contiguous()
        rank = len(indices)
        rank_pattern[name] = rank
        alpha = rank * scaling
        if not float(alpha).is_integer():
            raise ValueError(f"Non-integral alpha for {name}: rank={rank}, scaling={scaling}")
        alpha_pattern[name] = int(alpha)
        out_features, in_features = shapes[name]
        cost = out_features + in_features
        total_parameters += rank * cost
        f = scores[name]["F"]
        u = scores[name]["U"]
        total_f += float(f.sum().item())
        selected_f += float(f.index_select(0, index_tensor).sum().item())
        total_u += float(u.sum().item())
        selected_u += float(u.index_select(0, index_tensor).sum().item())
        module_records[name] = {
            "rank": rank,
            "atom_indices": indices,
            "shape": [out_features, in_features],
            "atom_parameter_cost": cost,
            "trainable_parameters": rank * cost,
            "calibration_energy_capture": float(f.index_select(0, index_tensor).sum().item() / max(f.sum().item(), 1e-12)),
            "u_score_capture": float(u.index_select(0, index_tensor).sum().item() / max(u.sum().item(), 1e-12)),
        }

    ranks = list(rank_pattern.values())
    max_layer = max(_layer_index(name) for name in rank_pattern)
    family_ranks: dict[str, list[int]] = defaultdict(list)
    segment_ranks: dict[str, list[int]] = defaultdict(list)
    for name, rank in rank_pattern.items():
        family_ranks[_module_family(name)].append(rank)
        segment_ranks[_layer_segment(_layer_index(name), max_layer)].append(rank)
    structure = {
        "active_rank_mean": sum(ranks) / len(ranks),
        "active_rank_min": min(ranks),
        "active_rank_max": max(ranks),
        "rank_by_family": {key: sum(values) / len(values) for key, values in sorted(family_ranks.items())},
        "rank_by_layer_segment": {key: sum(values) / len(values) for key, values in sorted(segment_ranks.items())},
        "calibration_energy_capture": selected_f / max(total_f, 1e-12),
        "u_score_capture": selected_u / max(total_u, 1e-12),
    }
    rank_map = {
        "schema_version": 4,
        "method": method,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "constant_scaling": scaling,
        "subspace_path": str((output_dir / "subspaces.safetensors").resolve()),
    }
    summary = {
        **rank_map,
        "uniform_r8_parameter_budget": uniform_budget,
        "trainable_parameters": total_parameters,
        "budget_residual": uniform_budget - total_parameters,
        "budget_respected": total_parameters <= uniform_budget,
        "structure": structure,
        "modules": module_records,
    }
    save_file(subspaces, output_dir / "subspaces.safetensors")
    (output_dir / "rank_map.json").write_text(json.dumps(rank_map, indent=2, sort_keys=True) + "\n")
    (output_dir / "allocation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_allocations(artifact_dir: Path, output_root: Path, *, r_min: int = 2, uniform_rank: int = 8) -> dict:
    validation = validate_spar_artifact(artifact_dir)
    probe = json.loads((artifact_dir / "probe_summary.json").read_text())
    r_max = int(probe["r_max"])
    scaling = float(probe["constant_scaling"])
    if not math.isclose(scaling, 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"SPAR v0 requires constant scaling=2, got {scaling}")
    if not 0 < r_min <= uniform_rank <= r_max:
        raise ValueError("Expected 0 < r_min <= uniform_rank <= r_max")

    candidates: dict[str, torch.Tensor] = {}
    module_scores: dict[str, dict[str, torch.Tensor]] = {}
    shapes = {name: tuple(item["shape"]) for name, item in probe["modules"].items()}
    with safe_open(artifact_dir / "candidates.safetensors", framework="pt", device="cpu") as candidate_file:
        with safe_open(artifact_dir / "atom_scores.safetensors", framework="pt", device="cpu") as score_file:
            for name in sorted(probe["modules"]):
                candidates[name] = candidate_file.get_tensor(name).float()
                module_scores[name] = {
                    label: score_file.get_tensor(f"{name}.{label}").float()
                    for label in ("F", "S", "R", "P", "U")
                }

    costs = {name: sum(shapes[name]) for name in shapes}
    budget = sum(costs[name] * uniform_rank for name in costs)
    uniform = {
        name: torch.argsort(values["U"], descending=True, stable=True)[:uniform_rank].tolist()
        for name, values in module_scores.items()
    }
    adaptive = _allocate_adaptive(
        {name: values["U"] for name, values in module_scores.items()},
        costs,
        r_min=r_min,
        r_max=r_max,
        budget=budget,
    )
    uniform_summary = _write_allocation(
        output_root / "uniform_r8",
        method="spar_lora_v0_positive_probe_uniform_r8",
        candidates=candidates,
        scores=module_scores,
        selected=uniform,
        shapes=shapes,
        uniform_budget=budget,
        scaling=scaling,
    )
    adaptive_summary = _write_allocation(
        output_root / "adaptive_eqr8",
        method="spar_lora_v0_positive_probe_adaptive_eqr8",
        candidates=candidates,
        scores=module_scores,
        selected=adaptive,
        shapes=shapes,
        uniform_budget=budget,
        scaling=scaling,
    )
    result = {"probe_validation": validation, "uniform": uniform_summary, "adaptive": adaptive_summary}
    (output_root / "allocation_comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r-min", type=int, default=2)
    parser.add_argument("--uniform-rank", type=int, default=8)
    args = parser.parse_args()
    result = build_allocations(args.artifact_dir.resolve(), args.output_root.resolve(), r_min=args.r_min, uniform_rank=args.uniform_rank)
    print(json.dumps({key: value.get("trainable_parameters", value) if isinstance(value, dict) else value for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
