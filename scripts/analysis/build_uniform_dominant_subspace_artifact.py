#!/usr/bin/env python3
"""Rebuild a uniform-rank control from an exported dominant-atom probe.

The dominant probe persists the clipped per-window sketches before rank
allocation.  Reconstructing the consensus basis from those sketches lets us
change only the rank allocation without collecting another set of rollouts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from verl.utils.peft_gradient_subspace import _dominant_consensus_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _window_factors(raw: torch.Tensor, *, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if raw.ndim != 2 or raw.shape[1] % window_size:
        raise ValueError(f"Invalid persisted window sketch shape: {tuple(raw.shape)}")
    probe_width = raw.shape[1] // window_size
    observations = raw.float().reshape(raw.shape[0], window_size, probe_width).permute(1, 0, 2).contiguous()
    mean = observations.mean(dim=0)
    signal = mean * math.sqrt(window_size)
    residuals = observations - mean.unsqueeze(0)
    noise = residuals.permute(1, 0, 2).reshape(mean.shape[0], -1) / math.sqrt(window_size - 1)
    return signal.contiguous(), noise.contiguous()


def build_uniform_artifact(probe_dir: Path, output_dir: Path, rank: int) -> dict[str, Any]:
    if rank <= 0:
        raise ValueError("--rank must be positive")

    source_summary = _load_json(probe_dir / "summary.json")
    source_rank_map = _load_json(probe_dir / "rank_map.json")
    dominant = source_summary.get("dominant_atoms", {})
    window_size = int(dominant["window_size"])
    discovery_windows = int(dominant["num_windows"])
    local_atoms = int(dominant["local_atoms"])
    gap_cap = float(dominant["gap_cap"])
    rank_bins = [int(value) for value in source_summary["rank_bins"]]
    max_rank = max(rank_bins)
    scaling = float(source_rank_map["constant_scaling"])
    if rank > max_rank:
        raise ValueError(f"Requested rank {rank} exceeds the persisted probe capacity {max_rank}")
    alpha = rank * scaling
    if not math.isclose(alpha, round(alpha), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"rank={rank} and scaling={scaling} do not produce integral alpha")

    module_names = sorted(source_rank_map["rank_pattern"])
    if not module_names:
        raise ValueError("Source rank map has no modules")
    source_subspaces_path = probe_dir / "subspaces.safetensors"
    sketches_path = probe_dir / "window_sketches.safetensors"
    output_tensors: dict[str, torch.Tensor] = {}
    reproduction_overlaps: list[float] = []

    with (
        safe_open(sketches_path, framework="pt", device="cpu") as sketches,
        safe_open(source_subspaces_path, framework="pt", device="cpu") as source_subspaces,
    ):
        available_sketches = set(sketches.keys())
        available_subspaces = set(source_subspaces.keys())
        missing_subspaces = sorted(set(module_names) - available_subspaces)
        if missing_subspaces:
            raise KeyError(f"Missing source subspaces: {missing_subspaces[:5]}")

        for module_name in module_names:
            signals: list[torch.Tensor] = []
            noises: list[torch.Tensor] = []
            for window_index in range(discovery_windows):
                key = f"{module_name}.window_{window_index:02d}"
                if key not in available_sketches:
                    raise KeyError(f"Missing discovery sketch: {key}")
                signal, noise = _window_factors(sketches.get_tensor(key), window_size=window_size)
                signals.append(signal)
                noises.append(noise)

            basis, _, _, _ = _dominant_consensus_basis(
                signals,
                noises,
                window_size=window_size,
                local_atoms=local_atoms,
                gap_cap=gap_cap,
                max_rank=max_rank,
            )
            selected = basis[:, :rank].T.contiguous()
            output_tensors[module_name] = selected

            source = source_subspaces.get_tensor(module_name).float()
            compare_rank = min(source.shape[0], selected.shape[0])
            overlap = (source[:compare_rank] @ selected[:compare_rank].T).square().sum() / compare_rank
            reproduction_overlaps.append(float(overlap.item()))

    if min(reproduction_overlaps) < 0.999:
        raise RuntimeError(
            "Reconstructed bases do not reproduce the exported prefixes: "
            f"minimum overlap={min(reproduction_overlaps):.6f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "subspaces.safetensors"
    temporary_tensor_path = tensor_path.with_suffix(".safetensors.tmp")
    save_file(output_tensors, str(temporary_tensor_path))
    os.replace(temporary_tensor_path, tensor_path)

    rank_pattern = {name: rank for name in module_names}
    alpha_pattern = {name: int(round(alpha)) for name in module_names}
    rank_map = {
        "schema_version": 3,
        "method": "rl_policy_gradient_dominant_atom_subspace_uniform_rank_control",
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "constant_scaling": scaling,
        "subspace_path": str(tensor_path.resolve()),
    }
    parameter_costs = {
        name: int(source_summary["modules"][name]["parameter_cost_per_rank"]) for name in module_names
    }
    used_parameters = sum(parameter_costs[name] * rank for name in module_names)
    summary = {
        **rank_map,
        "source_probe_dir": str(probe_dir.resolve()),
        "source_method": source_summary["method"],
        "allocation": "uniform",
        "module_count": len(module_names),
        "uniform_rank": rank,
        "adapter_parameter_count": used_parameters,
        "reconstruction_prefix_overlap_mean": sum(reproduction_overlaps) / len(reproduction_overlaps),
        "reconstruction_prefix_overlap_min": min(reproduction_overlaps),
        "dominant_atoms": {
            "window_size": window_size,
            "num_windows": discovery_windows,
            "local_atoms": local_atoms,
            "gap_cap": gap_cap,
            "max_reconstructed_rank": max_rank,
        },
    }
    _write_json_atomic(output_dir / "rank_map.json", rank_map)
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_uniform_artifact(args.probe_dir.resolve(), args.output_dir.resolve(), args.rank)
    print(
        "Built uniform dominant-atom artifact: "
        f"modules={summary['module_count']}, rank={summary['uniform_rank']}, "
        f"parameters={summary['adapter_parameter_count']}, "
        f"prefix_overlap_min={summary['reconstruction_prefix_overlap_min']:.6f}"
    )


if __name__ == "__main__":
    main()
