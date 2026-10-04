#!/usr/bin/env python3
"""Materialize the preregistered Phase-0.5 winner for uniform-r32 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from scripts.analysis.build_full_gradient_uniform_allocation import (
    build_uniform_allocation,
)
from scripts.analysis.evaluate_phase05_hybrid_gate import NONZERO_METHODS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare_phase05_training_artifacts(
    artifact_dir: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve()
    output_root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else artifact_dir / "training_allocations"
    )
    gate_path = artifact_dir / "phase05_gate_decision.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    decision = gate.get("decision")
    if decision not in {"go", "no-go"}:
        raise ValueError(f"Unknown Phase-0.5 gate decision: {decision!r}")
    result: dict[str, Any] = {
        "schema_version": 1,
        "gate_path": str(gate_path),
        "gate_sha256": _sha256(gate_path),
        "gate_decision": decision,
        "calibration_selected_method": gate.get("calibration_selected_method"),
        "uniform_rank": 32,
        "status": "skipped_no_go",
        "allocations": {},
    }

    if decision == "no-go":
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            artifact_dir / "phase05_training_preparation.json", result
        )
        return result

    selected = gate.get("calibration_selected_method")
    if selected not in NONZERO_METHODS:
        raise ValueError(
            "A Phase-0.5 go decision must select one preregistered nonzero hybrid"
        )

    for method in ("H0", selected):
        allocation_dir = output_root / f"{method}_uniform_r32"
        allocation = build_uniform_allocation(
            artifact_dir,
            allocation_dir,
            candidate_method=method,
            uniform_rank=32,
        )
        ranks = {
            int(item["rank"])
            for item in allocation["modules"].values()
        }
        if ranks != {32}:
            raise ValueError(f"{method} allocation is not uniformly rank 32: {ranks}")
        rank_map_path = allocation_dir / "rank_map.json"
        subspace_path = allocation_dir / "subspaces.safetensors"
        allocation_summary_path = allocation_dir / "allocation_summary.json"
        result["allocations"][method] = {
            "directory": str(allocation_dir),
            "module_count": len(allocation["modules"]),
            "trainable_parameters": int(allocation["trainable_parameters"]),
            "rank_map_path": str(rank_map_path),
            "rank_map_sha256": _sha256(rank_map_path),
            "subspace_path": str(subspace_path),
            "subspace_sha256": _sha256(subspace_path),
            "allocation_summary_path": str(allocation_summary_path),
            "allocation_summary_sha256": _sha256(allocation_summary_path),
        }

    result["status"] = "ready"
    _write_json_atomic(artifact_dir / "phase05_training_preparation.json", result)
    return result


def verify_prepared_phase05_allocation(
    artifact_dir: Path,
    method: str,
    *,
    expected_modules: int = 196,
    expected_rank: int = 32,
    atol: float = 3e-4,
) -> dict[str, Any]:
    """Revalidate a prepared allocation immediately before training."""

    artifact_dir = artifact_dir.expanduser().resolve()
    gate_path = artifact_dir / "phase05_gate_decision.json"
    validation_path = artifact_dir / "artifact_validation.json"
    preparation_path = artifact_dir / "phase05_training_preparation.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    selected = gate.get("calibration_selected_method")

    if gate.get("decision") != "go" or selected not in NONZERO_METHODS:
        raise ValueError("Training is forbidden unless the Phase-0.5 gate selected a nonzero hybrid")
    if validation.get("status") != "valid":
        raise ValueError("Strict Phase-0.5 artifact validation did not pass")
    if preparation.get("status") != "ready":
        raise ValueError("Phase-0.5 training allocations are not ready")
    if preparation.get("gate_sha256") != _sha256(gate_path):
        raise ValueError("Gate file changed after training allocations were prepared")
    if method not in {"H0", selected}:
        raise ValueError(f"Method {method} is not H0 or the calibration-selected {selected}")

    record = preparation.get("allocations", {}).get(method)
    if not isinstance(record, dict):
        raise ValueError(f"Prepared allocation is missing for {method}")
    paths = {
        "rank_map": Path(record["rank_map_path"]).expanduser().resolve(),
        "subspace": Path(record["subspace_path"]).expanduser().resolve(),
        "allocation_summary": Path(record["allocation_summary_path"]).expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty prepared {label}: {path}")
        expected_hash = record[f"{label}_sha256"]
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Prepared {label} hash changed for {method}")

    rank_map = json.loads(paths["rank_map"].read_text(encoding="utf-8"))
    allocation = json.loads(paths["allocation_summary"].read_text(encoding="utf-8"))
    ranks = rank_map.get("rank_pattern", {})
    alphas = rank_map.get("alpha_pattern", {})
    if len(ranks) != expected_modules or set(ranks) != set(alphas):
        raise ValueError(f"Prepared {method} rank map does not contain {expected_modules} modules")
    if {int(value) for value in ranks.values()} != {expected_rank}:
        raise ValueError(f"Prepared {method} allocation is not uniform-r{expected_rank}")
    if {int(value) for value in alphas.values()} != {expected_rank * 2}:
        raise ValueError(f"Prepared {method} alpha pattern is not 2r")
    if float(rank_map.get("constant_scaling", float("nan"))) != 2.0:
        raise ValueError(f"Prepared {method} constant scaling is not 2")
    if Path(rank_map["subspace_path"]).expanduser().resolve() != paths["subspace"]:
        raise ValueError(f"Prepared {method} rank map points to another subspace file")
    if allocation.get("candidate_method") != method:
        raise ValueError(f"Prepared allocation summary identifies another method")
    if int(allocation.get("trainable_parameters", -1)) != int(record["trainable_parameters"]):
        raise ValueError(f"Prepared {method} trainable parameter count changed")

    maximum_error = 0.0
    with safe_open(paths["subspace"], framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != set(ranks):
            raise ValueError(f"Prepared {method} subspace module keys differ from rank map")
        for name in tensors.keys():
            basis = tensors.get_tensor(name).float()
            if basis.ndim != 2 or basis.shape[0] != expected_rank:
                raise ValueError(f"Prepared {method} basis shape is invalid for {name}")
            if not bool(torch.isfinite(basis).all()):
                raise ValueError(f"Prepared {method} basis is non-finite for {name}")
            error = float(
                (basis @ basis.T - torch.eye(expected_rank)).abs().max().item()
            )
            if error > atol:
                raise ValueError(
                    f"Prepared {method} basis is non-orthogonal for {name}: {error}"
                )
            maximum_error = max(maximum_error, error)

    return {
        "status": "verified",
        "method": method,
        "selected_hybrid": selected,
        "module_count": len(ranks),
        "uniform_rank": expected_rank,
        "trainable_parameters": int(record["trainable_parameters"]),
        "orthogonality_error_max": maximum_error,
        "rank_map_path": str(paths["rank_map"]),
        "subspace_path": str(paths["subspace"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verify-method", choices=("H0", *NONZERO_METHODS))
    args = parser.parse_args()
    if args.verify_method:
        if args.output_root is not None:
            parser.error("--output-root cannot be combined with --verify-method")
        result = verify_prepared_phase05_allocation(
            args.artifact_dir, args.verify_method
        )
    else:
        result = prepare_phase05_training_artifacts(
            args.artifact_dir,
            output_root=args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
