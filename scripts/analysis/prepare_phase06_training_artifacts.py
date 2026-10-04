#!/usr/bin/env python3
"""Materialize the preregistered Phase-0.6 pair for uniform-r32 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from scripts.analysis.build_full_gradient_uniform_allocation import build_uniform_allocation


MATCHED_STANDARD = {"C0": "P0", "C2": "P2"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_phase06_training_artifacts(
    source_artifact: Path,
    replay_artifact: Path,
    *,
    output_root: Path | None = None,
    uniform_rank: int = 32,
) -> dict[str, Any]:
    source_artifact = source_artifact.expanduser().resolve()
    replay_artifact = replay_artifact.expanduser().resolve()
    output_root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else replay_artifact / "training_allocations"
    )
    gate_path = replay_artifact / "phase06_gate_decision.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    source_validation = json.loads(
        (source_artifact / "artifact_validation.json").read_text(encoding="utf-8")
    )
    replay_validation = json.loads(
        (replay_artifact / "artifact_validation.json").read_text(encoding="utf-8")
    )
    if source_validation.get("status") != "valid":
        raise ValueError("Phase-0.6 source validation did not pass")
    if replay_validation.get("status") != "valid":
        raise ValueError("Phase-0.6 replay validation did not pass")

    decision = gate.get("decision")
    if decision not in {"go", "no-go"}:
        raise ValueError(f"Unknown Phase-0.6 gate decision: {decision!r}")
    selected = gate.get("calibration_selected_method")
    result: dict[str, Any] = {
        "schema_version": 1,
        "source_artifact": str(source_artifact),
        "replay_artifact": str(replay_artifact),
        "source_validation_sha256": _sha256(source_artifact / "artifact_validation.json"),
        "replay_validation_sha256": _sha256(replay_artifact / "artifact_validation.json"),
        "gate_path": str(gate_path),
        "gate_sha256": _sha256(gate_path),
        "gate_decision": decision,
        "calibration_selected_method": selected,
        "matched_standard_method": MATCHED_STANDARD.get(selected),
        "uniform_rank": uniform_rank,
        "status": "skipped_no_go",
        "allocations": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = replay_artifact / "phase06_training_preparation.json"
    if decision == "no-go":
        _write_json_atomic(output_path, result)
        return result
    if selected not in MATCHED_STANDARD:
        raise ValueError("A Phase-0.6 go decision must select C0 or C2")

    standard = MATCHED_STANDARD[selected]
    for method, artifact in ((standard, source_artifact), (selected, replay_artifact)):
        allocation_dir = output_root / f"{method}_uniform_r{uniform_rank}"
        allocation = build_uniform_allocation(
            artifact,
            allocation_dir,
            candidate_method=method,
            uniform_rank=uniform_rank,
        )
        ranks = {int(item["rank"]) for item in allocation["modules"].values()}
        if ranks != {uniform_rank}:
            raise ValueError(f"{method} allocation is not uniformly rank {uniform_rank}")
        paths = {
            "rank_map": allocation_dir / "rank_map.json",
            "subspace": allocation_dir / "subspaces.safetensors",
            "allocation_summary": allocation_dir / "allocation_summary.json",
        }
        result["allocations"][method] = {
            "source_artifact": str(artifact),
            "directory": str(allocation_dir),
            "module_count": len(allocation["modules"]),
            "trainable_parameters": int(allocation["trainable_parameters"]),
            **{
                f"{label}_path": str(path)
                for label, path in paths.items()
            },
            **{
                f"{label}_sha256": _sha256(path)
                for label, path in paths.items()
            },
        }

    result["status"] = "ready"
    _write_json_atomic(output_path, result)
    return result


def verify_prepared_phase06_allocation(
    source_artifact: Path,
    replay_artifact: Path,
    method: str,
    *,
    expected_modules: int = 196,
    expected_rank: int = 32,
    atol: float = 3e-4,
) -> dict[str, Any]:
    source_artifact = source_artifact.expanduser().resolve()
    replay_artifact = replay_artifact.expanduser().resolve()
    gate_path = replay_artifact / "phase06_gate_decision.json"
    preparation_path = replay_artifact / "phase06_training_preparation.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    selected = gate.get("calibration_selected_method")
    standard = MATCHED_STANDARD.get(selected)
    if gate.get("decision") != "go" or standard is None:
        raise ValueError("Training is forbidden unless Phase 0.6 selected C0 or C2")
    if method not in {standard, selected}:
        raise ValueError(f"Method {method} is not the matched pair {standard}/{selected}")
    if preparation.get("status") != "ready":
        raise ValueError("Phase-0.6 training allocations are not ready")
    if preparation.get("gate_sha256") != _sha256(gate_path):
        raise ValueError("Phase-0.6 gate changed after allocations were prepared")
    checks = (
        (source_artifact / "artifact_validation.json", "source_validation_sha256"),
        (replay_artifact / "artifact_validation.json", "replay_validation_sha256"),
    )
    for path, field in checks:
        if preparation.get(field) != _sha256(path):
            raise ValueError(f"Phase-0.6 {field} changed after preparation")

    record = preparation.get("allocations", {}).get(method)
    if not isinstance(record, dict):
        raise ValueError(f"Prepared allocation is missing for {method}")
    paths = {
        label: Path(record[f"{label}_path"]).expanduser().resolve()
        for label in ("rank_map", "subspace", "allocation_summary")
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty prepared {label}: {path}")
        if _sha256(path) != record[f"{label}_sha256"]:
            raise ValueError(f"Prepared {label} hash changed for {method}")

    rank_map = json.loads(paths["rank_map"].read_text(encoding="utf-8"))
    allocation = json.loads(paths["allocation_summary"].read_text(encoding="utf-8"))
    ranks = rank_map.get("rank_pattern", {})
    alphas = rank_map.get("alpha_pattern", {})
    if len(ranks) != expected_modules or set(ranks) != set(alphas):
        raise ValueError(f"Prepared {method} rank map must contain {expected_modules} modules")
    if {int(value) for value in ranks.values()} != {expected_rank}:
        raise ValueError(f"Prepared {method} allocation is not uniform-r{expected_rank}")
    if {int(value) for value in alphas.values()} != {expected_rank * 2}:
        raise ValueError(f"Prepared {method} alpha pattern is not 2r")
    if float(rank_map.get("constant_scaling", float("nan"))) != 2.0:
        raise ValueError(f"Prepared {method} constant scaling is not 2")
    if Path(rank_map["subspace_path"]).expanduser().resolve() != paths["subspace"]:
        raise ValueError(f"Prepared {method} rank map points to another subspace file")
    if allocation.get("candidate_method") != method:
        raise ValueError(f"Prepared {method} allocation summary identifies another method")

    maximum_error = 0.0
    with safe_open(paths["subspace"], framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != set(ranks):
            raise ValueError(f"Prepared {method} subspace keys differ from the rank map")
        for name in tensors.keys():
            basis = tensors.get_tensor(name).float()
            if basis.ndim != 2 or tuple(basis.shape)[0] != expected_rank:
                raise ValueError(f"Prepared {method} basis shape is invalid for {name}")
            if not bool(torch.isfinite(basis).all()):
                raise ValueError(f"Prepared {method} basis is non-finite for {name}")
            error = float((basis @ basis.T - torch.eye(expected_rank)).abs().max().item())
            if error > atol:
                raise ValueError(f"Prepared {method} basis is non-orthogonal for {name}: {error}")
            maximum_error = max(maximum_error, error)
    return {
        "status": "verified",
        "method": method,
        "selected_crossfit": selected,
        "matched_standard": standard,
        "module_count": len(ranks),
        "uniform_rank": expected_rank,
        "orthogonality_error_max": maximum_error,
        "rank_map_path": str(paths["rank_map"]),
        "subspace_path": str(paths["subspace"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--replay-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verify-method", choices=("P0", "P2", "C0", "C2"))
    args = parser.parse_args()
    if args.verify_method:
        if args.output_root is not None:
            parser.error("--output-root cannot be combined with --verify-method")
        result = verify_prepared_phase06_allocation(
            args.source_artifact, args.replay_artifact, args.verify_method
        )
    else:
        result = prepare_phase06_training_artifacts(
            args.source_artifact,
            args.replay_artifact,
            output_root=args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
