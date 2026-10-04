#!/usr/bin/env python3
"""Prepare fixed-rank signal/random LoRA-A initializations after both probe gates fail."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from verl.utils.full_gradient_rl_probe import (
    signal_random_hybrid_basis,
    validate_full_gradient_probe_artifact,
)


PHASE1_METHODS = ("I0", "I8", "I16", "I32")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base_seed: int, module: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{module}:{purpose}".encode()).digest()
    return (base_seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _phase06_outcome(
    source_artifact: Path, replay_artifact: Path, signal_method: str
) -> dict[str, Any]:
    gate_path = replay_artifact / "phase06_gate_decision.json"
    invalidation_path = replay_artifact / "phase06_invalidation.json"
    if gate_path.exists() and invalidation_path.exists():
        raise ValueError("Phase 0.6 cannot have both a gate and an invalidation record")
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("decision") != "no-go":
            raise ValueError("Phase 1 requires a Phase-0.6 no-go gate")
        return {
            "kind": "formal_no_go_gate",
            "path": gate_path,
            "sha256": _sha256(gate_path),
        }
    if not invalidation_path.exists():
        raise ValueError("Phase 1 requires a Phase-0.6 gate or invalidation record")
    invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    expected = {
        "status": "invalid",
        "decision": "unavailable",
        "diagnosis": "crossfit_estimator_failure",
        "phase06_training_authorized": False,
        "selection_uses_audit": False,
    }
    for field, value in expected.items():
        if invalidation.get(field) != value:
            raise ValueError(f"Phase-0.6 invalidation changed: {field}")
    fallback = invalidation.get("phase1_fallback", {})
    if (
        fallback.get("authorized") is not True
        or fallback.get("signal_method") != signal_method
        or fallback.get("basis_source") != "validated_phase05_source_only"
        or fallback.get("authorization_uses_audit") is not False
        or set(fallback.get("allowed_methods", ())) != set(PHASE1_METHODS)
    ):
        raise ValueError(
            "Phase-0.6 invalidation does not authorize the Phase-1 fallback"
        )
    if (
        Path(invalidation.get("source_artifact", "")).expanduser().resolve()
        != source_artifact
    ):
        raise ValueError("Phase-0.6 invalidation points to another source artifact")
    if (
        Path(invalidation.get("replay_artifact", "")).expanduser().resolve()
        != replay_artifact
    ):
        raise ValueError("Phase-0.6 invalidation points to another replay artifact")
    pinned = {
        "source_probe_summary_sha256": source_artifact / "probe_summary.json",
        "replay_probe_summary_sha256": replay_artifact / "probe_summary.json",
        "source_validation_sha256": source_artifact / "artifact_validation.json",
        "replay_validation_sha256": replay_artifact / "artifact_validation.json",
        "phase05_gate_sha256": source_artifact / "phase05_gate_decision.json",
    }
    for field, path in pinned.items():
        if invalidation.get(field) != _sha256(path):
            raise ValueError(f"Phase-0.6 invalidation input changed: {field}")
    return {
        "kind": "invalid_replay_fallback",
        "path": invalidation_path,
        "sha256": _sha256(invalidation_path),
    }


def prepare_phase1_signal_random_artifacts(
    source_artifact: Path,
    replay_artifact: Path,
    output_root: Path,
    *,
    rank: int = 32,
    signal_counts: tuple[int, ...] = (0, 8, 16, 32),
    complement_seed: int = 42,
    signal_method: str = "P2",
) -> dict[str, Any]:
    source_artifact = source_artifact.expanduser().resolve()
    replay_artifact = replay_artifact.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    source_validation_path = source_artifact / "artifact_validation.json"
    replay_validation_path = replay_artifact / "artifact_validation.json"
    phase05_gate_path = source_artifact / "phase05_gate_decision.json"
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    replay_validation = json.loads(replay_validation_path.read_text(encoding="utf-8"))
    phase05_gate = json.loads(phase05_gate_path.read_text(encoding="utf-8"))
    if (
        source_validation.get("status") != "valid"
        or replay_validation.get("status") != "valid"
    ):
        raise ValueError("Phase 1 requires valid Phase-0.5 and Phase-0.6 artifacts")
    if phase05_gate.get("decision") != "no-go":
        raise ValueError("Phase 1 requires a Phase-0.5 no-go decision")
    if signal_method != "P2":
        raise ValueError("The preregistered Phase-1 signal source is P2")
    phase06_outcome = _phase06_outcome(source_artifact, replay_artifact, signal_method)
    if tuple(sorted(set(signal_counts))) != signal_counts:
        raise ValueError("signal_counts must be unique and increasing")
    if not signal_counts or signal_counts[0] != 0 or signal_counts[-1] != rank:
        raise ValueError(
            "signal_counts must include the random-only and signal-only endpoints"
        )
    if any(value < 0 or value > rank for value in signal_counts):
        raise ValueError("signal_counts must lie in [0, rank]")

    validation = validate_full_gradient_probe_artifact(
        source_artifact, candidate_method=signal_method
    )
    source_summary = json.loads(
        (source_artifact / "probe_summary.json").read_text(encoding="utf-8")
    )
    if int(source_summary.get("r_max", -1)) != rank:
        raise ValueError("Phase-1 rank differs from the source candidate rank")
    modules = source_summary.get("modules", {})
    signal_path = source_artifact / f"candidates_{signal_method}.safetensors"
    signals: dict[str, torch.Tensor] = {}
    with safe_open(signal_path, framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != set(modules):
            raise ValueError(
                "Phase-1 signal tensor keys differ from the source modules"
            )
        for name in sorted(modules):
            value = tensors.get_tensor(name).float().contiguous()
            if tuple(value.shape) != (rank, int(modules[name]["shape"][1])):
                raise ValueError(f"Phase-1 signal shape differs for {name}")
            signals[name] = value

    output_root.mkdir(parents=True, exist_ok=True)
    allocations: dict[str, Any] = {}
    for signal_count in signal_counts:
        method = f"I{signal_count}"
        allocation_dir = output_root / f"{method}_uniform_r{rank}"
        if allocation_dir.exists() and any(allocation_dir.iterdir()):
            raise ValueError(
                f"Refusing to overwrite Phase-1 allocation: {allocation_dir}"
            )
        allocation_dir.mkdir(parents=True, exist_ok=True)
        candidates = {
            name: signal_random_hybrid_basis(
                signal,
                rank,
                signal_directions=signal_count,
                seed=_stable_seed(
                    complement_seed, name, "phase1_signal_random_complement_v1"
                ),
            )
            for name, signal in signals.items()
        }
        if signal_count == rank and any(
            not torch.equal(candidates[name], signals[name]) for name in signals
        ):
            raise RuntimeError("Phase-1 signal-only endpoint does not exactly equal P2")
        if signal_count and any(
            not torch.equal(
                candidates[name][:signal_count], signals[name][:signal_count]
            )
            for name in signals
        ):
            raise RuntimeError(f"Phase-1 {method} did not preserve the signal prefix")

        subspace_path = allocation_dir / "subspaces.safetensors"
        save_file(candidates, subspace_path)
        rank_pattern = {name: rank for name in signals}
        rank_map = {
            "schema_version": 1,
            "method": "phase1_signal_random_orthogonal_a_v1",
            "candidate_method": method,
            "signal_method": signal_method,
            "signal_directions": signal_count,
            "random_complement_directions": rank - signal_count,
            "complement_seed": complement_seed,
            "rank_pattern": rank_pattern,
            "alpha_pattern": {name: rank * 2 for name in signals},
            "constant_scaling": 2.0,
            "subspace_path": str(subspace_path.resolve()),
        }
        rank_map_path = allocation_dir / "rank_map.json"
        _write_json_atomic(rank_map_path, rank_map)
        allocation_summary = {
            **rank_map,
            "module_count": len(signals),
            "uniform_rank": rank,
            "trainable_parameters": sum(
                rank * sum(int(value) for value in modules[name]["shape"])
                for name in signals
            ),
            "signal_prefix_exact": True,
            "orthogonality_error_max": max(
                float(
                    (basis @ basis.T - torch.eye(rank, dtype=torch.float32))
                    .abs()
                    .max()
                    .item()
                )
                for basis in candidates.values()
            ),
        }
        allocation_summary_path = allocation_dir / "allocation_summary.json"
        _write_json_atomic(allocation_summary_path, allocation_summary)
        allocations[method] = {
            "directory": str(allocation_dir),
            "signal_directions": signal_count,
            "random_complement_directions": rank - signal_count,
            "rank_map_path": str(rank_map_path),
            "rank_map_sha256": _sha256(rank_map_path),
            "subspace_path": str(subspace_path),
            "subspace_sha256": _sha256(subspace_path),
            "allocation_summary_path": str(allocation_summary_path),
            "allocation_summary_sha256": _sha256(allocation_summary_path),
        }

    result = {
        "schema_version": 1,
        "method": "phase1_signal_random_orthogonal_a_preparation_v1",
        "status": "ready",
        "source_artifact": str(source_artifact),
        "replay_artifact": str(replay_artifact),
        "signal_method": signal_method,
        "uniform_rank": rank,
        "complement_seed": complement_seed,
        "signal_counts": list(signal_counts),
        "source_probe_validation": validation,
        "source_validation_sha256": _sha256(source_validation_path),
        "replay_validation_sha256": _sha256(replay_validation_path),
        "phase05_gate_sha256": _sha256(phase05_gate_path),
        "phase06_outcome_kind": phase06_outcome["kind"],
        "phase06_outcome_path": str(phase06_outcome["path"]),
        "phase06_outcome_sha256": phase06_outcome["sha256"],
        "signal_candidate_sha256": _sha256(signal_path),
        "allocations": allocations,
    }
    _write_json_atomic(output_root / "phase1_training_preparation.json", result)
    return result


def verify_prepared_phase1_allocation(
    output_root: Path,
    method: str,
    *,
    expected_modules: int = 196,
    expected_rank: int = 32,
    expected_signal_counts: tuple[int, ...] = (0, 8, 16, 32),
    expected_complement_seed: int = 42,
    atol: float = 3e-4,
) -> dict[str, Any]:
    """Revalidate a Phase-1 allocation immediately before training."""

    output_root = output_root.expanduser().resolve()
    preparation_path = output_root / "phase1_training_preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("status") != "ready":
        raise ValueError("Phase-1 training allocations are not ready")
    if method not in preparation.get("allocations", {}):
        raise ValueError(f"Unknown or unprepared Phase-1 method: {method}")
    if int(preparation.get("uniform_rank", -1)) != expected_rank:
        raise ValueError("Phase-1 preparation rank differs from the training rank")
    if preparation.get("signal_method") != "P2":
        raise ValueError("Phase-1 preparation does not use the preregistered P2 signal")
    if tuple(preparation.get("signal_counts", ())) != expected_signal_counts:
        raise ValueError("Phase-1 preparation candidate set changed")
    if int(preparation.get("complement_seed", -1)) != expected_complement_seed:
        raise ValueError("Phase-1 preparation complement seed changed")
    expected_methods = {f"I{count}" for count in expected_signal_counts}
    if set(preparation.get("allocations", {})) != expected_methods:
        raise ValueError("Phase-1 preparation allocation set changed")

    source_artifact = Path(preparation["source_artifact"]).expanduser().resolve()
    replay_artifact = Path(preparation["replay_artifact"]).expanduser().resolve()
    immutable_inputs = {
        "source_validation": source_artifact / "artifact_validation.json",
        "replay_validation": replay_artifact / "artifact_validation.json",
        "phase05_gate": source_artifact / "phase05_gate_decision.json",
        "phase06_outcome": Path(preparation["phase06_outcome_path"])
        .expanduser()
        .resolve(),
        "signal_candidate": source_artifact / "candidates_P2.safetensors",
    }
    for label, path in immutable_inputs.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty Phase-1 input {label}: {path}")
        if _sha256(path) != preparation.get(f"{label}_sha256"):
            raise ValueError(f"Phase-1 input {label} changed after preparation")
    source_validation = json.loads(
        immutable_inputs["source_validation"].read_text(encoding="utf-8")
    )
    replay_validation = json.loads(
        immutable_inputs["replay_validation"].read_text(encoding="utf-8")
    )
    phase05_gate = json.loads(
        immutable_inputs["phase05_gate"].read_text(encoding="utf-8")
    )
    if source_validation.get("status") != "valid":
        raise ValueError("Strict Phase-0.5 validation is no longer valid")
    if replay_validation.get("status") != "valid":
        raise ValueError("Strict Phase-0.6 validation is no longer valid")
    if phase05_gate.get("decision") != "no-go":
        raise ValueError("Phase 1 requires a Phase-0.5 no-go decision")
    outcome = _phase06_outcome(source_artifact, replay_artifact, "P2")
    if preparation.get("phase06_outcome_kind") != outcome["kind"]:
        raise ValueError("Phase-0.6 outcome kind changed after preparation")
    if (
        Path(preparation.get("phase06_outcome_path", "")).expanduser().resolve()
        != outcome["path"]
    ):
        raise ValueError("Phase-0.6 outcome path changed after preparation")
    if preparation.get("phase06_outcome_sha256") != outcome["sha256"]:
        raise ValueError("Phase-0.6 outcome changed after preparation")

    record = preparation.get("allocations", {}).get(method)
    if not isinstance(record, dict):
        raise ValueError(f"Prepared Phase-1 allocation is missing for {method}")
    paths = {
        "rank_map": Path(record["rank_map_path"]).expanduser().resolve(),
        "subspace": Path(record["subspace_path"]).expanduser().resolve(),
        "allocation_summary": Path(record["allocation_summary_path"])
        .expanduser()
        .resolve(),
    }
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty prepared {label}: {path}")
        if _sha256(path) != record.get(f"{label}_sha256"):
            raise ValueError(f"Prepared {label} hash changed for {method}")

    rank_map = json.loads(paths["rank_map"].read_text(encoding="utf-8"))
    allocation = json.loads(paths["allocation_summary"].read_text(encoding="utf-8"))
    ranks = rank_map.get("rank_pattern", {})
    alphas = rank_map.get("alpha_pattern", {})
    expected_signal_count = int(method[1:])
    if rank_map.get("candidate_method") != method:
        raise ValueError(f"Prepared rank map identifies another method for {method}")
    if rank_map.get("method") != "phase1_signal_random_orthogonal_a_v1":
        raise ValueError(f"Prepared {method} rank map has the wrong method")
    if int(rank_map.get("complement_seed", -1)) != expected_complement_seed:
        raise ValueError(f"Prepared {method} complement seed changed")
    if int(rank_map.get("signal_directions", -1)) != expected_signal_count:
        raise ValueError(f"Prepared {method} signal count changed")
    if int(rank_map.get("random_complement_directions", -1)) != (
        expected_rank - expected_signal_count
    ):
        raise ValueError(f"Prepared {method} random-complement count changed")
    if len(ranks) != expected_modules or set(ranks) != set(alphas):
        raise ValueError(
            f"Prepared {method} rank map does not contain {expected_modules} modules"
        )
    if {int(value) for value in ranks.values()} != {expected_rank}:
        raise ValueError(
            f"Prepared {method} allocation is not uniform-r{expected_rank}"
        )
    if {int(value) for value in alphas.values()} != {expected_rank * 2}:
        raise ValueError(f"Prepared {method} alpha pattern is not 2r")
    if float(rank_map.get("constant_scaling", float("nan"))) != 2.0:
        raise ValueError(f"Prepared {method} constant scaling is not 2")
    if Path(rank_map["subspace_path"]).expanduser().resolve() != paths["subspace"]:
        raise ValueError(f"Prepared {method} rank map points to another subspace file")
    if allocation.get("candidate_method") != method:
        raise ValueError(f"Prepared allocation summary identifies another method")
    if int(allocation.get("module_count", -1)) != expected_modules:
        raise ValueError(
            f"Prepared allocation summary module count changed for {method}"
        )
    if int(allocation.get("signal_directions", -1)) != expected_signal_count:
        raise ValueError(
            f"Prepared allocation summary signal count changed for {method}"
        )
    if int(allocation.get("random_complement_directions", -1)) != (
        expected_rank - expected_signal_count
    ):
        raise ValueError(
            f"Prepared allocation summary random-complement count changed for {method}"
        )

    signal_path = immutable_inputs["signal_candidate"]
    maximum_error = 0.0
    maximum_prefix_error = 0.0
    with (
        safe_open(paths["subspace"], framework="pt", device="cpu") as tensors,
        safe_open(signal_path, framework="pt", device="cpu") as signals,
    ):
        if set(tensors.keys()) != set(ranks) or set(signals.keys()) != set(ranks):
            raise ValueError(f"Prepared {method} module keys differ from the source")
        for name in tensors.keys():
            basis = tensors.get_tensor(name).float()
            signal = signals.get_tensor(name).float()
            if basis.ndim != 2 or basis.shape[0] != expected_rank:
                raise ValueError(f"Prepared {method} basis shape is invalid for {name}")
            if basis.shape != signal.shape:
                raise ValueError(
                    f"Prepared {method} basis shape differs from P2 for {name}"
                )
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
            if expected_signal_count:
                prefix_error = float(
                    (basis[:expected_signal_count] - signal[:expected_signal_count])
                    .abs()
                    .max()
                    .item()
                )
                if prefix_error != 0.0:
                    raise ValueError(
                        f"Prepared {method} signal prefix changed for {name}: {prefix_error}"
                    )
                maximum_prefix_error = max(maximum_prefix_error, prefix_error)

    return {
        "status": "verified",
        "method": method,
        "module_count": len(ranks),
        "uniform_rank": expected_rank,
        "signal_directions": expected_signal_count,
        "random_complement_directions": expected_rank - expected_signal_count,
        "orthogonality_error_max": maximum_error,
        "signal_prefix_error_max": maximum_prefix_error,
        "rank_map_path": str(paths["rank_map"]),
        "subspace_path": str(paths["subspace"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path)
    parser.add_argument("--replay-artifact", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-method", choices=PHASE1_METHODS)
    args = parser.parse_args()
    if args.verify_method:
        if args.source_artifact is not None or args.replay_artifact is not None:
            parser.error(
                "--source-artifact/--replay-artifact cannot be combined with --verify-method"
            )
        result = verify_prepared_phase1_allocation(args.output_root, args.verify_method)
    else:
        if args.source_artifact is None or args.replay_artifact is None:
            parser.error(
                "--source-artifact and --replay-artifact are required for preparation"
            )
        result = prepare_phase1_signal_random_artifacts(
            args.source_artifact, args.replay_artifact, args.output_root
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
