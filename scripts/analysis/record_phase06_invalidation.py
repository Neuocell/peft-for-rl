#!/usr/bin/env python3
"""Record a failed Phase-0.6 replay invariant without manufacturing a gate result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from scripts.analysis.audit_phase06_artifact import audit_phase06_artifact


COMPARISONS = {"C0": "P0", "C2": "P2"}


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


def _candidate_invariant(source: Path, replay: Path, atol: float) -> dict[str, Any]:
    maximum_error = 0.0
    failing_tensors = 0
    tensor_count = 0
    with (
        safe_open(source, framework="pt", device="cpu") as first_file,
        safe_open(replay, framework="pt", device="cpu") as second_file,
    ):
        if list(first_file.keys()) != list(second_file.keys()):
            raise ValueError("P3/C3 candidate tensor keys differ")
        for key in first_file.keys():
            first = first_file.get_tensor(key).float()
            second = second_file.get_tensor(key).float()
            if first.shape != second.shape or first.ndim != 2:
                raise ValueError(f"P3/C3 candidate shape differs for {key}")
            direct = (first - second).abs().amax(dim=1)
            flipped = (first + second).abs().amax(dim=1)
            error = float(torch.minimum(direct, flipped).max().item())
            maximum_error = max(maximum_error, error)
            failing_tensors += int(error > atol)
            tensor_count += 1
    return {
        "passed": failing_tensors == 0,
        "atol": atol,
        "tensor_count": tensor_count,
        "failing_tensor_count": failing_tensors,
        "maximum_sign_invariant_row_error": maximum_error,
    }


def _score_invariant(
    source: Path, replay: Path, *, atol: float, rtol: float
) -> dict[str, Any]:
    maximum_error = 0.0
    failing_tensors = 0
    tensor_count = 0
    with (
        safe_open(source, framework="pt", device="cpu") as first_file,
        safe_open(replay, framework="pt", device="cpu") as second_file,
    ):
        if list(first_file.keys()) != list(second_file.keys()):
            raise ValueError("P3/C3 atom-score tensor keys differ")
        for key in first_file.keys():
            first = first_file.get_tensor(key).float()
            second = second_file.get_tensor(key).float()
            if first.shape != second.shape:
                raise ValueError(f"P3/C3 atom-score shape differs for {key}")
            error = float((first - second).abs().max().item())
            maximum_error = max(maximum_error, error)
            failing_tensors += int(
                not torch.allclose(first, second, atol=atol, rtol=rtol)
            )
            tensor_count += 1
    return {
        "passed": failing_tensors == 0,
        "atol": atol,
        "rtol": rtol,
        "tensor_count": tensor_count,
        "failing_tensor_count": failing_tensors,
        "maximum_absolute_error": maximum_error,
    }


def record_phase06_invalidation(
    source_artifact: Path,
    replay_artifact: Path,
    *,
    capture_tolerance: float = 0.002,
    overlap_tolerance: float = 0.02,
    replay_metric_atol: float = 1e-6,
    replay_candidate_atol: float = 3e-5,
    replay_score_rtol: float = 1e-5,
) -> dict[str, Any]:
    source_artifact = source_artifact.expanduser().resolve()
    replay_artifact = replay_artifact.expanduser().resolve()
    output = replay_artifact / "phase06_invalidation.json"
    gate_path = replay_artifact / "phase06_gate_decision.json"
    if output.exists():
        raise ValueError(f"Refusing to overwrite Phase-0.6 invalidation: {output}")
    if gate_path.exists():
        raise ValueError("Cannot invalidate Phase 0.6 after a gate record exists")

    source_summary_path = source_artifact / "probe_summary.json"
    replay_summary_path = replay_artifact / "probe_summary.json"
    source_validation_path = source_artifact / "artifact_validation.json"
    replay_validation_path = replay_artifact / "artifact_validation.json"
    phase05_gate_path = source_artifact / "phase05_gate_decision.json"
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_summary_path.read_text(encoding="utf-8"))
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    phase05_gate = json.loads(phase05_gate_path.read_text(encoding="utf-8"))
    if source_validation.get("status") != "valid":
        raise ValueError("Phase-0.5 source validation is not valid")
    if phase05_gate.get("decision") != "no-go":
        raise ValueError("Phase-0.5 did not produce a no-go decision")

    # Re-run the structural audit so the invalidation pins the current replay bytes.
    replay_validation = audit_phase06_artifact(
        source_artifact,
        replay_artifact,
        expected_discovery=int(source["discovery_prompts"]),
        expected_calibration=int(source["calibration_prompts"]),
        expected_audit=int(source["audit_prompts"]),
        expected_modules=len(source["modules"]),
        expected_rank=int(source["r_max"]),
    )
    if replay_validation.get("status") != "valid":
        raise ValueError("Phase-0.6 structural validation is not valid")
    if replay.get("selection_uses_audit") is not False:
        raise ValueError("Phase-0.6 replay used audit data for selection")

    candidate = _candidate_invariant(
        source_artifact / "candidates_P3.safetensors",
        replay_artifact / "candidates_C3.safetensors",
        replay_candidate_atol,
    )
    scores = _score_invariant(
        source_artifact / "atom_scores_P3.safetensors",
        replay_artifact / "atom_scores_C3.safetensors",
        atol=replay_metric_atol,
        rtol=replay_score_rtol,
    )
    global_metrics: dict[str, Any] = {}
    metrics_passed = True
    for field in ("calibration_capture", "audit_capture"):
        source_value = float(source["diagnostics"]["P3"][field])
        replay_value = float(replay["diagnostics"]["C3"][field])
        delta = replay_value - source_value
        passed = math.isfinite(delta) and abs(delta) <= replay_metric_atol
        metrics_passed = metrics_passed and passed
        global_metrics[field] = {
            "source": source_value,
            "replay": replay_value,
            "delta": delta,
            "passed": passed,
        }
    invariant_passed = candidate["passed"] and scores["passed"] and metrics_passed
    if invariant_passed:
        raise ValueError(
            "P3/C3 replay invariants passed; run the preregistered gate instead"
        )

    comparisons: dict[str, Any] = {}
    all_failed_calibration_screen = True
    for method, baseline_method in COMPARISONS.items():
        item = replay["diagnostics"][method]
        baseline = source["diagnostics"][baseline_method]
        calibration_delta = float(item["calibration_capture"]) - float(
            baseline["calibration_capture"]
        )
        audit_delta = float(item["audit_capture"]) - float(baseline["audit_capture"])
        overlap_delta = float(item["prompt_split_overlap"]["mean_cosine"]) - float(
            baseline["prompt_split_overlap"]["mean_cosine"]
        )
        admissible = (
            math.isfinite(calibration_delta)
            and math.isfinite(overlap_delta)
            and calibration_delta >= -capture_tolerance
            and overlap_delta >= -overlap_tolerance
        )
        all_failed_calibration_screen = all_failed_calibration_screen and not admissible
        comparisons[method] = {
            "matched_standard_method": baseline_method,
            "calibration_capture_delta": calibration_delta,
            "audit_capture_delta_reported_not_used_for_authorization": audit_delta,
            "prompt_split_mean_cosine_delta": overlap_delta,
            "calibration_admissible": admissible,
        }

    diagnosis = (
        "crossfit_estimator_failure"
        if all_failed_calibration_screen
        else "invalid_replay_inconclusive"
    )
    fallback_authorized = diagnosis == "crossfit_estimator_failure"
    result = {
        "schema_version": 1,
        "method": "phase06_failed_replay_invalidation_v1",
        "status": "invalid",
        "decision": "unavailable",
        "reason": "P3/C3 replay invariants failed before the preregistered gate",
        "protocol_deviation": {
            "present": True,
            "description": (
                "Phase 1 may use only the already validated P2 source after an "
                "invalid Phase-0.6 replay; no C candidate may enter training."
            ),
            "formal_phase06_gate_written": False,
        },
        "selection_uses_audit": False,
        "thresholds": {
            "calibration_capture_tolerance": capture_tolerance,
            "prompt_split_overlap_tolerance": overlap_tolerance,
            "replay_metric_atol": replay_metric_atol,
            "replay_candidate_atol": replay_candidate_atol,
            "replay_score_rtol": replay_score_rtol,
        },
        "P3_C3_replay_invariants": {
            "passed": False,
            "candidates": candidate,
            "atom_scores": scores,
            "global_metrics": global_metrics,
        },
        "diagnostic_comparisons": comparisons,
        "diagnosis": diagnosis,
        "phase06_training_authorized": False,
        "phase1_fallback": {
            "authorized": fallback_authorized,
            "signal_method": "P2" if fallback_authorized else None,
            "allowed_methods": (
                ["I0", "I8", "I16", "I32"] if fallback_authorized else []
            ),
            "basis_source": (
                "validated_phase05_source_only" if fallback_authorized else None
            ),
            "authorization_uses_audit": False,
        },
        "source_artifact": str(source_artifact),
        "replay_artifact": str(replay_artifact),
        "source_probe_summary_sha256": _sha256(source_summary_path),
        "replay_probe_summary_sha256": _sha256(replay_summary_path),
        "source_validation_sha256": _sha256(source_validation_path),
        "replay_validation_sha256": _sha256(replay_validation_path),
        "phase05_gate_sha256": _sha256(phase05_gate_path),
        "source_P3_candidates_sha256": _sha256(
            source_artifact / "candidates_P3.safetensors"
        ),
        "replay_C3_candidates_sha256": _sha256(
            replay_artifact / "candidates_C3.safetensors"
        ),
        "source_P3_scores_sha256": _sha256(
            source_artifact / "atom_scores_P3.safetensors"
        ),
        "replay_C3_scores_sha256": _sha256(
            replay_artifact / "atom_scores_C3.safetensors"
        ),
    }
    _write_json_atomic(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--replay-artifact", type=Path, required=True)
    args = parser.parse_args()
    result = record_phase06_invalidation(args.source_artifact, args.replay_artifact)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
