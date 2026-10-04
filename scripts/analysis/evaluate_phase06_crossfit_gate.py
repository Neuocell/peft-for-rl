#!/usr/bin/env python3
"""Apply the preregistered Phase-0.6 cross-fit mask ablation gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


COMPARISONS = {"C0": "P0", "C2": "P2"}


def _assert_replay_candidate_equivalent(
    source: Path,
    replay: Path,
    *,
    atol: float,
) -> dict[str, Any]:
    maximum_row_error = 0.0
    tensor_count = 0
    with safe_open(source, framework="pt", device="cpu") as source_file, safe_open(
        replay, framework="pt", device="cpu"
    ) as replay_file:
        if list(source_file.keys()) != list(replay_file.keys()):
            raise ValueError("P3/C3 candidate tensor keys differ")
        for key in source_file.keys():
            first = source_file.get_tensor(key).float()
            second = replay_file.get_tensor(key).float()
            if first.shape != second.shape or first.ndim != 2:
                raise ValueError(f"P3/C3 candidate shape differs for {key}")
            direct = (first - second).abs().amax(dim=1)
            flipped = (first + second).abs().amax(dim=1)
            error = float(torch.minimum(direct, flipped).max().item())
            if error > atol:
                raise ValueError(f"P3/C3 candidate differs for {key}: {error} > {atol}")
            maximum_row_error = max(maximum_row_error, error)
            tensor_count += 1
    return {
        "equivalent": True,
        "tensor_count": tensor_count,
        "maximum_sign_invariant_row_error": maximum_row_error,
    }


def _assert_replay_scores_equal(
    source: Path,
    replay: Path,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    maximum_absolute_error = 0.0
    tensor_count = 0
    with safe_open(source, framework="pt", device="cpu") as source_file, safe_open(
        replay, framework="pt", device="cpu"
    ) as replay_file:
        if list(source_file.keys()) != list(replay_file.keys()):
            raise ValueError("P3/C3 atom-score tensor keys differ")
        for key in source_file.keys():
            first = source_file.get_tensor(key).float()
            second = replay_file.get_tensor(key).float()
            if first.shape != second.shape or not torch.allclose(
                first, second, atol=atol, rtol=rtol
            ):
                error = (
                    float((first - second).abs().max().item())
                    if first.shape == second.shape
                    else float("inf")
                )
                raise ValueError(f"P3/C3 atom score differs for {key}: {error}")
            maximum_absolute_error = max(
                maximum_absolute_error,
                float((first - second).abs().max().item()),
            )
            tensor_count += 1
    return {
        "equal_within_tolerance": True,
        "tensor_count": tensor_count,
        "maximum_absolute_error": maximum_absolute_error,
    }


def evaluate(
    source_artifact: Path,
    replay_artifact: Path,
    *,
    capture_tolerance: float = 0.002,
    overlap_tolerance: float = 0.02,
    tie_tolerance: float = 5e-4,
    minimum_improvement: float = 0.001,
    replay_metric_atol: float = 1e-6,
    replay_candidate_atol: float = 3e-5,
    replay_score_rtol: float = 1e-5,
) -> dict[str, Any]:
    source_artifact = source_artifact.expanduser().resolve()
    replay_artifact = replay_artifact.expanduser().resolve()
    source = json.loads(
        (source_artifact / "probe_summary.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (replay_artifact / "probe_summary.json").read_text(encoding="utf-8")
    )
    if not {"P0", "P2", "P3"}.issubset(source.get("diagnostics", {})):
        raise ValueError("Source artifact is missing P0/P2/P3 diagnostics")
    if not {"C0", "C2", "C3"}.issubset(replay.get("diagnostics", {})):
        raise ValueError("Replay artifact is missing C0/C2/C3 diagnostics")
    if replay.get("selection_uses_audit") is not False:
        raise ValueError("Replay candidate construction used audit data")
    if replay.get("source_prompt_ids_sha256") != replay.get(
        "replayed_prompt_ids_sha256"
    ):
        raise ValueError("Replay prompt IDs do not match the source manifest")

    replay_invariants = {
        "candidates": _assert_replay_candidate_equivalent(
            source_artifact / "candidates_P3.safetensors",
            replay_artifact / "candidates_C3.safetensors",
            atol=replay_candidate_atol,
        ),
        "atom_scores": _assert_replay_scores_equal(
            source_artifact / "atom_scores_P3.safetensors",
            replay_artifact / "atom_scores_C3.safetensors",
            atol=replay_metric_atol,
            rtol=replay_score_rtol,
        ),
        "global_metrics": {},
    }
    for field in ("calibration_capture", "audit_capture"):
        source_value = float(source["diagnostics"]["P3"][field])
        replay_value = float(replay["diagnostics"]["C3"][field])
        delta = replay_value - source_value
        if not math.isfinite(delta) or abs(delta) > replay_metric_atol:
            raise ValueError(f"P3/C3 replay invariant failed for {field}: {delta}")
        replay_invariants["global_metrics"][field] = {
            "source": source_value,
            "replay": replay_value,
            "delta": delta,
        }

    candidates: dict[str, dict[str, Any]] = {}
    admissible: list[str] = []
    for method, baseline_method in COMPARISONS.items():
        item = replay["diagnostics"][method]
        baseline = source["diagnostics"][baseline_method]
        calibration = float(item["calibration_capture"])
        audit = float(item["audit_capture"])
        overlap = float(item["prompt_split_overlap"]["mean_cosine"])
        baseline_calibration = float(baseline["calibration_capture"])
        baseline_audit = float(baseline["audit_capture"])
        baseline_overlap = float(baseline["prompt_split_overlap"]["mean_cosine"])
        finite = all(
            math.isfinite(value)
            for value in (
                calibration,
                audit,
                overlap,
                baseline_calibration,
                baseline_audit,
                baseline_overlap,
            )
        )
        is_admissible = (
            finite
            and calibration >= baseline_calibration - capture_tolerance
            and overlap >= baseline_overlap - overlap_tolerance
        )
        candidates[method] = {
            "matched_standard_method": baseline_method,
            "calibration_capture": calibration,
            "audit_capture": audit,
            "prompt_split_mean_cosine": overlap,
            "calibration_delta_vs_standard": calibration - baseline_calibration,
            "audit_delta_vs_standard": audit - baseline_audit,
            "prompt_split_delta_vs_standard": overlap - baseline_overlap,
            "calibration_admissible": is_admissible,
        }
        if is_admissible:
            admissible.append(method)

    selected = None
    if admissible:
        best = max(candidates[method]["calibration_capture"] for method in admissible)
        tied = [
            method
            for method in admissible
            if best - candidates[method]["calibration_capture"] <= tie_tolerance
        ]
        selected = "C0" if "C0" in tied else tied[0]

    if selected is None:
        decision = "no-go"
        reason = "no cross-fit candidate passed the calibration admissibility screen"
    else:
        selected_result = candidates[selected]
        passed = (
            selected_result["calibration_delta_vs_standard"] >= minimum_improvement
            and selected_result["audit_delta_vs_standard"] >= minimum_improvement
        )
        decision = "go" if passed else "no-go"
        reason = (
            "calibration-selected cross-fit candidate passed the untouched audit gate"
            if passed
            else "calibration-selected cross-fit candidate did not improve both held-out splits"
        )

    c3_vs_c2 = {
        field: float(replay["diagnostics"]["C3"][field])
        - float(replay["diagnostics"]["C2"][field])
        for field in ("calibration_capture", "audit_capture")
    }
    if not admissible:
        diagnosis = "crossfit_estimator_failure"
    elif (
        candidates["C0"]["calibration_admissible"]
        and candidates["C2"]["calibration_admissible"]
        and c3_vs_c2["calibration_capture"] < -capture_tolerance
    ):
        diagnosis = "stable_band_mask_failure"
    else:
        diagnosis = "mixed_or_inconclusive"

    return {
        "schema_version": 1,
        "method": "phase06_crossfit_mask_preregistered_gate",
        "selection_uses_audit": False,
        "thresholds": {
            "calibration_capture_tolerance": capture_tolerance,
            "prompt_split_overlap_tolerance": overlap_tolerance,
            "calibration_tie_tolerance": tie_tolerance,
            "required_calibration_and_audit_improvement": minimum_improvement,
            "replay_metric_atol": replay_metric_atol,
            "replay_candidate_atol": replay_candidate_atol,
            "replay_score_rtol": replay_score_rtol,
        },
        "C3_P3_replay_invariants": replay_invariants,
        "candidates": candidates,
        "calibration_admissible_methods": admissible,
        "calibration_selected_method": selected,
        "C3_delta_vs_C2": c3_vs_c2,
        "diagnosis": diagnosis,
        "decision": decision,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--replay-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.source_artifact, args.replay_artifact)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else args.replay_artifact.expanduser().resolve() / "phase06_gate_decision.json"
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
