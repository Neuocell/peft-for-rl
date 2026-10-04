#!/usr/bin/env python3
"""Apply the preregistered Phase-0.5 hybrid selection and audit gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


NONZERO_METHODS = ("H2", "H4", "H8", "H16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-tolerance", type=float, default=0.002)
    parser.add_argument("--overlap-tolerance", type=float, default=0.02)
    parser.add_argument("--tie-tolerance", type=float, default=5e-4)
    parser.add_argument("--minimum-improvement", type=float, default=0.001)
    return parser.parse_args()


def assert_tensor_files_equal(first: Path, second: Path) -> dict[str, Any]:
    with safe_open(first, framework="pt", device="cpu") as first_file, safe_open(
        second, framework="pt", device="cpu"
    ) as second_file:
        first_keys = list(first_file.keys())
        second_keys = list(second_file.keys())
        if first_keys != second_keys:
            raise ValueError(f"Tensor keys differ: {first.name} vs {second.name}")
        for key in first_keys:
            if not torch.equal(first_file.get_tensor(key), second_file.get_tensor(key)):
                raise ValueError(
                    f"H0/P2 invariant failed for {key}: {first.name} vs {second.name}"
                )
    return {"equal": True, "tensor_count": len(first_keys)}


def evaluate(
    artifact_dir: Path,
    *,
    capture_tolerance: float,
    overlap_tolerance: float,
    tie_tolerance: float,
    minimum_improvement: float,
) -> dict[str, Any]:
    summary_path = artifact_dir / "probe_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    diagnostics = summary["diagnostics"]
    required = {"P2", "H0", *NONZERO_METHODS}
    missing = sorted(required - diagnostics.keys())
    if missing:
        raise ValueError(f"Phase-0.5 summary is missing candidates: {missing}")

    tensor_invariants = {
        "candidates": assert_tensor_files_equal(
            artifact_dir / "candidates_P2.safetensors",
            artifact_dir / "candidates_H0.safetensors",
        ),
        "atom_scores": assert_tensor_files_equal(
            artifact_dir / "atom_scores_P2.safetensors",
            artifact_dir / "atom_scores_H0.safetensors",
        ),
    }
    numeric_invariants = {}
    for field in ("calibration_capture", "audit_capture"):
        p2_value = float(diagnostics["P2"][field])
        h0_value = float(diagnostics["H0"][field])
        if p2_value != h0_value:
            raise ValueError(f"H0/P2 {field} differs: {h0_value} vs {p2_value}")
        numeric_invariants[field] = {"equal": True, "value": p2_value}

    baseline_capture = float(diagnostics["H0"]["calibration_capture"])
    baseline_overlap = float(
        diagnostics["H0"]["prompt_split_overlap"]["mean_cosine"]
    )
    candidates = {}
    admissible = []
    for method in NONZERO_METHODS:
        calibration = float(diagnostics[method]["calibration_capture"])
        audit = float(diagnostics[method]["audit_capture"])
        overlap = float(
            diagnostics[method]["prompt_split_overlap"]["mean_cosine"]
        )
        is_admissible = (
            calibration >= baseline_capture - capture_tolerance
            and overlap >= baseline_overlap - overlap_tolerance
        )
        candidates[method] = {
            "requested_p3_directions": int(method[1:]),
            "calibration_capture": calibration,
            "audit_capture": audit,
            "prompt_split_mean_cosine": overlap,
            "calibration_delta_vs_H0": calibration - baseline_capture,
            "audit_delta_vs_H0": audit
            - float(diagnostics["H0"]["audit_capture"]),
            "prompt_split_delta_vs_H0": overlap - baseline_overlap,
            "calibration_admissible": is_admissible,
        }
        if is_admissible:
            admissible.append(method)

    selected = None
    if admissible:
        best_calibration = max(
            candidates[method]["calibration_capture"] for method in admissible
        )
        tied = [
            method
            for method in admissible
            if best_calibration - candidates[method]["calibration_capture"]
            <= tie_tolerance
        ]
        selected = min(tied, key=lambda method: int(method[1:]))

    if selected is None:
        decision = "no-go"
        reason = "no nonzero hybrid passed the calibration admissibility screen"
    else:
        selected_result = candidates[selected]
        calibration_pass = (
            selected_result["calibration_delta_vs_H0"] >= minimum_improvement
        )
        audit_pass = selected_result["audit_delta_vs_H0"] >= minimum_improvement
        finite = torch.isfinite(
            torch.tensor(
                [
                    selected_result["calibration_capture"],
                    selected_result["audit_capture"],
                ]
            )
        ).all()
        decision = "go" if calibration_pass and audit_pass and bool(finite) else "no-go"
        reason = (
            "selected hybrid passed calibration selection and the untouched audit gate"
            if decision == "go"
            else "selected hybrid did not clear both preregistered capture improvements"
        )

    return {
        "schema_version": 1,
        "method": "phase05_stability_supported_hybrid_preregistered_gate",
        "selection_uses_audit": False,
        "thresholds": {
            "calibration_capture_tolerance": capture_tolerance,
            "prompt_split_overlap_tolerance": overlap_tolerance,
            "calibration_tie_tolerance": tie_tolerance,
            "required_calibration_and_audit_improvement": minimum_improvement,
        },
        "H0_P2_invariants": {
            "tensor_files": tensor_invariants,
            "global_capture": numeric_invariants,
        },
        "baseline": {
            "method": "H0",
            "calibration_capture": baseline_capture,
            "audit_capture": float(diagnostics["H0"]["audit_capture"]),
            "prompt_split_mean_cosine": baseline_overlap,
        },
        "candidates": candidates,
        "calibration_admissible_methods": admissible,
        "calibration_selected_method": selected,
        "decision": decision,
        "reason": reason,
    }


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else artifact_dir / "phase05_gate_decision.json"
    )
    result = evaluate(
        artifact_dir,
        capture_tolerance=args.capture_tolerance,
        overlap_tolerance=args.overlap_tolerance,
        tie_tolerance=args.tie_tolerance,
        minimum_improvement=args.minimum_improvement,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
