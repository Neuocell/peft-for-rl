#!/usr/bin/env python3
"""Strict structural audit for a completed Phase-0.6 cache replay."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from verl.utils.full_gradient_rl_probe import validate_full_gradient_probe_artifact


METHODS = ("C0", "C2", "C3")
REPLAY_METHOD_MASKS = {
    "C0": "none",
    "C2": "top_surprisal",
    "C3": "advantage_entropy_stable_band",
}


def _finite_count(value: Any, path: str = "root") -> int:
    if isinstance(value, dict):
        return sum(_finite_count(item, f"{path}.{key}") for key, item in value.items())
    if isinstance(value, list):
        return sum(_finite_count(item, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {path}: {value!r}")
        return 1
    return 0


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit_phase06_artifact(
    source_artifact: Path,
    replay_artifact: Path,
    *,
    expected_discovery: int = 64,
    expected_calibration: int = 32,
    expected_audit: int = 16,
    expected_modules: int = 196,
    expected_rank: int = 32,
) -> dict[str, Any]:
    source_artifact = source_artifact.expanduser().resolve()
    replay_artifact = replay_artifact.expanduser().resolve()
    source = json.loads((source_artifact / "probe_summary.json").read_text(encoding="utf-8"))
    replay = json.loads((replay_artifact / "probe_summary.json").read_text(encoding="utf-8"))
    source_validation = json.loads(
        (source_artifact / "artifact_validation.json").read_text(encoding="utf-8")
    )
    phase05_gate = json.loads(
        (source_artifact / "phase05_gate_decision.json").read_text(encoding="utf-8")
    )

    if source_validation.get("status") != "valid":
        raise ValueError("Phase 0.6 source did not pass the strict Phase-0.5 audit")
    if phase05_gate.get("decision") != "no-go":
        raise ValueError("Phase 0.6 is permitted only after a Phase-0.5 no-go decision")
    if int(replay.get("schema_version", -1)) != 1:
        raise ValueError(f"Unexpected Phase-0.6 schema: {replay.get('schema_version')}")
    if replay.get("method") != "phase06_sequential_crossfit_cache_replay_v1":
        raise ValueError(f"Unexpected Phase-0.6 method: {replay.get('method')!r}")
    if tuple(replay.get("candidate_methods", ())) != METHODS:
        raise ValueError("Phase-0.6 candidate method set or order differs")
    if replay.get("replay_method_masks") != REPLAY_METHOD_MASKS:
        raise ValueError("Phase-0.6 replay method/mask mapping differs")
    if replay.get("score_labels") != ["F", "P", "U"]:
        raise ValueError("Phase-0.6 score labels differ")
    if float(replay.get("constant_scaling", float("nan"))) != 2.0:
        raise ValueError("Phase-0.6 constant scaling differs")
    if replay.get("selection_uses_audit") is not False:
        raise ValueError("Phase-0.6 candidate construction used audit data")
    if replay.get("shared_rollouts_across_selectors") is not True:
        raise ValueError("Phase-0.6 selectors did not share rollout caches")
    if replay.get("held_out_scoring_space") != "replayed_unmasked_full_prompt_group_policy_gradient":
        raise ValueError("Phase-0.6 held-out scoring space differs")
    if Path(str(replay.get("source_artifact", ""))).expanduser().resolve() != source_artifact:
        raise ValueError("Phase-0.6 summary points to a different source artifact")

    expected_counts = {
        "discovery_prompts": expected_discovery,
        "calibration_prompts": expected_calibration,
        "audit_prompts": expected_audit,
    }
    for field, expected in expected_counts.items():
        if int(replay.get(field, -1)) != expected:
            raise ValueError(f"{field} differs: {replay.get(field)} != {expected}")
        if int(source.get(field, -1)) != expected:
            raise ValueError(f"Source {field} differs: {source.get(field)} != {expected}")

    split_fields = (
        "discovery_prompt_ids",
        "calibration_prompt_ids",
        "audit_prompt_ids",
    )
    for field in split_fields:
        if replay.get(field) != source.get(field):
            raise ValueError(f"Replayed {field} differs from the source")
    all_ids = [str(value) for field in split_fields for value in replay[field]]
    if len(all_ids) != expected_discovery + expected_calibration + expected_audit:
        raise ValueError("Phase-0.6 prompt manifest has the wrong length")
    if len(set(all_ids)) != len(all_ids) or replay.get("prompt_splits_disjoint") is not True:
        raise ValueError("Phase-0.6 prompt splits are not unique and disjoint")
    if replay.get("source_prompt_ids_sha256") != replay.get("replayed_prompt_ids_sha256"):
        raise ValueError("Phase-0.6 prompt manifest hashes differ")
    if replay.get("rollout_cache") != source.get("rollout_cache"):
        raise ValueError("Phase-0.6 did not preserve the exact source cache manifest")

    if int(replay.get("r_max", -1)) != expected_rank:
        raise ValueError("Phase-0.6 rank differs")
    if int(replay.get("r_max", -1)) != int(source.get("r_max", -2)):
        raise ValueError("Phase-0.6 rank differs from the source")
    if int(replay.get("sketch_width", -1)) != int(source.get("sketch_width", -2)):
        raise ValueError("Phase-0.6 sketch width differs from the source")
    if int(replay.get("crossfit_splits", -1)) != int(source.get("crossfit_splits", -2)):
        raise ValueError("Phase-0.6 cross-fit split count differs from the source")
    if replay.get("token_mask") != source.get("token_mask"):
        raise ValueError("Phase-0.6 token-mask configuration differs from the source")
    if replay.get("token_mask", {}).get("calibration_audit_mask") != "none":
        raise ValueError("Phase-0.6 calibration/audit gradients must be unmasked")

    modules = replay.get("modules", {})
    source_modules = source.get("modules", {})
    if len(modules) != expected_modules or set(modules) != set(source_modules):
        raise ValueError("Phase-0.6 module set differs")
    for name, item in modules.items():
        if item.get("shape") != source_modules[name].get("shape"):
            raise ValueError(f"Module shape differs for {name}")
        if int(item.get("candidate_rank", -1)) != expected_rank:
            raise ValueError(f"Candidate rank differs for {name}")
        if set(item.get("diagnostics", {})) != set(METHODS):
            raise ValueError(f"Module diagnostics differ for {name}")

    method_validation = {
        method: validate_full_gradient_probe_artifact(
            replay_artifact, candidate_method=method
        )
        for method in METHODS
    }
    finite_values = _finite_count(
        {
            "diagnostics": replay.get("diagnostics", {}),
            "modules": modules,
            "samples": replay.get("samples", {}),
        }
    )
    report = {
        "schema_version": 1,
        "status": "valid",
        "source_artifact": str(source_artifact),
        "replay_artifact": str(replay_artifact),
        "phase05_gate_decision": "no-go",
        "split_counts": {
            "discovery": expected_discovery,
            "calibration": expected_calibration,
            "audit": expected_audit,
        },
        "candidate_methods": list(METHODS),
        "module_count": len(modules),
        "method_validation": method_validation,
        "prompt_ids_exactly_replayed": True,
        "cache_manifest_exactly_replayed": True,
        "finite_numeric_values_checked": finite_values,
    }
    _write_json_atomic(replay_artifact / "artifact_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--replay-artifact", type=Path, required=True)
    parser.add_argument("--expected-discovery", type=int, default=64)
    parser.add_argument("--expected-calibration", type=int, default=32)
    parser.add_argument("--expected-audit", type=int, default=16)
    parser.add_argument("--expected-modules", type=int, default=196)
    parser.add_argument("--expected-rank", type=int, default=32)
    args = parser.parse_args()
    result = audit_phase06_artifact(
        args.source_artifact,
        args.replay_artifact,
        expected_discovery=args.expected_discovery,
        expected_calibration=args.expected_calibration,
        expected_audit=args.expected_audit,
        expected_modules=args.expected_modules,
        expected_rank=args.expected_rank,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
