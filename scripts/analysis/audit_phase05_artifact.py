#!/usr/bin/env python3
"""Strict independent audit for a completed formal Phase-0.5 artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from scripts.analysis.evaluate_phase05_hybrid_gate import assert_tensor_files_equal
from verl.utils.full_gradient_rl_probe import validate_full_gradient_probe_artifact


FORMAL_METHODS = (
    "P0",
    "P0_uncentered",
    "P1",
    "P1_uncentered",
    "P2",
    "P2_uncentered",
    "P3",
    "H0",
    "H2",
    "H4",
    "H8",
    "H16",
)
FORMAL_CACHE_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "advantages",
        "old_log_probs",
        "token_level_scores",
    }
)


def _assert_finite_numbers(value: Any, path: str = "root") -> int:
    count = 0
    if isinstance(value, dict):
        for key, item in value.items():
            count += _assert_finite_numbers(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            count += _assert_finite_numbers(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite numeric value at {path}: {value!r}")
        count += 1
    return count


def _phase_for_ordinal(
    ordinal: int, discovery: int, calibration: int, audit: int
) -> str:
    if ordinal < discovery:
        return "discovery"
    if ordinal < discovery + calibration:
        return "calibration"
    if ordinal < discovery + calibration + audit:
        return "audit"
    raise ValueError(f"Cache ordinal is outside the expected split sizes: {ordinal}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def audit_phase05_artifact(
    artifact_dir: Path,
    *,
    expected_discovery: int = 64,
    expected_calibration: int = 32,
    expected_audit: int = 16,
    expected_modules: int = 196,
    expected_rank: int = 32,
    expected_methods: tuple[str, ...] = FORMAL_METHODS,
    required_cache_keys: frozenset[str] = FORMAL_CACHE_KEYS,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve()
    summary_path = artifact_dir / "probe_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_total = expected_discovery + expected_calibration + expected_audit

    if int(summary.get("schema_version", -1)) != 4:
        raise ValueError(f"Expected Phase-0 schema 4, got {summary.get('schema_version')}")
    expected_counts = {
        "discovery_prompts": expected_discovery,
        "calibration_prompts": expected_calibration,
        "audit_prompts": expected_audit,
    }
    for field, expected in expected_counts.items():
        actual = int(summary.get(field, -1))
        if actual != expected:
            raise ValueError(f"{field} differs: {actual} != {expected}")
    if not summary.get("prompt_splits_disjoint", False):
        raise ValueError("Summary does not declare disjoint prompt splits")
    if not summary.get("shared_rollouts_across_selectors", False):
        raise ValueError("Selectors did not use shared rollouts")
    if summary.get("selection_uses_audit") is not False:
        raise ValueError("Candidate construction or selection used audit data")

    split_ids = {
        "discovery": [str(value) for value in summary["discovery_prompt_ids"]],
        "calibration": [str(value) for value in summary["calibration_prompt_ids"]],
        "audit": [str(value) for value in summary["audit_prompt_ids"]],
    }
    for phase, expected in (
        ("discovery", expected_discovery),
        ("calibration", expected_calibration),
        ("audit", expected_audit),
    ):
        if len(split_ids[phase]) != expected:
            raise ValueError(
                f"{phase} prompt ID count differs: {len(split_ids[phase])} != {expected}"
            )
        if len(set(split_ids[phase])) != expected:
            raise ValueError(f"{phase} prompt IDs contain duplicates")
    all_prompt_ids = split_ids["discovery"] + split_ids["calibration"] + split_ids["audit"]
    if len(set(all_prompt_ids)) != expected_total:
        raise ValueError("Prompt splits overlap despite the summary flag")

    methods = tuple(summary.get("candidate_methods", ()))
    if methods != expected_methods:
        raise ValueError(f"Candidate methods differ: {methods} != {expected_methods}")
    modules = summary.get("modules", {})
    if len(modules) != expected_modules:
        raise ValueError(f"Module count differs: {len(modules)} != {expected_modules}")
    if int(summary.get("r_max", -1)) != expected_rank:
        raise ValueError(f"Candidate rank differs: {summary.get('r_max')} != {expected_rank}")

    samples = summary.get("samples", {})
    response_counts = samples.get("response_counts", [])
    response_tokens = samples.get("response_tokens", [])
    advantage_rms = samples.get("advantage_rms", [])
    for label, values in (
        ("response_counts", response_counts),
        ("response_tokens", response_tokens),
        ("advantage_rms", advantage_rms),
    ):
        if len(values) != expected_total:
            raise ValueError(f"samples.{label} has {len(values)} != {expected_total} entries")
    if any(int(value) <= 0 for value in response_counts):
        raise ValueError("Every selected prompt must contain at least one response")
    if any(int(value) <= 0 for value in response_tokens):
        raise ValueError("Every selected prompt must contain at least one response token")
    if any(float(value) <= 0 for value in advantage_rms):
        raise ValueError("Every selected prompt must have positive advantage RMS")

    rollout_cache = summary.get("rollout_cache", [])
    if len(rollout_cache) != expected_total:
        raise ValueError(
            f"Rollout cache manifest has {len(rollout_cache)} != {expected_total} entries"
        )
    cache_dir = (artifact_dir / "rollout_cache").resolve()
    manifest_paths: set[Path] = set()
    cached_tensor_count = 0
    for ordinal, entry in enumerate(rollout_cache):
        if int(entry.get("ordinal", -1)) != ordinal:
            raise ValueError(f"Cache ordinal mismatch at index {ordinal}: {entry.get('ordinal')}")
        expected_phase = _phase_for_ordinal(
            ordinal, expected_discovery, expected_calibration, expected_audit
        )
        if entry.get("phase") != expected_phase:
            raise ValueError(
                f"Cache phase mismatch at ordinal {ordinal}: {entry.get('phase')} != {expected_phase}"
            )
        if str(entry.get("prompt_id")) != all_prompt_ids[ordinal]:
            raise ValueError(f"Cache prompt ID mismatch at ordinal {ordinal}")
        path = Path(entry["path"]).expanduser().resolve()
        if path.parent != cache_dir:
            raise ValueError(f"Cache path escapes the artifact cache directory: {path}")
        if path in manifest_paths:
            raise ValueError(f"Duplicate cache path in manifest: {path}")
        manifest_paths.add(path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty cache file: {path}")
        expected_prefix = f"{ordinal:04d}_{expected_phase}_"
        if not path.name.startswith(expected_prefix) or path.suffix != ".safetensors":
            raise ValueError(f"Malformed cache filename: {path.name}")
        manifest_keys = frozenset(str(key) for key in entry.get("tensor_keys", []))
        if manifest_keys != required_cache_keys:
            raise ValueError(
                f"Cache tensor manifest differs at ordinal {ordinal}: "
                f"{sorted(manifest_keys)} != {sorted(required_cache_keys)}"
            )
        with safe_open(path, framework="pt", device="cpu") as cache_file:
            file_keys = frozenset(cache_file.keys())
            if file_keys != required_cache_keys:
                raise ValueError(f"Cache tensor keys differ at ordinal {ordinal}")
            for key in sorted(file_keys):
                tensor = cache_file.get_tensor(key)
                if tensor.ndim == 0 or tensor.shape[0] != int(response_counts[ordinal]):
                    raise ValueError(
                        f"Cache {ordinal} tensor {key} has incompatible shape {tuple(tensor.shape)}"
                    )
                if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"Cache {ordinal} tensor {key} is non-finite")
                cached_tensor_count += 1
    disk_paths = {
        path.resolve() for path in cache_dir.glob("*.safetensors") if path.is_file()
    }
    if disk_paths != manifest_paths:
        missing = sorted(str(path) for path in manifest_paths - disk_paths)
        extra = sorted(str(path) for path in disk_paths - manifest_paths)
        raise ValueError(f"Cache manifest/files differ: missing={missing[:3]}, extra={extra[:3]}")

    method_validation: dict[str, dict[str, float]] = {}
    for method in methods:
        result = validate_full_gradient_probe_artifact(
            artifact_dir, candidate_method=method
        )
        if int(result["module_count"]) != expected_modules:
            raise ValueError(
                f"{method} candidate module count differs: {result['module_count']}"
            )
        method_validation[method] = result

    h0_p2 = {
        "candidates": assert_tensor_files_equal(
            artifact_dir / "candidates_P2.safetensors",
            artifact_dir / "candidates_H0.safetensors",
        ),
        "atom_scores": assert_tensor_files_equal(
            artifact_dir / "atom_scores_P2.safetensors",
            artifact_dir / "atom_scores_H0.safetensors",
        ),
    }
    for field in ("calibration_capture", "audit_capture"):
        if summary["diagnostics"]["P2"][field] != summary["diagnostics"]["H0"][field]:
            raise ValueError(f"H0/P2 global {field} differs")

    hybrid_diagnostics: dict[str, dict[str, float]] = {}
    for method in (method for method in methods if method.startswith("H")):
        requested = int(method[1:])
        items = [module["diagnostics"][method] for module in modules.values()]
        for item in items:
            selected = float(item["selected_p3_directions"])
            supported = float(item["p3_consensus_supported_rank"])
            fallback = float(item["fallback_p2_directions"])
            if not 0 <= selected <= min(requested, supported, expected_rank):
                raise ValueError(f"Invalid selected P3 directions for {method}: {selected}")
            if not math.isclose(selected + fallback, float(expected_rank), abs_tol=1e-9):
                raise ValueError(
                    f"Hybrid rank does not sum to {expected_rank} for {method}"
                )
        hybrid_diagnostics[method] = {
            "mean_selected_p3_directions": float(
                sum(float(item["selected_p3_directions"]) for item in items) / len(items)
            ),
            "mean_p3_consensus_supported_rank": float(
                sum(float(item["p3_consensus_supported_rank"]) for item in items)
                / len(items)
            ),
        }

    finite_numeric_values = _assert_finite_numbers(
        {
            "diagnostics": summary["diagnostics"],
            "module_diagnostics": {
                name: item["diagnostics"] for name, item in modules.items()
            },
            "samples": samples,
        }
    )
    report = {
        "schema_version": 1,
        "status": "valid",
        "artifact_dir": str(artifact_dir),
        "summary_schema_version": int(summary["schema_version"]),
        "split_counts": {
            "discovery": expected_discovery,
            "calibration": expected_calibration,
            "audit": expected_audit,
            "total": expected_total,
        },
        "prompt_ids_unique_and_disjoint": True,
        "rollout_cache": {
            "manifest_entries": len(rollout_cache),
            "files": len(disk_paths),
            "validated_tensors": cached_tensor_count,
        },
        "candidate_methods": list(methods),
        "module_count": len(modules),
        "method_validation": method_validation,
        "H0_P2_exact_tensor_invariants": h0_p2,
        "hybrid_diagnostics": hybrid_diagnostics,
        "finite_numeric_values_checked": finite_numeric_values,
    }
    _write_json_atomic(artifact_dir / "artifact_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-discovery", type=int, default=64)
    parser.add_argument("--expected-calibration", type=int, default=32)
    parser.add_argument("--expected-audit", type=int, default=16)
    parser.add_argument("--expected-modules", type=int, default=196)
    parser.add_argument("--expected-rank", type=int, default=32)
    args = parser.parse_args()
    result = audit_phase05_artifact(
        args.artifact_dir,
        expected_discovery=args.expected_discovery,
        expected_calibration=args.expected_calibration,
        expected_audit=args.expected_audit,
        expected_modules=args.expected_modules,
        expected_rank=args.expected_rank,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
