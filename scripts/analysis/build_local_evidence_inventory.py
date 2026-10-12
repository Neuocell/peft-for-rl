#!/usr/bin/env python3
"""Build a machine-readable inventory of local full-benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_BENCHMARKS = (
    "aime24",
    "aime25",
    "amc23",
    "hmmt_feb",
    "math500",
    "minerva",
)
EXPECTED_COUNTS = {
    "aime24": 960,
    "aime25": 960,
    "amc23": 1280,
    "hmmt_feb": 960,
    "math500": 2000,
    "minerva": 1088,
}
INVARIANT_FIELDS = ("id", "problem", "gold_answer", "samples_per_problem")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshot(path: Path) -> dict[tuple[str, int, int], tuple[str, ...]]:
    rows: dict[tuple[str, int, int], tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["benchmark"]),
                int(row["problem_index"]),
                int(row["sample_index"]),
            )
            rows[key] = tuple(str(row.get(field)) for field in INVARIANT_FIELDS)
    return rows


def audit_records(
    path: Path, snapshot: dict[tuple[str, int, int], tuple[str, ...]]
) -> dict[str, Any]:
    keys: set[tuple[str, int, int]] = set()
    duplicate_keys = 0
    invariant_mismatches: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    line_count = 0
    protocol_values: dict[str, set[str]] = {
        field: set()
        for field in ("seed", "temperature", "top_p", "max_new_tokens", "max_model_len")
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            line_count += 1
            row = json.loads(line)
            benchmark = str(row["benchmark"])
            key = (benchmark, int(row["problem_index"]), int(row["sample_index"]))
            if key in keys:
                duplicate_keys += 1
            keys.add(key)
            benchmark_counts[benchmark] += 1
            for field in protocol_values:
                protocol_values[field].add(str(row.get(field)))
            expected = snapshot.get(key)
            if expected is not None:
                actual = tuple(str(row.get(field)) for field in INVARIANT_FIELDS)
                for index, field in enumerate(INVARIANT_FIELDS):
                    if actual[index] != expected[index]:
                        invariant_mismatches[field] += 1
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "lines": line_count,
        "unique_keys": len(keys),
        "duplicate_keys": duplicate_keys,
        "benchmark_counts": dict(sorted(benchmark_counts.items())),
        "keys_match_snapshot": keys == snapshot.keys(),
        "invariant_mismatch_counts": dict(invariant_mismatches),
        "protocol_values": {
            field: sorted(values) for field, values in protocol_values.items()
        },
    }


def load_adapter_config(manifest: dict[str, Any]) -> dict[str, Any] | None:
    adapter = manifest.get("lora_adapter")
    if not adapter:
        return None
    adapter_dir = Path(str(adapter))
    candidates = (adapter_dir / "adapter_config.json", adapter_dir.parent / "adapter_config.json")
    for path in candidates:
        try:
            if path.is_file():
                config = json.loads(path.read_text(encoding="utf-8"))
                return {
                    "path": str(path.resolve()),
                    "peft_type": config.get("peft_type"),
                    "peft_version": config.get("peft_version"),
                    "rank": config.get("r"),
                    "alpha": config.get("lora_alpha"),
                    "dropout": config.get("lora_dropout"),
                    "target_modules": sorted(config.get("target_modules") or []),
                }
        except OSError:
            # Historical manifests may preserve absolute paths from another host.
            continue
    return None


def manifest_protocol(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    if not manifests:
        return {"present": False, "fixed_eval_match": False}
    first = manifests[0]
    expected = {
        "benchmarks": list(EXPECTED_BENCHMARKS),
        "total_requests": 7248,
        "num_shards": 4,
        "seed": 42,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 32768,
        "max_model_len": 34816,
        "samples_small": 32,
        "samples_large": 4,
        "limit_per_benchmark": None,
    }
    mismatches = {
        field: {"expected": value, "actual": first.get(field)}
        for field, value in expected.items()
        if first.get(field) != value
    }
    return {
        "present": True,
        "count": len(manifests),
        "base_model": first.get("base_model"),
        "lora_adapter": first.get("lora_adapter"),
        "fixed_eval_match": len(manifests) == 4 and not mismatches,
        "mismatches": mismatches,
        "benchmark_snapshot_sha256": first.get("benchmark_snapshot_sha256"),
    }


def infer_method(name: str, manifest: dict[str, Any]) -> str:
    lowered = name.lower()
    if lowered.startswith("base_fullbench") and not manifest.get("lora_adapter"):
        return "base_model"
    for method in ("standard_lora", "gradtop", "chr_rlpo", "rlpo", "geora", "adalora", "biso", "boet", "oft", "skew", "spo"):
        if method in lowered:
            return method
    if "lora" in lowered:
        return "standard_lora"
    if "phase1_i" in lowered:
        match = re.search(r"phase1_(i\d+)", lowered)
        return match.group(1).upper() if match else "phase1"
    return "other_adapter"


def phase1_postverify_map(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    verification = root / "phase1-signal-random-v1/analysis/verification"
    for path in sorted(verification.glob("*_postverify.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        result[str(artifact.get("evaluation_name"))] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "status": artifact.get("status"),
            "records_sha256": artifact.get("artifacts", {}).get("records", {}).get("sha256"),
        }
    return result


def classify(
    name: str,
    summary: dict[str, Any],
    protocol: dict[str, Any],
    records: dict[str, Any] | None,
    postverify: dict[str, Any] | None,
    base_audit: dict[str, Any],
) -> tuple[str, list[str], str]:
    reasons = []
    if name == base_audit.get("evaluation_name"):
        return "A-reference", ["Strict base-model compatibility audit passed."], "absolute no-RL reference"
    if postverify and postverify.get("status") == "verified":
        return "A", ["Independent training-contract and full-benchmark postverification passed."], "frozen Phase-1 primary evidence"
    if summary.get("num_samples") != 7248 or summary.get("num_benchmarks") != 6:
        return "D", ["Not a complete 7,248-record six-benchmark result."], "screen, repair, or incomplete evidence"
    if name == "rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1":
        return "D", ["Known corrupted HMMT source; use the separate repaired HMMT result and historical report."], "known-damaged aggregate"
    if not protocol.get("fixed_eval_match"):
        reasons.append("Generation manifest does not match the frozen evaluation protocol.")
    if records is None:
        reasons.append("Only summary/manifests are local; paired or parser-level reanalysis is unavailable.")
        return "C", reasons, "descriptive historical result"
    if not records.get("keys_match_snapshot") or records.get("invariant_mismatch_counts"):
        reasons.append("Record request identity does not exactly match the frozen snapshot.")
        return "D", reasons, "incompatible record-level result"
    if protocol.get("fixed_eval_match"):
        reasons.append("Frozen generation protocol and request identity match, but training is outside the Phase-1 matched contract.")
        return "B", reasons, "historical sensitivity or method context"
    return "C", reasons, "descriptive historical result"


def inventory_root(
    label: str,
    root: Path,
    snapshot: dict[tuple[str, int, int], tuple[str, ...]],
    postverify_by_name: dict[str, dict[str, Any]],
    base_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = []
    for summary_path in sorted(root.glob("**/summary/*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = str(summary.get("checkpoint") or summary_path.stem)
        eval_dir = summary_path.parent.parent
        record_candidates = sorted((eval_dir / "records").glob("*.jsonl"))
        manifest_paths = sorted((eval_dir / "manifests").glob("*.json"))
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
        protocol = manifest_protocol(manifests)
        records = audit_records(record_candidates[0], snapshot) if len(record_candidates) == 1 else None
        postverify = postverify_by_name.get(name)
        grade, reasons, use = classify(name, summary, protocol, records, postverify, base_audit)
        first_manifest = manifests[0] if manifests else {}
        entries.append(
            {
                "source": label,
                "evaluation_name": name,
                "evaluation_directory": str(eval_dir.resolve()),
                "summary": {
                    "path": str(summary_path.resolve()),
                    "sha256": sha256(summary_path),
                    "num_samples": summary.get("num_samples"),
                    "num_benchmarks": summary.get("num_benchmarks"),
                    "macro_avg_at_k": summary.get("macro_avg_at_k_over_benchmarks"),
                    "benchmark_scores": {
                        row["benchmark"]: row.get("macro_avg_at_k")
                        for row in summary.get("benchmarks", [])
                    },
                },
                "method_family": infer_method(name, first_manifest),
                "training_step": int(re.search(r"step(\d+)", name).group(1)) if re.search(r"step(\d+)", name) else None,
                "evaluation_protocol": protocol,
                "records": records,
                "manifest_paths": [str(path.resolve()) for path in manifest_paths],
                "adapter_config": load_adapter_config(first_manifest),
                "postverification": postverify,
                "evidence_grade": grade,
                "grade_reasons": reasons,
                "permitted_use": use,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = load_snapshot(args.snapshot)
    base_audit_artifact = json.loads(args.base_audit.read_text(encoding="utf-8"))
    base_audit = {
        "evaluation_name": Path(base_audit_artifact["artifacts"]["evaluation_directory"]).name,
        "status": base_audit_artifact["status"],
        "path": str(args.base_audit.resolve()),
        "sha256": sha256(args.base_audit),
    }
    postverify = phase1_postverify_map(args.active_root)
    entries = inventory_root("active_repo", args.active_root, snapshot, postverify, base_audit)
    entries.extend(inventory_root("legacy_tina", args.legacy_root, snapshot, postverify, base_audit))
    grade_counts = Counter(entry["evidence_grade"] for entry in entries)
    source_counts = Counter(entry["source"] for entry in entries)
    records_counts = Counter(
        "with_records" if entry["records"] else "summary_only" for entry in entries
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "scope": "Local full-benchmark evidence inventory; grades constrain reuse and claims rather than ranking methods.",
        "grading_policy": {
            "A": "Frozen Phase-1 protocol with independent contract/full-benchmark postverification.",
            "A-reference": "Strictly audited base-model reference; no training-seed pairing role.",
            "B": "Record-level frozen evaluation compatible, but training is outside the matched Phase-1 contract.",
            "C": "Complete descriptive historical result without locally reusable record-level evidence or with a different evaluation protocol.",
            "D": "Screen, repair-only, incomplete, incompatible, or known-damaged result.",
        },
        "frozen_snapshot": {"path": str(args.snapshot.resolve()), "sha256": sha256(args.snapshot), "records": len(snapshot)},
        "base_reference_audit": base_audit,
        "counts": {
            "entries": len(entries),
            "by_source": dict(sorted(source_counts.items())),
            "by_grade": dict(sorted(grade_counts.items())),
            "by_record_availability": dict(sorted(records_counts.items())),
        },
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
