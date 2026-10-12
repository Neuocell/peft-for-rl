#!/usr/bin/env python3
"""Audit whether a legacy base-model full-benchmark run is reusable."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["benchmark"]),
        int(row["problem_index"]),
        int(row["sample_index"]),
    )


def finite_mean(values: list[Any]) -> float:
    return sum(float(value) for value in values) / len(values)


def recompute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_benchmark[str(row["benchmark"])].append(row)

    summaries = []
    for benchmark in sorted(by_benchmark):
        bench_rows = by_benchmark[benchmark]
        by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bench_rows:
            by_problem[str(row["id"])].append(row)
        problem_avg = [
            finite_mean([row["correct"] for row in group])
            for group in by_problem.values()
        ]
        problem_pass = [
            any(bool(row["correct"]) for row in group)
            for group in by_problem.values()
        ]
        summaries.append(
            {
                "benchmark": benchmark,
                "num_problems": len(by_problem),
                "num_samples": len(bench_rows),
                "samples_per_problem": max(
                    int(row["samples_per_problem"]) for row in bench_rows
                ),
                "avg_at_k": finite_mean([row["correct"] for row in bench_rows]),
                "macro_avg_at_k": finite_mean(problem_avg),
                "pass_at_k": finite_mean(problem_pass),
                "parse_rate": finite_mean(
                    [row["format_valid"] for row in bench_rows]
                ),
                "hit_max_rate": finite_mean(
                    [row["hit_max_new_tokens"] for row in bench_rows]
                ),
                "no_answer_rate": finite_mean(
                    [not row["format_valid"] for row in bench_rows]
                ),
                "length_mean": finite_mean(
                    [row["completion_length"] for row in bench_rows]
                ),
                "length_median": statistics.median(
                    [row["completion_length"] for row in bench_rows]
                ),
                "maxed_unparseable_rate": finite_mean(
                    [
                        row["hit_max_new_tokens"] and not row["format_valid"]
                        for row in bench_rows
                    ]
                ),
            }
        )
    return {
        "num_benchmarks": len(summaries),
        "num_samples": len(rows),
        "benchmarks": summaries,
        "macro_avg_at_k_over_benchmarks": finite_mean(
            [row["macro_avg_at_k"] for row in summaries]
        ),
        "macro_pass_at_k_over_benchmarks": finite_mean(
            [row["pass_at_k"] for row in summaries]
        ),
        "parse_rate_overall": finite_mean(
            [row["format_valid"] for row in rows]
        ),
        "hit_max_rate_overall": finite_mean(
            [row["hit_max_new_tokens"] for row in rows]
        ),
        "length_mean_overall": finite_mean(
            [row["completion_length"] for row in rows]
        ),
    }


def compare_summary(recomputed: dict[str, Any], stored: dict[str, Any]) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    top_fields = (
        "num_benchmarks",
        "num_samples",
        "macro_avg_at_k_over_benchmarks",
        "macro_pass_at_k_over_benchmarks",
        "parse_rate_overall",
        "hit_max_rate_overall",
        "length_mean_overall",
    )
    for field in top_fields:
        if recomputed[field] != stored.get(field):
            differences.append(
                {"scope": "overall", "field": field, "recomputed": recomputed[field], "stored": stored.get(field)}
            )
    stored_benchmarks = {
        str(row["benchmark"]): row for row in stored.get("benchmarks", [])
    }
    for row in recomputed["benchmarks"]:
        benchmark = row["benchmark"]
        stored_row = stored_benchmarks.get(benchmark, {})
        for field, value in row.items():
            if field != "benchmark" and value != stored_row.get(field):
                differences.append(
                    {"scope": benchmark, "field": field, "recomputed": value, "stored": stored_row.get(field)}
                )
    return differences


def parser_recheck(
    rows: list[dict[str, Any]],
    evaluator: Callable[[str, str], tuple[str | None, bool]] | None,
) -> dict[str, Any]:
    if evaluator is None:
        return {"performed": False}
    mismatches: Counter[str] = Counter()
    examples = []
    for line_number, row in enumerate(rows, start=1):
        extracted, correct = evaluator(row["completion"], row["gold_answer"])
        values = {
            "extracted_answer": extracted,
            "correct": bool(correct),
            "format_valid": extracted is not None,
        }
        for field, value in values.items():
            if row.get(field) != value:
                mismatches[field] += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "line": line_number,
                            "field": field,
                            "stored": row.get(field),
                            "recomputed": value,
                        }
                    )
    return {
        "performed": True,
        "records": len(rows),
        "mismatch_counts": dict(mismatches),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expected-base-model-sha256", required=True)
    parser.add_argument("--legacy-evaluator", type=Path, required=True)
    parser.add_argument("--active-evaluator", type=Path, required=True)
    parser.add_argument("--legacy-parser", type=Path, required=True)
    parser.add_argument("--active-parser", type=Path, required=True)
    parser.add_argument("--recheck-parser", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_paths = sorted((args.eval_dir / "summary").glob("*.json"))
    records_paths = sorted((args.eval_dir / "records").glob("*.jsonl"))
    shard_paths = sorted((args.eval_dir / "shards").glob("*.jsonl"))
    manifest_paths = sorted((args.eval_dir / "manifests").glob("*.json"))
    if not (len(summary_paths) == len(records_paths) == 1):
        raise ValueError("Expected exactly one summary and one merged records file")
    if len(shard_paths) != 4 or len(manifest_paths) != 4:
        raise ValueError("Expected exactly four shards and four manifests")

    summary_path = summary_paths[0]
    records_path = records_paths[0]
    stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(records_path)
    snapshot_rows = read_jsonl(args.snapshot)
    records_by_key = {record_key(row): row for row in rows}
    snapshot_by_key = {record_key(row): row for row in snapshot_rows}

    invariant_mismatches: Counter[str] = Counter()
    for key in records_by_key.keys() & snapshot_by_key.keys():
        for field in INVARIANT_FIELDS:
            if str(records_by_key[key].get(field)) != str(snapshot_by_key[key].get(field)):
                invariant_mismatches[field] += 1

    shard_rows = [row for path in shard_paths for row in read_jsonl(path)]
    shard_by_key = {record_key(row): row for row in shard_rows}
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    fixed_manifest_fields = {
        "benchmarks": list(EXPECTED_BENCHMARKS),
        "total_requests": 7248,
        "shard_requests": 1812,
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
    manifest_mismatches = []
    for index, manifest in enumerate(manifests):
        for field, expected in fixed_manifest_fields.items():
            if manifest.get(field) != expected:
                manifest_mismatches.append(
                    {"shard": index, "field": field, "expected": expected, "actual": manifest.get(field)}
                )
        if manifest.get("shard_index") != index:
            manifest_mismatches.append(
                {"shard": index, "field": "shard_index", "expected": index, "actual": manifest.get("shard_index")}
            )
        if manifest.get("lora_adapter") is not None:
            manifest_mismatches.append(
                {"shard": index, "field": "lora_adapter", "expected": None, "actual": manifest.get("lora_adapter")}
            )

    evaluator = None
    if args.recheck_parser:
        from tina.analysis.rollout_utils import evaluate_math_answer

        evaluator = evaluate_math_answer
    parser_check = parser_recheck(rows, evaluator)
    recomputed = recompute_summary(rows)
    summary_differences = compare_summary(recomputed, stored_summary)
    base_model_sha256 = sha256(args.base_model)
    result = {
        "schema_version": 1,
        "status": "verified_reusable_base_reference",
        "scope": "Read-only compatibility audit of a legacy base-model full-benchmark run against the frozen Phase-1 evaluation protocol.",
        "artifacts": {
            "evaluation_directory": str(args.eval_dir.resolve()),
            "records": {"path": str(records_path.resolve()), "sha256": sha256(records_path)},
            "summary": {"path": str(summary_path.resolve()), "sha256": sha256(summary_path)},
            "shards": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in shard_paths],
            "manifests": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in manifest_paths],
            "snapshot": {"path": str(args.snapshot.resolve()), "sha256": sha256(args.snapshot)},
            "base_model": {"path": str(args.base_model.resolve()), "sha256": base_model_sha256},
        },
        "checks": {
            "base_model_hash_matches": base_model_sha256 == args.expected_base_model_sha256,
            "merged_records": len(rows),
            "unique_merged_record_keys": len(records_by_key),
            "snapshot_records": len(snapshot_rows),
            "unique_snapshot_record_keys": len(snapshot_by_key),
            "record_keys_exact_match_snapshot": records_by_key.keys() == snapshot_by_key.keys(),
            "invariant_mismatch_counts": dict(invariant_mismatches),
            "benchmark_record_counts": dict(sorted(Counter(str(row["benchmark"]) for row in rows).items())),
            "benchmark_record_counts_match": Counter(str(row["benchmark"]) for row in rows) == Counter(EXPECTED_COUNTS),
            "manifest_mismatches": manifest_mismatches,
            "shard_record_counts": [len(read_jsonl(path)) for path in shard_paths],
            "shard_union_keys_match_merged": shard_by_key.keys() == records_by_key.keys(),
            "shard_rows_exactly_match_merged": all(shard_by_key[key] == records_by_key[key] for key in records_by_key),
            "summary_recompute_differences": summary_differences,
            "parser_code_hash_match": sha256(args.legacy_parser) == sha256(args.active_parser),
            "parser_recheck": parser_check,
        },
        "code_provenance": {
            "legacy_evaluator_sha256": sha256(args.legacy_evaluator),
            "active_evaluator_sha256": sha256(args.active_evaluator),
            "evaluator_difference_scope": "The active evaluator adds immutable snapshot loading and safer output handling; prompt construction, generation parameters, answer scoring call, record schema, and aggregation formulas are unchanged.",
            "legacy_parser_sha256": sha256(args.legacy_parser),
            "active_parser_sha256": sha256(args.active_parser),
        },
        "result": {
            "macro_avg_at_k": stored_summary["macro_avg_at_k_over_benchmarks"],
            "reuse_decision": "reuse_without_rerun",
            "role": "absolute_no_rl_reference",
            "not_valid_for": "a paired training-seed comparison",
        },
    }
    required_checks = (
        result["checks"]["base_model_hash_matches"],
        result["checks"]["merged_records"] == 7248,
        result["checks"]["unique_merged_record_keys"] == 7248,
        result["checks"]["record_keys_exact_match_snapshot"],
        not result["checks"]["invariant_mismatch_counts"],
        result["checks"]["benchmark_record_counts_match"],
        not result["checks"]["manifest_mismatches"],
        result["checks"]["shard_record_counts"] == [1812, 1812, 1812, 1812],
        result["checks"]["shard_union_keys_match_merged"],
        result["checks"]["shard_rows_exactly_match_merged"],
        not result["checks"]["summary_recompute_differences"],
        result["checks"]["parser_code_hash_match"],
        not parser_check.get("mismatch_counts", {}),
    )
    if not all(required_checks):
        result["status"] = "incompatible_or_incomplete"
        result["result"]["reuse_decision"] = "do_not_reuse"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
