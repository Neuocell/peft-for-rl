#!/usr/bin/env python3
"""Verify that a reusable full-benchmark run matches the fixed protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
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
EXPECTED_BENCHMARK_SIZES = {
    "aime24": 30,
    "aime25": 30,
    "amc23": 40,
    "hmmt_feb": 30,
    "math500": 500,
    "minerva": 272,
}
SMALL_BENCHMARKS = {"aime24", "aime25", "amc23", "hmmt_feb"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(value: Any) -> Path:
    return Path(str(value or "")).expanduser().resolve()


def verify_full_bench_run(
    summary_path: Path,
    records_path: Path,
    eval_dir: Path,
    eval_name: str,
    adapter_dir: Path,
    base_model: Path,
    snapshot_path: Path,
    snapshot_sha256: str,
    *,
    expected_samples: int = 7248,
    expected_benchmarks: tuple[str, ...] = EXPECTED_BENCHMARKS,
    expected_benchmark_sizes: dict[str, int] = EXPECTED_BENCHMARK_SIZES,
    expected_shards: int = 4,
    expected_seed: int = 42,
    expected_temperature: float = 0.6,
    expected_top_p: float = 0.95,
    expected_max_new_tokens: int = 32768,
    expected_max_model_len: int = 34816,
    expected_samples_small: int = 32,
    expected_samples_large: int = 4,
) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    records_path = records_path.expanduser().resolve()
    eval_dir = eval_dir.expanduser().resolve()
    adapter_dir = adapter_dir.expanduser().resolve()
    base_model = base_model.expanduser().resolve()
    snapshot_path = snapshot_path.expanduser().resolve()
    if not eval_name:
        raise ValueError("Evaluation name cannot be empty")
    for label, path in (
        ("summary", summary_path),
        ("records", records_path),
        ("snapshot", snapshot_path),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty full-benchmark {label}: {path}")
    actual_snapshot_sha256 = _sha256(snapshot_path)
    if actual_snapshot_sha256 != snapshot_sha256:
        raise ValueError(
            "Benchmark snapshot SHA-256 changed: "
            f"expected={snapshot_sha256} actual={actual_snapshot_sha256}"
        )

    expected_set = set(expected_benchmarks)
    if set(expected_benchmark_sizes) != expected_set:
        raise ValueError("Expected benchmark sizes do not match the benchmark set")
    expected_record_counts = {
        benchmark: size
        * (
            expected_samples_small
            if benchmark in SMALL_BENCHMARKS
            else expected_samples_large
        )
        for benchmark, size in expected_benchmark_sizes.items()
    }
    if sum(expected_record_counts.values()) != expected_samples:
        raise ValueError("Expected benchmark protocol does not sum to expected samples")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_rows = summary.get("benchmarks", [])
    summary_names = [row.get("benchmark") for row in summary_rows]
    if (
        summary.get("checkpoint") != eval_name
        or summary.get("num_samples") != expected_samples
        or len(summary_names) != len(expected_benchmarks)
        or set(summary_names) != expected_set
    ):
        raise ValueError("Full-benchmark summary does not match the fixed protocol")
    for row in summary_rows:
        benchmark = row["benchmark"]
        if row.get("num_problems") != expected_benchmark_sizes[benchmark]:
            raise ValueError(f"Summary problem count changed for {benchmark}")
        if row.get("num_samples") != expected_record_counts[benchmark]:
            raise ValueError(f"Summary sample count changed for {benchmark}")

    record_counts: Counter[str] = Counter()
    record_keys: set[tuple[str, int, int]] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                benchmark = str(record["benchmark"])
                key = (
                    benchmark,
                    int(record["problem_index"]),
                    int(record["sample_index"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid full-benchmark record at line {line_number}"
                ) from exc
            if benchmark not in expected_set:
                raise ValueError(f"Unexpected benchmark in records: {benchmark}")
            if record.get("checkpoint") != eval_name:
                raise ValueError(f"Record checkpoint changed at line {line_number}")
            if key in record_keys:
                raise ValueError(f"Duplicate full-benchmark record key: {key}")
            record_keys.add(key)
            record_counts[benchmark] += 1
    if len(record_keys) != expected_samples:
        raise ValueError(
            f"Incomplete full-benchmark records: samples={len(record_keys)}"
        )
    if dict(record_counts) != expected_record_counts:
        raise ValueError("Full-benchmark record counts changed")

    expected_manifests = [
        eval_dir
        / "manifests"
        / f"{eval_name}.shard-{index:02d}-of-{expected_shards:02d}.json"
        for index in range(expected_shards)
    ]
    actual_manifests = set(
        (eval_dir / "manifests").glob(f"{eval_name}.shard-*-of-*.json")
    )
    if actual_manifests != set(expected_manifests):
        missing = sorted(
            str(path) for path in set(expected_manifests) - actual_manifests
        )
        extra = sorted(str(path) for path in actual_manifests - set(expected_manifests))
        raise ValueError(
            f"Evaluation manifest set changed: missing={missing} extra={extra}"
        )

    shard_requests = 0
    for index, path in enumerate(expected_manifests):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_shard_requests = expected_samples // expected_shards + (
            index < expected_samples % expected_shards
        )
        expected_fields = {
            "checkpoint": eval_name,
            "benchmarks": list(expected_benchmarks),
            "total_requests": expected_samples,
            "shard_requests": expected_shard_requests,
            "num_shards": expected_shards,
            "shard_index": index,
            "seed": expected_seed,
            "temperature": expected_temperature,
            "top_p": expected_top_p,
            "max_new_tokens": expected_max_new_tokens,
            "max_model_len": expected_max_model_len,
            "samples_small": expected_samples_small,
            "samples_large": expected_samples_large,
            "limit_per_benchmark": None,
            "benchmark_sizes": expected_benchmark_sizes,
            "benchmark_snapshot_sha256": snapshot_sha256,
        }
        for field, expected in expected_fields.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Evaluation manifest field changed in {path}: {field}"
                )
        if _resolved(manifest.get("lora_adapter")) != adapter_dir:
            raise ValueError(f"Evaluation adapter changed in {path}")
        if _resolved(manifest.get("base_model")) != base_model:
            raise ValueError(f"Evaluation base model changed in {path}")
        if _resolved(manifest.get("benchmark_snapshot_records")) != snapshot_path:
            raise ValueError(f"Evaluation snapshot path changed in {path}")
        shard_requests += int(manifest["shard_requests"])
    if shard_requests != expected_samples:
        raise ValueError("Evaluation shard request counts do not sum to the full run")

    return {
        "status": "verified",
        "evaluation_name": eval_name,
        "num_samples": expected_samples,
        "benchmarks": list(expected_benchmarks),
        "benchmark_record_counts": expected_record_counts,
        "num_shards": expected_shards,
        "snapshot_sha256": snapshot_sha256,
        "adapter_path": str(adapter_dir),
        "base_model_path": str(base_model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    args = parser.parse_args()
    result = verify_full_bench_run(
        args.summary,
        args.records,
        args.eval_dir,
        args.eval_name,
        args.adapter,
        args.base_model,
        args.snapshot,
        args.snapshot_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
