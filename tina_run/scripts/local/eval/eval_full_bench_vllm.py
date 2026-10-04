#!/usr/bin/env python
"""Run full math benchmark evaluation with vLLM, optional LoRA, and Avg@k aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets import load_dataset  # noqa: E402
from tina.analysis.rollout_utils import evaluate_math_answer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.lora.request import LoRARequest  # noqa: E402

MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()

DEFAULT_BENCHMARKS = ["aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva"]
SMALL_BENCHMARKS = {"aime24", "aime25", "amc23", "hmmt_feb"}
BENCHMARK_ALIASES = {
    "math_500": "math500",
    "math-500": "math500",
    "hmmt": "hmmt_feb",
    "hmmt-feb": "hmmt_feb",
}
DIRECT_OPD_EVAL_ROOT = Path(os.environ.get("DIRECT_OPD_EVAL_ROOT", "/home/wangls/Direct-OPD/datasets/eval"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_adapter", default=None)
    parser.add_argument("--checkpoint_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=34816)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_lora_rank", type=int, default=32)
    parser.add_argument("--samples_small", type=int, default=32)
    parser.add_argument("--samples_large", type=int, default=4)
    parser.add_argument("--benchmark_snapshot_records", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit_per_benchmark", type=int, default=None)
    parser.add_argument("--dry_run_no_model", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def normalize_benchmark(name: str) -> str:
    key = name.strip().lower()
    return BENCHMARK_ALIASES.get(key, key)


def parse_benchmarks(value: str) -> list[str]:
    benchmarks = [normalize_benchmark(item) for item in value.split(",") if item.strip()]
    unknown = [bench for bench in benchmarks if bench not in DEFAULT_BENCHMARKS]
    if unknown:
        raise SystemExit(f"Unknown benchmark(s): {unknown}. Supported: {DEFAULT_BENCHMARKS}")
    return benchmarks


def sample_count(benchmark: str, args: argparse.Namespace) -> int:
    return args.samples_small if benchmark in SMALL_BENCHMARKS else args.samples_large


def first_message_content(value: Any) -> str:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    if hasattr(value, "tolist"):
        return first_message_content(value.tolist())
    return str(value or "")


def load_hf_rows(repo: str, subset: str, split: str, benchmark: str, answer_key: str) -> list[dict[str, Any]]:
    dataset = load_dataset(repo, subset, split=split)
    rows = []
    for idx, row in enumerate(dataset):
        problem = row.get("problem") or row.get("question") or row.get("Question")
        gold = row.get(answer_key) or row.get("answer") or row.get("solution")
        rows.append(
            {
                "id": str(row.get("unique_id") or row.get("idx") or row.get("id") or idx),
                "benchmark": benchmark,
                "problem": str(problem),
                "gold_answer": str(gold),
            }
        )
    return rows


def load_direct_opd_parquet(path: str, benchmark: str) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    flat_schema = {"problem", "answer"}.issubset(df.columns)
    direct_opd_schema = {"prompt", "reward_model"}.issubset(df.columns)
    if not flat_schema and not direct_opd_schema:
        raise ValueError(
            f"Unsupported evaluation parquet schema in {path}: columns={sorted(df.columns.tolist())}"
        )

    rows = []
    for idx, row in df.iterrows():
        if flat_schema:
            problem_value = row["problem"]
            problem = "" if problem_value is None else str(problem_value)
            gold = row["answer"]
            row_id = row.get("problem_idx", idx)
        else:
            reward_model = row.get("reward_model")
            extra_info = row.get("extra_info")
            reward_model = reward_model if isinstance(reward_model, Mapping) else {}
            extra_info = extra_info if isinstance(extra_info, Mapping) else {}
            problem = first_message_content(row.get("prompt"))
            gold = reward_model.get("ground_truth")
            row_id = extra_info.get("index", idx)

        if not problem.strip() or problem.strip().lower() in {"nan", "none"}:
            raise ValueError(f"Empty problem at row {idx} in {path}")
        if gold is None or not str(gold).strip() or str(gold).strip().lower() in {"nan", "none"}:
            raise ValueError(f"Missing ground-truth answer at row {idx} in {path}")
        rows.append(
            {
                "id": str(row_id),
                "benchmark": benchmark,
                "problem": problem,
                "gold_answer": str(gold),
            }
        )
    return rows


def load_benchmark(benchmark: str) -> list[dict[str, Any]]:
    if benchmark == "aime24":
        try:
            return load_hf_rows("HuggingFaceH4/aime_2024", "default", "train", benchmark, "answer")
        except Exception:
            return load_direct_opd_parquet(str(DIRECT_OPD_EVAL_ROOT / "aime24.parquet"), benchmark)
    if benchmark == "aime25":
        try:
            return load_hf_rows("yentinglin/aime_2025", "default", "train", benchmark, "answer")
        except Exception:
            return load_direct_opd_parquet(str(DIRECT_OPD_EVAL_ROOT / "aime25.parquet"), benchmark)
    if benchmark == "amc23":
        return load_hf_rows("knoveleng/AMC-23", "default", "train", benchmark, "answer")
    if benchmark == "hmmt_feb":
        return load_direct_opd_parquet(str(DIRECT_OPD_EVAL_ROOT / "hmmt_feb.parquet"), benchmark)
    if benchmark == "math500":
        return load_hf_rows("HuggingFaceH4/MATH-500", "default", "test", benchmark, "answer")
    if benchmark == "minerva":
        return load_hf_rows("knoveleng/Minerva-Math", "default", "train", benchmark, "solution")
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_benchmark_snapshot_records(
    path: str, benchmarks: list[str]
) -> dict[str, list[dict[str, Any]]]:
    requested = set(benchmarks)
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as records:
        for line_number, line in enumerate(records, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            benchmark = normalize_benchmark(str(record.get("benchmark", "")))
            if benchmark not in requested:
                continue
            try:
                problem_index = int(record["problem_index"])
                row = {
                    "id": str(record["id"]),
                    "benchmark": benchmark,
                    "problem": str(record["problem"]),
                    "gold_answer": str(record["gold_answer"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid benchmark snapshot record at line {line_number} in {path}"
                ) from exc
            key = (benchmark, problem_index)
            previous = by_key.setdefault(key, row)
            if previous != row:
                raise ValueError(
                    f"Conflicting benchmark snapshot records for {benchmark} problem_index={problem_index}"
                )

    rows_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for benchmark in benchmarks:
        indexed = sorted(
            (index, row)
            for (row_benchmark, index), row in by_key.items()
            if row_benchmark == benchmark
        )
        if not indexed:
            raise ValueError(f"Benchmark snapshot has no rows for {benchmark}: {path}")
        indices = [index for index, _ in indexed]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Benchmark snapshot has non-contiguous problem_index values for {benchmark}: {indices[:5]}"
            )
        rows_by_benchmark[benchmark] = [row for _, row in indexed]
    return rows_by_benchmark


def build_prompt(tokenizer: Any, problem: str) -> str:
    messages = [{"role": "user", "content": MATH_QUERY_TEMPLATE.format(Question=problem)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def expanded_requests(rows_by_benchmark: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    requests = []
    for benchmark in parse_benchmarks(args.benchmarks):
        rows = rows_by_benchmark[benchmark]
        if args.limit_per_benchmark is not None:
            rows = rows[: args.limit_per_benchmark]
        for problem_index, row in enumerate(rows):
            for sample_index in range(sample_count(benchmark, args)):
                requests.append(
                    {
                        **row,
                        "problem_index": problem_index,
                        "sample_index": sample_index,
                        "samples_per_problem": sample_count(benchmark, args),
                    }
                )
    return requests


def shard_requests(requests: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise SystemExit("--num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit("--shard_index must be in [0, num_shards)")
    return [row for pos, row in enumerate(requests) if pos % num_shards == shard_index]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def finite_mean(values: list[float | int | bool | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def finite_median(values: list[float | int | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.median(vals) if vals else None


def aggregate(output_dir: Path, checkpoint_name: str, num_shards: int) -> dict[str, Any]:
    shard_paths = [output_dir / "shards" / f"{checkpoint_name}.shard-{idx:02d}-of-{num_shards:02d}.jsonl" for idx in range(num_shards)]
    missing = [str(path) for path in shard_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing shard output(s): {missing}")

    rows = []
    for path in shard_paths:
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda r: (r["benchmark"], int(r["problem_index"]), int(r["sample_index"])))
    write_jsonl(output_dir / "records" / f"{checkpoint_name}.jsonl", rows)

    summaries = []
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)

    for benchmark in sorted(by_benchmark):
        bench_rows = by_benchmark[benchmark]
        by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bench_rows:
            by_problem[str(row["id"])].append(row)
        problem_avg = [finite_mean([r["correct"] for r in group]) for group in by_problem.values()]
        problem_pass = [any(bool(r["correct"]) for r in group) for group in by_problem.values()]
        lengths = [r["completion_length"] for r in bench_rows]
        summary = {
            "checkpoint": checkpoint_name,
            "benchmark": benchmark,
            "num_problems": len(by_problem),
            "num_samples": len(bench_rows),
            "samples_per_problem": max((int(r["samples_per_problem"]) for r in bench_rows), default=None),
            "avg_at_k": finite_mean([r["correct"] for r in bench_rows]),
            "macro_avg_at_k": finite_mean(problem_avg),
            "pass_at_k": finite_mean(problem_pass),
            "parse_rate": finite_mean([r["format_valid"] for r in bench_rows]),
            "hit_max_rate": finite_mean([r["hit_max_new_tokens"] for r in bench_rows]),
            "no_answer_rate": finite_mean([not r["format_valid"] for r in bench_rows]),
            "length_mean": finite_mean(lengths),
            "length_median": finite_median(lengths),
            "maxed_unparseable_rate": finite_mean([r["hit_max_new_tokens"] and not r["format_valid"] for r in bench_rows]),
        }
        summaries.append(summary)

    overall = {
        "checkpoint": checkpoint_name,
        "num_benchmarks": len(summaries),
        "num_samples": len(rows),
        "benchmarks": summaries,
        "macro_avg_at_k_over_benchmarks": finite_mean([row["macro_avg_at_k"] for row in summaries]),
        "macro_pass_at_k_over_benchmarks": finite_mean([row["pass_at_k"] for row in summaries]),
        "parse_rate_overall": finite_mean([row["format_valid"] for row in rows]),
        "hit_max_rate_overall": finite_mean([row["hit_max_new_tokens"] for row in rows]),
        "length_mean_overall": finite_mean([row["completion_length"] for row in rows]),
    }
    write_json(output_dir / "summary" / f"{checkpoint_name}.json", overall)

    csv_path = output_dir / "summary" / f"{checkpoint_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()) if summaries else [])
        if summaries:
            writer.writeheader()
            writer.writerows(summaries)
    return overall


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    benchmarks = parse_benchmarks(args.benchmarks)

    if args.aggregate_only:
        summary = aggregate(output_dir, args.checkpoint_name, args.num_shards)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.benchmark_snapshot_records:
        rows_by_benchmark = load_benchmark_snapshot_records(
            args.benchmark_snapshot_records, benchmarks
        )
    else:
        rows_by_benchmark = {benchmark: load_benchmark(benchmark) for benchmark in benchmarks}
    requests = expanded_requests(rows_by_benchmark, args)
    shard = shard_requests(requests, args.num_shards, args.shard_index)
    manifest = {
        "checkpoint": args.checkpoint_name,
        "base_model": args.base_model,
        "lora_adapter": args.lora_adapter,
        "benchmarks": benchmarks,
        "total_requests": len(requests),
        "shard_requests": len(shard),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "samples_small": args.samples_small,
        "samples_large": args.samples_large,
        "limit_per_benchmark": args.limit_per_benchmark,
        "benchmark_snapshot_records": args.benchmark_snapshot_records,
        "benchmark_snapshot_sha256": (
            sha256_file(args.benchmark_snapshot_records)
            if args.benchmark_snapshot_records
            else None
        ),
        "benchmark_sizes": {benchmark: len(rows_by_benchmark[benchmark]) for benchmark in benchmarks},
    }
    write_json(output_dir / "manifests" / f"{args.checkpoint_name}.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json", manifest)
    if args.dry_run_no_model:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    prompts = [build_prompt(tokenizer, row["problem"]) for row in shard]
    llm_kwargs = {
        "model": args.base_model,
        "dtype": args.dtype,
        "seed": args.seed + args.shard_index,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
    }
    lora_request = None
    if args.lora_adapter:
        llm_kwargs.update({"enable_lora": True, "max_loras": 1, "max_lora_rank": args.max_lora_rank})
        lora_request = LoRARequest(args.checkpoint_name, 1, args.lora_adapter)
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed + args.shard_index,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=True)

    result_rows = []
    for request, prompt, output in zip(shard, prompts, outputs, strict=True):
        completion_output = output.outputs[0]
        completion = completion_output.text
        completion_length = len(completion_output.token_ids or [])
        pred, correct = evaluate_math_answer(completion, request["gold_answer"])
        result_rows.append(
            {
                "checkpoint": args.checkpoint_name,
                "checkpoint_type": "lora_adapter" if args.lora_adapter else "base_model",
                "base_model": args.base_model,
                "lora_adapter": args.lora_adapter,
                "benchmark": request["benchmark"],
                "id": request["id"],
                "problem_index": request["problem_index"],
                "sample_index": request["sample_index"],
                "samples_per_problem": request["samples_per_problem"],
                "problem": request["problem"],
                "gold_answer": request["gold_answer"],
                "prompt": prompt,
                "completion": completion,
                "extracted_answer": pred,
                "correct": bool(correct),
                "format_valid": bool(pred is not None),
                "completion_length": completion_length,
                "hit_max_new_tokens": completion_length >= args.max_new_tokens,
                "seed": args.seed,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len,
            }
        )
    write_jsonl(output_dir / "shards" / f"{args.checkpoint_name}.shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl", result_rows)
    print(f"Wrote {len(result_rows)} rows for shard {args.shard_index}/{args.num_shards}: {output_dir}")


if __name__ == "__main__":
    main()
