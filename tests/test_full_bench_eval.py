from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.compare_paired_full_bench import compare_paired_records
from scripts.analysis.aggregate_multiseed_full_bench import aggregate
from scripts.analysis.evaluate_single_seed_screen import evaluate as evaluate_single_seed_screen
from scripts.analysis.select_phase1_seed42_candidate import (
    select_phase1_seed42_candidate,
)
from scripts.analysis.verify_full_bench_run import verify_full_bench_run

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tina_run"
    / "scripts"
    / "local"
    / "eval"
    / "eval_full_bench_vllm.py"
)


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_full_bench_vllm", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def eval_module():
    return load_eval_module()


def write_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "benchmark.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_full_bench_run_verifier_locks_sampling_protocol(tmp_path: Path) -> None:
    benchmarks = ("aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva")
    eval_name = "candidate_fullbench"
    eval_dir = tmp_path / "eval"
    summary_path = eval_dir / "summary" / f"{eval_name}.json"
    records_path = eval_dir / "records" / f"{eval_name}.jsonl"
    manifest_dir = eval_dir / "manifests"
    summary_path.parent.mkdir(parents=True)
    records_path.parent.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    adapter_dir = tmp_path / "adapter"
    base_model = tmp_path / "base-model"
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot_path.write_text("fixed snapshot\n", encoding="utf-8")
    snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    benchmark_sizes = {benchmark: 1 for benchmark in benchmarks}
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": eval_name,
                "num_samples": 6,
                "benchmarks": [
                    {
                        "benchmark": benchmark,
                        "num_problems": 1,
                        "num_samples": 1,
                    }
                    for benchmark in benchmarks
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records_path.write_text(
        "".join(
            json.dumps(
                {
                    "checkpoint": eval_name,
                    "benchmark": benchmark,
                    "problem_index": 0,
                    "sample_index": 0,
                }
            )
            + "\n"
            for benchmark in benchmarks
        ),
        encoding="utf-8",
    )
    for index in range(2):
        manifest = {
            "checkpoint": eval_name,
            "base_model": str(base_model),
            "lora_adapter": str(adapter_dir),
            "benchmarks": list(benchmarks),
            "total_requests": 6,
            "shard_requests": 3,
            "num_shards": 2,
            "shard_index": index,
            "seed": 42,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 32768,
            "max_model_len": 34816,
            "samples_small": 1,
            "samples_large": 1,
            "limit_per_benchmark": None,
            "benchmark_snapshot_records": str(snapshot_path),
            "benchmark_snapshot_sha256": snapshot_sha256,
            "benchmark_sizes": benchmark_sizes,
        }
        (manifest_dir / f"{eval_name}.shard-{index:02d}-of-02.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )

    verified = verify_full_bench_run(
        summary_path,
        records_path,
        eval_dir,
        eval_name,
        adapter_dir,
        base_model,
        snapshot_path,
        snapshot_sha256,
        expected_samples=6,
        expected_benchmarks=benchmarks,
        expected_benchmark_sizes=benchmark_sizes,
        expected_shards=2,
        expected_samples_small=1,
        expected_samples_large=1,
    )
    assert verified["status"] == "verified"
    assert verified["num_samples"] == 6

    first_manifest = manifest_dir / f"{eval_name}.shard-00-of-02.json"
    tampered = json.loads(first_manifest.read_text(encoding="utf-8"))
    tampered["top_p"] = 0.9
    first_manifest.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="field changed.*top_p"):
        verify_full_bench_run(
            summary_path,
            records_path,
            eval_dir,
            eval_name,
            adapter_dir,
            base_model,
            snapshot_path,
            snapshot_sha256,
            expected_samples=6,
            expected_benchmarks=benchmarks,
            expected_benchmark_sizes=benchmark_sizes,
            expected_shards=2,
            expected_samples_small=1,
            expected_samples_large=1,
        )


def test_load_flat_hmmt_schema(eval_module, tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path,
        [
            {"problem_idx": 7, "problem": "What is 1 + 1?", "answer": "2"},
            {"problem_idx": 8, "problem": "What is 2 + 2?", "answer": "4"},
        ],
    )

    rows = eval_module.load_direct_opd_parquet(str(path), "hmmt_feb")

    assert rows == [
        {"id": "7", "benchmark": "hmmt_feb", "problem": "What is 1 + 1?", "gold_answer": "2"},
        {"id": "8", "benchmark": "hmmt_feb", "problem": "What is 2 + 2?", "gold_answer": "4"},
    ]


def test_load_nested_direct_opd_schema(eval_module, tmp_path: Path) -> None:
    path = write_parquet(
        tmp_path,
        [
            {
                "prompt": [{"role": "user", "content": "Compute 3 + 4."}],
                "reward_model": {"ground_truth": "7"},
                "extra_info": {"index": 11},
            }
        ],
    )

    rows = eval_module.load_direct_opd_parquet(str(path), "aime24")

    assert rows == [
        {"id": "11", "benchmark": "aime24", "problem": "Compute 3 + 4.", "gold_answer": "7"}
    ]


def test_rejects_unknown_schema(eval_module, tmp_path: Path) -> None:
    path = write_parquet(tmp_path, [{"question": "What is 1 + 1?", "target": "2"}])

    with pytest.raises(ValueError, match="Unsupported evaluation parquet schema"):
        eval_module.load_direct_opd_parquet(str(path), "hmmt_feb")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"problem": "", "answer": "2"}, "Empty problem"),
        ({"problem": None, "answer": "2"}, "Empty problem"),
        ({"problem": "What is 1 + 1?", "answer": None}, "Missing ground-truth answer"),
        ({"problem": "What is 1 + 1?", "answer": float("nan")}, "Missing ground-truth answer"),
    ],
)
def test_rejects_incomplete_flat_rows(eval_module, tmp_path: Path, row: dict, message: str) -> None:
    path = write_parquet(tmp_path, [row])

    with pytest.raises(ValueError, match=message):
        eval_module.load_direct_opd_parquet(str(path), "hmmt_feb")


def test_loads_and_deduplicates_benchmark_snapshot_records(eval_module, tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {"benchmark": "aime24", "problem_index": 1, "id": "b", "problem": "p1", "gold_answer": "a1"},
        {"benchmark": "aime24", "problem_index": 0, "id": "a", "problem": "p0", "gold_answer": "a0"},
        {"benchmark": "aime24", "problem_index": 0, "id": "a", "problem": "p0", "gold_answer": "a0"},
        {"benchmark": "math500", "problem_index": 0, "id": "m", "problem": "mp", "gold_answer": "ma"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    loaded = eval_module.load_benchmark_snapshot_records(
        str(path), ["aime24", "math500"]
    )

    assert [row["id"] for row in loaded["aime24"]] == ["a", "b"]
    assert [row["id"] for row in loaded["math500"]] == ["m"]


def test_rejects_conflicting_benchmark_snapshot_records(eval_module, tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [
        {"benchmark": "aime24", "problem_index": 0, "id": "a", "problem": "p0", "gold_answer": "a0"},
        {"benchmark": "aime24", "problem_index": 0, "id": "a", "problem": "changed", "gold_answer": "a0"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    with pytest.raises(ValueError, match="Conflicting benchmark snapshot records"):
        eval_module.load_benchmark_snapshot_records(str(path), ["aime24"])


def test_paired_full_bench_bootstrap_matches_records_and_clusters_by_problem(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_rows = []
    candidate_rows = []
    for benchmark in ("aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva"):
        for problem_index in range(2):
            for sample_index in range(2):
                common = {
                    "benchmark": benchmark,
                    "problem_index": problem_index,
                    "sample_index": sample_index,
                    "id": f"{benchmark}-{problem_index}",
                    "problem": f"problem-{benchmark}-{problem_index}",
                    "gold_answer": "1",
                    "samples_per_problem": 2,
                }
                baseline_rows.append({**common, "checkpoint": "H0", "correct": False})
                candidate_rows.append(
                    {
                        **common,
                        "checkpoint": "H2",
                        "correct": problem_index == 0,
                    }
                )
    baseline_path.write_text(
        "".join(json.dumps(row) + "\n" for row in baseline_rows), encoding="utf-8"
    )
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
    )

    result = compare_paired_records(
        baseline_path, candidate_path, bootstrap_samples=200, seed=7
    )

    assert result["record_keys_exactly_matched"] is True
    assert result["num_samples"] == 24
    assert result["num_problems"] == 12
    assert result["macro_delta_avg_at_k"] == pytest.approx(0.5)
    assert result["ci95_low"] >= 0.0
    assert result["interpretation"] == "single_seed_screen_only"

    screen = evaluate_single_seed_screen(result)
    assert screen["decision"] == "advance_multiseed"


def test_multiseed_hierarchical_bootstrap_applies_preregistered_gate(
    tmp_path: Path,
) -> None:
    baseline_paths = []
    candidate_paths = []
    for seed in (42, 43, 44):
        baseline_path = tmp_path / f"baseline-{seed}.jsonl"
        candidate_path = tmp_path / f"candidate-{seed}.jsonl"
        baseline_rows = []
        candidate_rows = []
        for benchmark in ("aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva"):
            for problem_index in range(3):
                common = {
                    "benchmark": benchmark,
                    "problem_index": problem_index,
                    "sample_index": 0,
                    "id": f"{benchmark}-{problem_index}",
                    "problem": f"problem-{benchmark}-{problem_index}",
                    "gold_answer": "1",
                    "samples_per_problem": 1,
                }
                baseline_rows.append({**common, "correct": False})
                candidate_rows.append({**common, "correct": True})
        baseline_path.write_text(
            "".join(json.dumps(row) + "\n" for row in baseline_rows), encoding="utf-8"
        )
        candidate_path.write_text(
            "".join(json.dumps(row) + "\n" for row in candidate_rows), encoding="utf-8"
        )
        baseline_paths.append(baseline_path)
        candidate_paths.append(candidate_path)

    result = aggregate(
        baseline_paths,
        candidate_paths,
        [42, 43, 44],
        bootstrap_samples=100,
        bootstrap_seed=3,
        expected_samples=18,
    )

    assert result["mean_macro_delta_avg_at_k"] == pytest.approx(1.0)
    assert result["hierarchical_ci95_low"] == pytest.approx(1.0)
    assert result["positive_seed_count"] == 3
    assert result["positive_benchmark_count"] == 6
    assert result["decision"] == "stable_generalizable"

    with pytest.raises(ValueError, match="exactly 19 samples"):
        aggregate(
            baseline_paths,
            candidate_paths,
            [42, 43, 44],
            bootstrap_samples=10,
            expected_samples=19,
        )


def test_phase1_seed42_selection_excludes_signal_only_control_and_breaks_ties() -> None:
    def comparison(delta: float, probability: float) -> dict:
        return {
            "record_keys_exactly_matched": True,
            "interpretation": "single_seed_screen_only",
            "num_samples": 7248,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 42,
            "benchmarks": [
                {"benchmark": name}
                for name in (
                    "aime24",
                    "aime25",
                    "amc23",
                    "hmmt_feb",
                    "math500",
                    "minerva",
                )
            ],
            "macro_delta_avg_at_k": delta,
            "probability_delta_gt_zero": probability,
        }

    result = select_phase1_seed42_candidate(
        {
            "I8": comparison(0.012, 0.95),
            "I16": comparison(0.018, 0.94),
            "I32": comparison(0.030, 0.99),
        }
    )
    assert result["decision"] == "advance_multiseed"
    assert result["selected_method"] == "I16"
    assert result["candidates"]["I32"]["eligible_for_selection"] is False

    tie = select_phase1_seed42_candidate(
        {
            "I8": comparison(0.012, 0.95),
            "I16": comparison(0.012, 0.95),
            "I32": comparison(0.030, 0.99),
        }
    )
    assert tie["selected_method"] == "I8"

    stopped = select_phase1_seed42_candidate(
        {
            "I8": comparison(0.004, 0.99),
            "I16": comparison(0.010, 0.89),
            "I32": comparison(0.030, 0.99),
        }
    )
    assert stopped["decision"] == "stop_single_seed"
    assert stopped["selected_method"] is None

    incomplete = comparison(0.02, 0.99)
    incomplete["num_samples"] = 7247
    with pytest.raises(ValueError, match="7248 samples"):
        select_phase1_seed42_candidate(
            {
                "I8": incomplete,
                "I16": comparison(0.02, 0.99),
                "I32": comparison(0.02, 0.99),
            }
        )
