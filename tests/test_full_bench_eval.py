from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

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
