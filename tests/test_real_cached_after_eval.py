import json
from pathlib import Path

import pytest

from scripts.local import run_real_cached_after_eval as cached


def setup_dependency(tmp_path, monkeypatch, stage="train", alive=True):
    dependency = {"experiment": "real64", "status": str(tmp_path / "status.json"),
                  "result": str(tmp_path / "result.json"), "eval_output_dir": str(tmp_path),
                  "process": {"pid": 100, "start_ticks": 123}}
    Path(dependency["status"]).write_text(json.dumps({"experiment": "real64", "status": stage}))
    Path(dependency["result"]).write_text(json.dumps({"status": "verified_real_step50_comparison",
        "evaluation_name": "real64_step50_fullbench_32768_seed42", "num_samples": 7248}))
    (tmp_path / "verification.json").write_text(json.dumps({"status": "verified", "num_samples": 7248}))
    monkeypatch.setattr(cached.queue, "process_info", lambda pid: {"state": "S", "start_ticks": 123} if alive else None)
    return {"predecessor": dependency}


def test_stale_verified_output_does_not_start_training(tmp_path, monkeypatch):
    assert not cached.predecessor_ready(setup_dependency(tmp_path, monkeypatch))


def test_completion_waits_for_process_exit(tmp_path, monkeypatch):
    plan = setup_dependency(tmp_path, monkeypatch, "complete")
    assert not cached.predecessor_ready(plan)
    monkeypatch.setattr(cached.queue, "process_info", lambda pid: None)
    assert cached.predecessor_ready(plan)


def test_dead_or_reused_process_before_completion_fails(tmp_path, monkeypatch):
    plan = setup_dependency(tmp_path, monkeypatch, alive=False)
    with pytest.raises(RuntimeError, match="exited without verified"):
        cached.predecessor_ready(plan)
    monkeypatch.setattr(cached.queue, "process_info", lambda pid: {"state": "S", "start_ticks": 999})
    with pytest.raises(RuntimeError, match="exited without verified"):
        cached.predecessor_ready(plan)


def test_incomplete_verification_blocks_launch(tmp_path, monkeypatch):
    plan = setup_dependency(tmp_path, monkeypatch, "complete", alive=False)
    (tmp_path / "verification.json").write_text(json.dumps({"status": "verified", "num_samples": 7000}))
    with pytest.raises(RuntimeError, match="verification failed"):
        cached.predecessor_ready(plan)


def test_predecessor_failure_is_reported(tmp_path, monkeypatch):
    plan = setup_dependency(tmp_path, monkeypatch, "error")
    with pytest.raises(RuntimeError, match="Predecessor failed"):
        cached.predecessor_ready(plan)


def test_weighted_average_uses_response_count(tmp_path):
    (tmp_path / "summary").mkdir()
    rows = [{"benchmark": k, "num_samples": n, "avg_at_k": float(k == "math500")}
            for k, n in cached.BENCHMARKS.items()]
    path = tmp_path / "summary/test.json"
    path.write_text(json.dumps({"checkpoint": "test", "benchmarks": rows}))
    assert cached.weighted_summary(tmp_path)["weighted_avg"] == pytest.approx(2000 / 7248)
    rows[0]["num_samples"] -= 1
    path.write_text(json.dumps({"checkpoint": "test", "benchmarks": rows}))
    with pytest.raises(ValueError, match="sample counts"):
        cached.weighted_summary(tmp_path)
