import json
from pathlib import Path

import pytest

from scripts.local import run_real_after_eval as queue


def prerequisite(tmp_path):
    status = tmp_path / "baseline_status.json"
    verification = tmp_path / "verification.json"
    comparison = tmp_path / "comparison.json"
    verification.write_text(json.dumps({"status": "verified", "num_samples": 7248}))
    comparison.write_text(json.dumps({"status": "verified_three_checkpoints"}))
    state = {"experiment": "baseline", "status": "evaluating_step50", "completed_steps": [100],
             "current_step": 50, "num_records": 7248, "summary": "stale_step100_summary.json",
             "comparison": str(comparison)}
    status.write_text(json.dumps(state))
    plan = {"prerequisite_status": str(status), "baseline_experiment": "baseline",
            "prerequisite_process": {"pid": 123, "start_ticks": 456},
            "prerequisite_verifications": [str(verification)]}
    return plan, state


def test_stale_summary_and_count_cannot_trigger_launch(tmp_path, monkeypatch):
    plan, _ = prerequisite(tmp_path)
    monkeypatch.setattr(queue, "process_info", lambda pid: {"state": "S", "start_ticks": 456})
    assert queue.prerequisite_ready(plan) is False


def test_dead_or_reused_controller_without_completion_fails(tmp_path, monkeypatch):
    plan, _ = prerequisite(tmp_path)
    monkeypatch.setattr(queue, "process_info", lambda pid: {"state": "S", "start_ticks": 999})
    with pytest.raises(RuntimeError, match="exited without verified"):
        queue.prerequisite_ready(plan)


def test_complete_records_wait_for_controller_exit(tmp_path, monkeypatch):
    plan, state = prerequisite(tmp_path)
    state.update(status="all_evaluations_complete", completed_steps=[100, 50], current_step=None)
    Path(plan["prerequisite_status"]).write_text(json.dumps(state))
    monkeypatch.setattr(queue, "process_info", lambda pid: {"state": "S", "start_ticks": 456})
    assert queue.prerequisite_ready(plan) is False
    monkeypatch.setattr(queue, "process_info", lambda pid: None)
    assert queue.prerequisite_ready(plan) is True


def test_incomplete_verification_blocks_complete_status(tmp_path, monkeypatch):
    plan, state = prerequisite(tmp_path)
    state.update(status="all_evaluations_complete", completed_steps=[100, 50], current_step=None)
    Path(plan["prerequisite_status"]).write_text(json.dumps(state))
    Path(plan["prerequisite_verifications"][0]).write_text(json.dumps({"status": "verified", "num_samples": 7000}))
    monkeypatch.setattr(queue, "process_info", lambda pid: None)
    with pytest.raises(RuntimeError, match="verification failed"):
        queue.prerequisite_ready(plan)


def test_training_marker_requires_all_resumable_outputs(tmp_path):
    plan = {"checkpoint_root": str(tmp_path), "target_step": 50}
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("50\n")
    assert queue.training_complete(plan) is False
    checkpoint = tmp_path / "global_step_50"
    for relative in ["data.pt", "actor/peft_adapter/adapter_config.json", "actor/peft_adapter/adapter_model.safetensors"]:
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    for rank in range(4):
        for kind in ("model", "optim", "extra_state"):
            (checkpoint / "actor" / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"test")
    assert queue.training_complete(plan) is True


def test_queue_rejects_code_changes_after_plan_was_prepared(tmp_path):
    source = tmp_path / "probe.py"
    source.write_text("original")
    plan = {"code_sha256": {str(source): queue.sha256(source)}}
    queue.verify_code(plan)
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="Queued implementation changed"):
        queue.verify_code(plan)
