from pathlib import Path

import pytest

from scripts.local import real_early_step100 as early


def test_process_identity_rejects_pid_reuse_and_zombie(monkeypatch):
    identity = {"pid": 123, "start_ticks": 456}
    for current in [None, {"state": "S", "start_ticks": 789}, {"state": "Z", "start_ticks": 456}]:
        monkeypatch.setattr(early.queue, "process_info", lambda pid: current)
        assert not early.same_live_process(identity)
    monkeypatch.setattr(early.queue, "process_info", lambda pid: {"state": "S", "start_ticks": 456})
    assert early.same_live_process(identity)


def test_takeover_refuses_missing_original_process(monkeypatch):
    monkeypatch.setattr(early, "same_live_process", lambda identity: False)
    with pytest.raises(RuntimeError, match="missing or PID reused"):
        early.validate_takeover({"takeover": {"controller": {}, "training": {}}})


def test_pruning_preserves_adapter_and_successor(tmp_path, monkeypatch):
    old = tmp_path / "global_step_50"
    (old / "actor/peft_adapter").mkdir(parents=True)
    adapter = old / "actor/peft_adapter/adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    (old / "data.pt").write_bytes(b"dataloader")
    model = old / "actor/model_world_size_4_rank_0.pt"
    model.write_bytes(b"model")
    successor = tmp_path / "global_step_100"
    successor.mkdir()
    (successor / "data.pt").write_bytes(b"successor")
    calls = []
    monkeypatch.setattr(early, "validate_resume", lambda plan, step: calls.append(step))
    early.prune_step50_training_state({"checkpoint_root": str(tmp_path), "job_dir": str(tmp_path)})
    assert calls == [100]
    assert adapter.read_bytes() == b"adapter"
    assert (successor / "data.pt").read_bytes() == b"successor"
    assert not model.exists()


def test_pruning_cannot_run_before_successor_validation(tmp_path, monkeypatch):
    def reject(*args):
        raise RuntimeError("incomplete successor")
    monkeypatch.setattr(early, "validate_resume", reject)
    with pytest.raises(RuntimeError, match="incomplete successor"):
        early.prune_step50_training_state({"checkpoint_root": str(tmp_path)})
    assert list(tmp_path.iterdir()) == []
