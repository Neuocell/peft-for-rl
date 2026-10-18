"""CPU checks for queue ownership, recovery bounds and fresh-run isolation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.local import run_gradtop_clean_after_real as clean


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def predecessor(tmp_path, monkeypatch, *, status="train", alive=True, error=None):
    original = {"root": str(tmp_path), "experiment": "real32", "target_step": 150,
                "job_dir": str(tmp_path / "original"), "checkpoint_root": str(tmp_path / "checkpoints"),
                "continuation": {"evaluation_steps": [100, 150]}}
    original_path = tmp_path / "plan.json"
    state_path = tmp_path / "original/status.json"
    write(original_path, original)
    state = {"experiment": "real32", "status": status, "completed_evaluations": [50, 100, 150],
             "controller_pid": 123, "controller_start_ticks": 45}
    if error:
        state["error"] = error
    write(state_path, state)
    plan = {"predecessor": {"plan": str(original_path), "plan_sha256": clean.queue.sha256(original_path),
                           "status": str(state_path), "process": {"pid": 123, "start_ticks": 45}}}
    monkeypatch.setattr(clean.queue, "process_info", lambda pid: {"state": "S", "start_ticks": 45} if alive else None)
    monkeypatch.setattr(clean, "verified_evaluation", lambda *args, **kwargs: ({}, Path("records")))
    monkeypatch.setattr(clean, "validate_complete_checkpoint", lambda *a: {})
    return plan, original, state


def test_stale_evaluations_do_not_override_live_predecessor(tmp_path, monkeypatch):
    plan, _, _ = predecessor(tmp_path, monkeypatch, status="complete")
    assert clean.predecessor_state(plan) == "waiting"
    monkeypatch.setattr(clean.queue, "process_info", lambda pid: None)
    assert clean.predecessor_state(plan) == "complete"


def test_reused_pid_is_not_a_live_owner(tmp_path, monkeypatch):
    plan, _, _ = predecessor(tmp_path, monkeypatch)
    monkeypatch.setattr(clean.queue, "process_info", lambda pid: {"state": "S", "start_ticks": 999})
    with pytest.raises(RuntimeError, match="exited without verified"):
        clean.predecessor_state(plan)


def test_changed_predecessor_plan_cannot_redirect_recovery(tmp_path, monkeypatch):
    plan, _, _ = predecessor(tmp_path, monkeypatch)
    Path(plan["predecessor"]["plan"]).write_text("{}")
    with pytest.raises(RuntimeError, match="Queued plan changed"):
        clean.predecessor_state(plan)


@pytest.mark.parametrize("error", ["RuntimeError: GPU failure", "RuntimeError: bad training Queued implementation changed:"])
def test_unrelated_failure_is_not_recoverable(tmp_path, monkeypatch, error):
    plan, _, _ = predecessor(tmp_path, monkeypatch, status="error", alive=False, error=error)
    with pytest.raises(RuntimeError, match="exited without verified"):
        clean.predecessor_state(plan)


def test_code_drift_failure_never_runs_fallback_recovery(tmp_path, monkeypatch):
    plan, _, _ = predecessor(tmp_path, monkeypatch, status="error", error="RuntimeError: Queued implementation changed: actor.py")
    assert clean.predecessor_state(plan) == "waiting"
    monkeypatch.setattr(clean.queue, "process_info", lambda pid: None)
    with pytest.raises(RuntimeError, match="exited without verified"):
        clean.predecessor_state(plan)


def test_superseded_predecessor_is_not_automatically_followed(tmp_path, monkeypatch):
    plan, _, state = predecessor(tmp_path, monkeypatch, status="superseded_by_early_step100", alive=False)
    state["replacement_status"] = "/some/unpinned/replacement.json"
    write(Path(plan["predecessor"]["status"]), state)
    with pytest.raises(RuntimeError, match="superseded_by_early_step100"):
        clean.predecessor_state(plan)


def test_completed_predecessor_requires_both_verified_evaluations(tmp_path, monkeypatch):
    plan, _, state = predecessor(tmp_path, monkeypatch, status="complete", alive=False)
    state["completed_evaluations"] = [50, 100]
    write(Path(plan["predecessor"]["status"]), state)
    with pytest.raises(RuntimeError, match="omits required"):
        clean.predecessor_state(plan)
    state["completed_evaluations"] = [50, 100, 150]
    write(Path(plan["predecessor"]["status"]), state)
    monkeypatch.setattr(clean, "verified_evaluation", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad records")))
    with pytest.raises(RuntimeError, match="bad records"):
        clean.predecessor_state(plan)


def test_disk_shortage_waits_without_removing_files(tmp_path, monkeypatch):
    existing = tmp_path / "keep.pt"
    existing.write_bytes(b"user checkpoint")
    disk = iter([SimpleNamespace(free=1), SimpleNamespace(free=clean.MIN_FREE_BYTES)])
    monkeypatch.setattr(clean.shutil, "disk_usage", lambda root: next(disk))
    monkeypatch.setattr(clean.queue, "wait_idle", lambda status: None)
    sleeps, stages = [], []
    monkeypatch.setattr(clean.time, "sleep", sleeps.append)
    clean.wait_resources({"root": str(tmp_path)}, lambda stage, **details: stages.append(stage))
    assert stages == ["waiting_for_disk_space"]
    assert sleeps == [30]
    assert existing.read_bytes() == b"user checkpoint"


def test_training_stage_does_not_retry_failed_child(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, "wait_resources", lambda *args: None)
    calls = []
    def fail(*args):
        calls.append(True)
        raise RuntimeError("REAL train exited1 without complete outputs")
    monkeypatch.setattr(clean.queue, "run_stage", fail)
    with pytest.raises(RuntimeError, match="exited1"):
        clean.run_training_stage({"job_dir": str(tmp_path)}, "train", [], {}, lambda: False,
                                 lambda *a, **k: None, lambda: None)
    assert len(calls) == 1


def test_probe_requires16_observations_and_zero_optimizer_updates(tmp_path):
    for name in ("rank_map.json", "summary.json", "subspaces.safetensors"):
        (tmp_path / name).write_text("{}")
    plan = {"artifact_dir": str(tmp_path)}
    assert not clean.probe_complete(plan)
    write(tmp_path / "summary.json", {"observations": 16, "optimizer_updates": 0})
    assert clean.probe_complete(plan)
    write(tmp_path / "summary.json", {"observations": 16, "optimizer_updates": 16})
    assert not clean.probe_complete(plan)


def test_snapshot_uses_direct_step_directory_and_trainer_step(tmp_path, monkeypatch):
    plan = {"snapshot_root": str(tmp_path), "base_model": "base",
            "checkpoint_root": str(tmp_path / "ckpt"), "job_dir": str(tmp_path / "job")}
    snapshot = tmp_path / "step_010"
    manifest = {"artifact_kind": "evaluation_snapshot", "trainer_step": 10, "optimizer_update": 40,
                "rank": 32, "alpha": 64.0, "module_count": 196, "tensor_count": 392,
                "snapshot": "adapter_model.safetensors"}
    write(snapshot / "manifest.json", manifest)
    (snapshot / "adapter_model.safetensors").write_bytes(b"snapshot10weights")
    config = Path(plan["checkpoint_root"]) / "global_step_50/actor/peft_adapter/adapter_config.json"
    write(config, {"r": 32})
    checked = []
    monkeypatch.setattr(clean, "verify_snapshot_compatibility", lambda *a, **k: {})
    monkeypatch.setattr(clean, "_check_adapter", lambda adapter, base: checked.append(adapter))
    adapter = clean.evaluation_adapter(plan, 10)
    assert adapter == Path(plan["job_dir"]) / "evaluation_adapters/step_010"
    assert checked == [adapter, adapter]
    assert (adapter / "adapter_model.safetensors").read_bytes() == b"snapshot10weights"
    assert not (snapshot / "adapter_config.json").exists()
    provenance = json.loads((adapter / "export_provenance.json").read_text())
    assert provenance["source_weights_sha256"] == provenance["export_weights_sha256"]
    manifest["trainer_step"] = 20
    write(snapshot / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="manifest"):
        clean.evaluation_adapter(plan, 10)


def test_environment_isolates_probe_from_existing_warm_start(tmp_path, monkeypatch):
    plan = {"root": str(tmp_path), "python": "/bin/python", "job_dir": str(tmp_path),
            "run_dir": str(tmp_path / "run"), "artifact_dir": str(tmp_path / "artifact"),
            "protocol_file": "protocol.json", "experiment": "gradtop_clean"}
    monkeypatch.setenv("LORA_ADAPTER_PATH", "old-adapter")
    monkeypatch.setenv("WARM_START_GLOBAL_STEP", "150")
    monkeypatch.setenv("REAL_ARTIFACT_DIR", "old-real")
    monkeypatch.setenv("GRADIENT_PROBE_VARIANT", "prompt_sign")
    env = clean.environment(plan, "probe")
    assert not {"LORA_ADAPTER_PATH", "WARM_START_GLOBAL_STEP", "REAL_ARTIFACT_DIR", "GRADIENT_PROBE_VARIANT"} & env.keys()
    assert env["METHOD"] == "gradtop" and env["PHASE"] == "probe"
    assert env["RESUME_MODE"] == "disable" and env["RAY_NUM_CPUS"] == "20"
    assert env["RAY_TEMP_BASE"] == "/tmp/gt32_1015"


def test_existing_outputs_are_never_reused_for_fresh_run(tmp_path):
    plan = {"root": str(tmp_path), "experiment": "new", "evaluations": [50, 10]}
    for key in ("artifact_dir", "run_dir", "checkpoint_root", "snapshot_root"):
        plan[key] = str(tmp_path / key)
    Path(plan["run_dir"]).mkdir()
    (Path(plan["run_dir"]) / "old_probe.json").write_text("old")
    with pytest.raises(FileExistsError, match="existing experiment output"):
        clean.require_clean_outputs(plan)


def test_check_missing_artifact_is_preview_and_performs_no_launch_or_write(tmp_path, monkeypatch):
    plan = {"root": str(tmp_path), "comparison_eval_dirs": {}}
    for name in ("validate_plan", "verify_eval_assets", "require_clean_outputs"):
        monkeypatch.setattr(clean, name, lambda *args: None)
    monkeypatch.setattr(clean.queue, "verify_code", lambda plan: None)
    monkeypatch.setattr(clean.queue, "verify_assets", lambda plan: None)
    monkeypatch.setattr(clean.queue, "gpu_pids", lambda: [99])
    monkeypatch.setattr(clean, "protocol_check", lambda *a, **k: {"status": "preview_pending_gradtop_artifact"})
    monkeypatch.setattr(clean, "predecessor_state", lambda plan: "waiting")
    monkeypatch.setattr(clean.subprocess, "Popen", lambda *a, **k: pytest.fail("must not start a process"))
    monkeypatch.setattr(clean.queue, "write_json", lambda *a, **k: pytest.fail("must not write"))
    monkeypatch.setattr(clean.time, "sleep", lambda *a: pytest.fail("must not wait"))
    result = clean.check(plan)
    assert result["artifact_validation"] == "pending_probe"
    assert result["predecessor"] == "waiting" and result["gpu_pids"] == [99]
    assert list(tmp_path.iterdir()) == []


def test_predecessor_status_cannot_switch_to_a_new_owner(tmp_path, monkeypatch):
    plan, _, state = predecessor(tmp_path, monkeypatch, status="complete", alive=False)
    state["controller_pid"] = 777
    write(Path(plan["predecessor"]["status"]), state)
    with pytest.raises(RuntimeError, match="status owner differs"):
        clean.predecessor_state(plan)


def test_verified_evaluation_does_not_replace_complete_checkpoint(tmp_path, monkeypatch):
    plan, _, _ = predecessor(tmp_path, monkeypatch, status="complete", alive=False)
    monkeypatch.setattr(clean, "validate_complete_checkpoint", lambda *a: (_ for _ in ()).throw(RuntimeError("missing optimizer")))
    with pytest.raises(RuntimeError, match="missing optimizer"):
        clean.predecessor_state(plan)


def test_evaluation_has_smaller_disk_budget_than_training(tmp_path, monkeypatch):
    monkeypatch.setattr(clean.queue, "wait_idle", lambda status: None)
    monkeypatch.setattr(clean.shutil, "disk_usage", lambda root: SimpleNamespace(free=3 * 1024**3))
    monkeypatch.setattr(clean.time, "sleep", lambda seconds: pytest.fail("evaluation should fit without waiting"))
    clean.wait_resources({"root": str(tmp_path)}, lambda *a, **k: None, clean.MIN_EVAL_FREE_BYTES)


def test_snapshot_compatibility_checks_values_separately_from_serialization(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    source, target = tmp_path / "snapshot.safetensors", tmp_path / "checkpoint.safetensors"
    save_file({"weight": np.ones((2, 3), dtype=np.float32)}, source, metadata={"kind": "snapshot"})
    save_file({"weight": np.ones((2, 3), dtype=np.float32)}, target, metadata={"kind": "checkpoint"})
    assert clean.queue.sha256(source) != clean.queue.sha256(target)
    assert clean.verify_snapshot_compatibility(source, target, equal_values=True)["equal_values"]
    save_file({"weight": np.zeros((2, 3), dtype=np.float32)}, target)
    clean.verify_snapshot_compatibility(source, target)
    with pytest.raises(RuntimeError, match="differs from full checkpoint"):
        clean.verify_snapshot_compatibility(source, target, equal_values=True)


@pytest.mark.parametrize("step,updates", [(50, 200), (150, 600)])
def test_full_checkpoint_requires_four_optimizer_updates_per_trainer_step(tmp_path, monkeypatch, step, updates):
    import sys
    support = tmp_path / "support.py"
    support.write_text("pinned validator")
    plan = {"checkpoint_root": str(tmp_path), "eval_support_path": str(support),
            "eval_support_sha256": clean.queue.sha256(support)}
    monkeypatch.setattr(clean.queue, "training_complete", lambda plan: True)
    monkeypatch.setattr(clean, "_load_file", lambda *a: SimpleNamespace(validate_checkpoint=lambda *args: {"status": "verified"}))
    torch = SimpleNamespace(load=lambda *a, **k: {"state": {0: {"step": updates}}})
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert clean.validate_complete_checkpoint(plan, step)["logical_optimizer_updates"] == updates
    torch.load = lambda *a, **k: {"state": {0: {"step": step}}}
    with pytest.raises(RuntimeError, match="optimizer updates"):
        clean.validate_complete_checkpoint(plan, step)
