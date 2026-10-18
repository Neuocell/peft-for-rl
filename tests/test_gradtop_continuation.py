"""CPU regressions for exact resume, stage ordering, retention and reporting."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.local import continue_gradtop_r32 as cont


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def contract(step):
    return {"execution": {"target_step": step, "full_checkpoints_to_keep": 2},
            "experiment_variables": {"method": "gradtop", "rank": 32, "learning_rate": "1e-5"},
            "gradtop_artifact": {"freeze_a": False, "orthogonality_error_max": 2.562999725341797e-05,
                                 "files": {name: {"sha256": "original"} for name in
                                           ("rank_map.json", "summary.json", "subspaces.safetensors")}}}


def test_contract_extension_changes_only_target_step():
    cont.assert_contract_extension(contract(50), contract(100), 50, 100)
    changed = contract(100)
    changed["gradtop_artifact"]["freeze_a"] = True
    with pytest.raises(RuntimeError, match="more than target_step"):
        cont.assert_contract_extension(contract(50), changed, 50, 100)
    with pytest.raises(RuntimeError, match="source or target"):
        cont.assert_contract_extension(contract(50), contract(150), 50, 100)


def test_latest_is_exact_not_any_complete_checkpoint(tmp_path):
    plan = {"checkpoint_root": str(tmp_path)}
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("50\n")
    cont.ensure_latest(plan, 50)
    with pytest.raises(RuntimeError, match="exactly100"):
        cont.ensure_latest(plan, 100)
    (tmp_path / "global_step_100").mkdir()
    with pytest.raises(RuntimeError, match="future checkpoint"):
        cont.ensure_latest(plan, 50)


def test_existing_post50_snapshot_prevents_duplicate_continuation(tmp_path):
    plan = {"root": str(tmp_path), "experiment": "test", "checkpoint_root": str(tmp_path / "ckpt"),
            "snapshot_root": str(tmp_path / "snapshots")}
    (Path(plan["snapshot_root"]) / "step_060").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="post-step50 snapshot"):
        cont.require_new_endpoints(plan)


def test_training_environment_is_same_method_with_full_auto_resume(tmp_path):
    plan = {"root": str(tmp_path), "python": "/bin/python", "job_dir": str(tmp_path),
            "run_dir": "original-run", "artifact_dir": "original-artifact", "protocol_file": "original-protocol",
            "experiment": "original-exp", "execution": {"ray_train_temp_base": "/tmp/gt32_cont"}}
    env = cont.training_environment(plan, 50, 100)
    assert env["METHOD"] == "gradtop" and env["RESUME_MODE"] == "auto"
    assert env["TARGET_STEP"] == "100" and env["LORA_RANK"] == "32" and env["LR"] == "1e-5"
    assert env["GRADTOP_ARTIFACT_DIR"] == "original-artifact"
    assert not {"RESUME_FROM_PATH", "WARM_START_GLOBAL_STEP", "LORA_ADAPTER_PATH"} & env.keys()
    with pytest.raises(ValueError, match="registered"):
        cont.training_environment(plan, 50, 150)


def test_resume_log_requires_full_state_and_correct_source(tmp_path):
    path = tmp_path / "train.log"
    ckpt = tmp_path / "global_step_50"
    assert not cont.verify_resume_log(path, ckpt)
    with pytest.raises(RuntimeError, match="Missing"):
        cont.verify_resume_log(path, ckpt, required=True)
    path.write_text(f"Load from checkpoint folder: {ckpt}\n" + "\n".join(
        f"Loaded {name} from {ckpt}/actor/file.pt" for name in ("model", "optimizer", "rng", "lr_scheduler")))
    assert cont.verify_resume_log(path, ckpt, required=True)
    with pytest.raises(RuntimeError, match="different resume"):
        cont.verify_resume_log(path, tmp_path / "global_step_100", required=True)
    path.write_text(f"Load from checkpoint folder: {ckpt}\nLoaded model from model.pt\n")
    with pytest.raises(RuntimeError, match="full model"):
        cont.verify_resume_log(path, ckpt, required=True)


@pytest.mark.parametrize("message", ["Training from scratch", "Warning: No dataloader state found at data.pt"])
def test_resume_log_rejects_silent_fresh_start(tmp_path, message):
    log = tmp_path / "train.log"
    log.write_text(message)
    with pytest.raises(RuntimeError, match="from scratch"):
        cont.verify_resume_log(log, tmp_path / "global_step_50")


def test_adapter_backup_survives_retention_without_faking_full_resume(tmp_path):
    initial = tmp_path / "contract.json"
    write(initial, contract(50))
    job = tmp_path / "job"
    job.mkdir()
    source = tmp_path / "checkpoints/global_step_50/actor/peft_adapter"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text('{"r":32}')
    (source / "adapter_model.safetensors").write_bytes(b"learned weights")
    (source / "README.md").write_text("source adapter metadata")
    plan = {"job_dir": str(job), "checkpoint_root": str(tmp_path / "checkpoints"),
            "source_contract": {"path": str(initial), "sha256": cont.queue.sha256(initial)}}
    cont.preserve_source(plan)
    cont.shutil.rmtree(source.parent)
    cont.restore_step50_adapter(plan)
    assert (source / "adapter_model.safetensors").read_bytes() == b"learned weights"
    assert (source / "README.md").exists()
    assert not (source.parent / "optim_world_size_4_rank_0.pt").exists()
    cont.restore_step50_adapter(plan)
    (source / "adapter_model.safetensors").write_bytes(b"unexpected weights")
    with pytest.raises(RuntimeError, match="refusing overwrite"):
        cont.restore_step50_adapter(plan)


def disk_fixture(tmp_path):
    plan = {"root": str(tmp_path), "checkpoint_root": str(tmp_path / "checkpoints"),
            "snapshot_root": str(tmp_path / "snapshots")}
    for step in (50, 100):
        actor = Path(plan["checkpoint_root"]) / f"global_step_{step}/actor"
        actor.mkdir(parents=True)
        (actor / "model.pt").write_bytes(b"x" * 8192)
        adapter = actor / "peft_adapter"
        adapter.mkdir()
        (adapter / "adapter.bin").write_bytes(b"y" * 4096)
        snapshot = Path(plan["snapshot_root"]) / f"step_{step:03d}"
        snapshot.mkdir(parents=True)
        (snapshot / "adapter.bin").write_bytes(b"z" * 4096)
    return plan


def test_disk_budget_credits_only_pre150_retention(tmp_path, monkeypatch):
    plan = disk_fixture(tmp_path)
    monkeypatch.setattr(cont, "protocol_report", lambda p, step: contract(step))
    first = cont.stage_disk_budget(plan, 50, 100)
    second = cont.stage_disk_budget(plan, 100, 150)
    assert first["verified_pre_save_retention_credit_bytes"] == 0
    assert second["verified_pre_save_retention_credit_bytes"] > 0
    assert second["restored_adapter_bytes"] == 4096
    assert second["required_free_bytes"] < first["required_free_bytes"]
    assert second["required_free_bytes"] >= 2 * 1024**3 + second["projected_snapshot_bytes"]


def test_hardlinked_files_do_not_count_as_freed_space(tmp_path, monkeypatch):
    plan = disk_fixture(tmp_path)
    monkeypatch.setattr(cont, "protocol_report", lambda p, step: contract(step))
    before = cont.stage_disk_budget(plan, 100, 150)
    target = Path(plan["checkpoint_root"]) / "global_step_50/actor/model.pt"
    (tmp_path / "other_reference.pt").hardlink_to(target)
    after = cont.stage_disk_budget(plan, 100, 150)
    assert after["verified_pre_save_retention_credit_bytes"] < before["verified_pre_save_retention_credit_bytes"]
    assert after["required_free_bytes"] > before["required_free_bytes"]


def test_disk_credit_requires_retention2(tmp_path, monkeypatch):
    plan = disk_fixture(tmp_path)
    monkeypatch.setattr(cont, "protocol_report", lambda *a: {"execution": {"full_checkpoints_to_keep": 3}})
    with pytest.raises(RuntimeError, match="retention of two"):
        cont.stage_disk_budget(plan, 100, 150)


def test_source_owner_and_complete_state_are_verified(tmp_path, monkeypatch):
    original = {"job_dir": str(tmp_path)}
    plan = {"experiment": "gt", "source_run": {"status": str(tmp_path / "status.json"),
            "process": {"pid": 1, "start_ticks": 2}}}
    state = {"experiment": "gt", "controller_pid": 1, "controller_start_ticks": 2,
             "status": "complete", "completed_evaluations": [50, 10]}
    write(Path(plan["source_run"]["status"]), state)
    monkeypatch.setattr(cont, "source_plan", lambda p: original)
    monkeypatch.setattr(cont.clean, "process_alive", lambda p: True)
    assert not cont.source_ready(plan)
    monkeypatch.setattr(cont.clean, "process_alive", lambda p: False)
    assert cont.source_ready(plan)
    state["controller_pid"] = 99
    write(Path(plan["source_run"]["status"]), state)
    with pytest.raises(RuntimeError, match="identity changed"):
        cont.source_ready(plan)


def test_report_keeps_weighted_average_distinct_from_macro(tmp_path, monkeypatch):
    output = tmp_path / "evaluation"
    rows = [{"benchmark": key, "num_samples": count, "avg_at_k": float(key == "math500"),
             "pass_at_k": float(key == "math500")} for key, count in cont.BENCHMARKS.items()]
    data = {"checkpoint": "gt_step50_eval", "benchmarks": rows,
            "macro_avg_at_k_over_benchmarks": 1 / 6, "macro_pass_at_k_over_benchmarks": 1 / 6,
            "parse_rate_overall": 0.9, "hit_max_rate_overall": 0.1, "length_mean_overall": 5000}
    write(output / "summary/result.json", data)
    monkeypatch.setattr(cont.clean, "verified_evaluation", lambda *a, **k: (data, Path("records")))
    result = cont.metrics_row({}, output, 50)
    assert result["weighted_avg"] == pytest.approx(2000 / 7248)
    assert result["macro_avg_at_k"] == pytest.approx(1 / 6)
    assert result["logical_optimizer_updates"] == 200


def test_train100_then_eval100_precedes_resume150(tmp_path, monkeypatch):
    initial = tmp_path / "canonical_contract.json"
    write(initial, contract(50))
    write(tmp_path / "contract_step50_original.json", contract(50))
    plan = {"root": str(tmp_path), "job_dir": str(tmp_path), "checkpoint_root": str(tmp_path / "checkpoints"),
            "source_contract": {"path": str(initial)}, "base_model": "base", "experiment": "gt"}
    events = []
    for name in ("check", "preserve_source", "write_results", "ensure_latest", "restore_step50_adapter"):
        monkeypatch.setattr(cont, name, lambda *a, **k: None)
    monkeypatch.setattr(cont, "stage_disk_budget", lambda *a: {"required_free_bytes": 1})
    monkeypatch.setattr(cont.clean, "wait_resources", lambda *a: None)
    monkeypatch.setattr(cont.clean, "validate_complete_checkpoint", lambda p, step: {"logical_optimizer_updates": 4 * step})
    monkeypatch.setattr(cont, "protocol_report", lambda p, target: contract(target))
    monkeypatch.setattr(cont, "training_environment", lambda *a: {})
    monkeypatch.setattr(cont, "_check_adapter", lambda *a: None)
    monkeypatch.setattr(cont, "compare_endpoint", lambda *a: {})
    def train(stage_plan, name, *args):
        events.append(("train", stage_plan["target_step"]))
        write(initial, contract(stage_plan["target_step"]))
    monkeypatch.setattr(cont, "run_continuation_stage", train)
    monkeypatch.setattr(cont.clean, "run_evaluation", lambda p, step, *a, **k: events.append(("eval", step)))
    cont.execute(plan, lambda *a, **k: None, lambda: None)
    assert events == [("train", 100), ("eval", 100), ("train", 150), ("eval", 150)]
    assert json.loads((tmp_path / "checkpoint_step150_verification.json").read_text())["logical_optimizer_updates"] == 600


def test_failed_stage_cleans_only_its_owned_token(tmp_path, monkeypatch):
    plan = {"job_dir": str(tmp_path), "root": str(tmp_path)}
    captured, cleaned = {}, []
    child = SimpleNamespace(pid=1234, returncode=1, poll=lambda: 1)
    def popen(*args, **kwargs):
        captured.update(kwargs)
        return child
    monkeypatch.setattr(cont.clean, "wait_resources", lambda *a: None)
    monkeypatch.setattr(cont.subprocess, "Popen", popen)
    monkeypatch.setattr(cont.queue, "cleanup_stage", cleaned.append)
    with pytest.raises(RuntimeError, match="without complete outputs"):
        cont.run_continuation_stage(plan, "train_to100", ["unused"], {"OTHER_OWNER": "safe"}, lambda: False,
                                    lambda *a, **k: None, lambda: None, {"required_free_bytes": 123})
    assert captured["start_new_session"] is True
    assert cleaned == [captured["env"]["REAL_PIPELINE_STAGE_TOKEN"]]
    assert captured["env"]["OTHER_OWNER"] == "safe"


def test_identical_artifact_hashes_allow_only_bounded_recomputed_diagnostic():
    original, target = contract(50), contract(100)
    target["gradtop_artifact"]["orthogonality_error_max"] = 2.6941299438476562e-05
    cont.assert_contract_extension(original, target, 50, 100)
    cont.assert_contract_extension(target, contract(100), 100, 100)
    target["gradtop_artifact"]["files"]["subspaces.safetensors"]["sha256"] = "changed"
    with pytest.raises(RuntimeError, match="more than target_step"):
        cont.assert_contract_extension(original, target, 50, 100)


@pytest.mark.parametrize("diagnostic", [float("nan"), float("inf"), 1.01e-4, -1e-5])
def test_diagnostic_exception_does_not_relax_orthogonality_bound(diagnostic):
    target = contract(100)
    target["gradtop_artifact"]["orthogonality_error_max"] = diagnostic
    with pytest.raises(RuntimeError, match="protocol bound"):
        cont.assert_contract_extension(contract(50), target, 50, 100)
