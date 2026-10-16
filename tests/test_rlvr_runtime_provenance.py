"""Protect experiment provenance when correcting rollout export implementation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/record_rlvr_runtime.py"
SPEC = importlib.util.spec_from_file_location("rlvr_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


@pytest.fixture
def run(tmp_path):
    identity = {
        "experiment_variables": {"method": "dora", "learning_rate": "1e-5", "rank": 32},
        "assets": {"source_data": {"sha256": "frozen-complete-prompt-hash"}},
        "rollout_export_version": "fsdp-canonical-merge-v1",
    }
    return identity, tmp_path / "runtime.json", tmp_path / "checkpoints", tmp_path / "snapshots"


def record(identity, manifest):
    manifest.write_text(json.dumps({"identity": identity, "trajectory_origin": "fresh_base"}))


def test_legacy_affected_checkpoint_cannot_resume(run):
    identity, manifest, checkpoints, snapshots = run
    (checkpoints / "global_step_50").mkdir(parents=True)
    with pytest.raises(ValueError, match="lacks corrected-exporter provenance"):
        runtime.validate_run(*run, "auto")
    assert not manifest.exists()


def test_snapshot_only_never_resumes_or_overwrites(run):
    identity, manifest, checkpoints, snapshots = run
    (snapshots / "step_010").mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot resume"):
        runtime.validate_run(*run, "auto")
    with pytest.raises(ValueError, match="new EXP_NAME"):
        runtime.validate_run(*run, "disable")


@pytest.mark.parametrize("change", ["prompt", "exporter", "method"])
def test_identity_drift_rejected(run, change):
    identity, manifest, checkpoints, snapshots = run
    record(identity, manifest)
    changed = copy.deepcopy(identity)
    if change == "prompt":
        changed["assets"]["source_data"]["sha256"] = "same-questions-different-instructions"
    elif change == "exporter":
        changed["rollout_export_version"] = "old-exporter"
    else:
        changed["experiment_variables"]["method"] = "nora_dora"
    with pytest.raises(ValueError, match="Runtime identity changed"):
        runtime.validate_run(changed, manifest, checkpoints, snapshots, "auto")


def test_valid_resume_and_pretraining_retry_preserve_record(run):
    identity, manifest, checkpoints, snapshots = run
    assert runtime.validate_run(*run, "disable") == "fresh_base"
    assert not manifest.exists()
    record(identity, manifest)
    before = manifest.read_bytes()
    assert runtime.validate_run(*run, "disable") == "fresh_base"
    (checkpoints / "global_step_20").mkdir(parents=True)
    assert runtime.validate_run(*run, "auto") == "fresh_base"
    assert manifest.read_bytes() == before


def test_legacy_lora_does_not_silently_switch_prompt(run):
    identity, manifest, checkpoints, snapshots = run
    identity["experiment_variables"]["method"] = "lora"
    (checkpoints / "global_step_50").mkdir(parents=True)
    with pytest.raises(ValueError, match="verify and migrate its original"):
        runtime.validate_run(*run, "auto")
    assert not manifest.exists()


def test_target_step_is_execution_control_not_identity(tmp_path):
    contract = {
        "status": "verified", "protocol_id": "v1", "experiment_variables": {"method": "lora"},
        "assets": {}, "data_order": {}, "execution": {"target_step": 20},
    }
    protocol = {"protocol_id": "v1", "training": {"seed": 42}}
    before = runtime.make_identity(contract, protocol, tmp_path)
    contract["execution"]["target_step"] = 50
    assert runtime.make_identity(contract, protocol, tmp_path) == before


@pytest.mark.parametrize("method", ["dora", "nora_dora", "nora_dora_continuous", "nora_init"])
def test_shared_initializer_drift_is_part_of_runtime_identity(tmp_path, method):
    source = tmp_path / "verl/utils"
    source.mkdir(parents=True)
    (source / "fsdp_utils.py").write_text('CUSTOM_PEFT_ROLLOUT_EXPORT_VERSION = "test-v1"\n')
    (source / "peft_nora_dora.py").write_text("# wrapper version\n")
    (source / "peft_lora_init.py").write_text("# native initializer\n")
    helper = source / "adapter_initialization.py"
    helper.write_text("# reference initial definition\n")
    contract = {
        "status": "verified", "protocol_id": "v1", "experiment_variables": {"method": method},
        "assets": {}, "data_order": {},
    }
    protocol = {"protocol_id": "v1", "training": {"seed": 42}}
    before = runtime.make_identity(contract, protocol, tmp_path)
    assert before["initialization_version"] == "peft_nora_reference_v2"
    helper.write_text("# changed initialization definition\n")
    after = runtime.make_identity(contract, protocol, tmp_path)
    assert before != after
    assert before["initialization_source_sha256"] != after["initialization_source_sha256"]
