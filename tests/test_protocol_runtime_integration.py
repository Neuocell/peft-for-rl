"""Exercise launcher/provenance integration in an isolated repository.

Asset validation and the final training launcher are substitutes; the protocol
launcher and runtime provenance checker are the production scripts.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    for name in [
        "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh",
        "scripts/analysis/record_rlvr_runtime.py",
        "verl/utils/fsdp_utils.py",
        "verl/utils/adapter_initialization.py",
        "verl/utils/peft_lora_init.py",
        "verl/utils/peft_nora_dora.py",
        "verl/utils/peft_nora_doft.py",
    ]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"protocol_id": "test", "training": {"seed": 42}}))
    (tmp_path / "scripts/analysis/verify_rlvr_protocol.py").write_text('''
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
for flag in ["protocol", "method", "learning-rate", "rank", "target-step", "output", "real-artifact-dir", "gradtop-artifact-dir"]:
    p.add_argument("--" + flag)
p.add_argument("--preview-missing-gradtop-artifact", action="store_true")
a = p.parse_args()
contract = {
    "status": "preview_pending_gradtop_artifact" if a.preview_missing_gradtop_artifact else "verified",
    "protocol_id": "test", "protocol_path": a.protocol,
    "experiment_variables": {"method": a.method, "rank": a.rank, "learning_rate": a.learning_rate},
    "assets": {}, "data_order": {}, "execution": {"target_step": int(a.target_step)},
}
if a.method == "real" or a.method.startswith("gradtop"):
    key = "real_artifact" if a.method == "real" else "gradtop_artifact"
    contract[key] = {"files": {
        name: {"path": "/probe/" + name, "sha256": os.environ.get("TEST_ARTIFACT_HASH", "original") + name}
        for name in ["rank_map.json", "summary.json", "subspaces.safetensors"]
    }}
Path(a.output).write_text(json.dumps(contract))
''')
    (tmp_path / "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh").write_text(
        '#!/usr/bin/env bash\nprintf "TRAINING_STUB method=%s orig=%s freeze_a=%s\\n" "$METHOD" "$USE_ORIG_PARAMS" "$LORA_FREEZE_A"\n'
    )

    def run(**overrides):
        env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
               "PYTHON_BIN": sys.executable, "PROTOCOL_FILE": str(protocol),
               "EXP_NAME": "integration", "DRY_RUN": "1", "METHOD": "lora",
               "GRADTOP_ARTIFACT_DIR": str(tmp_path / "probe"),
               "REAL_ARTIFACT_DIR": str(tmp_path / "probe"), **overrides}
        return subprocess.run(
            ["bash", str(tmp_path / "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh")],
            cwd=tmp_path, env=env, text=True, capture_output=True,
        )

    return tmp_path, run


def test_preview_preserves_formal_contract_and_never_records_provenance(launch):
    root, run = launch
    contracts = root / "runs/protocol_contracts"
    contracts.mkdir(parents=True)
    formal = contracts / "integration.json"
    formal.write_text("original formal contract")
    result = run(METHOD="gradtop", GRADTOP_PREVIEW_MISSING_ARTIFACT="1")
    assert result.returncode == 0, result.stderr
    assert "TRAINING_STUB" in result.stdout
    assert formal.read_text() == "original formal contract"
    assert json.loads((contracts / "integration.preview.json").read_text())["status"] == "preview_pending_gradtop_artifact"
    assert not (contracts / "integration.runtime.json").exists()
    assert not list(contracts.glob("*.check.*"))
    result = run(METHOD="gradtop", GRADTOP_PREVIEW_MISSING_ARTIFACT="1", DRY_RUN="0")
    assert result.returncode != 0
    assert "TRAINING_STUB" not in result.stdout


@pytest.mark.parametrize("method,orig,freeze", [("gradtop", "False", "False"), ("real", "True", "True")])
def test_actual_launch_records_artifact_then_rejects_changed_resume(launch, method, orig, freeze):
    root, run = launch
    result = run(METHOD=method, DRY_RUN="0")
    assert result.returncode == 0, result.stderr
    assert f"orig={orig} freeze_a={freeze}" in result.stdout
    contracts = root / "runs/protocol_contracts"
    manifest = contracts / "integration.runtime.json"
    formal = contracts / "integration.json"
    before = (manifest.read_bytes(), formal.read_bytes())
    (root / "runs/ckpts/verl/peft-for-rl/integration/global_step_50").mkdir(parents=True)
    result = run(METHOD=method, RESUME_MODE="auto", TARGET_STEP="100")
    assert result.returncode == 0, result.stderr
    assert manifest.read_bytes() == before[0]
    formal_before_failure = formal.read_bytes()
    result = run(METHOD=method, RESUME_MODE="auto", TEST_ARTIFACT_HASH="changed", DRY_RUN="0")
    assert result.returncode != 0
    assert "Runtime identity changed" in result.stderr
    assert "TRAINING_STUB" not in result.stdout
    assert manifest.read_bytes() == before[0]
    assert formal.read_bytes() == formal_before_failure
    assert not list(contracts.glob("*.check.*"))


def test_preview_flag_for_lora_cannot_bypass_legacy_checkpoint_guard(launch):
    root, run = launch
    (root / "runs/ckpts/verl/peft-for-rl/integration/global_step_50").mkdir(parents=True)
    result = run(GRADTOP_PREVIEW_MISSING_ARTIFACT="1", RESUME_MODE="auto")
    assert result.returncode != 0
    assert "Legacy checkpoint lacks runtime input provenance" in result.stderr
    assert "TRAINING_STUB" not in result.stdout


def test_experiment_name_preview_suffix_cannot_bypass_provenance(launch):
    root, run = launch
    (root / "runs/ckpts/verl/peft-for-rl/integration.preview/global_step_50").mkdir(parents=True)
    result = run(EXP_NAME="integration.preview", RESUME_MODE="auto")
    assert result.returncode != 0
    assert "Legacy checkpoint lacks runtime input provenance" in result.stderr


def test_verified_dry_run_does_not_create_launch_record(launch):
    root, run = launch
    result = run()
    assert result.returncode == 0, result.stderr
    contracts = root / "runs/protocol_contracts"
    assert (contracts / "integration.json").exists()
    assert not (contracts / "integration.runtime.json").exists()
