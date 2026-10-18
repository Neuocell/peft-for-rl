"""GradTop contracts reject stale, mismatched, and malformed probe artifacts before training."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gradtop_protocol_verifier", ROOT / "scripts/analysis/verify_rlvr_protocol.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
PROTOCOL = json.loads((ROOT / "configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json").read_text())
DEFINITION = PROTOCOL["method_definitions"]["gradtop_prompt_sign"]
METHODS = ["gradtop", "gradtop_short", "gradtop_prompt_mean", "gradtop_prompt_sign", "gradtop_masked"]


@pytest.fixture
def artifact(tmp_path):
    model_dir = tmp_path / "base"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"]
    shapes = [(6, 8), (8, 12)]
    save_file({name + ".weight": torch.zeros(shape) for name, shape in zip(names, shapes)}, str(weights))
    directory = tmp_path / "artifact"
    directory.mkdir()
    bases = {name: torch.eye(shape[1])[:2].contiguous() for name, shape in zip(names, shapes)}
    allocation = {
        "schema_version": 1, "method": "rl_policy_gradient_subspace",
        "rank_pattern": dict.fromkeys(names, 2), "alpha_pattern": dict.fromkeys(names, 64),
        "constant_scaling": 32, "subspace_path": str(directory / "subspaces.safetensors"),
    }
    summary = {
        **copy.deepcopy(allocation), **copy.deepcopy(DEFINITION["observation"]), "rank_bins": [2],
        "source_model_path": str(model_dir), "sampled_prompts": 128, "sampled_rollouts": 1024,
        "observed_prompt_groups": 128, "formal_trainable_parameters": 68, "completed_basis_directions": 0,
        "observation_metadata": [{"prompt_groups": 16, "responses": 128, "valid_tokens": 500} for _ in range(8)],
        "modules": {
            name: {"selected_rank": 2, "input_dimension": input_dim, "output_dimension": output_dim,
                   "observations": 8, "parameter_cost_per_rank": output_dim + input_dim,
                   "sketch_supported_rank": 2, "basis_completion_rank": 0, "basis_completion_method": "none"}
            for name, (output_dim, input_dim) in zip(names, shapes)
        },
    }
    return directory, weights, allocation, summary, bases


def validate(artifact, definition=DEFINITION):
    directory, weights, allocation, summary, bases = artifact
    (directory / "rank_map.json").write_text(json.dumps(allocation))
    (directory / "summary.json").write_text(json.dumps(summary))
    save_file(bases, str(directory / "subspaces.safetensors"))
    return VERIFIER.validate_gradtop_artifact(directory, rank=2, alpha=64, model_weights=weights, definition=definition)


def test_valid_artifact_preserves_both_trainable_factors_and_hashes(artifact):
    result = validate(artifact)
    assert result["freeze_a"] is False
    assert result["initial_b"] == "zero"
    assert result["formal_trainable_parameters"] == 68
    assert result["optimizer_updates"] == 0
    assert result["observations"] == 8
    assert result["module_count"] == 2
    assert result["orthogonality_error_max"] == 0
    assert set(result["files"]) == {"rank_map.json", "summary.json", "subspaces.safetensors"}
    assert all(len(item["sha256"]) == 64 for item in result["files"].values())
    artifact[3]["observation_metadata"][0]["valid_tokens"] += 1
    assert result["files"]["summary.json"]["sha256"] != validate(artifact)["files"]["summary.json"]["sha256"]


@pytest.mark.parametrize(("field", "value"), [
    ("optimizer_updates", 8), ("optimizer_updates", False), ("observations", 16),
    ("requested_observations", 16), ("trainer_step", 4), ("gradient_probe_variant", "prompt_mean"),
    ("probe_width", 16), ("capacity", 32), ("seed", 7), ("sign_seed", 7),
    ("gradient_clipping", True), ("freeze_a", True), ("initial_b", "random"),
    ("rank_bins", [2, 4]), ("stopped_by_stability", True), ("source_model_path", "/wrong/model"),
    ("observed_prompt_groups", 16), ("sampled_rollouts", 4096), ("formal_trainable_parameters", 28),
])
def test_wrong_provenance_is_rejected(artifact, field, value):
    artifact[3][field] = value
    with pytest.raises(ValueError):
        validate(artifact)


@pytest.mark.parametrize("mutation", ["coverage", "shape", "nonfinite", "orthogonality", "count", "rank", "alpha", "path", "contract", "metadata", "empty_tokens"])
def test_invalid_initialization_is_rejected(artifact, mutation):
    _, _, allocation, summary, bases = artifact
    name = next(iter(bases))
    if mutation == "coverage":
        bases.pop(name)
    elif mutation == "shape":
        bases[name] = bases[name][:, :-1].contiguous()
    elif mutation == "nonfinite":
        bases[name][0, 0] = float("nan")
    elif mutation == "orthogonality":
        bases[name][1] = bases[name][0]
    elif mutation == "count":
        summary["modules"][name]["observations"] = 1
    elif mutation in {"rank", "alpha"}:
        allocation[f"{mutation}_pattern"][name] = 1
        summary[f"{mutation}_pattern"] = copy.deepcopy(allocation[f"{mutation}_pattern"])
    elif mutation == "path":
        allocation["subspace_path"] = summary["subspace_path"] = "/other/subspaces.safetensors"
    elif mutation == "contract":
        summary["probe_contract"]["balance_batch"] = True
    elif mutation == "metadata":
        summary["observation_metadata"][0]["prompt_groups"] = 4
    else:
        summary["observation_metadata"][0]["valid_tokens"] = 0
    with pytest.raises(ValueError):
        validate(artifact)


def test_sibling_method_cannot_consume_wrong_variant_or_budget(artifact):
    for method in ["gradtop", "gradtop_short", "gradtop_prompt_mean", "gradtop_masked"]:
        with pytest.raises(ValueError):
            validate(artifact, PROTOCOL["method_definitions"][method])


def test_missing_artifact_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="Missing GradTop artifact"):
        VERIFIER.validate_gradtop_artifact(tmp_path, rank=32, alpha=64, model_weights=tmp_path / "base", definition=DEFINITION)


def run_launcher(script, **overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GRADIENT_", "GRADTOP_"))}
    env.update({"PYTHON_BIN": sys.executable, "DRY_RUN": "1", **overrides})
    return subprocess.run(["bash", str(ROOT / script)], env=env, cwd=ROOT, text=True, capture_output=True)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"METHOD": "gradtop_prompt_sign", "GRADIENT_PROBE_EXPECTED_OBSERVATIONS": "16"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "GRADIENT_PROBE_MIN_STEPS": "9"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "GRADIENT_PROBE_WIDTH": "16"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "LORA_ADAPTER_PATH": "/stale/adapter"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "LORA_RANK_PATTERN_PATH": "/stale/ranks"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "WARM_START_GLOBAL_STEP": "10"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "WARM_START_DATA_PATH": "/stale/data"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "SAVE_STEPS": "[1]"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "RAY_ADDRESS": "auto"}, "Frozen GradTop recipe"),
    ({"METHOD": "gradtop", "PHASE": "train"}, "requires GRADTOP_ARTIFACT_DIR"),
    ({"METHOD": "gradtop", "LORA_RANK": "65"}, "sketch capacity"),
    ({"METHOD": "gradtop", "LR": "2e-5"}, "LR must"),
    ({"METHOD": "gradtop", "TARGET_STEP": "11"}, "TARGET_STEP must"),
    ({"METHOD": "gradtop", "DRY_RUN": "0", "GRADTOP_PREVIEW_MISSING_ARTIFACT": "1"}, "cannot be deferred"),
])
def test_pipeline_rejects_unregistered_overrides_before_launch(overrides, message):
    result = run_launcher("scripts/local/run_gradtop_candidates_4gpu.sh", **overrides)
    assert result.returncode != 0
    assert message in result.stderr
    assert "main_ppo" not in result.stdout


def test_pipeline_will_not_reuse_or_overwrite_probe_artifact(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    result = run_launcher("scripts/local/run_gradtop_candidates_4gpu.sh", GRADTOP_ARTIFACT_DIR=str(tmp_path))
    assert result.returncode != 0
    assert "Refusing to overwrite or silently reuse" in result.stderr


def test_formal_wrapper_never_defers_artifact_checks_for_execution(tmp_path):
    result = run_launcher("scripts/local/run_rlvr_protocol_1p5b_4gpu.sh", METHOD="gradtop",
                          GRADTOP_ARTIFACT_DIR=str(tmp_path), GRADTOP_PREVIEW_MISSING_ARTIFACT="1", DRY_RUN="0")
    assert result.returncode != 0
    assert "allowed only with DRY_RUN=1" in result.stderr


@pytest.mark.parametrize("method", METHODS)
def test_formal_command_uses_grad_subspace_with_both_factors_trainable(tmp_path, method):
    train_file = tmp_path / "train.parquet"
    pq.write_table(pa.table({"x": [1]}), train_file)
    result = run_launcher("scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh", METHOD=method,
                          RUNTIME_ROOT=str(tmp_path), MODEL_PATH=str(tmp_path / "base"),
                          TRAIN_FILE=str(train_file), TEST_FILE=str(train_file), LORA_DROPOUT="0.0",
                          GRADIENT_SUBSPACE_RANK_MAP_PATH=str(tmp_path / "rank_map.json"),
                          GRADIENT_SUBSPACE_PATH=str(tmp_path / "subspaces.safetensors"))
    assert result.returncode == 0, result.stderr
    assert "+actor_rollout_ref.model.peft_type=grad_subspace" in result.stdout
    assert "+actor_rollout_ref.model.lora_freeze_a=False" in result.stdout
    assert "actor_rollout_ref.actor.fsdp_config.use_orig_params=False" in result.stdout
    assert "+actor_rollout_ref.model.gradient_subspace_path=" in result.stdout
    assert "actor_rollout_ref.model.lora_alpha=64" in result.stdout
    assert "+trainer.optimizer_updates_per_step=4" in result.stdout
    assert "DRY_RUN=1, command not executed" in result.stdout


def test_registered_controls_change_only_budget_or_probe_objective():
    definitions = PROTOCOL["method_definitions"]
    for method in METHODS:
        definition = definitions[method]
        assert definition["freeze_a"] is False
        assert definition["rank_allocation"] == "uniform"
        assert definition["observation"]["optimizer_updates"] == 0
    assert definitions["gradtop"]["observation"]["observations"] == 16
    assert definitions["gradtop_masked"]["observation"]["observations"] == 16
    for method in ["gradtop_short", "gradtop_prompt_mean", "gradtop_prompt_sign"]:
        assert definitions[method]["observation"]["observations"] == 8
    mean = copy.deepcopy(definitions["gradtop_prompt_mean"]["observation"])
    sign = copy.deepcopy(definitions["gradtop_prompt_sign"]["observation"])
    mean.pop("gradient_probe_variant")
    sign.pop("gradient_probe_variant")
    mean["probe_contract"].pop("sign_distribution")
    sign["probe_contract"].pop("sign_distribution")
    assert mean == sign


def test_reported_coordinate_completion_is_allowed_without_spectral_gate(artifact):
    name = next(iter(artifact[3]["modules"]))
    artifact[3]["modules"][name].update(sketch_supported_rank=0, basis_completion_rank=2, basis_completion_method="deterministic_coordinate_qr")
    artifact[3]["completed_basis_directions"] = 2
    assert validate(artifact)["completed_basis_directions"] == 2
    artifact[3]["completed_basis_directions"] = 0
    with pytest.raises(ValueError, match="completed_basis_directions"):
        validate(artifact)
