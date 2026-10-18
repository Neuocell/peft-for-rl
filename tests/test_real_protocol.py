"""Preflight failures must stop a REAL run before it claims training resources."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_rlvr_protocol", ROOT / "scripts/analysis/verify_rlvr_protocol.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
DEFINITION = json.loads((ROOT / "configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json").read_text())["method_definitions"]["real"]


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
        "schema_version": 1,
        "method": "real_reward_gradient_aligned_lora",
        "freeze_a": True,
        "rank_pattern": dict.fromkeys(names, 2),
        "alpha_pattern": dict.fromkeys(names, 64),
        "constant_scaling": 32,
        "subspace_path": str(directory / "subspaces.safetensors"),
    }
    summary = {
        **allocation,
        **{key: value for key, value in DEFINITION["observation"].items() if key != "sketch_width_multiple_of_rank"},
        "probe_method": "real",
        "trainer_step": 64,
        "input_sketch_width": 8,
        "output_sketch_width": 8,
        "source_model_path": str(model_dir),
        "distributed_observation": "complete_prompt_group_replicated_on_data_parallel_ranks",
        "prompt_groups": 2,
        "prompt_ids": ["10", "20"],
        "formal_trainable_parameters": 28,
        "modules": {
            name: {
                "selected_rank": 2,
                "input_dimension": input_dim,
                "prompt_groups": 2,
                "parameter_cost_per_rank": output_dim + input_dim,
                "b_only_parameter_cost_per_rank": output_dim,
            }
            for name, (output_dim, input_dim) in zip(names, shapes)
        },
    }
    return directory, weights, allocation, summary, bases


def validate(artifact):
    directory, weights, allocation, summary, bases = artifact
    (directory / "rank_map.json").write_text(json.dumps(allocation))
    (directory / "summary.json").write_text(json.dumps(summary))
    save_file(bases, str(directory / "subspaces.safetensors"))
    return VERIFIER.validate_real_artifact(directory, rank=2, alpha=64, model_weights=weights, definition=DEFINITION)


def test_complete_real_artifact_has_content_hashes_and_b_only_parameter_budget(artifact):
    result = validate(artifact)
    assert result["freeze_a"] is True
    assert result["initial_b"] == "zero"
    assert result["formal_trainable_parameters"] == 28
    assert result["module_count"] == 2
    assert result["orthogonality_error_max"] == 0
    assert set(result["files"]) == {"rank_map.json", "summary.json", "subspaces.safetensors"}
    assert all(len(item["sha256"]) == 64 for item in result["files"].values())
    artifact[3]["prompt_ids"] = ["11", "21"]
    assert result["files"]["summary.json"]["sha256"] != validate(artifact)["files"]["summary.json"]["sha256"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_updates", 1, "optimizer_updates"),
        ("objective", "token-mean", "objective"),
        ("seed", 7, "seed"),
        ("requested_prompt_groups", 5, "requested_prompt_groups"),
        ("trainer_step", 63, "step budget"),
        ("input_sketch_width", 4, "sketch widths"),
        ("prompt_ids", ["10", "10"], "unique ID"),
        ("formal_trainable_parameters", 68, "trainable parameter count"),
        ("source_model_path", "/wrong/model", "source model"),
    ],
)
def test_wrong_observation_or_training_contract_is_rejected(artifact, field, value, message):
    artifact[3][field] = value
    with pytest.raises(ValueError, match=message):
        validate(artifact)


@pytest.mark.parametrize("mutation", ["missing", "shape", "nonfinite", "orthogonality", "count", "rank", "alpha", "frozen"])
def test_bad_initialization_cannot_pass_preflight(artifact, mutation):
    _, _, allocation, summary, bases = artifact
    name = next(iter(bases))
    if mutation == "missing":
        bases.pop(name)
    elif mutation == "shape":
        bases[name] = bases[name][:, :-1].contiguous()
    elif mutation == "nonfinite":
        bases[name][0, 0] = float("nan")
    elif mutation == "orthogonality":
        bases[name][1] = bases[name][0]
    elif mutation == "count":
        summary["modules"][name]["prompt_groups"] = 1
    elif mutation in {"rank", "alpha"}:
        allocation[f"{mutation}_pattern"][name] = 1
        summary[f"{mutation}_pattern"] = copy.deepcopy(allocation[f"{mutation}_pattern"])
    else:
        allocation["freeze_a"] = summary["freeze_a"] = False
    with pytest.raises(ValueError):
        validate(artifact)


def test_missing_artifact_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing REAL artifact"):
        VERIFIER.validate_real_artifact(tmp_path, rank=32, alpha=64, model_weights=tmp_path / "base", definition=DEFINITION)
