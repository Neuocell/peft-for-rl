"""Exercise fixed-budget observation -> exported basis -> fresh LoRA on CPU."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from verl.utils.peft_gradient_subspace import (
    GradientProbeCollector,
    apply_gradient_subspace_initialization,
    initialize_gradient_probe,
)


class TinyProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(6, 6, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


def model():
    return get_peft_model(TinyProjection(), LoraConfig(r=2, lora_alpha=2, target_modules=["proj"]))


def config(tmp_path):
    return {
        "path": str(tmp_path / "base"),
        "gradient_probe_output_dir": str(tmp_path / "probe"),
        "gradient_probe_variant": "prompt_sign",
        "gradient_probe_width": 2,
        "gradient_probe_capacity": 4,
        "gradient_probe_rank_bins": [2],
        "gradient_probe_expected_observations": 2,
        "gradient_probe_min_steps": 2,
        "gradient_probe_max_steps": 2,
        "gradient_probe_stability_patience": 1,
        "gradient_probe_overlap_threshold": 0.0,
        "gradient_subspace_scaling": 2.0,
    }


def contract():
    return {"responses_per_prompt": 2, "ppo_minibatch_size": 2, "prompts_per_step": 2}


def backward_and_capture(probe, collector):
    probe.zero_grad(set_to_none=True)
    # Independent signed contributions give a nontrivial finite sketch.
    inputs = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 24
    probe(inputs).square().sum().backward()
    return collector.capture_and_refresh(
        observation_metadata={"prompt_groups": 2, "responses": 4, "valid_tokens": 16}
    )


def test_fixed_budget_exports_zero_updates_and_loadable_function_preserving_adapter(tmp_path):
    torch.manual_seed(42)
    probe = model()
    cfg = config(tmp_path)
    inputs = torch.randn(3, 6)
    baseline = probe(inputs).detach().clone()
    initialize_gradient_probe(probe, cfg)
    collector = GradientProbeCollector(probe, cfg)
    collector.set_runtime_contract(contract())
    for step in (1, 2):
        metrics = backward_and_capture(probe, collector)
        assert metrics["gradient_probe/optimizer_updates"] == 0
        assert metrics["gradient_probe/observations"] == step
        torch.testing.assert_close(probe(inputs), baseline)
        assert collector.finish_trainer_step(step)["gradient_probe/artifact_ready"] == (step == 2)
        if step == 1:
            assert not (tmp_path / "probe/summary.json").exists()
    summary = json.loads((tmp_path / "probe/summary.json").read_text())
    assert summary["observations"] == summary["requested_observations"] == 2
    assert summary["optimizer_updates"] == 0
    assert not summary["stopped_by_stability"]
    assert summary["sampled_prompts"] == summary["observed_prompt_groups"] == 4
    assert summary["sampled_rollouts"] == 8
    assert summary["formal_trainable_parameters"] == 24
    assert summary["probe_contract"] == contract()
    assert summary["modules"]["proj"]["input_dimension"] == 6
    fresh = model()
    before = fresh(inputs).detach().clone()
    basis_path = tmp_path / "probe/subspaces.safetensors"
    apply_gradient_subspace_initialization(fresh, {"gradient_subspace_path": str(basis_path)})
    torch.testing.assert_close(fresh(inputs), before)
    layer = fresh.base_model.model.proj
    assert layer.lora_A["default"].weight.requires_grad
    assert layer.lora_B["default"].weight.requires_grad
    torch.testing.assert_close(layer.lora_A["default"].weight, load_file(str(basis_path))["proj"])
    with pytest.raises(ValueError, match="exceeded"):
        backward_and_capture(probe, collector)


def test_incomplete_budget_and_contract_drift_are_rejected(tmp_path):
    probe = model()
    cfg = config(tmp_path)
    initialize_gradient_probe(probe, cfg)
    collector = GradientProbeCollector(probe, cfg)
    collector.set_runtime_contract(contract())
    backward_and_capture(probe, collector)
    with pytest.raises(ValueError, match="expected 2 observations, captured 1"):
        collector.finish_trainer_step(2)
    with pytest.raises(ValueError, match="settings changed"):
        collector.set_runtime_contract({**contract(), "prompts_per_step": 4})
    with pytest.raises(ValueError, match="incomplete prompt"):
        collector.capture_and_refresh(
            observation_metadata={"prompt_groups": 2, "responses": 3, "valid_tokens": 12}
        )


@pytest.mark.parametrize("overrides", [
    {"gradient_probe_expected_observations": 0},
    {"gradient_probe_min_steps": 1},
    {"gradient_probe_rank_bins": [2, 4]},
    {"gradient_probe_method": "stable_snr"},
])
def test_invalid_candidate_recipe_fails_before_observation(tmp_path, overrides):
    with pytest.raises(ValueError, match="GradTop"):
        GradientProbeCollector(model(), {**config(tmp_path), **overrides})


def test_zero_signal_completion_is_disclosed_without_a_coverage_gate(tmp_path):
    probe = model()
    cfg = config(tmp_path)
    initialize_gradient_probe(probe, cfg)
    collector = GradientProbeCollector(probe, cfg)
    collector.set_runtime_contract(contract())
    for step in (1, 2):
        probe.zero_grad(set_to_none=True)
        (probe(torch.zeros(4, 6)).sum() * 0).backward()
        collector.capture_and_refresh(
            observation_metadata={"prompt_groups": 2, "responses": 4, "valid_tokens": 16}
        )
        collector.finish_trainer_step(step)
    summary = json.loads((tmp_path / "probe/summary.json").read_text())
    module = summary["modules"]["proj"]
    assert module["sketch_supported_rank"] == 0
    assert module["basis_completion_rank"] == summary["completed_basis_directions"] == 2
    assert module["basis_completion_method"] == "deterministic_coordinate_qr"


def test_real_collector_export_passes_protocol_validator(tmp_path):
    """Use actual collector output, not a hand-built verifier-shaped fixture."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("collector_protocol", root / "scripts/analysis/verify_rlvr_protocol.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    protocol = json.loads((root / "configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json").read_text())
    definition = protocol["method_definitions"]["gradtop_prompt_sign"]
    base = TinyProjection()
    # Match the architecture-shaped names the frozen protocol validates.
    base.model = torch.nn.Module()
    base.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    base.model.layers[0].self_attn = torch.nn.Module()
    base.model.layers[0].self_attn.q_proj = base.proj
    del base.proj
    base.forward = lambda x: base.model.layers[0].self_attn.q_proj(x)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    weights = base_dir / "model.safetensors"
    save_file(base.state_dict(), str(weights))
    probe = get_peft_model(base, LoraConfig(r=8, lora_alpha=8, target_modules=["q_proj"]))
    cfg = {
        **config(tmp_path), "gradient_probe_width": 8, "gradient_probe_capacity": 64,
        "gradient_probe_expected_observations": 8, "gradient_subspace_scaling": 32.0,
    }
    initialize_gradient_probe(probe, cfg)
    collector = GradientProbeCollector(probe, cfg)
    collector.set_runtime_contract(definition["observation"]["probe_contract"])
    for step in (1, 2):
        for _ in range(4):
            probe.zero_grad(set_to_none=True)
            probe(torch.randn(4, 6)).square().sum().backward()
            collector.capture_and_refresh(
                observation_metadata={"prompt_groups": 16, "responses": 128, "valid_tokens": 400}
            )
        collector.finish_trainer_step(step)
    verified = verifier.validate_gradtop_artifact(
        tmp_path / "probe", rank=2, alpha=64, model_weights=weights, definition=definition
    )
    assert verified["observations"] == 8
    assert verified["module_count"] == 1
