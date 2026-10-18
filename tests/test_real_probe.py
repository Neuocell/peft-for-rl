from __future__ import annotations

import copy
import json
import math

import pytest
import torch
from peft import LoraConfig, get_peft_model

from verl.utils.peft_gradient_subspace import (
    apply_gradient_subspace_initialization,
    freeze_lora_a_factors,
)
from verl.utils.peft_oracle_lora import load_oracle_lora_patterns
from verl.utils.peft_real import (
    RealProbeCollector,
    RealSubspaceAccumulator,
    initialize_real_probe,
    nystrom_subspace,
    real_probe_dimensions,
    reward_contrast_coefficients,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].self_attn = torch.nn.Module()
        self.model.layers[0].self_attn.q_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0].self_attn.q_proj(inputs)


def tiny_lora(*, rank: int = 4, width: int = 6):
    return get_peft_model(
        TinyPolicy(width),
        LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=["q_proj"], lora_dropout=0.0, bias="none"),
    )


def lora_layer(model):
    return next(module for module in model.modules() if hasattr(module, "lora_A"))


def config_for(tmp_path, **overrides):
    return {
        "real_probe_rank": 2,
        "real_probe_output_width": 4,
        "real_probe_input_width": 6,
        "real_probe_prompts": 2,
        "real_probe_seed": 42,
        "gradient_probe_output_dir": str(tmp_path),
        "gradient_subspace_scaling": 2.0,
        **overrides,
    }


def synthetic_gradient(model, sign: float = 1.0):
    layer = lora_layer(model)
    gradient = torch.zeros_like(layer.lora_A["default"].weight)
    gradient[0, 0], gradient[1, 1] = 3.0 * sign, 2.0 * sign
    layer.lora_A["default"].weight.grad = gradient * layer.scaling["default"]


def test_equation6_uses_per_response_lengths_and_population_standard_deviation():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    lengths = torch.tensor([2, 3, 5, 4, 6, 1])
    mask = torch.arange(6)[None, :] < lengths[:, None]
    coefficients = reward_contrast_coefficients(rewards, mask)
    scores = torch.arange(36, dtype=torch.float64).reshape(6, 6).sin()
    sequence_scores = (scores * mask).sum(-1) / lengths
    observed = (scores * mask * coefficients[:, None]).sum()
    contrast = math.sqrt(2 * 4) / 6 * (sequence_scores[rewards == 1].mean() - sequence_scores[rewards == 0].mean())
    assert observed == pytest.approx(contrast, abs=1e-12)
    expected_advantage = (rewards - rewards.mean()) / rewards.std(unbiased=False)
    assert torch.allclose(coefficients * 6 * lengths, expected_advantage, atol=1e-12)
    group_token_mean = (scores * mask * expected_advantage[:, None]).sum() / lengths.sum()
    assert not torch.isclose(observed, group_token_mean)


@pytest.mark.parametrize("rewards", [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_binary_paper_group_filter_requires_two_of_each_outcome(rewards):
    with pytest.raises(ValueError, match="correct.*incorrect"):
        reward_contrast_coefficients(torch.tensor(rewards), torch.ones(4, 3))


@pytest.mark.parametrize("invalid", ["nonbinary", "nonfinite", "empty_response", "fractional_mask"])
def test_reward_contrast_rejects_malformed_groups(invalid):
    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0])
    mask = torch.ones(4, 3)
    if invalid == "nonbinary":
        rewards[-1] = 0.5
    elif invalid == "nonfinite":
        rewards[-1] = float("nan")
    elif invalid == "empty_response":
        mask[-1] = 0
    else:
        mask[-1, -1] = 0.5
    with pytest.raises(ValueError):
        reward_contrast_coefficients(rewards, mask)


def test_default_dimensions_match_four_times_final_rank():
    assert real_probe_dimensions({}) == (32, 128, 128)
    assert real_probe_dimensions({"real_probe_rank": 16}) == (16, 64, 64)
    with pytest.raises(ValueError, match="input_width"):
        real_probe_dimensions({"real_probe_rank": 8, "real_probe_input_width": 4})


def test_probe_is_function_preserving_and_recovers_projected_dense_gradient(tmp_path):
    torch.manual_seed(7)
    model = tiny_lora()
    layer = lora_layer(model)
    base_before = layer.base_layer.weight.detach().clone()
    stats = initialize_real_probe(model, config_for(tmp_path))
    inputs, upstream = torch.randn(5, 6), torch.randn(5, 6)
    actual_output = model(inputs)
    expected_output = layer.base_layer(inputs)
    assert torch.equal(actual_output, expected_output)
    assert stats.rank == 4 and stats.a_abs_max == 0.0
    assert torch.equal(base_before, layer.base_layer.weight)
    assert all(parameter.requires_grad == ("lora_A" in name) for name, parameter in model.named_parameters())
    (actual_output * upstream).sum().backward()
    expected_gradient = layer.lora_B["default"].weight.detach().T @ (upstream.T @ inputs)
    observed_gradient = layer.lora_A["default"].weight.grad / layer.scaling["default"]
    torch.testing.assert_close(observed_gradient, expected_gradient, atol=1e-6, rtol=1e-5)
    assert layer.base_layer.weight.grad is None
    assert layer.lora_B["default"].weight.grad is None


def test_per_prompt_energy_preserves_directions_cancelled_by_a_batch_mean():
    accumulator = RealSubspaceAccumulator(rank=2, input_width=6, seed=19)
    h = torch.zeros(3, 6)
    h[0, 0], h[1, 1] = 3.0, 2.0
    accumulator.add("layer", h)
    accumulator.add("layer", -h)
    basis, diagnostics = accumulator.snapshot()
    assert torch.count_nonzero((h + (-h)) / 2) == 0
    projector = basis["layer"].T @ basis["layer"]
    expected = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(projector, expected, atol=2e-6, rtol=2e-6)
    assert diagnostics["layer"]["output_sketch_energy"] == pytest.approx(13.0)


def test_streaming_sketch_matches_explicit_two_sided_second_moment():
    generator = torch.Generator().manual_seed(29)
    accumulator = RealSubspaceAccumulator(rank=2, input_width=7, seed=31)
    exact_second_moment = torch.zeros(7, 7)
    for _ in range(5):
        gradient = torch.randn(5, 7, generator=generator)
        psi = torch.randn(5, 3, generator=generator) / math.sqrt(3)
        h = psi.T @ gradient
        accumulator.add("layer", h)
        exact_second_moment += gradient.T @ psi @ psi.T @ gradient
    torch.testing.assert_close(
        accumulator.sketches["layer"], exact_second_moment @ accumulator.omegas["layer"], atol=5e-5, rtol=3e-6
    )
    basis, _ = accumulator.snapshot()
    _, eigenvectors = torch.linalg.eigh(exact_second_moment)
    expected = eigenvectors[:, -2:] @ eigenvectors[:, -2:].T
    torch.testing.assert_close(basis["layer"].T @ basis["layer"], expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("scale", [1e-24, 1.0, 1e24])
def test_nystrom_matches_full_covariance_oracle_and_is_scale_invariant(scale):
    generator = torch.Generator().manual_seed(73)
    directions, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator, dtype=torch.float64))
    eigenvalues = torch.tensor([9.0, 4.0, 1.0, 0.5, 0.2, 0.1], dtype=torch.float64)
    covariance = (directions * eigenvalues) @ directions.T
    omega = torch.eye(6, dtype=torch.float64)
    basis, spectrum, diagnostics = nystrom_subspace(scale * covariance @ omega, omega, 2)
    torch.testing.assert_close(basis.T @ basis, (directions[:, :2] @ directions[:, :2].T).float(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(spectrum / scale, eigenvalues, atol=1e-10, rtol=1e-10)
    assert diagnostics["orthogonality_error"] < 1e-6


def test_nystrom_truncates_roundoff_but_rejects_indefinite_or_insufficient_rank():
    omega = torch.eye(4)
    covariance = torch.diag(torch.tensor([3.0, 1.0, 1e-9, -1e-8]))
    basis, _, diagnostics = nystrom_subspace(covariance, omega, 2)
    assert basis.shape == (2, 4)
    assert diagnostics["nystrom_core_rank"] == 2
    with pytest.raises(ValueError, match="numerical rank"):
        nystrom_subspace(covariance, omega, 3)
    with pytest.raises(ValueError, match="positive semidefinite"):
        nystrom_subspace(torch.diag(torch.tensor([3.0, 1.0, 0.0, -0.2])), omega, 2)
    with pytest.raises(ValueError, match="zero energy"):
        nystrom_subspace(torch.zeros(4, 4), omega, 1)
    with pytest.raises(FloatingPointError):
        nystrom_subspace(torch.full((4, 4), float("nan")), omega, 1)


def test_fresh_output_projection_is_deterministic_per_group_and_keeps_zero_function(tmp_path):
    first, second = tiny_lora(), tiny_lora()
    configuration = config_for(tmp_path)
    initialize_real_probe(first, configuration)
    initialize_real_probe(second, configuration)
    first_layer, second_layer = lora_layer(first), lora_layer(second)
    before = first_layer.lora_B["default"].weight.detach().clone()
    collector1, collector2 = RealProbeCollector(first, configuration), RealProbeCollector(second, configuration)
    synthetic_gradient(first)
    synthetic_gradient(second)
    collector1.capture_and_refresh("one")
    collector2.capture_and_refresh("one")
    assert not torch.equal(before, first_layer.lora_B["default"].weight)
    assert torch.equal(first_layer.lora_B["default"].weight, second_layer.lora_B["default"].weight)
    assert torch.count_nonzero(first_layer.lora_A["default"].weight) == 0
    assert not first_layer.lora_B["default"].weight.requires_grad


def test_collector_exports_consumable_fixed_a_artifact_only_after_observation_budget(tmp_path):
    model = tiny_lora()
    configuration = config_for(tmp_path)
    initialize_real_probe(model, configuration)
    collector = RealProbeCollector(model, configuration)
    synthetic_gradient(model)
    collector.capture_and_refresh("first")
    assert not collector.complete
    assert collector.finish_trainer_step(1)["real_probe/artifact_ready"] == 0.0
    assert not (tmp_path / "rank_map.json").exists()
    synthetic_gradient(model, -1.0)
    collector.capture_and_refresh("second")
    assert collector.complete and not collector.exported
    assert collector.finish_trainer_step(2)["real_probe/artifact_ready"] == 1.0
    patterns = load_oracle_lora_patterns(tmp_path / "rank_map.json", base_rank=2, base_alpha=4)
    assert patterns.rank_mean == 2 and patterns.scaling_ratio == 2.0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["prompt_groups"] == 2
    assert summary["freeze_a"] and summary["optimizer_updates"] == 0
    assert summary["persistent_accumulator_bytes"] == 2 * 6 * 6 * 4
    assert summary["formal_trainable_parameters"] == 2 * 6
    trained = tiny_lora(rank=2)
    base = copy.deepcopy(trained)
    stats = apply_gradient_subspace_initialization(trained, {"gradient_subspace_path": str(tmp_path / "subspaces.safetensors")})
    frozen = freeze_lora_a_factors(trained)
    assert stats.orthogonality_error_max < 1e-6 and stats.b_abs_max == 0.0
    assert frozen.b_parameters == 12
    inputs = torch.randn(4, 6)
    torch.testing.assert_close(trained(inputs), base(inputs), rtol=0, atol=0)
    trained(inputs).square().sum().backward()
    layer = lora_layer(trained)
    assert layer.lora_A["default"].weight.grad is None
    assert layer.lora_B["default"].weight.grad is not None
    assert collector.finish_trainer_step(3)["real_probe/artifact_ready"] == 1.0
    with pytest.raises(RuntimeError, match="budget is complete"):
        collector.capture_and_refresh("third")


def test_step_budget_collects_all_available_groups_and_reports_empty_observations(tmp_path):
    configuration = config_for(tmp_path, real_probe_prompts=0, real_probe_steps=2)
    model = tiny_lora()
    initialize_real_probe(model, configuration)
    collector = RealProbeCollector(model, configuration)
    for index in range(3):
        synthetic_gradient(model)
        collector.capture_and_refresh(index)
    assert not collector.complete
    assert collector.finish_trainer_step(1)["real_probe/artifact_ready"] == 0.0
    assert collector.finish_trainer_step(2)["real_probe/artifact_ready"] == 1.0
    assert json.loads((tmp_path / "summary.json").read_text())["prompt_groups"] == 3
    empty_configuration = config_for(tmp_path / "empty", real_probe_prompts=0, real_probe_steps=1)
    empty_model = tiny_lora()
    initialize_real_probe(empty_model, empty_configuration)
    empty = RealProbeCollector(empty_model, empty_configuration)
    with pytest.raises(RuntimeError, match="without an eligible prompt group"):
        empty.finish_trainer_step(1)


def test_collector_refuses_missing_gradients_or_modified_probe_parameters(tmp_path):
    configuration = config_for(tmp_path)
    model = tiny_lora()
    initialize_real_probe(model, configuration)
    collector = RealProbeCollector(model, configuration)
    with pytest.raises(RuntimeError, match="missing a full LoRA A gradient"):
        collector.capture_and_refresh("missing")
    synthetic_gradient(model)
    with torch.no_grad():
        lora_layer(model).lora_A["default"].weight[0, 0] = 1
    with pytest.raises(RuntimeError, match="probe A was modified"):
        collector.capture_and_refresh("changed")


def test_failed_rank_recovery_does_not_write_an_artifact(tmp_path):
    configuration = config_for(tmp_path, real_probe_prompts=1)
    model = tiny_lora()
    initialize_real_probe(model, configuration)
    collector = RealProbeCollector(model, configuration)
    layer = lora_layer(model)
    layer.lora_A["default"].weight.grad = torch.zeros_like(layer.lora_A["default"].weight)
    layer.lora_A["default"].weight.grad[0, 0] = 1.0
    collector.capture_and_refresh("rank_one")
    with pytest.raises(RuntimeError, match="numerical rank"):
        collector.finish_trainer_step(1)
    assert not (tmp_path / "rank_map.json").exists()
    assert not (tmp_path / "subspaces.safetensors").exists()


@pytest.mark.parametrize("output_dir", [None, "", "   "])
def test_collector_rejects_missing_output_directory(tmp_path, output_dir):
    configuration = config_for(tmp_path, gradient_probe_output_dir=output_dir)
    model = tiny_lora()
    initialize_real_probe(model, configuration)
    with pytest.raises(ValueError, match="gradient_probe_output_dir is required"):
        RealProbeCollector(model, configuration)
