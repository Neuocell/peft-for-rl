from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model

from verl.utils.peft_gradient_subspace import (
    DominantAtomGradientSubspaceAccumulator,
    GradientProbeCollector,
    GradientSubspaceAccumulator,
    StableSNRGradientSubspaceAccumulator,
    _allocate_rank_budget,
    apply_gradient_subspace_initialization,
    freeze_lora_a_factors,
    initialize_gradient_probe,
    normalize_target_name,
)
from verl.utils.peft_oracle_lora import load_oracle_lora_patterns
from verl.workers.actor.dp_actor import DataParallelPPOActor

from scripts.analysis.build_uniform_dominant_subspace_artifact import _window_factors

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyAttention(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.q_proj(inputs)


class TinyLayer(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = TinyAttention(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.self_attn(inputs)


class TinyStack(torch.nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(width)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0](inputs)


def tiny_lora_model(*, rank: int = 2, alpha: int = 2, width: int = 6):
    return get_peft_model(
        TinyStack(width),
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=["q_proj"],
            bias="none",
        ),
    )


def lora_layer(model):
    return next(module for module in model.modules() if hasattr(module, "lora_A") and "default" in module.lora_A)


def test_probe_is_zero_function_and_preserves_base_weight() -> None:
    model = tiny_lora_model(rank=3, alpha=3)
    layer = lora_layer(model)
    base_before = layer.base_layer.weight.detach().clone()

    stats = initialize_gradient_probe(model, {"gradient_probe_width": 3, "gradient_probe_seed": 17})
    inputs = torch.randn(4, 6)
    adapter_output = layer.lora_B["default"](layer.lora_A["default"](inputs))

    assert stats.num_layers == 1
    assert stats.a_abs_max == 0.0
    assert torch.count_nonzero(adapter_output).item() == 0
    assert torch.equal(layer.base_layer.weight, base_before)
    assert torch.count_nonzero(layer.lora_B["default"].weight).item() > 0


def test_probe_gradient_is_randomized_dense_gradient_sketch() -> None:
    torch.manual_seed(0)
    model = tiny_lora_model(rank=3, alpha=6)
    initialize_gradient_probe(model, {"gradient_probe_width": 3, "gradient_probe_seed": 7})
    layer = lora_layer(model)
    inputs = torch.randn(5, 6)
    upstream = torch.randn(5, 6)

    output = model(inputs)
    (output * upstream).sum().backward()

    dense_gradient = upstream.T @ inputs
    b = layer.lora_B["default"].weight.detach().float()
    scaling = float(layer.scaling["default"])
    observed = layer.lora_A["default"].weight.grad.detach().float().T / scaling
    expected = dense_gradient.T @ b
    assert torch.allclose(observed, expected, atol=1e-5, rtol=1e-5)


def test_actor_optimizer_boundary_can_measure_gradient_without_updating() -> None:
    module = torch.nn.Linear(4, 3, bias=False)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = SimpleNamespace(grad_clip=1.0)
    actor.actor_module = module
    actor.actor_optimizer = optimizer
    actor.scaler = None
    inputs = torch.randn(5, 4)
    module(inputs).square().sum().backward()
    before = module.weight.detach().clone()

    grad_norm = actor._optimizer_step(apply_update=False)

    assert torch.isfinite(grad_norm)
    assert torch.equal(module.weight, before)


def test_accumulator_recovers_dominant_subspace_and_quantizes_energy_rank() -> None:
    accumulator = GradientSubspaceAccumulator(
        capacity=4,
        target_energy=0.90,
        rank_bins=[1, 2, 4],
        scaling_ratio=2.0,
    )
    sketch = torch.zeros(6, 2)
    sketch[0, 0] = 3.0
    sketch[1, 1] = 2.0
    accumulator.add("model.layers.0.self_attn.q_proj", sketch)
    accumulator.compress()
    diagnostics, bases, metrics = accumulator.snapshot()

    item = diagnostics["model.layers.0.self_attn.q_proj"]
    basis = bases["model.layers.0.self_attn.q_proj"]
    assert item["raw_energy_rank"] == 2
    assert item["selected_rank"] == 2
    assert item["retained_energy"] == pytest.approx(1.0)
    assert metrics["rank_mean"] == 2.0
    expected_projection = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(basis @ basis.T, expected_projection, atol=1e-6, rtol=1e-6)


def test_stable_snr_probe_keeps_signed_signal_and_rejects_alternating_noise() -> None:
    name = "model.layers.0.self_attn.q_proj"
    accumulator = StableSNRGradientSubspaceAccumulator(
        window_size=2,
        num_windows=3,
        rank_bins=[1, 2],
        scaling_ratio=2.0,
        target_mean_rank=1.0,
        snr_ridge=0.05,
        clip_factor=10.0,
        parameter_costs={name: 8},
    )
    for _ in range(3):
        positive_noise = torch.tensor([[2.0], [4.0], [0.0], [0.0]])
        negative_noise = torch.tensor([[2.0], [-4.0], [0.0], [0.0]])
        accumulator.add(name, positive_noise)
        assert accumulator.finish_observation() is False
        accumulator.add(name, negative_noise)
        assert accumulator.finish_observation() is True

    diagnostics, bases, metrics = accumulator.snapshot()

    assert diagnostics[name]["selected_rank"] == 1
    assert diagnostics[name]["heldout_capture_selected"] == pytest.approx(1.0, abs=1e-5)
    assert bases[name].shape == (4, 1)
    assert bases[name][0, 0].abs() == pytest.approx(1.0, abs=1e-5)
    assert bases[name][1:, 0].abs().max().item() < 1e-5
    assert metrics["rank_mean"] == 1.0
    assert metrics["rank_loo_mae"] == 0.0


def test_global_rank_budget_prefers_heldout_gain() -> None:
    captures = {
        "stable": {1: 0.50, 2: 0.95},
        "noisy": {1: 0.50, 2: 0.51},
    }
    ranks, used, budget = _allocate_rank_budget(
        captures,
        parameter_costs={"stable": 10, "noisy": 10},
        rank_bins=(1, 2),
        target_mean_rank=1.5,
    )

    assert ranks == {"stable": 2, "noisy": 1}
    assert used == budget == 30


def test_grouped_normalized_budget_prevents_cross_family_rank_collapse() -> None:
    captures = {
        "layer.0.q_proj": {1: 0.40, 2: 0.90},
        "layer.1.q_proj": {1: 0.50, 2: 0.80},
        "layer.0.down_proj": {1: 0.01, 2: 0.02},
        "layer.1.down_proj": {1: 0.02, 2: 0.10},
    }
    groups = {name: name.rsplit(".", 1)[-1] for name in captures}
    ranks, used, budget = _allocate_rank_budget(
        captures,
        parameter_costs={
            "layer.0.q_proj": 10,
            "layer.1.q_proj": 10,
            "layer.0.down_proj": 100,
            "layer.1.down_proj": 100,
        },
        rank_bins=(1, 2),
        target_mean_rank=1.5,
        normalize_by_max_capture=True,
        allocation_groups=groups,
    )

    assert sum(rank for name, rank in ranks.items() if name.endswith("q_proj")) == 3
    assert sum(rank for name, rank in ranks.items() if name.endswith("down_proj")) == 3
    assert ranks["layer.1.down_proj"] == 2
    assert used == budget == 330


def test_stable_snr_uses_separate_discovery_calibration_and_validation_windows() -> None:
    name = "model.layers.0.self_attn.q_proj"
    accumulator = StableSNRGradientSubspaceAccumulator(
        window_size=2,
        num_windows=3,
        calibration_windows=1,
        validation_windows=1,
        rank_bins=[1, 2],
        scaling_ratio=2.0,
        target_mean_rank=1.0,
        snr_ridge=1.0,
        clip_factor=10.0,
        parameter_costs={name: 8},
        normalize_rank_utility=True,
        balance_rank_by_module_type=True,
    )
    discovery_or_calibration = torch.tensor([[3.0], [0.0], [0.0], [0.0]])
    independent_validation = torch.tensor([[0.0], [3.0], [0.0], [0.0]])

    for observation in range(10):
        value = discovery_or_calibration if observation < 8 else independent_validation
        accumulator.add(name, value)
        accumulator.finish_observation()
        assert accumulator.ready is (observation == 9)

    diagnostics, bases, metrics = accumulator.snapshot()

    assert diagnostics[name]["heldout_capture_selected"] == pytest.approx(1.0)
    assert diagnostics[name]["validation_capture_selected"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["discovery_windows"] == 3.0
    assert metrics["calibration_windows"] == 1.0
    assert metrics["validation_windows"] == 1.0
    assert metrics["validation_capture_selected_mean"] == pytest.approx(0.0, abs=1e-6)
    assert bases[name][:, 0].abs() == pytest.approx(torch.tensor([1.0, 0.0, 0.0, 0.0]), abs=1e-6)


def test_dominant_atoms_prefer_recurrent_direction_over_stronger_window_specific_atoms(
    tmp_path: Path,
) -> None:
    name = "model.layers.0.self_attn.q_proj"
    accumulator = DominantAtomGradientSubspaceAccumulator(
        window_size=2,
        num_windows=4,
        calibration_windows=1,
        validation_windows=1,
        rank_bins=[1, 2],
        scaling_ratio=2.0,
        target_mean_rank=1.0,
        clip_factor=10.0,
        parameter_costs={name: 8},
        local_atoms=2,
        gap_cap=4.0,
    )

    for window in range(4):
        value = torch.zeros(7, 2)
        value[0, 0] = -2.0 if window % 2 else 2.0
        value[window + 1, 1] = 4.0
        accumulator.add(name, value)
        accumulator.finish_observation()
        accumulator.add(name, value)
        accumulator.finish_observation()

    calibration = torch.zeros(7, 2)
    calibration[0, 0] = 3.0
    for _ in range(2):
        accumulator.add(name, calibration)
        accumulator.finish_observation()
    validation = torch.zeros(7, 2)
    validation[6, 0] = 3.0
    for _ in range(2):
        accumulator.add(name, validation)
        accumulator.finish_observation()

    diagnostics, bases, metrics = accumulator.snapshot()

    assert accumulator.ready
    assert diagnostics[name]["selected_rank"] == 1
    assert bases[name][:, 0].abs() == pytest.approx(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), abs=1e-5)
    assert diagnostics[name]["heldout_dominant_capture_selected"] == pytest.approx(1.0, abs=1e-5)
    assert diagnostics[name]["validation_dominant_capture_selected"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["rank_mean"] == 1.0
    sketches, atoms = accumulator.artifact_tensors()
    assert len(sketches) == 6
    assert len(atoms) == 12
    assert all(torch.isfinite(value).all() for value in (*sketches.values(), *atoms.values()))

    collector = GradientProbeCollector.__new__(GradientProbeCollector)
    collector.output_dir = tmp_path
    collector.probe_method = "dominant_atoms"
    collector.accumulator = accumulator
    collector.optimizer_updates = 12
    collector.stable_windows = 0
    collector.last_diagnostics = diagnostics
    collector.last_bases = bases
    collector.last_snapshot_metrics = metrics
    collector._export(12, stopped_by_stability=False)
    assert (tmp_path / "subspaces.safetensors").is_file()
    assert (tmp_path / "rank_map.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "window_sketches.safetensors").is_file()
    assert (tmp_path / "window_atoms.safetensors").is_file()
    assert (tmp_path / "window_metrics.json").is_file()


def test_persisted_window_sketch_round_trip() -> None:
    observations = torch.randn(2, 7, 3)
    mean = observations.mean(dim=0)
    signal = mean * math.sqrt(2)
    residuals = observations - mean.unsqueeze(0)
    noise = residuals.permute(1, 0, 2).reshape(7, -1)
    persisted = observations.permute(1, 0, 2).reshape(7, -1)

    reconstructed_signal, reconstructed_noise = _window_factors(persisted, window_size=2)

    assert torch.allclose(reconstructed_signal, signal)
    assert torch.allclose(reconstructed_noise, noise)


def test_collector_builds_dominant_atom_probe_from_runtime_config(tmp_path: Path) -> None:
    model = tiny_lora_model(rank=8, alpha=8, width=32)
    initialize_gradient_probe(model, {"gradient_probe_width": 8, "gradient_probe_seed": 42})
    collector = GradientProbeCollector(
        model,
        {
            "gradient_probe_output_dir": str(tmp_path),
            "gradient_probe_method": "dominant_atoms",
            "gradient_probe_width": 8,
            "gradient_probe_rank_bins": [4, 8, 12, 16],
            "gradient_probe_window_size": 2,
            "gradient_probe_num_windows": 4,
            "gradient_probe_calibration_windows": 1,
            "gradient_probe_validation_windows": 1,
            "gradient_probe_target_mean_rank": 8.0,
            "gradient_probe_local_atoms": 2,
            "gradient_probe_gap_cap": 4.0,
            "gradient_subspace_scaling": 2.0,
        },
    )

    assert collector.probe_method == "dominant_atoms"
    assert isinstance(collector.accumulator, DominantAtomGradientSubspaceAccumulator)
    assert collector.accumulator.required_windows == 6
    assert collector.accumulator.local_atoms == 2


def _make_unwrapped_collector(tmp_path: Path) -> GradientProbeCollector:
    collector = GradientProbeCollector.__new__(GradientProbeCollector)
    collector.output_dir = tmp_path
    collector.min_steps = 1
    collector.max_steps = 10
    collector.patience = 2
    collector.overlap_threshold = 0.99
    collector.rank_tolerance = 0.0
    collector.probe_method = "energy"
    collector.accumulator = GradientSubspaceAccumulator(
        capacity=2,
        target_energy=0.90,
        rank_bins=[1, 2],
        scaling_ratio=2.0,
    )
    collector.module_names = {0: "model.layers.0.self_attn.q_proj"}
    collector.optimizer_updates = 3
    collector.stable_windows = 0
    collector.ready = False
    collector.last_diagnostics = {}
    collector.last_bases = {}
    collector.last_snapshot_metrics = {}
    return collector


def test_stability_patience_exports_loadable_artifacts(tmp_path: Path) -> None:
    collector = _make_unwrapped_collector(tmp_path)
    sketch = torch.tensor([[4.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

    readiness = []
    for step in (1, 2, 3):
        collector.accumulator.add("model.layers.0.self_attn.q_proj", sketch)
        readiness.append(collector.finish_trainer_step(step)["gradient_probe/artifact_ready"])

    assert readiness == [0.0, 0.0, 1.0]
    assert (tmp_path / "subspaces.safetensors").is_file()
    assert (tmp_path / "rank_map.json").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_by_stability"] is True
    patterns = load_oracle_lora_patterns(tmp_path / "rank_map.json", base_rank=2, base_alpha=4)
    assert patterns.rank_pattern == {"model.layers.0.self_attn.q_proj": 1}
    assert patterns.alpha_pattern == {"model.layers.0.self_attn.q_proj": 2}


def test_exported_basis_initializes_fresh_lora_exactly(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    basis = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    tensor_path = tmp_path / "subspaces.safetensors"
    save_file({"model.layers.0.self_attn.q_proj": basis}, tensor_path)
    model = tiny_lora_model(rank=2, alpha=4)

    stats = apply_gradient_subspace_initialization(
        model,
        SimpleNamespace(gradient_subspace_path=str(tensor_path)),
    )
    layer = lora_layer(model)
    actual_a = layer.lora_A["default"].weight.detach().float()
    actual_b = layer.lora_B["default"].weight.detach().float()

    assert torch.equal(actual_a, basis)
    assert torch.count_nonzero(actual_b).item() == 0
    assert torch.allclose(actual_a @ actual_a.T, torch.eye(2))
    assert stats.num_layers == 1
    assert stats.orthogonality_error_max == 0.0
    assert stats.b_abs_max == 0.0


def test_fixed_a_lora_only_updates_b() -> None:
    torch.manual_seed(3)
    model = tiny_lora_model(rank=2, alpha=4)
    layer = lora_layer(model)
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(torch.eye(2, 6))
        layer.lora_B["default"].weight.zero_()
    stats = freeze_lora_a_factors(model)
    a_before = layer.lora_A["default"].weight.detach().clone()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2)

    model(torch.randn(5, 6)).square().sum().backward()
    assert layer.lora_A["default"].weight.grad is None
    assert layer.lora_B["default"].weight.grad is not None
    assert torch.isfinite(layer.lora_B["default"].weight.grad).all()
    optimizer.step()

    assert stats.a_parameters == 12
    assert stats.b_parameters == 12
    assert stats.unexpected_trainable_parameters == 0
    assert torch.equal(layer.lora_A["default"].weight, a_before)
    assert torch.count_nonzero(layer.lora_B["default"].weight).item() > 0


def test_integration_points_and_launchers_are_wired() -> None:
    actor_source = (REPO_ROOT / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    worker_source = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
    trainer_source = (REPO_ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    base_launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh").read_text(
        encoding="utf-8"
    )
    final_launcher = (
        REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_gradient_subspace_init_1p5b_4gpu_8k.sh"
    ).read_text(encoding="utf-8")

    assert "capture_and_refresh()" in actor_source
    assert "_optimizer_step(apply_update=not is_gradient_probe)" in actor_source
    assert 'self._peft_type == "grad_probe"' in worker_source
    assert 'self._peft_type == "grad_subspace"' in worker_source
    assert 'is_lora=self._is_peft and self._peft_type != "grad_probe"' in worker_source
    assert 'if self._peft_type == "grad_probe":' in worker_source
    assert "self.use_orig_params = True" in worker_source
    assert "gradient_probe/artifact_ready" in trainer_source
    assert "gradient_probe_output_dir" in base_launcher
    assert 'export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"' in final_launcher
    assert 'export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"' in final_launcher
    assert 'export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"' in final_launcher
    assert (REPO_ROOT / "scripts/local/start_gradient_probe_4gpu.sh").is_file()
    assert (REPO_ROOT / "scripts/local/start_gradient_probe_stable_snr_4gpu.sh").is_file()
    assert (REPO_ROOT / "scripts/local/start_gradient_probe_dominant_atoms_4gpu.sh").is_file()
    assert (REPO_ROOT / "scripts/local/start_gradient_subspace_init_4gpu.sh").is_file()
    assert (REPO_ROOT / "scripts/local/start_stable_snr_fixed_a_4gpu.sh").is_file()
    assert (REPO_ROOT / "scripts/local/start_dominant_atoms_trainable_a_4gpu.sh").is_file()


def test_target_name_normalization_is_stable_across_peft_and_fsdp_wrappers() -> None:
    raw = "_fsdp_wrapped_module.base_model.model.model.layers.7.mlp.down_proj"
    assert normalize_target_name(raw) == "model.layers.7.mlp.down_proj"
