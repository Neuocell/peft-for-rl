from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.analysis.build_full_gradient_uniform_allocation import (
    build_adaptive_allocation,
    build_uniform_allocation,
)
from verl.utils.full_gradient_rl_probe import (
    FullGradientRLProbeCollector,
    WindowedAdamConsensusCollector,
    configure_full_gradient_probe_parameters,
    deterministic_response_offset,
    unbiased_single_response_scale,
    validate_full_gradient_probe_artifact,
    virtual_adam_update,
)
from verl.utils.peft_gradient_subspace import apply_gradient_subspace_initialization


class TinyPolicy(torch.nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(width, width, bias=False)
        layer.mlp = torch.nn.Module()
        layer.mlp.up_proj = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.model.layers[0].self_attn.q_proj(inputs)
        hidden = torch.tanh(self.model.layers[0].mlp.up_proj(hidden))
        return self.output(hidden)


def _probe_config(path: Path) -> dict:
    return {
        "target_modules": "all-linear",
        "gradient_probe_output_dir": str(path),
        "gradient_probe_seed": 17,
        "full_gradient_probe_rank": 2,
        "full_gradient_probe_sketch_width": 4,
        "full_gradient_probe_svd_oversample": 2,
        "full_gradient_probe_svd_niter": 1,
        "full_gradient_probe_svd_device": "cpu",
        "full_gradient_probe_discovery_prompts": 2,
        "full_gradient_probe_calibration_prompts": 2,
        "full_gradient_probe_audit_prompts": 1,
        "full_gradient_probe_clip_factor": 2.5,
        "full_gradient_probe_confidence_z": 1.0,
    }


def _windowed_probe_config(path: Path) -> dict:
    config = _probe_config(path)
    config.update(
        {
            "full_gradient_probe_mode": "windowed_adam_consensus",
            "full_gradient_probe_window_prompts": 2,
            "full_gradient_probe_discovery_windows": 3,
            "full_gradient_probe_calibration_windows": 2,
            "full_gradient_probe_audit_windows": 2,
            "full_gradient_probe_local_atoms": 1,
            "full_gradient_probe_adam_beta1": 0.5,
            "full_gradient_probe_adam_beta2": 0.75,
            "full_gradient_probe_adam_eps": 1e-6,
            "full_gradient_probe_future_lcb_z": 1.0,
        }
    )
    return config


def test_single_response_estimator_is_unbiased_and_deterministic() -> None:
    response_losses = torch.tensor([-0.5, 0.25, 1.5, -0.75])
    full_group_loss = response_losses.sum()
    estimates = torch.stack(
        [
            response_losses[index]
            * unbiased_single_response_scale(len(response_losses), 1)
            for index in range(len(response_losses))
        ]
    )
    assert estimates.mean() == pytest.approx(float(full_group_loss))
    first = deterministic_response_offset(42, "prompt-7", 3, 8)
    assert first == deterministic_response_offset(42, "prompt-7", 3, 8)
    assert 0 <= first < 8


def test_virtual_adam_uses_bias_correction() -> None:
    first = torch.zeros(2)
    second = torch.zeros(2)
    gradient = torch.tensor([2.0, -4.0])
    update = virtual_adam_update(
        first,
        second,
        gradient,
        step=1,
        beta1=0.5,
        beta2=0.75,
        eps=0.0,
    )
    assert torch.allclose(update, torch.tensor([1.0, -1.0]))


def test_windowed_adam_consensus_exports_cross_fit_candidates(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = TinyPolicy()
    config = _windowed_probe_config(tmp_path)
    configure_full_gradient_probe_parameters(model, config)
    collector = WindowedAdamConsensusCollector(model, config)

    total_windows = 7
    for window in range(total_windows):
        model.zero_grad(set_to_none=True)
        inputs = torch.randn(4, 6) + window / 10
        loss = model(inputs).square().mean() * (1.0 + window / 5)
        loss.backward()
        collector.capture_window(
            loss=float(loss.item()),
            prompt_ids=[f"prompt-{window}-0", f"prompt-{window}-1"],
            response_counts=[8, 8],
            response_tokens=[24 + window, 28 + window],
            selected_response_offsets=[window % 8, (window + 1) % 8],
            advantage_rms=[0.5 + window, 0.75 + window],
        )

    assert collector.ready
    summary = json.loads((tmp_path / "probe_summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["selection_uses_audit"] is False
    assert summary["held_out_scoring_space"] == "raw_full_policy_gradient"
    assert len(summary["discovery_prompt_ids"]) == 6
    assert len(summary["calibration_prompt_ids"]) == 4
    assert len(summary["audit_prompt_ids"]) == 4
    assert summary["prompt_splits_disjoint"] is True

    for method in ("raw_momentum", "adam_update", "consensus_hybrid"):
        validation = validate_full_gradient_probe_artifact(
            tmp_path, candidate_method=method
        )
        assert validation["module_count"] == 2
        with safe_open(
            tmp_path / f"atom_scores_{method}.safetensors",
            framework="pt",
            device="cpu",
        ) as score_file:
            for name in summary["modules"]:
                p_lcb = score_file.get_tensor(f"{name}.P_lcb")
                recurrence = score_file.get_tensor(f"{name}.recurrence")
                utility = score_file.get_tensor(f"{name}.U")
                assert torch.allclose(utility, recurrence * p_lcb.clamp_min(0))
                assert torch.isfinite(score_file.get_tensor(f"{name}.audit_U")).all()

    allocation = build_uniform_allocation(
        tmp_path,
        tmp_path / "windowed-allocation",
        candidate_method="consensus_hybrid",
        uniform_rank=1,
        selection_utility="future_lcb",
    )
    assert allocation["selection_utility"] == "future_lcb"
    assert allocation["constant_scaling"] == pytest.approx(2.0)


def test_full_gradient_probe_exports_three_cross_fit_candidate_sets(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4)
    model = TinyPolicy()
    config = _probe_config(tmp_path)
    setup = configure_full_gradient_probe_parameters(model, config)
    assert setup["num_layers"] == 2
    assert model.output.weight.requires_grad is False
    collector = FullGradientRLProbeCollector(model, config)

    for index in range(5):
        model.zero_grad(set_to_none=True)
        inputs = torch.randn(4, 6) + index / 5
        signed_target = torch.randn(4, 6)
        loss = ((model(inputs) - signed_target) * (1 if index % 2 else -1)).mean()
        loss.backward()
        collector.capture_group(
            loss=float(loss.item()),
            prompt_id=f"prompt-{index}",
            response_count=2,
            response_tokens=20 + index,
            advantage_rms=0.5 + index,
        )

    assert collector.ready
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))
    assert summary["prompt_splits_disjoint"] is True
    assert summary["discovery_prompt_ids"] == ["prompt-0", "prompt-1"]
    assert summary["calibration_prompt_ids"] == ["prompt-2", "prompt-3"]
    assert summary["audit_prompt_ids"] == ["prompt-4"]

    for method in ("mean", "covariance", "hybrid"):
        validation = validate_full_gradient_probe_artifact(
            tmp_path, candidate_method=method
        )
        assert validation["module_count"] == 2
        with safe_open(
            tmp_path / f"atom_scores_{method}.safetensors", framework="pt", device="cpu"
        ) as scores:
            name = next(iter(summary["modules"]))
            gain = scores.get_tensor(f"{name}.gain")
            gain_lcb = scores.get_tensor(f"{name}.gain_lcb")
            utility = scores.get_tensor(f"{name}.U")
            assert torch.allclose(utility, gain_lcb.clamp_min(0))
            assert torch.isfinite(gain).all()
            assert torch.isfinite(scores.get_tensor(f"{name}.audit_gain")).all()

    score_path = tmp_path / "atom_scores_mean.safetensors"
    with safe_open(score_path, framework="pt", device="cpu") as score_file:
        tensors = {key: score_file.get_tensor(key) for key in score_file.keys()}
    name = next(iter(summary["modules"]))
    tensors[f"{name}.audit_F"][0] = float("nan")
    save_file(tensors, score_path)
    with pytest.raises(ValueError, match="audit_F"):
        validate_full_gradient_probe_artifact(tmp_path, candidate_method="mean")


def test_full_gradient_uniform_allocation_uses_lcb_and_scaling_two(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    model = TinyPolicy()
    config = _probe_config(artifact)
    configure_full_gradient_probe_parameters(model, config)
    collector = FullGradientRLProbeCollector(model, config)
    torch.manual_seed(9)
    for index in range(5):
        model.zero_grad(set_to_none=True)
        output = model(torch.randn(3, 6))
        loss = output.square().mean() * (index + 1)
        loss.backward()
        collector.capture_group(
            loss=float(loss.item()),
            prompt_id=str(index),
            response_count=2,
            response_tokens=12,
            advantage_rms=1.0,
        )

    result = build_uniform_allocation(
        artifact,
        tmp_path / "allocation",
        candidate_method="covariance",
        uniform_rank=1,
    )
    assert result["candidate_method"] == "covariance"
    assert result["structure"]["active_rank_mean"] == pytest.approx(1.0)
    assert result["constant_scaling"] == pytest.approx(2.0)
    rank_map = json.loads(
        (tmp_path / "allocation/rank_map.json").read_text(encoding="utf-8")
    )
    assert all(rank == 1 for rank in rank_map["rank_pattern"].values())
    assert all(alpha == 2 for alpha in rank_map["alpha_pattern"].values())

    uniform_r2 = build_uniform_allocation(
        artifact,
        tmp_path / "uniform-r2",
        candidate_method="covariance",
        uniform_rank=2,
    )
    adaptive = build_adaptive_allocation(
        artifact,
        tmp_path / "adaptive",
        candidate_method="covariance",
        uniform_rank=2,
        r_min=1,
    )
    assert adaptive["allocation_mode"] == "adaptive"
    assert adaptive["budget_respected"] is True
    assert adaptive["trainable_parameters"] <= uniform_r2["trainable_parameters"]
    assert adaptive["constant_scaling"] == pytest.approx(2.0)
    assert all(item["rank"] >= 1 for item in adaptive["modules"].values())
    assert all(item["rank"] <= 4 for item in adaptive["modules"].values())
    assert sum(item["rank"] for item in adaptive["modules"].values()) == 4
    assert all(
        len(item["atom_indices"]) == item["rank"]
        for item in adaptive["modules"].values()
    )

    stable_energy = build_adaptive_allocation(
        artifact,
        tmp_path / "adaptive-stable-energy",
        candidate_method="covariance",
        uniform_rank=2,
        r_min=1,
        adaptive_utility="stable_energy",
    )
    assert stable_energy["adaptive_utility"] == "stable_energy"
    assert stable_energy["budget_respected"] is True
    assert stable_energy["trainable_parameters"] <= uniform_r2["trainable_parameters"]

    matched_uniform = build_uniform_allocation(
        artifact,
        tmp_path / "uniform-stable-energy",
        candidate_method="covariance",
        uniform_rank=1,
        selection_utility="stable_energy",
    )
    assert matched_uniform["selection_utility"] == "stable_energy"
    assert matched_uniform["trainable_parameters"] == result["trainable_parameters"]
    with safe_open(
        artifact / "atom_scores_covariance.safetensors", framework="pt", device="cpu"
    ) as score_file:
        for name, item in matched_uniform["modules"].items():
            utility = score_file.get_tensor(f"{name}.P") * score_file.get_tensor(
                f"{name}.R"
            )
            assert item["atom_indices"] == [int(utility.argmax().item())]


def test_full_gradient_probe_integration_is_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    actor = (root / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    worker = (root / "verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
    assert "_build_full_gradient_probe_batch" in trainer
    assert "run_full_gradient_probe" in actor
    assert 'self._peft_type == "full_gradient_probe"' in worker
    assert (root / "scripts/local/start_full_gradient_rl_probe_4gpu.sh").is_file()
    assert (root / "scripts/local/start_full_gradient_uniform_r8_4gpu.sh").is_file()
    assert (root / "scripts/local/start_full_gradient_adaptive_eqr8_4gpu.sh").is_file()
    windowed_probe = root / "scripts/local/start_windowed_adam_consensus_probe_4gpu.sh"
    windowed_train = root / "scripts/local/start_windowed_consensus_uniform_r8_50_4gpu.sh"
    assert windowed_probe.is_file()
    assert windowed_train.is_file()
    probe_launcher = windowed_probe.read_text()
    train_launcher = windowed_train.read_text()
    for launcher in (probe_launcher, train_launcher):
        assert "export TRAIN_PROMPT_BSZ=64" in launcher
        assert "export TRAIN_PROMPT_MINI_BSZ=16" in launcher
        assert "export N_RESP_PER_PROMPT=8" in launcher
    assert "FULL_GRADIENT_PROBE_MODE=windowed_adam_consensus" in probe_launcher
    assert "export LORA_RANK=8" in train_launcher
    assert "export LORA_ALPHA=16" in train_launcher
    uniform_launcher = (root / "scripts/local/start_full_gradient_uniform_r8_4gpu.sh").read_text()
    assert 'export LORA_RANK="${LORA_RANK:-8}"' in uniform_launcher
    assert 'export LORA_ALPHA="${LORA_ALPHA:-16}"' in uniform_launcher
    adaptive_launcher = (root / "scripts/local/start_full_gradient_adaptive_eqr8_4gpu.sh").read_text()
    assert "export LORA_RANK=32" in adaptive_launcher
    assert "export LORA_ALPHA=64" in adaptive_launcher


def test_heterogeneous_probe_lora_saves_restores_and_merges(tmp_path: Path) -> None:
    torch.manual_seed(13)
    base = TinyPolicy()
    restored_base = copy.deepcopy(base)
    model = get_peft_model(
        base,
        LoraConfig(
            r=1,
            lora_alpha=2,
            target_modules=["q_proj", "up_proj"],
            rank_pattern={"q_proj": 1, "up_proj": 2},
            alpha_pattern={"q_proj": 2, "up_proj": 4},
        ),
    )
    subspace_path = tmp_path / "subspaces.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj": torch.eye(6)[:1].contiguous(),
            "model.layers.0.mlp.up_proj": torch.eye(6)[:2].contiguous(),
        },
        subspace_path,
    )
    apply_gradient_subspace_initialization(
        model, SimpleNamespace(gradient_subspace_path=str(subspace_path))
    )
    inputs = torch.randn(3, 6)
    with model.disable_adapter():
        base_output = model(inputs).detach()
    assert torch.equal(model(inputs), base_output)
    layers = [module for module in model.modules() if hasattr(module, "lora_A")]
    assert sorted(module.r["default"] for module in layers) == [1, 2]
    assert all(module.scaling["default"] == 2 for module in layers)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.01
    )
    model(inputs).square().mean().backward()
    optimizer.step()
    trained_output = model(inputs).detach()
    assert not torch.equal(trained_output, base_output)

    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir)
    restored = PeftModel.from_pretrained(restored_base, adapter_dir)
    assert torch.allclose(restored(inputs), trained_output, atol=1e-6)
    merged = restored.merge_and_unload()
    assert torch.allclose(merged(inputs), trained_output, atol=1e-6)
