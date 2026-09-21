from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from scripts.analysis.build_full_gradient_uniform_allocation import (
    build_uniform_allocation,
)
from verl.utils.full_gradient_rl_probe import (
    FullGradientRLProbeCollector,
    configure_full_gradient_probe_parameters,
    validate_full_gradient_probe_artifact,
)


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
