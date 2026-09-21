from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.analysis.build_spar_lora_allocations import build_allocations
from scripts.analysis.summarize_spar_lora_runs import parse_step_metrics, summarize_run
from verl.utils.peft_spar_lora import (
    SparPositiveProbeCollector,
    initialize_spar_probe,
    validate_spar_artifact,
)


class TinyBlock(torch.nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].self_attn = torch.nn.Module()
        self.model.layers[0].self_attn.q_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0].self_attn.q_proj(inputs)


def tiny_model(rank: int = 2, scaling: float = 2.0, width: int = 6):
    return get_peft_model(
        TinyBlock(width),
        LoraConfig(
            r=rank,
            lora_alpha=int(rank * scaling),
            lora_dropout=0.0,
            target_modules=["q_proj"],
            bias="none",
        ),
    )


def lora_layer(model):
    return next(module for module in model.modules() if hasattr(module, "lora_A") and "default" in module.lora_A)


def test_spar_probe_is_zero_function_with_constant_scaling() -> None:
    torch.manual_seed(1)
    model = tiny_model(rank=3, scaling=2.0)
    layer = lora_layer(model)
    inputs = torch.randn(4, 6)
    with model.disable_adapter():
        base = model(inputs).detach()

    stats = initialize_spar_probe(
        model,
        {"spar_r_max": 3, "gradient_probe_seed": 9, "gradient_subspace_scaling": 2.0},
    )

    assert stats["scaling"] == 2.0
    assert float(layer.scaling["default"]) == 2.0
    assert torch.count_nonzero(layer.lora_A["default"].weight).item() == 0
    assert torch.count_nonzero(layer.lora_B["default"].weight).item() > 0
    assert torch.equal(model(inputs), base)


def test_positive_probe_exports_cpu_valid_f_s_r_p_u(tmp_path: Path) -> None:
    torch.manual_seed(2)
    model = tiny_model(rank=2, scaling=2.0)
    initialize_spar_probe(
        model,
        {"spar_r_max": 2, "gradient_probe_seed": 7, "gradient_subspace_scaling": 2.0},
    )
    collector = SparPositiveProbeCollector(
        model,
        {
            "gradient_probe_output_dir": str(tmp_path),
            "spar_r_max": 2,
            "spar_discovery_samples": 2,
            "spar_calibration_samples": 2,
            "spar_sample_clip_factor": 2.5,
            "gradient_subspace_scaling": 2.0,
        },
    )
    layer = lora_layer(model)
    base_weight = layer.base_layer.weight.detach().clone()

    for index in range(4):
        model.zero_grad(set_to_none=True)
        inputs = torch.randn(3, 6) + index
        target = torch.randn(3, 6)
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        collector.capture_sample(
            loss=float(loss.item()),
            prompt_id=f"prompt-{index}",
            response_length=10 + index,
        )

    assert collector.ready
    assert torch.equal(layer.base_layer.weight, base_weight)
    assert torch.count_nonzero(layer.lora_B["default"].weight).item() == 0
    assert validate_spar_artifact(tmp_path)["module_count"] == 1.0
    summary = json.loads((tmp_path / "probe_summary.json").read_text())
    assert summary["prompt_splits_disjoint"] is True
    assert summary["discovery_prompt_ids"] == ["prompt-0", "prompt-1"]
    assert summary["calibration_prompt_ids"] == ["prompt-2", "prompt-3"]

    name = next(iter(summary["modules"]))
    with safe_open(tmp_path / "atom_scores.safetensors", framework="pt", device="cpu") as scores:
        f_score = scores.get_tensor(f"{name}.F").double()
        s_score = scores.get_tensor(f"{name}.S").double()
        r_score = scores.get_tensor(f"{name}.R").double()
        p_score = scores.get_tensor(f"{name}.P").double()
        u_score = scores.get_tensor(f"{name}.U").double()
    assert torch.allclose(r_score, s_score / (f_score + 1e-12), atol=1e-6, rtol=1e-5)
    assert torch.allclose(p_score, f_score / (f_score.sum() + 1e-12), atol=1e-6, rtol=1e-5)
    assert torch.allclose(u_score, p_score * r_score, atol=1e-6, rtol=1e-5)


def test_allocator_uses_complete_atoms_and_respects_uniform_budget(tmp_path: Path) -> None:
    artifact = tmp_path / "probe"
    artifact.mkdir()
    names = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
    shapes = {names[0]: (4, 4), names[1]: (8, 4)}
    candidates = {name: torch.eye(4) for name in names}
    save_file(candidates, artifact / "candidates.safetensors")
    score_tensors = {}
    for offset, name in enumerate(names):
        f_score = torch.tensor([4.0, 3.0, 2.0, 1.0]) + offset
        s_score = f_score * torch.tensor([0.9, 0.8, 0.2, 0.1])
        r_score = s_score / f_score
        p_score = f_score / f_score.sum()
        u_score = p_score * r_score
        for label, value in (("F", f_score), ("S", s_score), ("R", r_score), ("P", p_score), ("U", u_score)):
            score_tensors[f"{name}.{label}"] = value
        score_tensors[f"{name}.singular_values"] = torch.ones(4)
    save_file(score_tensors, artifact / "atom_scores.safetensors")
    summary = {
        "schema_version": 1,
        "method": "test",
        "r_max": 4,
        "constant_scaling": 2.0,
        "prompt_splits_disjoint": True,
        "modules": {
            name: {"shape": list(shapes[name]), "candidate_rank": 4}
            for name in names
        },
    }
    (artifact / "probe_summary.json").write_text(json.dumps(summary))

    result = build_allocations(artifact, tmp_path / "allocations", r_min=1, uniform_rank=2)

    uniform = result["uniform"]
    adaptive = result["adaptive"]
    assert adaptive["trainable_parameters"] <= uniform["trainable_parameters"]
    assert adaptive["budget_respected"] is True
    assert all(item["rank"] >= 1 for item in adaptive["modules"].values())
    assert all(item["rank"] <= 4 for item in adaptive["modules"].values())
    assert all(len(item["atom_indices"]) == item["rank"] for item in adaptive["modules"].values())
    assert all(alpha / rank == pytest.approx(2.0) for rank, alpha in zip(
        adaptive["rank_pattern"].values(), adaptive["alpha_pattern"].values(), strict=True
    ))


def test_spar_integration_files_and_settings_are_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "verl/trainer/ppo/ray_trainer.py").read_text()
    actor = (root / "verl/workers/actor/dp_actor.py").read_text()
    worker = (root / "verl/workers/fsdp_workers.py").read_text()
    assert "_build_spar_probe_batch" in trainer
    assert "Checkpoint step {self.global_steps} already reached total training steps" in trainer
    assert "run_spar_probe" in actor
    assert 'self._peft_type == "spar_probe"' in worker
    assert (root / "scripts/analysis/build_spar_lora_allocations.py").is_file()
    assert (root / "scripts/local/start_spar_probe_4gpu.sh").is_file()
    assert (root / "scripts/local/start_spar_uniform_r8_4gpu.sh").is_file()
    assert (root / "scripts/local/start_spar_adaptive_eqr8_4gpu.sh").is_file()


def test_spar_log_summary_computes_normalized_reward_auc(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    lines = []
    for step in range(1, 51):
        reward = step / 100
        lines.append(
            f"\x1b[36m(TaskRunner pid=1)\x1b[0m step:{step}"
            f" - critic/score/mean:{reward}"
            f" - reward_extra/acc/mean:{reward}"
            " - response_length/mean:100"
            " - actor/entropy:0.5"
            " - actor/ppo_kl:0.01"
            " - actor/pg_loss:0.02"
            " - actor/grad_norm:0.03"
            " - actor/pg_clipfrac:0.04"
            " - timing_s/step:10"
            " - perf/max_memory_allocated_gb:20"
            " - actor/active_parameter_count:123"
            " - spar_structure/rank_mean:8"
        )
    log.write_text("\n".join(lines) + "\n")

    records = parse_step_metrics(log)
    summary = summarize_run(log)

    assert records[20]["critic/score/mean"] == pytest.approx(0.2)
    assert summary["reward_at_50"] == pytest.approx(0.5)
    assert summary["reward_auc_1_20"] == pytest.approx(0.105)
    assert summary["reward_auc_1_50"] == pytest.approx(0.255)

    screen = summarize_run(log, expected_steps=10)
    assert screen["reward_at_10"] == pytest.approx(0.1)
    assert screen["reward_auc_1_10"] == pytest.approx(0.055)
    assert screen["reward_at_20"] is None
    assert screen["reward_auc_1_20"] is None
    assert screen["reward_at_50"] is None
    assert screen["reward_auc_1_50"] is None
    assert summary["step_time_mean_s_1_50"] == pytest.approx(10.0)
    assert summary["peak_allocated_memory_gb"] == pytest.approx(5.0)
    assert summary["peak_allocated_memory_gb_aggregate"] == pytest.approx(20.0)
