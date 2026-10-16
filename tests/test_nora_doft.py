from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from verl.utils.fsdp_utils import collect_nora_doft_adapter_params, collect_nora_doft_full_params
from verl.utils.peft_nora_doft import NoRADoftLinear, apply_nora_doft_adapters

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self, width: int = 8, out_features: int = 6):
        super().__init__()
        self.q_proj = torch.nn.Linear(width, out_features, bias=True)

    def forward(self, inputs):
        return self.q_proj(inputs)


def nora_doft_config(**overrides):
    config = {
        "nora_doft_rank": 2,
        "nora_doft_block_size": 4,
        "nora_doft_alpha": 2,
        "nora_doft_normalization": "init",
        "nora_doft_use_magnitude": True,
        "nora_doft_eps": 1e-6,
        "nora_doft_seed": 17,
        "target_modules": "q_proj",
    }
    config.update(overrides)
    return config


def test_nora_doft_is_exact_identity_with_nora_initialization():
    torch.manual_seed(3)
    model = TinyModel()
    inputs = torch.randn(4, 8)
    base_output = model(inputs).detach().clone()

    stats = apply_nora_doft_adapters(model, nora_doft_config())
    wrapped = model.q_proj

    assert isinstance(wrapped, NoRADoftLinear)
    assert torch.equal(model(inputs), base_output)
    assert torch.count_nonzero(wrapped.nora_doft_B).item() == 0
    column_norms = torch.linalg.vector_norm(wrapped.nora_doft_A.float(), dim=-2)
    assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-6, rtol=0)
    assert torch.equal(wrapped.nora_doft_magnitude, wrapped.nora_doft_base_row_norm)
    assert stats.num_layers == 1
    assert stats.num_parameters == 38
    assert stats.normalization_error_max <= 1e-6


def test_nora_doft_first_step_gradient_structure():
    model = TinyModel()
    apply_nora_doft_adapters(model, nora_doft_config())
    wrapped = model.q_proj

    model(torch.randn(5, 8)).square().mean().backward()

    assert torch.count_nonzero(wrapped.nora_doft_A.grad).item() == 0
    assert torch.count_nonzero(wrapped.nora_doft_B.grad).item() > 0
    assert torch.count_nonzero(wrapped.nora_doft_magnitude.grad).item() > 0
    assert torch.isfinite(wrapped.nora_doft_B.grad).all()
    assert torch.isfinite(wrapped.nora_doft_magnitude.grad).all()


def test_nora_doft_rotation_is_orthogonal_and_merge_matches_forward():
    model = TinyModel()
    apply_nora_doft_adapters(model, nora_doft_config())
    wrapped = model.q_proj
    with torch.no_grad():
        wrapped.nora_doft_B.normal_(mean=0.0, std=0.2)
        wrapped.nora_doft_magnitude.mul_(1.1)

    rotations = wrapped.rotations()
    eye = torch.eye(wrapped.block_size).expand_as(rotations)
    assert torch.allclose(rotations.transpose(-1, -2) @ rotations, eye, atol=1e-5, rtol=1e-5)

    inputs = torch.randn(2, 3, 8)
    direct = wrapped(inputs)
    merged_weight, merged_bias = wrapped.merged_weight_bias()
    merged = F.linear(inputs, merged_weight, merged_bias)
    assert torch.allclose(direct, merged, atol=1e-5, rtol=1e-5)
    assert torch.equal(merged_bias, wrapped.base_layer.bias)


def test_nora_doft_continuous_normalization_and_disable_adapter():
    model = TinyModel()
    inputs = torch.randn(3, 8)
    base_output = model(inputs).detach().clone()
    apply_nora_doft_adapters(model, nora_doft_config(nora_doft_normalization="continuous"))
    wrapped = model.q_proj
    with torch.no_grad():
        wrapped.nora_doft_A.mul_(7.0)
        wrapped.nora_doft_B.normal_(std=0.1)

    effective_norms = torch.linalg.vector_norm(wrapped.normalized_a(), dim=-2)
    assert torch.allclose(effective_norms, torch.ones_like(effective_norms), atol=1e-6, rtol=0)
    assert not torch.allclose(model(inputs), base_output)
    with model.disable_adapter():
        assert torch.equal(model(inputs), base_output)


def test_nora_doft_state_dict_and_full_weight_collection_round_trip():
    torch.manual_seed(5)
    base = TinyModel()
    clone = TinyModel()
    clone.load_state_dict(base.state_dict())
    apply_nora_doft_adapters(base, nora_doft_config())
    apply_nora_doft_adapters(clone, nora_doft_config())
    with torch.no_grad():
        base.q_proj.nora_doft_B.normal_(std=0.15)
        base.q_proj.nora_doft_magnitude.add_(0.2)
    clone.load_state_dict(base.state_dict())

    inputs = torch.randn(4, 8)
    assert torch.equal(base(inputs), clone(inputs))

    params = collect_nora_doft_full_params(base)
    assert set(params) == {"q_proj.weight", "q_proj.bias"}
    collected = F.linear(inputs, params["q_proj.weight"], params["q_proj.bias"])
    assert torch.allclose(base(inputs), collected, atol=1e-5, rtol=1e-5)

    adapter = collect_nora_doft_adapter_params(base)
    assert set(adapter) == {
        "q_proj.nora_doft_A",
        "q_proj.nora_doft_B",
        "q_proj.nora_doft_magnitude",
    }
    restored = TinyModel()
    restored.q_proj.load_state_dict(base.q_proj.base_layer.state_dict())
    apply_nora_doft_adapters(restored, nora_doft_config())
    incompatible = restored.load_state_dict(adapter, strict=False)
    assert not incompatible.unexpected_keys
    assert torch.equal(base(inputs), restored(inputs))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"nora_doft_rank": 0}, "generator_rank must be positive"),
        ({"nora_doft_rank": 5}, "exceeds block_size"),
        ({"nora_doft_block_size": 3}, "in_features divisible"),
        ({"nora_doft_normalization": "rows"}, "normalization mode"),
    ],
)
def test_nora_doft_validates_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        apply_nora_doft_adapters(TinyModel(), nora_doft_config(**overrides))


def test_nora_doft_launcher_and_fsdp_paths_are_wired():
    launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_nora_doft_1p5b_4gpu_8k.sh").read_text()
    base_launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh").read_text()
    paper_clean_launcher = (REPO_ROOT / "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh").read_text()
    worker = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text()
    engine = (REPO_ROOT / "verl/workers/engine/fsdp/transformer_impl.py").read_text()
    trainer = (REPO_ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    merger = (REPO_ROOT / "tina_run/scripts/local/eval/merge_nora_doft_wrapped_hf.py").read_text()

    assert "export PEFT_TYPE=nora_doft" in launcher
    assert 'export NORA_DOFT_RANK="${NORA_DOFT_RANK:-8}"' in launcher
    assert 'export NORA_DOFT_BLOCK_SIZE="${NORA_DOFT_BLOCK_SIZE:-32}"' in launcher
    assert 'export NORA_DOFT_NORMALIZATION="${NORA_DOFT_NORMALIZATION:-init}"' in launcher
    assert 'export NORA_DOFT_USE_MAGNITUDE="${NORA_DOFT_USE_MAGNITUDE:-True}"' in launcher
    assert "+actor_rollout_ref.model.nora_doft_rank" in base_launcher
    assert 'nora_doft)\n    PEFT_TYPE="nora_doft"\n    LORA_RANK=0' in paper_clean_launcher
    assert 'elif self._peft_type == "nora_doft":' in worker
    assert "collect_nora_doft_full_params" in worker
    assert 'self._peft_type in {"tinylora", "nora_doft", "nora_dora"}' in worker
    assert 'elif self._peft_type == "nora_doft":' in engine
    assert "collect_nora_doft_full_params" in engine
    assert '{"oft", "nora_doft", "nora_dora"}' in trainer
    assert "apply_nora_doft_adapters" in merger
    assert "collect_nora_doft_full_params" in merger
    assert "model.safetensors.index.json" in merger
