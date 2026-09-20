from pathlib import Path

import pytest
import torch

from verl.utils.peft_tinylora import TinyLoRALinear, apply_tinylora_adapters

REPO_ROOT = Path(__file__).resolve().parents[1]


class ProjectionBlock(torch.nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.q_proj = torch.nn.Linear(width, width, bias=True)
        self.v_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs):
        return self.v_proj(torch.tanh(self.q_proj(inputs)))


class TinyModel(torch.nn.Module):
    def __init__(self, num_layers: int = 3, width: int = 6):
        super().__init__()
        self.layers = torch.nn.ModuleList([ProjectionBlock(width) for _ in range(num_layers)])

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def tinylora_config(**overrides):
    config = {
        "tinylora_rank": 2,
        "tinylora_projection_dim": 1,
        "tinylora_tie_factor": 2,
        "tinylora_tie_strategy": "tiled",
        "tinylora_seed": 7,
        "tinylora_svd_device": "cpu",
        "tinylora_svd_method": "exact",
        "tinylora_svd_oversample": 2,
        "tinylora_svd_niter": 1,
        "tinylora_projection_std": 1.0,
        "target_modules": "q_proj,v_proj",
    }
    config.update(overrides)
    return config


def adapters(model):
    return [module for module in model.modules() if isinstance(module, TinyLoRALinear)]


def test_tinylora_is_zero_function_and_only_shared_vectors_train():
    torch.manual_seed(3)
    model = TinyModel(num_layers=3)
    inputs = torch.randn(4, 6)
    base_output = model(inputs).detach().clone()

    stats = apply_tinylora_adapters(model, tinylora_config())
    adapted_output = model(inputs)

    assert torch.equal(adapted_output, base_output)
    assert stats.num_layers == 6
    assert stats.num_vectors == 3
    assert stats.num_parameters == 3
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(trainable) == 3
    assert sum(parameter.numel() for parameter in trainable) == 3

    adapted_output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert any(torch.count_nonzero(parameter.grad).item() > 0 for parameter in trainable)


def test_tinylora_tiled_sharing_and_merged_weight_match_forward():
    model = TinyModel(num_layers=2)
    apply_tinylora_adapters(model, tinylora_config(tinylora_tie_factor=3))
    wrapped = adapters(model)

    assert wrapped[0].tinylora_vector is wrapped[1].tinylora_vector
    assert wrapped[1].tinylora_vector is wrapped[2].tinylora_vector
    assert wrapped[2].tinylora_vector is not wrapped[3].tinylora_vector

    with torch.no_grad():
        wrapped[0].tinylora_vector.fill_(0.125)
    inputs = torch.randn(5, 6)
    direct = wrapped[0](inputs)
    merged_weight, merged_bias = wrapped[0].merged_weight_bias()
    merged = torch.nn.functional.linear(inputs, merged_weight, merged_bias)
    assert torch.allclose(direct, merged, atol=1e-5, rtol=1e-5)


def test_tinylora_structured_sharing_groups_same_module_types():
    model = TinyModel(num_layers=3)
    stats = apply_tinylora_adapters(
        model,
        tinylora_config(tinylora_tie_strategy="structured", tinylora_tie_factor=2),
    )
    wrapped = adapters(model)

    q0, v0, q1, v1, q2, v2 = wrapped
    assert q0.tinylora_vector is q1.tinylora_vector
    assert v0.tinylora_vector is v1.tinylora_vector
    assert q0.tinylora_vector is not v0.tinylora_vector
    assert q2.tinylora_vector is not q0.tinylora_vector
    assert v2.tinylora_vector is not v0.tinylora_vector
    assert stats.num_vectors == 4


def test_tinylora_lowrank_svd_is_deterministic():
    torch.manual_seed(11)
    first = TinyModel(num_layers=1)
    second = TinyModel(num_layers=1)
    second.load_state_dict(first.state_dict())
    config = tinylora_config(tinylora_svd_method="lowrank")

    apply_tinylora_adapters(first, config)
    apply_tinylora_adapters(second, config)

    for left, right in zip(adapters(first), adapters(second), strict=True):
        assert torch.equal(left.tinylora_u_sigma, right.tinylora_u_sigma)
        assert torch.equal(left.tinylora_vh, right.tinylora_vh)
        assert torch.equal(left.tinylora_projection, right.tinylora_projection)


def test_tinylora_validates_rank_and_tying():
    with pytest.raises(ValueError, match="rank=7 exceeds max rank 6"):
        apply_tinylora_adapters(TinyModel(num_layers=1), tinylora_config(tinylora_rank=7))
    with pytest.raises(ValueError, match="tie_factor must be positive"):
        apply_tinylora_adapters(TinyModel(num_layers=1), tinylora_config(tinylora_tie_factor=0))


def test_tinylora_launchers_and_fsdp_paths_are_wired():
    launcher = (
        REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_tinylora_1p5b_4gpu_8k.sh"
    ).read_text()
    starter = (REPO_ROOT / "scripts/local/start_tinylora_13p_4gpu.sh").read_text()
    base_launcher = (
        REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
    ).read_text()
    worker = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text()
    engine = (REPO_ROOT / "verl/workers/engine/fsdp/transformer_impl.py").read_text()

    assert "export PEFT_TYPE=tinylora" in launcher
    assert 'export TINY_LORA_RANK="${TINY_LORA_RANK:-2}"' in launcher
    assert 'export TINY_LORA_TIE_FACTOR="${TINY_LORA_TIE_FACTOR:-16}"' in launcher
    assert 'export TINY_LORA_TIE_STRATEGY="${TINY_LORA_TIE_STRATEGY:-tiled}"' in launcher
    assert 'export LR="${LR:-2e-4}"' in launcher
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"' in starter
    assert "+actor_rollout_ref.model.tinylora_projection_dim" in base_launcher
    assert 'elif self._peft_type == "tinylora":' in worker
    assert "collect_tinylora_full_params" in worker
    assert 'elif self._peft_type == "tinylora":' in engine
    assert "collect_tinylora_full_params" in engine
