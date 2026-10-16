from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from verl.utils.fsdp_utils import collect_nora_dora_adapter_params, collect_nora_dora_full_params
from verl.utils.peft_nora_dora import NoRADoraLinear, apply_nora_dora_adapters

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self, width: int = 8, out_features: int = 6):
        super().__init__()
        self.q_proj = torch.nn.Linear(width, out_features, bias=True)

    def forward(self, inputs):
        return self.q_proj(inputs)


def nora_dora_config(**overrides):
    config = {
        "nora_dora_rank": 2,
        "nora_dora_alpha": 2,
        "nora_dora_normalization": "init",
        "nora_dora_eps": 1e-6,
        "nora_dora_seed": 17,
        "target_modules": "q_proj",
    }
    config.update(overrides)
    return config


def test_nora_dora_is_exact_identity_with_nora_initialization():
    torch.manual_seed(3)
    model = TinyModel()
    inputs = torch.randn(4, 8)
    base_output = model(inputs).detach().clone()

    stats = apply_nora_dora_adapters(model, nora_dora_config())
    wrapped = model.q_proj

    assert isinstance(wrapped, NoRADoraLinear)
    assert torch.equal(model(inputs), base_output)
    assert torch.count_nonzero(wrapped.nora_dora_B).item() == 0
    column_norms = torch.linalg.vector_norm(wrapped.nora_dora_A.float(), dim=0)
    assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-6, rtol=0)
    expected_magnitude = torch.linalg.vector_norm(wrapped.base_layer.weight.float(), dim=1)
    assert torch.equal(wrapped.nora_dora_magnitude, expected_magnitude)
    assert stats.num_layers == 1
    assert stats.num_parameters == 34
    assert stats.normalization_error_max <= 1e-6
    assert not wrapped.base_layer.weight.requires_grad
    assert not wrapped.base_layer.bias.requires_grad


def test_standard_dora_matches_seeded_standard_lora_initialization():
    from peft import LoraConfig

    from verl.utils.peft_lora_init import create_standard_lora_model

    torch.manual_seed(3)
    dora_model = TinyModel()
    lora_model = TinyModel()
    lora_model.load_state_dict(dora_model.state_dict())
    inputs = torch.randn(4, 8)
    base_output = dora_model(inputs).detach().clone()

    apply_nora_dora_adapters(
        dora_model,
        nora_dora_config(nora_dora_normalization="standard"),
    )
    lora_model, _ = create_standard_lora_model(
        lora_model,
        LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, target_modules=["q_proj"]),
        init_seed=17,
    )

    wrapped = dora_model.q_proj
    peft_layer = lora_model.base_model.model.q_proj
    assert torch.equal(wrapped.nora_dora_A, peft_layer.lora_A["default"].weight)
    assert torch.equal(wrapped.nora_dora_B, peft_layer.lora_B["default"].weight)
    assert torch.equal(dora_model(inputs), base_output)
    assert not torch.allclose(
        torch.linalg.vector_norm(wrapped.nora_dora_A.float(), dim=0),
        torch.ones(wrapped.base_layer.in_features),
    )


def test_nora_dora_first_step_gradient_structure():
    model = TinyModel()
    apply_nora_dora_adapters(model, nora_dora_config())
    wrapped = model.q_proj

    model(torch.randn(5, 8)).square().mean().backward()

    assert torch.count_nonzero(wrapped.nora_dora_A.grad).item() == 0
    assert torch.count_nonzero(wrapped.nora_dora_B.grad).item() > 0
    assert torch.count_nonzero(wrapped.nora_dora_magnitude.grad).item() > 0
    assert torch.isfinite(wrapped.nora_dora_B.grad).all()
    assert torch.isfinite(wrapped.nora_dora_magnitude.grad).all()


def test_nora_dora_matches_reference_formula_and_merge():
    model = TinyModel()
    apply_nora_dora_adapters(model, nora_dora_config())
    wrapped = model.q_proj
    with torch.no_grad():
        wrapped.nora_dora_B.normal_(mean=0.0, std=0.2)
        wrapped.nora_dora_magnitude.mul_(1.1)

    inputs = torch.randn(2, 3, 8)
    a = wrapped.normalized_a()
    delta = wrapped.scaling * (wrapped.nora_dora_B.float() @ a)
    direction = wrapped.base_layer.weight.float() + delta
    scale = wrapped.nora_dora_magnitude.float() / torch.linalg.vector_norm(direction, dim=1).detach()
    reference = F.linear(inputs, direction * scale[:, None], wrapped.base_layer.bias)
    direct = wrapped(inputs)
    assert torch.allclose(direct, reference, atol=1e-5, rtol=1e-5)

    merged_weight, merged_bias = wrapped.merged_weight_bias()
    merged = F.linear(inputs, merged_weight, merged_bias)
    assert torch.allclose(direct, merged, atol=1e-5, rtol=1e-5)
    assert torch.equal(merged_bias, wrapped.base_layer.bias)


def test_nora_dora_direction_norm_is_detached():
    model = TinyModel()
    apply_nora_dora_adapters(model, nora_dora_config())
    wrapped = model.q_proj
    with torch.no_grad():
        wrapped.nora_dora_B.normal_(std=0.2)

    inputs = torch.randn(2, 8)
    wrapped(inputs).sum().backward()
    actual_b_grad = wrapped.nora_dora_B.grad.detach().clone()

    a = wrapped.normalized_a().detach().requires_grad_(True)
    b = wrapped.nora_dora_B.detach().requires_grad_(True)
    direction = wrapped.base_layer.weight.detach().float() + wrapped.scaling * (b @ a)
    scale = wrapped.nora_dora_magnitude.detach().float() / torch.linalg.vector_norm(
        direction, dim=1
    ).detach()
    reference = F.linear(inputs, direction * scale[:, None], wrapped.base_layer.bias).sum()
    reference.backward()
    assert torch.allclose(actual_b_grad, b.grad, atol=1e-6, rtol=1e-5)


def test_nora_dora_continuous_normalization_and_disable_adapter():
    model = TinyModel()
    inputs = torch.randn(3, 8)
    base_output = model(inputs).detach().clone()
    apply_nora_dora_adapters(model, nora_dora_config(nora_dora_normalization="continuous"))
    wrapped = model.q_proj
    with torch.no_grad():
        wrapped.nora_dora_A.mul_(7.0)
        wrapped.nora_dora_B.normal_(std=0.1)

    effective_norms = torch.linalg.vector_norm(wrapped.normalized_a(), dim=0)
    assert torch.allclose(effective_norms, torch.ones_like(effective_norms), atol=1e-6, rtol=0)
    assert not torch.allclose(model(inputs), base_output)
    with model.disable_adapter():
        assert torch.equal(model(inputs), base_output)


def test_nora_dora_state_dict_and_full_weight_collection_round_trip():
    torch.manual_seed(5)
    base = TinyModel()
    clone = TinyModel()
    clone.load_state_dict(base.state_dict())
    apply_nora_dora_adapters(base, nora_dora_config())
    apply_nora_dora_adapters(clone, nora_dora_config())
    with torch.no_grad():
        base.q_proj.nora_dora_B.normal_(std=0.15)
        base.q_proj.nora_dora_magnitude.add_(0.2)
    clone.load_state_dict(base.state_dict())

    inputs = torch.randn(4, 8)
    assert torch.equal(base(inputs), clone(inputs))
    params = collect_nora_dora_full_params(base)
    assert set(params) == {"q_proj.weight", "q_proj.bias"}
    collected = F.linear(inputs, params["q_proj.weight"], params["q_proj.bias"])
    assert torch.allclose(base(inputs), collected, atol=1e-5, rtol=1e-5)

    adapter = collect_nora_dora_adapter_params(base)
    assert set(adapter) == {
        "q_proj.nora_dora_A",
        "q_proj.nora_dora_B",
        "q_proj.nora_dora_magnitude",
    }
    restored = TinyModel()
    restored.q_proj.load_state_dict(base.q_proj.base_layer.state_dict())
    apply_nora_dora_adapters(restored, nora_dora_config())
    incompatible = restored.load_state_dict(adapter, strict=False)
    assert not incompatible.unexpected_keys
    assert torch.equal(base(inputs), restored(inputs))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"nora_dora_rank": 0}, "rank must be positive"),
        ({"nora_dora_rank": 7}, "rank exceeds"),
        ({"nora_dora_alpha": 0}, "alpha must be"),
        ({"nora_dora_normalization": "rows"}, "normalization mode"),
    ],
)
def test_nora_dora_validates_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        apply_nora_dora_adapters(TinyModel(), nora_dora_config(**overrides))


def test_nora_dora_launcher_and_fsdp_paths_are_wired():
    launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_nora_dora_1p5b_4gpu_8k.sh").read_text()
    base_launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh").read_text()
    paper_clean_launcher = (REPO_ROOT / "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh").read_text()
    worker = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text()
    engine = (REPO_ROOT / "verl/workers/engine/fsdp/transformer_impl.py").read_text()
    trainer = (REPO_ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    merger = (REPO_ROOT / "tina_run/scripts/local/eval/merge_nora_dora_wrapped_hf.py").read_text()

    assert "export PEFT_TYPE=nora_dora" in launcher
    assert 'export NORA_DORA_RANK="${NORA_DORA_RANK:-8}"' in launcher
    assert 'export NORA_DORA_NORMALIZATION="${NORA_DORA_NORMALIZATION:-init}"' in launcher
    assert "['model','optimizer','extra']" in launcher
    assert "+actor_rollout_ref.model.nora_dora_rank" in base_launcher
    assert 'nora_dora)\n    PEFT_TYPE="nora_dora"\n    LORA_RANK=0' in paper_clean_launcher
    assert 'dora)\n    PEFT_TYPE="nora_dora"\n    LORA_RANK=0' in paper_clean_launcher
    assert 'elif self._peft_type == "nora_dora":' in worker
    assert "collect_nora_dora_full_params" in worker
    assert '"nora_dora"' in engine
    assert "collect_nora_dora_full_params" in engine
    assert '{"oft", "nora_doft", "nora_dora"}' in trainer
    assert "apply_nora_dora_adapters" in merger
    assert "collect_nora_dora_full_params" in merger
    assert "model.safetensors.index.json" in merger
