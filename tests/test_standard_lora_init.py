import json

import pytest
import torch
from peft import LoraConfig

from verl.utils.peft_lora_init import create_standard_lora_model, standard_lora_use_meta_tensor


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(6, 5, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


def build(seed: int, audit_path=None, init_mode="standard"):
    return create_standard_lora_model(
        TinyModel(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["proj"],
            bias="none",
        ),
        init_seed=seed,
        audit_path=audit_path,
        init_mode=init_mode,
    )


def test_fixed_seed_is_bitwise_identical_and_b_is_zero(tmp_path):
    _, first = build(42, tmp_path / "audit.json")
    _, second = build(42)

    assert first["canonical_sha256"] == second["canonical_sha256"]
    assert first["b_abs_max"] == 0.0
    assert first["module_count"] == 1
    assert first["all_ranks_agree"]
    assert json.loads((tmp_path / "audit.json").read_text()) == first


def test_different_init_seed_changes_a_but_keeps_b_zero():
    _, first = build(42)
    _, second = build(43)

    assert first["canonical_sha256"] != second["canonical_sha256"]
    assert second["b_abs_max"] == 0.0


def test_scoped_seed_restores_outer_cpu_rng_state():
    model = TinyModel()
    config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["proj"],
        bias="none",
    )
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)
    create_standard_lora_model(model, config, init_seed=42)
    observed = torch.rand(4)

    assert torch.equal(observed, expected)


def test_fixed_init_seed_requires_materialized_weights():
    assert standard_lora_use_meta_tensor(True, None)
    assert not standard_lora_use_meta_tensor(True, 42)
    assert not standard_lora_use_meta_tensor(False, None)


def test_random_orthogonal_starts_from_same_a_and_preserves_norm():
    standard_model, standard = build(42)
    orthogonal_model, orthogonal = build(42, init_mode="random_orthogonal")

    assert orthogonal["pre_orthogonal_canonical_sha256"] == standard["canonical_sha256"]
    assert orthogonal["canonical_sha256"] != standard["canonical_sha256"]
    assert orthogonal["b_abs_max"] == 0.0
    assert orthogonal["orthogonalization"]["max_relative_norm_error"] < 1e-6
    assert orthogonal["orthogonalization"]["max_normalized_row_gram_error"] < 1e-5

    standard_a = next(parameter for name, parameter in standard_model.named_parameters() if ".lora_A." in name)
    orthogonal_a = next(parameter for name, parameter in orthogonal_model.named_parameters() if ".lora_A." in name)
    assert torch.allclose(standard_a.norm(), orthogonal_a.norm(), rtol=1e-6, atol=1e-7)
    gram = orthogonal_a @ orthogonal_a.mT
    expected = torch.eye(gram.shape[0]).mul(orthogonal_a.square().sum() / gram.shape[0])
    assert torch.allclose(gram, expected, rtol=1e-5, atol=1e-7)


def test_random_orthogonal_requires_seed():
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], bias="none")
    with pytest.raises(ValueError, match="requires lora_init_seed"):
        create_standard_lora_model(TinyModel(), config, init_mode="random_orthogonal")


def test_nora_init_normalizes_each_a_column_and_keeps_b_zero():
    model, audit = build(42, init_mode="nora_init")

    a = next(parameter for name, parameter in model.named_parameters() if ".lora_A." in name)
    b = next(parameter for name, parameter in model.named_parameters() if ".lora_B." in name)
    column_norms = torch.linalg.vector_norm(a.float(), dim=0)

    assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-6, rtol=0)
    assert torch.count_nonzero(b).item() == 0
    assert audit["init_mode"] == "nora_init"
    assert audit["method"] == "nora_init_rank_dimension_column_normalization_v1"
    assert audit["nora_normalization"]["normalization_dimension"] == "rank"
    assert audit["nora_normalization"]["max_column_norm_error"] < 1e-6


def test_nora_init_requires_seed():
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"], bias="none")
    with pytest.raises(ValueError, match="requires lora_init_seed"):
        create_standard_lora_model(TinyModel(), config, init_mode="nora_init")
