"""Compare new initialization with real PEFT and pinned official NoRA formulas."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig
from torch.nn import functional as F

from verl.utils.adapter_initialization import LEGACY_INIT_VERSION, REFERENCE_INIT_VERSION
from verl.utils.peft_lora_init import create_standard_lora_model
from verl.utils.peft_nora_dora import apply_nora_dora_adapters, nora_dora_snapshot_config


class UnequalLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(9, 5)
        self.k_proj = torch.nn.Linear(5, 7)
        self.v_proj = torch.nn.Linear(7, 4)

    def forward(self, x):
        return self.v_proj(self.k_proj(self.q_proj(x)))


@pytest.mark.parametrize("mode", ["standard", "init", "continuous", "STANDARD"])
def test_multilayer_raw_and_effective_a_match_real_seeded_peft(mode):
    model = UnequalLayers()
    base = copy.deepcopy(model)
    config = LoraConfig(r=3, lora_alpha=6, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj"])
    reference, _ = create_standard_lora_model(copy.deepcopy(base), config, init_seed=42)
    standalone, _ = create_standard_lora_model(copy.deepcopy(base), config, init_seed=42, init_mode="nora_init")
    torch.manual_seed(192)
    outer_rng = torch.random.get_rng_state().clone()
    apply_nora_dora_adapters(model, {
        "nora_dora_rank": 3, "nora_dora_alpha": 6, "nora_dora_normalization": mode,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    })
    assert torch.equal(outer_rng, torch.random.get_rng_state())
    for name in ("q_proj", "k_proj", "v_proj"):
        child = getattr(model, name)
        ref = getattr(reference.base_model.model, name)
        raw = ref.lora_A["default"].weight
        normalized = raw / (raw.norm(dim=0, keepdim=True) + 1e-6)
        expected_stored = normalized if mode == "init" else raw
        expected_effective = normalized if mode in {"init", "continuous"} else raw
        assert child.initialization_version == REFERENCE_INIT_VERSION
        torch.testing.assert_close(child.nora_dora_A, expected_stored, rtol=0, atol=0)
        torch.testing.assert_close(child.normalized_a(), expected_effective, rtol=0, atol=0)
        assert torch.equal(child.nora_dora_B, ref.lora_B["default"].weight)
        if mode == "init":
            assert torch.equal(child.nora_dora_A, getattr(standalone.base_model.model, name).lora_A["default"].weight)
    inputs = torch.randn(2, 9)
    torch.testing.assert_close(model(inputs), base(inputs), rtol=0, atol=0)


def test_continuous_official_formula_forward_gradients_and_merge():
    model = UnequalLayers()
    apply_nora_dora_adapters(model, {
        "nora_dora_rank": 3, "nora_dora_normalization": "continuous", "target_modules": "q_proj",
    })
    child = model.q_proj
    with torch.no_grad():
        child.nora_dora_A.mul_(1e-5)  # make additive vs clamped epsilon observable
        child.nora_dora_B.normal_(std=0.1)
        child.nora_dora_magnitude.mul_(1.1)
    a = child.nora_dora_A.detach().clone().requires_grad_(True)
    b = child.nora_dora_B.detach().clone().requires_grad_(True)
    effective = a / (a.norm(dim=0, keepdim=True) + 1e-6)
    direction = child.base_layer.weight.detach() + child.scaling * (b @ effective)
    scale = child.nora_dora_magnitude.detach() / direction.norm(dim=1).detach()
    x = torch.randn(4, 9)
    expected = F.linear(x, direction * scale[:, None], child.base_layer.bias)
    actual = child(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(child.nora_dora_A.grad, a.grad, rtol=2e-5, atol=1e-2)
    torch.testing.assert_close(child.nora_dora_B.grad, b.grad, rtol=2e-5, atol=1e-6)
    weight, bias = child.merged_weight_bias()
    torch.testing.assert_close(F.linear(x, weight, bias), actual, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("version", [None, REFERENCE_INIT_VERSION])
def test_snapshot_roundtrip_preserves_continuous_semantics(tmp_path, version):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "merge_nora_dora", root / "tina_run/scripts/local/eval/merge_nora_dora_wrapped_hf.py"
    )
    merger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merger)
    method = {"rank": 3, "alpha": 6, "normalization": "continuous", "eps": 1e-6, "seed": 42}
    if version is not None:
        method["initialization_version"] = version
    (tmp_path / "manifest.json").write_text(json.dumps({"method_config": method}))
    config = merger.resolve_adapter_config(tmp_path, SimpleNamespace(target_modules="q_proj"))
    assert config["nora_dora_initialization_version"] == (version or LEGACY_INIT_VERSION)
    source = UnequalLayers()
    restored = copy.deepcopy(source)
    apply_nora_dora_adapters(source, config)
    with torch.no_grad():
        source.q_proj.nora_dora_A.mul_(1e-7)
        source.q_proj.nora_dora_B.normal_()
    state = source.state_dict()
    apply_nora_dora_adapters(restored, config)
    restored.load_state_dict(state)
    a = source.q_proj.nora_dora_A
    norm = a.norm(dim=0, keepdim=True)
    expected = a / ((norm + 1e-6) if version else norm.clamp_min(1e-6))
    torch.testing.assert_close(restored.q_proj.normalized_a(), expected, rtol=0, atol=0)
    x = torch.randn(2, 9)
    torch.testing.assert_close(restored(x), source(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match="conflicts|requires legacy"):
        merger.resolve_adapter_config(tmp_path, SimpleNamespace(
            target_modules="q_proj", initialization_version="legacy_v1" if version else REFERENCE_INIT_VERSION
        ))


def test_unknown_snapshot_version_fails_closed():
    with pytest.raises(ValueError, match="Unsupported snapshot"):
        nora_dora_snapshot_config({"initialization_version": "future_unsupported"})
