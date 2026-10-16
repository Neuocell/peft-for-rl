"""Regression coverage for custom-adapter exports through real nested FSDP."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from transformers import Qwen2Config, Qwen2ForCausalLM

import verl.utils.fsdp_utils as fsdp_utils
from verl.utils.peft_nora_doft import NoRADoftLinear, apply_nora_doft_adapters
from verl.utils.peft_nora_dora import NoRADoraLinear, apply_nora_dora_adapters


def _adapter_spec(method):
    if method == "nora_doft":
        return (
            NoRADoftLinear,
            apply_nora_doft_adapters,
            fsdp_utils.collect_nora_doft_full_params,
            {"nora_doft_rank": 4, "nora_doft_alpha": 8, "nora_doft_block_size": 8},
        )
    return (
        NoRADoraLinear,
        apply_nora_dora_adapters,
        fsdp_utils.collect_nora_dora_full_params,
        {"nora_dora_rank": 4, "nora_dora_alpha": 8, "nora_dora_normalization": method},
    )


def _modified_model(method):
    wrapper_type, apply, collect, config = _adapter_spec(method)
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0,
        )
    ).float().eval()
    apply(model, config)
    base_weights = {}
    expected_merged = {}
    for name, child in model.named_modules():
        if not isinstance(child, wrapper_type):
            continue
        with torch.no_grad():
            prefix = "nora_doft" if method == "nora_doft" else "nora_dora"
            getattr(child, f"{prefix}_B").normal_(std=0.02)
            getattr(child, f"{prefix}_magnitude").mul_(1.1)
            if child.base_layer.bias is not None:
                child.base_layer.bias.fill_(0.07)
        base_weights[f"{name}.weight"] = child.base_layer.weight.detach().clone()
        weight, bias = child.merged_weight_bias()
        expected_merged[f"{name}.weight"] = weight.clone()
        if bias is not None:
            expected_merged[f"{name}.bias"] = bias.clone()
    return model, collect, base_weights, expected_merged


def _nested_fsdp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=90)
    )
    try:
        for method in ("standard", "init", "continuous", "nora_doft"):
            model, collect, base_weights, expected = _modified_model(method)
            inputs = torch.tensor([[1, 4, 7, 3]])
            with torch.no_grad():
                actor_logits = model(inputs).logits
            dense = Qwen2ForCausalLM(model.config).float().eval()
            policy = fsdp_utils.get_fsdp_wrap_policy(model, {"min_num_params": 0}, is_lora=True)
            wrapped = FSDP(
                model,
                auto_wrap_policy=policy,
                device_id=torch.device("cpu"),
                use_orig_params=True,
                device_mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",)),
            )
            FSDP.set_state_dict_type(wrapped, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig())
            assert sum(isinstance(child, FSDP) for child in wrapped.modules()) > 1
            assert any("._fsdp_wrapped_module." in name for name, _ in model.named_modules())
            with patch.object(fsdp_utils, "get_torch_device", return_value=SimpleNamespace(empty_cache=lambda: None)):
                exported = collect(wrapped)
            assert len(base_weights) == 14
            assert sum(name.endswith(".bias") for name in expected) == 6
            for name, tensor in expected.items():
                torch.testing.assert_close(exported[name], tensor, rtol=0, atol=0, msg=f"{method}: {name}")
            assert all(not torch.equal(exported[name], base) for name, base in base_weights.items())
            dense.load_state_dict(exported, strict=True)
            with torch.no_grad():
                torch.testing.assert_close(dense(inputs).logits, actor_logits, rtol=1e-5, atol=1e-6)
            dist.barrier()
            del wrapped, model, dense
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU/Gloo distributed support is required")
def test_nested_fsdp_exports_all_modified_adapters_on_every_rank(tmp_path):
    """Exercise the production wrap policy and SHARDED_STATE_DICT on two CPU ranks."""
    mp.spawn(_nested_fsdp_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("method", ["standard", "init", "continuous", "nora_doft"])
def test_export_merges_once_and_owns_weight_and_bias_storage(method):
    model, collect, _, expected = _modified_model(method)
    wrapper_type = _adapter_spec(method)[0]
    calls = []
    original_merge = wrapper_type.merged_weight_bias

    def counted_merge(self):
        calls.append(id(self))
        return original_merge(self)

    with patch.object(wrapper_type, "merged_weight_bias", counted_merge):
        exported = collect(model)
    assert len(calls) == len(set(calls)) == 14
    for name, value in expected.items():
        torch.testing.assert_close(exported[name], value, rtol=0, atol=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    for name, value in expected.items():
        torch.testing.assert_close(exported[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["init", "nora_doft"])
def test_export_rejects_unmatched_wrapped_parameter(method):
    model, collect, _, _ = _modified_model(method)
    model.unknown_wrapper = nn.Module()
    model.unknown_wrapper.base_layer = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="Unmatched wrapped parameter"):
        collect(model)


@pytest.mark.parametrize("method", ["init", "nora_doft"])
@pytest.mark.parametrize("parameter_name", ["weight", "bias"])
def test_export_rejects_missing_adapter_weight_or_bias(method, parameter_name):
    model, collect, _, _ = _modified_model(method)

    def remove_parameter(module, state_dict, prefix, local_metadata):
        del state_dict[f"model.layers.0.self_attn.q_proj.base_layer.{parameter_name}"]

    model.register_state_dict_post_hook(remove_parameter)
    with pytest.raises(RuntimeError, match="Incomplete.*merge"):
        collect(model)


@pytest.mark.parametrize("method", ["init", "nora_doft"])
def test_export_rejects_model_without_expected_adapters(method):
    collect = _adapter_spec(method)[2]
    with pytest.raises(RuntimeError, match="No.*modules found"):
        collect(nn.Linear(4, 4))
