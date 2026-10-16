"""A receiver that silently keeps base weights must never pass the rollout gate."""

import ast
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from verl.utils.rollout_weight_verification import verify_rollout_weight_values


class PackedReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.attn.o_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
        self.mlp.down_proj = nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()

    def load_weights(self, weights):
        state = dict(weights)
        # Fixed layouts for a tiny GQA fixture: Q has four rows, K and V two.
        with torch.no_grad():
            self.attn.qkv_proj.weight.copy_(
                torch.cat([state["attn.q_proj.weight"], state["attn.k_proj.weight"], state["attn.v_proj.weight"]])
            )
            self.attn.o_proj.weight.copy_(state["attn.o_proj.weight"])
            self.mlp.gate_up_proj.weight.copy_(torch.cat([state["mlp.gate_proj.weight"], state["mlp.up_proj.weight"]]))
            self.mlp.down_proj.weight.copy_(state["mlp.down_proj.weight"])


@pytest.fixture
def payload():
    shapes = {
        "attn.q_proj": (4, 4),
        "attn.k_proj": (2, 4),
        "attn.v_proj": (2, 4),
        "attn.o_proj": (4, 4),
        "mlp.gate_proj": (6, 4),
        "mlp.up_proj": (6, 4),
        "mlp.down_proj": (4, 6),
    }
    return {
        f"{name}.weight": torch.full(shape, (i + 1) * 0.137, dtype=torch.float32)
        for i, (name, shape) in enumerate(shapes.items())
    }


def test_exact_readback_all_projections_in_receiver_bf16(payload):
    receiver = PackedReceiver()
    receiver.load_weights(iter(payload.items()))
    result = verify_rollout_weight_values(receiver, payload, list(payload))
    assert result["status"] == "verified"
    assert result["tensor_count"] == 7
    assert result["load_dtypes"] == ["torch.bfloat16"]
    assert len(result["loaded_weights_sha256"]) == 64


@pytest.mark.parametrize("projection", ["q", "k", "v", "gate", "up", "o", "down"])
def test_rejects_one_stale_projection_among_other_correct_weights(payload, projection):
    receiver = PackedReceiver()
    receiver.load_weights(iter(payload.items()))
    with torch.no_grad():
        target = {
            "q": receiver.attn.qkv_proj.weight[:4],
            "k": receiver.attn.qkv_proj.weight[4:6],
            "v": receiver.attn.qkv_proj.weight[6:],
            "gate": receiver.mlp.gate_up_proj.weight[:6],
            "up": receiver.mlp.gate_up_proj.weight[6:],
            "o": receiver.attn.o_proj.weight,
            "down": receiver.mlp.down_proj.weight,
        }[projection]
        target.zero_()
    with pytest.raises(RuntimeError, match="refusing to generate"):
        verify_rollout_weight_values(receiver, payload, list(payload))


def test_rejects_missing_payload_and_nonfinite_weights(payload):
    receiver = PackedReceiver()
    receiver.load_weights(iter(payload.items()))
    names = list(payload)
    missing = dict(payload)
    missing.pop(names[0])
    with pytest.raises(RuntimeError, match="Missing merged adapter payload"):
        verify_rollout_weight_values(receiver, missing, names)
    payload[names[0]].fill_(float("inf"))
    receiver.load_weights(iter(payload.items()))
    with pytest.raises(RuntimeError, match="refusing to generate"):
        verify_rollout_weight_values(receiver, payload, names)


@pytest.mark.parametrize("names", [[], ["attn.q_proj.weight", "attn.q_proj.weight"]])
def test_no_empty_or_duplicate_verification_receipts(payload, names):
    with pytest.raises(RuntimeError, match="nonempty unique"):
        verify_rollout_weight_values(PackedReceiver(), payload, names)


def _production_update_weights():
    """Execute the actual method body without initializing vLLM's GPU runtime."""
    path = Path(__file__).resolve().parents[1] / "verl/workers/rollout/vllm_rollout/vllm_rollout.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMAsyncRollout")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "update_weights")
    method.args.args[1].annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "torch": torch,
        "json": json,
        "is_fp8_model": lambda _: False,
        "logger": SimpleNamespace(info=lambda *a: None),
        "asdict": asdict,
        "VLLM_LORA_INT_ID": 1,
        "VLLM_LORA_NAME": "actor",
        "VLLM_LORA_PATH": "actor",
        "TensorLoRARequest": lambda **kw: SimpleNamespace(**kw),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["update_weights"]


@pytest.mark.parametrize("drop_load", [False, True])
def test_actual_update_method_checks_receiver_before_returning_receipt(payload, drop_load, monkeypatch):
    import sys

    monkeypatch.setitem(
        sys.modules, "verl.utils.vllm.patch", SimpleNamespace(patch_vllm_moe_model_weight_loader=lambda model: None)
    )
    model = PackedReceiver()
    if drop_load:
        model.load_weights = lambda weights: list(weights)
    runner = SimpleNamespace(model=model, vllm_config=object())
    worker = SimpleNamespace(inference_engine=SimpleNamespace(worker=SimpleNamespace(model_runner=runner)))
    call = _production_update_weights()(
        worker, iter(payload.items()), verify_weight_names=list(payload), verification_context="global_step_20/actor"
    )
    if drop_load:
        with pytest.raises(RuntimeError, match="refusing to generate"):
            asyncio.run(call)
    else:
        receipt = asyncio.run(call)
        assert receipt["checkpoint_source"] == "global_step_20/actor"
        assert receipt["tensor_count"] == 7


def test_native_adapter_rejected_by_engine_aborts_sync():
    @dataclass
    class AdapterConfig:
        r: int = 2

    engine_worker = SimpleNamespace(remove_lora=lambda _: True, add_lora=lambda _: False)
    rollout = SimpleNamespace(inference_engine=SimpleNamespace(worker=engine_worker))
    call = _production_update_weights()(
        rollout,
        iter([("proj.lora_A.weight", torch.ones(2, 4))]),
        peft_config=AdapterConfig(),
        base_sync_done=True,
    )
    with pytest.raises(RuntimeError, match="vLLM rejected"):
        asyncio.run(call)


def test_checkpoint_restore_invalidates_prior_sync_receipt():
    path = Path(__file__).resolve().parents[1] / "verl/workers/fsdp_workers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActorRolloutRefWorker")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_checkpoint")
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    loaded = []
    worker = SimpleNamespace(
        _is_actor=True,
        _is_rollout=True,
        _is_peft=True,
        _is_offload_param=False,
        _is_offload_optimizer=False,
        _custom_rollout_weights_verified=True,
        checkpoint_manager=SimpleNamespace(should_load_model=True, load_checkpoint=lambda **kw: loaded.append(kw)),
    )
    namespace["load_checkpoint"](worker, "global_step_20/actor")
    assert loaded[0]["local_path"] == "global_step_20/actor"
    assert worker._custom_rollout_weights_verified is False
    assert worker._rollout_checkpoint_source == "global_step_20/actor"
    worker.checkpoint_manager.should_load_model = False
    with pytest.raises(RuntimeError, match="include 'model'"):
        namespace["load_checkpoint"](worker, "global_step_50/actor")
    assert len(loaded) == 1
