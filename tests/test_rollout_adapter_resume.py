"""Execute production async methods on CPU without importing the CUDA vLLM engine."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model
from torch import nn

from verl.utils import fsdp_utils
from verl.utils.model import convert_weight_keys
from verl.utils.peft_adalora import adalora_config_for_vllm
from verl.utils.peft_nora_dora import apply_nora_dora_adapters

REPO = Path(__file__).resolve().parents[1]


def _production_method(path, class_name, method_name, namespace):
    """Compile the complete, unmodified method body with CPU transport doubles."""
    tree = ast.parse((REPO / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name)
    method.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(REPO / path), "exec"), namespace)
    return namespace[method_name]


class _Config(dict):
    __getattr__ = dict.__getitem__


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


def _actor(adalora=False):
    torch.manual_seed(17)
    config = (
        AdaLoraConfig(init_r=2, target_r=1, lora_alpha=4, target_modules=["proj"], total_step=100)
        if adalora
        else LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])
    )
    actor = get_peft_model(_LinearModel(), config).eval()
    # Mimic a checkpoint with a trained adapter; a zero-init test masks the bug.
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if "lora_" in name:
                parameter.fill_(0.2 if "lora_A" in name else 0.3)
    return actor


class _Rollout:
    def __init__(self, actor, *, preloaded, sleep_level=1, fail_on=None):
        self.sleep_level = sleep_level
        self.events = []
        self.base = actor.base_model.model.proj.base_layer.weight.detach().clone() if preloaded else None
        self.adapter = None
        self.fail_on = fail_on

    async def resume(self, tags):
        self.events.append(("resume", tuple(tags)))

    async def update_weights(self, weights, **kwargs):
        values = dict(weights)
        kind = "adapter" if kwargs.get("peft_config") is not None and kwargs.get("base_sync_done") else "base"
        self.events.append((kind, tuple(values)))
        if kind == self.fail_on:
            raise RuntimeError(f"injected {kind} sync failure")
        if kind == "base":
            self.base = next(iter(values.values())).detach().clone()
        else:
            assert self.base is not None, "adapter installation must follow base installation"
            config = kwargs["peft_config"]
            a = next(value for name, value in values.items() if name.endswith("lora_A.weight"))
            b = next(value for name, value in values.items() if name.endswith("lora_B.weight"))
            self.adapter = b @ a * config.lora_alpha / config.r

    def generate(self, inputs):
        self.events.append(("generate",))
        return inputs @ (self.base + (0 if self.adapter is None else self.adapter)).T


def _worker_and_method(*, preloaded=False, sleep_level=1, adalora=False, free_cache=True, fail_on=None):
    actor = _actor(adalora)
    rollout = _Rollout(actor, preloaded=preloaded, sleep_level=sleep_level, fail_on=fail_on)
    rng = SimpleNamespace(get_rng_state=lambda: "actor_rng", set_rng_state=lambda _: None)
    namespace = {
        "aggressive_empty_cache": lambda **_: None,
        "log_gpu_memory_usage": lambda *_, **__: None,
        "logger": None,
        "collect_lora_params": fsdp_utils.collect_lora_params,
        "replace_lora_wrapper": fsdp_utils.replace_lora_wrapper,
        "convert_weight_keys": convert_weight_keys,
        "adalora_config_for_vllm": adalora_config_for_vllm,
        "set_expandable_segments": lambda _: None,
        "get_device_id": lambda: "cpu",
        "get_torch_device": lambda: rng,
        "DTensor": type("UnusedDTensor", (), {}),
    }
    method = _production_method("verl/workers/fsdp_workers.py", "ActorRolloutRefWorker", "rollout_mode", namespace)
    worker = SimpleNamespace(
        actor_module_fsdp=actor,
        rollout=rollout,
        config=SimpleNamespace(
            rollout=_Config(
                name="vllm",
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                layered_summon=False,
                free_cache_engine=free_cache,
            )
        ),
        _is_offload_param=False,
        _is_lora=True,
        _is_adalora=adalora,
        _peft_type="adalora" if adalora else "lora",
        base_sync_done=preloaded,
        layered_summon=False,
        gen_random_states="rollout_rng",
    )
    return worker, method, namespace


@pytest.mark.parametrize("adalora", [False, True])
@pytest.mark.parametrize("preloaded", [False, True])
@pytest.mark.parametrize("free_cache", [False, True])
def test_first_rollout_uses_nonzero_restored_adapter(adalora, preloaded, free_cache):
    worker, method, _ = _worker_and_method(adalora=adalora, preloaded=preloaded, free_cache=free_cache)
    inputs = torch.tensor([[0.5, -0.1, 0.9]])
    expected = worker.actor_module_fsdp(inputs)
    base = worker.actor_module_fsdp.base_model.model.proj.base_layer(inputs)
    assert not torch.allclose(expected, base)

    asyncio.run(method(worker))
    actual = worker.rollout.generate(inputs)

    torch.testing.assert_close(actual, expected)
    assert worker.base_sync_done
    updates = [entry[0] for entry in worker.rollout.events if entry[0] in {"base", "adapter"}]
    assert updates == (["adapter"] if preloaded else ["base", "adapter"])
    assert worker.rollout.events[-1] == ("generate",)


@pytest.mark.parametrize("freeze_a", [False, True], ids=["gradtop", "real"])
@pytest.mark.parametrize("sleep_level", [1, 2])
def test_gradient_subspace_restored_policy_uses_adapter(freeze_a, sleep_level):
    worker, method, _ = _worker_and_method(sleep_level=sleep_level)
    worker._peft_type = "grad_subspace"
    worker.actor_module_fsdp.base_model.model.proj.lora_A["default"].weight.requires_grad_(not freeze_a)
    inputs = torch.tensor([[0.5, -0.1, 0.9]])
    expected = worker.actor_module_fsdp(inputs)
    assert not torch.allclose(expected, worker.actor_module_fsdp.base_model.model.proj.base_layer(inputs))
    asyncio.run(method(worker))
    torch.testing.assert_close(worker.rollout.generate(inputs), expected)
    assert [event[0] for event in worker.rollout.events if event[0] in {"base", "adapter"}] == ["base", "adapter"]


@pytest.mark.parametrize("sleep_level", [1, 2])
def test_subsequent_rollout_refreshes_adapter_and_reloads_destroyed_base(sleep_level):
    worker, method, _ = _worker_and_method(sleep_level=sleep_level)
    asyncio.run(method(worker))
    with torch.no_grad():
        worker.actor_module_fsdp.base_model.model.proj.lora_B["default"].weight.add_(0.4)
    worker.rollout.events.clear()
    if sleep_level == 2:
        worker.rollout.base = None
    asyncio.run(method(worker))
    updates = [entry[0] for entry in worker.rollout.events if entry[0] in {"base", "adapter"}]
    assert updates == (["base", "adapter"] if sleep_level == 2 else ["adapter"])
    inputs = torch.tensor([[0.5, -0.1, 0.9]])
    torch.testing.assert_close(worker.rollout.generate(inputs), worker.actor_module_fsdp(inputs))


def test_sleep_level_two_materializes_base_dtensor_after_prior_sync():
    worker, method, namespace = _worker_and_method(preloaded=True, sleep_level=2)
    original_collect = namespace["collect_lora_params"]
    materialized_on = []

    class CpuDTensor:
        def __init__(self, value):
            self.value = value

        def to(self, device, non_blocking):
            materialized_on.append(device)
            return self

        def full_tensor(self):
            return self.value

    def collect(**kwargs):
        values = original_collect(**kwargs)
        return values if kwargs["base_sync_done"] else {key: CpuDTensor(value) for key, value in values.items()}

    namespace["DTensor"] = CpuDTensor
    namespace["collect_lora_params"] = collect
    asyncio.run(method(worker))
    assert materialized_on == ["cpu"]


@pytest.mark.parametrize("failure", ["base", "adapter"])
def test_failed_first_sync_never_marks_ready_or_enables_generation(failure):
    worker, method, _ = _worker_and_method(fail_on=failure)
    with pytest.raises(RuntimeError, match=f"injected {failure}"):
        asyncio.run(method(worker))
    assert not worker.base_sync_done
    assert ("resume", ("kv_cache",)) not in worker.rollout.events
    if failure == "base":
        assert worker.rollout.adapter is None


def test_empty_adapter_collection_fails_before_loading_base():
    worker, method, namespace = _worker_and_method()
    namespace["collect_lora_params"] = lambda **_: {}
    with pytest.raises(RuntimeError, match="No LoRA tensors"):
        asyncio.run(method(worker))
    assert worker.rollout.events == []
    assert not worker.base_sync_done


def test_missing_default_adapter_config_fails_before_sync():
    worker, method, _ = _worker_and_method()
    worker.actor_module_fsdp.peft_config = {}
    with pytest.raises(RuntimeError, match="no default PEFT configuration"):
        asyncio.run(method(worker))
    assert worker.rollout.events == []
    assert not worker.base_sync_done


@pytest.mark.parametrize("receipt", [None, {}, {"status": "verified", "tensor_count": 0}])
def test_custom_first_sync_requires_complete_backend_verification_receipt(receipt):
    worker, method, namespace = _worker_and_method()
    worker.actor_module_fsdp = _LinearModel()
    apply_nora_dora_adapters(
        worker.actor_module_fsdp,
        {"nora_dora_rank": 2, "nora_dora_alpha": 4, "target_modules": ["proj"]},
    )
    worker._is_lora = False
    worker._peft_type = "nora_dora"
    namespace["collect_nora_dora_full_params"] = fsdp_utils.collect_nora_dora_full_params

    async def ignored_verification(weights, **kwargs):
        assert dict(weights)
        assert kwargs["verify_weight_names"] == ["proj.weight"]
        return receipt

    worker.rollout.update_weights = ignored_verification
    with pytest.raises(RuntimeError, match="did not acknowledge verification"):
        asyncio.run(method(worker))
    assert not getattr(worker, "_custom_rollout_weights_verified", False)
    assert not worker.base_sync_done
    assert ("resume", ("kv_cache",)) not in worker.rollout.events


def test_custom_weights_verified_on_first_sync_and_after_reset():
    worker, method, namespace = _worker_and_method()
    worker.actor_module_fsdp = _LinearModel()
    apply_nora_dora_adapters(
        worker.actor_module_fsdp,
        {"nora_dora_rank": 2, "nora_dora_alpha": 4, "target_modules": ["proj"]},
    )
    worker._is_lora = False
    worker._peft_type = "nora_dora"
    namespace["collect_nora_dora_full_params"] = fsdp_utils.collect_nora_dora_full_params
    verification_calls = []

    async def verify(weights, **kwargs):
        assert dict(weights)
        verification_calls.append((kwargs["verify_weight_names"], kwargs["verification_context"]))
        return {"status": "verified", "tensor_count": 1}

    worker.rollout.update_weights = verify
    asyncio.run(method(worker))
    assert worker._custom_rollout_weights_verified
    asyncio.run(method(worker))
    # The real checkpoint loader resets this flag after restoring actor state.
    worker._custom_rollout_weights_verified = False
    worker._rollout_checkpoint_source = "global_step_20/actor"
    asyncio.run(method(worker))
    assert verification_calls == [
        (["proj.weight"], "initial_actor"),
        (None, "initial_actor"),
        (["proj.weight"], "global_step_20/actor"),
    ]


class _ServerEngine:
    def __init__(self, loaded):
        self.loaded = loaded
        self.requests = []

    async def list_loras(self):
        return self.loaded

    async def generate(self, **kwargs):
        self.requests.append(kwargs)
        yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7], finish_reason="stop")])


@pytest.mark.parametrize(
    "mode,rank,loaded,expect_failure",
    [("hybrid", 2, [], True), ("hybrid", 2, [1], False), ("hybrid", 0, [], False), ("standalone", 2, [], False)],
)
def test_hybrid_generation_requires_loaded_training_adapter(mode, rank, loaded, expect_failure):
    namespace = {
        "SamplingParams": lambda **kwargs: SimpleNamespace(**kwargs),
        "TokensPrompt": lambda **kwargs: kwargs,
        "LoRARequest": lambda **kwargs: SimpleNamespace(**kwargs),
        "TokenOutput": lambda **kwargs: SimpleNamespace(**kwargs),
        "_qwen2_5_vl_dedup_image_tokens": lambda tokens, _: tokens,
        "VLLM_LORA_INT_ID": 1,
        "VLLM_LORA_NAME": "training",
        "VLLM_LORA_PATH": "/adapter",
        "RolloutMode": SimpleNamespace(HYBRID="hybrid"),
    }
    method = _production_method(
        "verl/workers/rollout/vllm_rollout/vllm_async_server.py", "vLLMHttpServerBase", "generate", namespace
    )
    server = SimpleNamespace(
        config=_Config(max_model_len=16, response_length=4, prompt_length=4, enable_rollout_routing_replay=False),
        model_config=SimpleNamespace(lora_rank=rank, processor=None),
        rollout_mode=mode,
        engine=_ServerEngine(loaded),
    )
    if expect_failure:
        with pytest.raises(RuntimeError, match="active training LoRA adapter is missing"):
            asyncio.run(method(server, [1, 2], {}, "test"))
        assert not server.engine.requests
    else:
        result = asyncio.run(method(server, [1, 2], {}, "test"))
        assert result.token_ids == [7]
        request = server.engine.requests[0]["lora_request"]
        assert (request is not None) == bool(rank and loaded)
