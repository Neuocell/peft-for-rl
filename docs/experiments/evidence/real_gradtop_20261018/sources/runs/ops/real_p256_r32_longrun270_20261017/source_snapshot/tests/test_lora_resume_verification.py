"""CPU faults for native LoRA resume gates, including packed active buffers."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from verl.utils import lora_resume_verification as verify


@pytest.fixture
def resumed(tmp_path, monkeypatch):
    tensors = {}
    for index, name in enumerate(("q_proj", "k_proj", "v_proj", "o_proj")):
        tensors[f"base_model.model.model.layers.0.self_attn.{name}.lora_A.weight"] = (
            torch.arange(6, dtype=torch.float32).reshape(2, 3) + index
        )
        tensors[f"base_model.model.model.layers.0.self_attn.{name}.lora_B.weight"] = (
            torch.arange(8, dtype=torch.float32).reshape(4, 2) + index + 0.25
        )
    source = tmp_path / "adapter_model.safetensors"
    save_file(tensors, str(source))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    values = {"SOURCE": str(source), "SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "OUTPUT_DIR": str(tmp_path / "receipts"), "TOKEN": "test-token", "WORLD_SIZE": "1", "TENSOR_COUNT": "8"}
    for key, value in values.items():
        monkeypatch.setenv(verify.PREFIX + key, value)
    checkpoint = str(tmp_path / "global_step_150" / "actor")
    receipt = verify.verify_actor_payload(tensors, checkpoint)
    canonical = verify.canonical_tensors(tensors)
    modules, installed = {}, {}
    qkv = "model.layers.0.self_attn.qkv_proj"
    o_proj = "model.layers.0.self_attn.o_proj"
    packed = {qkv: [f"model.layers.0.self_attn.{name}_proj" for name in ("q", "k", "v")]}
    for name, names in ((qkv, packed[qkv]), (o_proj, [o_proj])):
        aa = [canonical[f"{sub}.lora_A.weight"].to(torch.bfloat16) for sub in names]
        bb = [canonical[f"{sub}.lora_B.weight"].to(torch.bfloat16) * 2 for sub in names]
        a_stacked, b_stacked = [], []
        for a, b in zip(aa, bb):
            a_buffer, b_buffer = torch.zeros(1, 1, 4, 3, dtype=a.dtype), torch.zeros(1, 1, 4, 4, dtype=b.dtype)
            a_buffer[0, 0, :2, :] = a
            b_buffer[0, 0, :, :2] = b
            a_stacked.append(a_buffer)
            b_stacked.append(b_buffer)
        modules[name] = NS(tp_size=1, lora_a_stacked=a_stacked, lora_b_stacked=b_stacked)
        installed[name] = NS(lora_a=aa if name == qkv else aa[0], lora_b=bb if name == qkv else bb[0])
    registered = NS(loras=installed)
    manager = NS(modules=modules, packed_modules=packed, lora_index_to_id=[1], get_adapter=lambda _: registered)
    worker = NS(model_runner=NS(
        vllm_config=NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1), model_config=NS(quantization=None)),
        lora_manager=NS(_adapter_manager=manager, list_adapters=lambda: {1}),
    ))
    config = {"r": 2, "lora_alpha": 4, "rank_pattern": {name: 2 for name in packed[qkv]},
              "alpha_pattern": {name: 4 for name in packed[qkv]}}
    return NS(tensors=tensors, receipt=receipt, worker=worker, config=config, manager=manager,
              checkpoint=checkpoint, directory=tmp_path / "receipts")


def engine(resumed):
    return verify.verify_engine_adapter(resumed.worker, resumed.config, resumed.receipt, 1, require_cuda=False)


def test_packed_registered_and_active_device_readback(resumed):
    receipt = engine(resumed)
    assert receipt["tensor_count"] == 8
    assert receipt["registered_adapter_sha256"] == receipt["active_device_adapter_sha256"]
    assert receipt["device_types"] == ["cpu"]  # The fixture is not GPU acceptance evidence.
    verify.acknowledge_ready(receipt, resumed.receipt)


@pytest.mark.parametrize("fault", ["source_hash", "actor_value", "names", "zero_b", "checkpoint"])
def test_actor_source_gate_rejects_faults(resumed, monkeypatch, fault):
    tensors = copy.deepcopy(resumed.tensors)
    checkpoint = resumed.checkpoint
    if fault == "source_hash":
        monkeypatch.setenv(verify.PREFIX + "SHA256", "wrong")
    elif fault == "actor_value":
        next(iter(tensors.values())).add_(1)
    elif fault == "names":
        tensors.pop(next(iter(tensors)))
    elif fault == "zero_b":
        for name, tensor in tensors.items():
            if ".lora_B." in name:
                tensor.zero_()
    else:
        checkpoint = None
    with pytest.raises(RuntimeError):
        verify.verify_actor_payload(tensors, checkpoint)


@pytest.mark.parametrize("fault", ["cpu_value", "device_value", "padding", "slot", "missing_layer", "tp", "varying_rank"])
def test_readback_rejects_missing_stale_and_partial_adapter(resumed, fault):
    name = "model.layers.0.self_attn.qkv_proj"
    if fault == "cpu_value":
        resumed.manager.get_adapter(1).loras[name].lora_b[0].add_(1)
    elif fault == "device_value":
        resumed.manager.modules[name].lora_a_stacked[0][0, 0, 0, 0] += 1
    elif fault == "padding":
        resumed.manager.modules[name].lora_a_stacked[0][0, 0, 3, 0] = 1
    elif fault == "slot":
        resumed.manager.lora_index_to_id = [None]
    elif fault == "missing_layer":
        resumed.manager.get_adapter(1).loras.pop(name)
    elif fault == "tp":
        resumed.worker.model_runner.vllm_config.parallel_config.tensor_parallel_size = 2
    else:
        resumed.config["rank_pattern"]["q_proj"] = 1
    with pytest.raises(RuntimeError):
        engine(resumed)


def test_cpu_readback_cannot_pass_production_cuda_gate(resumed):
    with pytest.raises(RuntimeError, match="Expected CUDA"):
        verify.verify_engine_adapter(resumed.worker, resumed.config, resumed.receipt, 1)


def test_missing_backend_acknowledgement_is_rejected(resumed):
    with pytest.raises(RuntimeError, match="acknowledge"):
        verify.acknowledge_ready(None, resumed.receipt)


def _ready_fixture(resumed):
    receipt = engine(resumed)
    # Simulates metadata from a real GPU worker solely for exercising the
    # request/controller checks. The test never claims to run CUDA.
    receipt["device_types"] = ["cuda"]
    verify._emit(verify.settings(), "engine.rank0", receipt)
    verify.acknowledge_ready(receipt, resumed.receipt)


def test_request_requires_all_rank_receipts_and_records_adapter_id(resumed):
    with pytest.raises(FileNotFoundError):
        verify.verify_first_request(0, "request-151", 1, {1})
    _ready_fixture(resumed)
    result = verify.verify_first_request(0, "request-151", 1, {1})
    assert result["lora_request_id"] == 1
    assert result["verified_worker_ranks"] == [0]
    summary = verify.validate_receipt_directory(
        resumed.directory, expected_world_size=1, expected_checkpoint=resumed.checkpoint, required=True
    )
    assert summary["status"] == "verified"


@pytest.mark.parametrize("fault", ["missing_adapter", "token", "source", "wrong_checkpoint"])
def test_request_and_controller_reject_invalid_receipts(resumed, fault):
    _ready_fixture(resumed)
    if fault == "missing_adapter":
        with pytest.raises(RuntimeError):
            verify.verify_first_request(0, "request-151", 1, set())
        return
    path = resumed.directory / "ready.rank0.json"
    receipt = json.loads(path.read_text())
    receipt[{"token": "token", "source": "source_file_sha256", "wrong_checkpoint": "checkpoint_source"}[fault]] = "wrong"
    path.write_text(json.dumps(receipt))
    if fault != "wrong_checkpoint":
        with pytest.raises(RuntimeError):
            verify.verify_first_request(0, "request-151", 1, {1})
    with pytest.raises(RuntimeError):
        verify.validate_receipt_directory(
            resumed.directory, expected_world_size=1, expected_checkpoint=resumed.checkpoint
        )


def test_pending_and_required_controller_checks(resumed):
    assert verify.validate_receipt_directory(
        resumed.directory, expected_world_size=1, expected_checkpoint=resumed.checkpoint
    ) is None
    with pytest.raises(RuntimeError, match="Missing native"):
        verify.validate_receipt_directory(
            resumed.directory, expected_world_size=1, expected_checkpoint=resumed.checkpoint, required=True
        )


def test_fresh_zero_b_then_updated_actor_and_stale_engine(resumed, monkeypatch, tmp_path):
    from verl.utils import lora_fresh_verification as fresh
    initial = copy.deepcopy(resumed.tensors)
    for name, value in initial.items():
        if ".lora_B." in name:
            value.zero_()
    source = tmp_path / "initial_source.safetensors"
    save_file(initial, str(source))
    monkeypatch.delenv(verify.PREFIX + "SOURCE")
    root = tmp_path / "fresh"
    for key, value in {
        "DIR": str(root), "INITIAL_SOURCE": str(source),
        "INITIAL_SHA256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "TENSOR_COUNT": "8", "WORLD_SIZE": "1", "TOKEN": "fresh-test",
    }.items():
        monkeypatch.setenv("VERL_LORA_FRESH_VERIFY_" + key, value)
    receipt0 = fresh.verify_actor_payload(initial, NS(r=2, lora_alpha=4), 0)
    assert receipt0["b_nonzero"] is False
    # The existing nonzero device buffers cannot pass a zero-initialization gate.
    with pytest.raises(RuntimeError, match="differs from source"):
        verify.verify_engine_adapter(resumed.worker, resumed.config, receipt0, 1, require_cuda=False)
    receipt1 = fresh.verify_actor_payload(resumed.tensors, NS(r=2, lora_alpha=4), 1)
    assert receipt1["b_nonzero"] is True
    assert receipt0["token"] != receipt1["token"]
    engine1 = verify.verify_engine_adapter(resumed.worker, resumed.config, receipt1, 1, require_cuda=False)
    assert engine1["registered_adapter_sha256"] == engine1["active_device_adapter_sha256"]
    with pytest.raises(RuntimeError, match="mismatched"):
        verify.verify_engine_adapter(resumed.worker, resumed.config, receipt0, 1, require_cuda=False)


@pytest.mark.parametrize("fault", ["zero_update", "changed_a", "missing_factor", "nonzero_initial"])
def test_fresh_factor_faults(resumed, fault):
    from verl.utils.lora_fresh_verification import check_factors
    initial = verify.canonical_tensors(copy.deepcopy(resumed.tensors))
    for name, value in initial.items():
        if ".lora_B." in name:
            value.zero_()
    actual = verify.canonical_tensors(copy.deepcopy(resumed.tensors))
    phase = 1
    if fault == "zero_update":
        actual = copy.deepcopy(initial)
    elif fault == "changed_a":
        next(v for k, v in actual.items() if ".lora_A." in k).add_(1)
    elif fault == "missing_factor":
        actual.pop(next(iter(actual)))
    else:
        phase = 0
    with pytest.raises(RuntimeError):
        check_factors(actual, initial, phase)


def test_opt_in_disabled_and_partial_settings(monkeypatch):
    monkeypatch.delenv(verify.PREFIX + "SOURCE", raising=False)
    assert verify.settings() is None
    monkeypatch.setenv(verify.PREFIX + "SOURCE", "/not-real")
    with pytest.raises(RuntimeError, match="Incomplete"):
        verify.settings()


def test_production_rollout_fails_before_transport_when_source_wrong(resumed):
    from test_rollout_adapter_resume import _worker_and_method

    worker, method, _ = _worker_and_method()
    worker._rollout_checkpoint_source = resumed.checkpoint
    import asyncio

    with pytest.raises(RuntimeError, match="tensor names differ"):
        asyncio.run(method(worker))
    assert worker.rollout.events == []
    assert not worker.base_sync_done


def test_production_server_blocks_first_request_before_engine_generation(resumed):
    import asyncio
    from test_rollout_adapter_resume import _Config, _production_method, _ServerEngine

    namespace = {
        "SamplingParams": lambda **kwargs: NS(**kwargs), "TokensPrompt": lambda **kwargs: kwargs,
        "LoRARequest": lambda **kwargs: NS(**kwargs), "TokenOutput": lambda **kwargs: NS(**kwargs),
        "_qwen2_5_vl_dedup_image_tokens": lambda tokens, _: tokens,
        "VLLM_LORA_INT_ID": 1, "VLLM_LORA_NAME": "training", "VLLM_LORA_PATH": "/adapter",
        "RolloutMode": NS(HYBRID="hybrid"),
    }
    method = _production_method(
        "verl/workers/rollout/vllm_rollout/vllm_async_server.py", "vLLMHttpServerBase", "generate", namespace
    )
    server = NS(config=_Config(max_model_len=16, response_length=4, prompt_length=4,
                              enable_rollout_routing_replay=False),
                model_config=NS(lora_rank=2, processor=None), rollout_mode="hybrid",
                engine=_ServerEngine([1]), replica_rank=0)
    with pytest.raises(FileNotFoundError):
        asyncio.run(method(server, [1, 2], {}, "request-151"))
    assert server.engine.requests == []
    _ready_fixture(resumed)
    asyncio.run(method(server, [1, 2], {}, "request-151"))
    assert server.engine.requests[0]["lora_request"].lora_int_id == 1


def test_launcher_passes_verifier_as_string_ray_environment(resumed):
    # Execute the exact production injection block, then parse the resulting
    # overrides using Hydra's actual grammar. No training command is executed.
    from hydra.core.override_parser.overrides_parser import OverridesParser

    path = Path(__file__).resolve().parents[1] / "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh"
    source = path.read_text()
    start = source.index('if [[ -n "${VERL_LORA_RESUME_VERIFY_SOURCE:-}" ]]')
    end = source.index("\nfi", start) + len("\nfi")
    command = "set -eu\nCMD=()\n" + source[start:end] + '\nprintf "%s\\n" "${CMD[@]}"\n'
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    overrides = OverridesParser.create().parse_overrides(result.stdout.splitlines())
    assert len(overrides) == 6
    for override in overrides:
        key = override.key_or_group.rsplit(".", 1)[1]
        assert isinstance(override.value(), str)
        assert override.value() == os.environ[key]
