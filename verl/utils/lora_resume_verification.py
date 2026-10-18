"""Opt-in first-rollout verification for a restored native LoRA adapter.

This deliberately supports unquantized TP=PP=1 linear LoRA only. It checks the
actor export against an independently named safetensors snapshot, then reads
both the registered vLLM adapter and the active device buffers. Unsupported
layouts fail closed. No vLLM imports or GPU initialization are needed to import
this module or run its CPU transport tests.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

PREFIX = "VERL_LORA_RESUME_VERIFY_"


def settings():
    if os.environ.get("VERL_LORA_FRESH_VERIFY_DIR"):
        if os.environ.get(PREFIX + "SOURCE"):
            raise RuntimeError("Fresh and resumed LoRA verification cannot be combined")
        path = Path(os.environ["VERL_LORA_FRESH_VERIFY_DIR"]) / "active.json"
        return json.loads(path.read_text())
    if not os.environ.get(PREFIX + "SOURCE"):
        return None
    required = ("SOURCE", "SHA256", "OUTPUT_DIR", "TOKEN", "WORLD_SIZE", "TENSOR_COUNT")
    missing = [key for key in required if not os.environ.get(PREFIX + key)]
    if missing:
        raise RuntimeError(f"Incomplete LoRA resume verification settings: {missing}")
    cfg = {key.lower(): os.environ[PREFIX + key] for key in required}
    cfg["source"] = str(Path(cfg["source"]).resolve())
    cfg["world_size"] = int(cfg["world_size"])
    cfg["tensor_count"] = int(cfg["tensor_count"])
    if cfg["world_size"] <= 0 or cfg["tensor_count"] <= 0:
        raise RuntimeError("LoRA verification counts must be positive")
    return cfg


def canonical_name(name):
    name = name.replace("_fsdp_wrapped_module.", "").replace(".default.", ".")
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    if not name.endswith((".lora_A.weight", ".lora_B.weight")):
        raise RuntimeError(f"Unsupported LoRA verification tensor: {name}")
    return name


def canonical_tensors(tensors):
    result = {}
    for name, value in tensors.items():
        key = canonical_name(name)
        if key in result:
            raise RuntimeError(f"Duplicate canonical LoRA tensor: {key}")
        if not torch.is_tensor(value) or value.ndim != 2:
            raise RuntimeError(f"Unsupported LoRA tensor shape: {key}")
        value = value.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite LoRA tensor: {key}")
        result[key] = value
    if not result:
        raise RuntimeError("Empty LoRA verification payload")
    return result


def tensor_fingerprint(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _source(cfg):
    from safetensors.torch import load_file

    path = Path(cfg["source"])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != cfg["sha256"]:
        raise RuntimeError("Resume adapter snapshot SHA256 does not match the frozen expected SHA256")
    tensors = canonical_tensors(load_file(str(path), device="cpu"))
    if len(tensors) != cfg["tensor_count"]:
        raise RuntimeError("Resume adapter snapshot tensor count mismatch")
    return tensors


def _emit(cfg, kind, receipt):
    receipt = {
        **receipt,
        "schema_version": 1,
        "token": cfg["token"],
        "source": cfg["source"],
        "source_file_sha256": cfg["sha256"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    directory = Path(cfg["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{kind}.json"
    temporary = directory / f".{kind}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    temporary.replace(destination)
    print("[lora_resume_verification] " + json.dumps({"event": kind, **receipt}, sort_keys=True), flush=True)
    return receipt


def verify_actor_payload(weights, checkpoint_source):
    cfg = settings()
    if cfg is None:
        raise RuntimeError("Native LoRA resume verification was requested without settings")
    if not checkpoint_source or checkpoint_source == "initial_actor":
        raise RuntimeError("Resume verification requires a restored actor checkpoint")
    source, actual = _source(cfg), canonical_tensors(weights)
    if set(source) != set(actual):
        raise RuntimeError("Actor export and resume snapshot LoRA tensor names differ")
    for name, value in actual.items():
        expected = source[name].to(value.dtype)
        if value.shape != expected.shape or not torch.equal(value, expected):
            raise RuntimeError(f"Actor export differs from resume snapshot: {name}")
    if not any(torch.count_nonzero(value).item() for name, value in actual.items() if ".lora_B." in name):
        raise RuntimeError("Resume verification requires a nonzero trained LoRA B")
    return _emit(cfg, f"actor.rank{_rank()}", {
        "status": "source_verified", "rank": _rank(), "checkpoint_source": str(checkpoint_source),
        "tensor_count": len(actual), "actor_payload_sha256": tensor_fingerprint(actual),
        "actor_dtypes": {name: str(value.dtype).removeprefix("torch.") for name, value in actual.items()},
        "comparison": "exact_after_snapshot_cast_to_actor_dtype",
    })


def verify_engine_adapter(worker, peft_config, source_receipt, adapter_id, *, require_cuda=True):
    """Read registered adapter and active stack; do not trust add_lora alone."""
    cfg = settings()
    if cfg is None or source_receipt.get("token") != cfg["token"]:
        raise RuntimeError("Missing/mismatched actor verification receipt")
    if source_receipt.get("source_file_sha256") != cfg["sha256"]:
        raise RuntimeError("Actor verification used a different snapshot")
    runner = worker.model_runner
    parallel = runner.vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        raise RuntimeError("Native LoRA exact readback supports TP=PP=1 only")
    if getattr(runner.vllm_config.model_config, "quantization", None):
        raise RuntimeError("Native LoRA exact readback requires an unquantized engine")
    if any(peft_config.get(key) for key in ("use_rslora", "use_dora")):
        raise RuntimeError("Native LoRA exact readback requires ordinary constant rank/alpha LoRA")
    rank, alpha = int(peft_config["r"]), float(peft_config["lora_alpha"])
    if any(int(value) != rank for value in (peft_config.get("rank_pattern") or {}).values()) or any(
        float(value) != alpha for value in (peft_config.get("alpha_pattern") or {}).values()
    ):
        raise RuntimeError("Native LoRA exact readback requires constant rank/alpha patterns")
    config_path = Path(cfg["source"]).with_name("adapter_config.json")
    source_config = json.loads(config_path.read_text())
    if int(source_config["r"]) != rank or float(source_config["lora_alpha"]) != alpha:
        raise RuntimeError("Restored adapter rank/alpha differs from the source snapshot")
    manager = runner.lora_manager._adapter_manager
    registered = manager.get_adapter(adapter_id)
    if registered is None or adapter_id not in runner.lora_manager.list_adapters():
        raise RuntimeError("vLLM did not register the restored adapter")
    if manager.lora_index_to_id.count(adapter_id) != 1:
        raise RuntimeError("Restored LoRA has no unique active device slot")
    slot = manager.lora_index_to_id.index(adapter_id)
    source = _source(cfg)
    actual_cpu, actual_device, seen = {}, {}, set()
    devices = set()
    for module_name, installed in registered.loras.items():
        module = manager.modules.get(module_name)
        if module is None or getattr(module, "tp_size", None) != 1:
            raise RuntimeError(f"Unsupported installed LoRA module: {module_name}")
        names = manager.packed_modules.get(module_name, [module_name])
        aa = installed.lora_a if isinstance(installed.lora_a, (list, tuple)) else [installed.lora_a]
        bb = installed.lora_b if isinstance(installed.lora_b, (list, tuple)) else [installed.lora_b]
        if not (len(names) == len(aa) == len(bb) == len(module.lora_a_stacked) == len(module.lora_b_stacked)):
            raise RuntimeError(f"Unsupported packed LoRA layout: {module_name}")
        for index, name in enumerate(names):
            for factor, values, buffers in (("A", aa, module.lora_a_stacked), ("B", bb, module.lora_b_stacked)):
                key = canonical_name(f"{name}.lora_{factor}.weight")
                if key not in source or key in seen or values[index] is None:
                    raise RuntimeError(f"Unexpected/missing registered LoRA tensor: {key}")
                value = values[index].detach().cpu()
                actor_dtype = getattr(torch, source_receipt["actor_dtypes"][key])
                expected = source[key].to(actor_dtype).to(value.dtype).clone()
                if factor == "B":
                    expected.mul_(alpha / rank)
                if value.shape != expected.shape or not torch.equal(value, expected):
                    raise RuntimeError(f"Registered LoRA differs from source: {key}")
                buffer = buffers[index]
                devices.add(buffer.device.type)
                if require_cuda and buffer.device.type != "cuda":
                    raise RuntimeError(f"Expected CUDA LoRA active buffer: {key}")
                if buffer.ndim != 4 or buffer.shape[1] != 1:
                    raise RuntimeError(f"Unsupported LoRA stack layout: {key}")
                # Check the full slot, including padding that must remain zero.
                active = buffer[slot, 0].detach().cpu()
                padded = torch.zeros_like(active)
                if any(a < b for a, b in zip(active.shape, expected.shape)):
                    raise RuntimeError(f"LoRA active buffer is too small: {key}")
                padded[: expected.shape[0], : expected.shape[1]] = expected
                if not torch.equal(active, padded):
                    raise RuntimeError(f"Active device LoRA differs from source: {key}")
                actual_cpu[key] = value
                actual_device[key] = active[: expected.shape[0], : expected.shape[1]]
                seen.add(key)
    if seen != set(source):
        raise RuntimeError("vLLM readback did not cover every source LoRA tensor")
    return _emit(cfg, f"engine.rank{_rank()}", {
        "status": "verified", "rank": _rank(), "checkpoint_source": source_receipt["checkpoint_source"],
        "tensor_count": len(seen), "adapter_id": adapter_id, "active_slot": slot,
        "actor_payload_sha256": source_receipt["actor_payload_sha256"],
        "registered_adapter_sha256": tensor_fingerprint(actual_cpu),
        "active_device_adapter_sha256": tensor_fingerprint(actual_device),
        "device_types": sorted(devices), "comparison": "exact_registered_and_full_active_slot_with_scaled_B",
    })


def acknowledge_ready(engine_receipt, source_receipt):
    cfg = settings()
    if (not isinstance(engine_receipt, dict) or engine_receipt.get("status") != "verified"
            or engine_receipt.get("token") != cfg["token"]
            or engine_receipt.get("tensor_count") != cfg["tensor_count"]
            or engine_receipt.get("actor_payload_sha256") != source_receipt["actor_payload_sha256"]):
        raise RuntimeError("Rollout did not acknowledge complete native LoRA resume verification")
    return _emit(cfg, f"ready.rank{_rank()}", engine_receipt)


def verify_first_request(replica_rank, request_id, adapter_id, loaded_ids):
    cfg = settings()
    if adapter_id not in loaded_ids:
        raise RuntimeError("First resumed request has no installed training adapter")
    ready = []
    for rank in range(cfg["world_size"]):
        path = Path(cfg["output_dir"]) / f"ready.rank{rank}.json"
        receipt = json.loads(path.read_text())
        if (receipt.get("token") != cfg["token"] or receipt.get("status") != "verified"
                or receipt.get("rank") != rank or receipt.get("adapter_id") != adapter_id
                or receipt.get("source_file_sha256") != cfg["sha256"]
                or receipt.get("tensor_count") != cfg["tensor_count"]
                or receipt.get("device_types") != ["cuda"]):
            raise RuntimeError(f"First request lacks valid LoRA readback for rank {rank}")
        ready.append(receipt)
    if len({receipt["actor_payload_sha256"] for receipt in ready}) != 1:
        raise RuntimeError("Actor ranks disagree on resumed LoRA payload")
    return _emit(cfg, f"request.replica{replica_rank}", {
        "status": "request_verified", "replica_rank": replica_rank, "request_id": request_id,
        "lora_request_id": adapter_id, "loaded_adapter_ids": sorted(loaded_ids),
        "verified_worker_ranks": list(range(cfg["world_size"])),
        "actor_payload_sha256": ready[0]["actor_payload_sha256"],
    })


def validate_receipt_directory(directory, *, expected_world_size, expected_checkpoint, required=False):
    """Controller-side acceptance; missing evidence is pending, invalid evidence raises."""
    directory = Path(directory)
    expected_checkpoint = str(Path(expected_checkpoint).resolve())
    records = {"actor": [], "engine": [], "ready": [], "request": []}
    missing = []
    for rank in range(expected_world_size):
        for kind in records:
            name = f"{kind}.{'replica' if kind == 'request' else 'rank'}{rank}.json"
            path = directory / name
            if not path.exists():
                missing.append(name)
                continue
            receipt = json.loads(path.read_text())
            expected_status = "source_verified" if kind == "actor" else (
                "request_verified" if kind == "request" else "verified"
            )
            if receipt.get("status") != expected_status:
                raise RuntimeError(f"Invalid LoRA resume receipt: {name}")
            if kind == "request":
                if (receipt.get("replica_rank") != rank or not receipt.get("request_id")
                        or receipt.get("lora_request_id") not in receipt.get("loaded_adapter_ids", [])
                        or receipt.get("verified_worker_ranks") != list(range(expected_world_size))):
                    raise RuntimeError(f"Incomplete LoRA request proof: {name}")
            else:
                if (receipt.get("rank") != rank
                        or str(Path(receipt.get("checkpoint_source", "")).resolve()) != expected_checkpoint):
                    raise RuntimeError(f"LoRA receipt restored the wrong checkpoint: {name}")
                if kind != "actor" and (receipt.get("device_types") != ["cuda"]
                        or receipt.get("registered_adapter_sha256") != receipt.get("active_device_adapter_sha256")):
                    raise RuntimeError(f"LoRA GPU readback proof is incomplete: {name}")
            records[kind].append(receipt)
    all_records = [record for values in records.values() for record in values]
    for field in ("token", "source", "source_file_sha256", "actor_payload_sha256"):
        values = {record.get(field) for record in all_records}
        if None in values or len(values) > 1:
            raise RuntimeError(f"LoRA resume receipts disagree on {field}")
    cfg = settings()
    if cfg and all_records and (all_records[0]["token"] != cfg["token"]
            or all_records[0]["source_file_sha256"] != cfg["sha256"]):
        raise RuntimeError("LoRA resume receipts do not match the controller launch settings")
    if missing:
        if required:
            raise RuntimeError(f"Missing native LoRA resume receipts: {missing}")
        return None
    counts = {record.get("tensor_count") for kind in ("actor", "engine", "ready") for record in records[kind]}
    if len(counts) != 1 or None in counts or next(iter(counts)) <= 0:
        raise RuntimeError("LoRA resume receipt tensor counts disagree")
    return {
        "status": "verified", "checkpoint_source": expected_checkpoint,
        "world_size": expected_world_size, "tensor_count": next(iter(counts)),
        "token": all_records[0]["token"], "source": all_records[0]["source"],
        "source_file_sha256": all_records[0]["source_file_sha256"],
        "actor_payload_sha256": all_records[0]["actor_payload_sha256"],
        "receipts": records,
    }
