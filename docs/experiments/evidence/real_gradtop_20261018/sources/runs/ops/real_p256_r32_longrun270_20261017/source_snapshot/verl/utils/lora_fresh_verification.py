"""Two rollout acceptance gates for a fresh fixed-A REAL run.

The initial source is independently built from the probe artifact. The second
source is rank zero's live actor export after the first optimizer update block;
all actor ranks and actual CUDA adapter slots must match it before sampling.
This does not turn a zero-B initialization into a resumed-checkpoint receipt.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from verl.utils import lora_resume_verification as verify


def check_factors(actual, initial, phase):
    if phase not in (0, 1) or set(actual) != set(initial):
        raise RuntimeError("Invalid fresh LoRA phase or tensor coverage")
    for name, value in actual.items():
        expected = initial[name].to(value.dtype)
        if value.shape != expected.shape:
            raise RuntimeError(f"Fresh LoRA shape mismatch: {name}")
        if ".lora_A." in name and not torch.equal(value, expected):
            raise RuntimeError(f"Fixed REAL A differs from probe initialization: {name}")
        if phase == 0 and not torch.equal(value, expected):
            raise RuntimeError(f"Fresh LoRA initialization mismatch: {name}")
    initial_b = [v for k, v in initial.items() if ".lora_B." in k]
    nonzero = any(torch.count_nonzero(v).item() for k, v in actual.items() if ".lora_B." in k)
    if not initial_b or any(torch.count_nonzero(v).item() for v in initial_b):
        raise RuntimeError("Fresh reference must have zero B")
    if nonzero != (phase == 1):
        raise RuntimeError("Expected zero initial B and nonzero B after the first training step")


def verify_actor_payload(weights, peft_config, phase, checkpoint_source=None):
    if checkpoint_source not in (None, "initial_actor"):
        raise RuntimeError("Fresh verification cannot attest a resumed actor")
    root = Path(os.environ["VERL_LORA_FRESH_VERIFY_DIR"])
    source = Path(os.environ["VERL_LORA_FRESH_VERIFY_INITIAL_SOURCE"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != os.environ["VERL_LORA_FRESH_VERIFY_INITIAL_SHA256"]:
        raise RuntimeError("Fresh initialization source hash changed")
    initial = verify.canonical_tensors(load_file(str(source)))
    actual = verify.canonical_tensors(weights)
    if len(actual) != int(os.environ["VERL_LORA_FRESH_VERIFY_TENSOR_COUNT"]):
        raise RuntimeError("Fresh LoRA tensor count mismatch")
    check_factors(actual, initial, phase)
    directory = root / ("initial" if phase == 0 else "after_step1")
    if verify._rank() == 0:
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = directory / "adapter_model.safetensors"
        if phase == 0:
            os.link(source, snapshot)
        else:
            save_file(actual, str(snapshot))
        (directory / "adapter_config.json").write_text(json.dumps(
            {"r": int(peft_config.r), "lora_alpha": float(peft_config.lora_alpha)}) + "\n")
        cfg = {"source": str(snapshot), "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
               "output_dir": str(directory), "tensor_count": len(actual),
               "world_size": int(os.environ["VERL_LORA_FRESH_VERIFY_WORLD_SIZE"]),
               "token": os.environ["VERL_LORA_FRESH_VERIFY_TOKEN"] + f":{phase}",
               "fresh_phase": phase}
        temporary = root / "active.tmp"
        temporary.write_text(json.dumps(cfg) + "\n")
        temporary.replace(root / "active.json")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    cfg = verify.settings()
    if cfg["fresh_phase"] != phase:
        raise RuntimeError("Fresh verification phases disagree")
    reference = verify._source(cfg)
    for name, value in actual.items():
        if not torch.equal(value, reference[name].to(value.dtype)):
            raise RuntimeError(f"Fresh actor ranks disagree on {name}")
    return verify._emit(cfg, f"actor.rank{verify._rank()}", {
        "status": "source_verified", "rank": verify._rank(),
        "checkpoint_source": str(directory), "fresh_phase": phase,
        "source_kind": "probe_initialization" if phase == 0 else "live_actor_after_first_update_block",
        "tensor_count": len(actual), "actor_payload_sha256": verify.tensor_fingerprint(actual),
        "actor_dtypes": {k: str(v.dtype).removeprefix("torch.") for k, v in actual.items()},
        "fixed_a_matches_probe": True, "b_nonzero": phase == 1,
    })
