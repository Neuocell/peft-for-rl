"""Deterministic construction and auditing for protocol-matched standard LoRA."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def standard_lora_use_meta_tensor(base_use_meta: bool, init_seed: int | None) -> bool:
    """Audited LoRA initialization needs materialized tensors on every rank."""

    return base_use_meta and init_seed is None


def _canonical_lora_state(model: torch.nn.Module) -> dict[str, Any]:
    tensors = []
    digest = hashlib.sha256()
    a_count = 0
    b_count = 0
    b_abs_max = 0.0
    trainable_parameters = 0

    for name, parameter in sorted(model.named_parameters()):
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        tensor = parameter.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\n")
        factor = "A" if ".lora_A." in name else "B"
        if factor == "A":
            a_count += 1
        else:
            b_count += 1
            if tensor.numel():
                b_abs_max = max(b_abs_max, float(tensor.abs().max().item()))
        trainable_parameters += tensor.numel()
        tensors.append(
            {
                "name": name,
                "factor": factor,
                "shape": shape,
                "dtype": dtype,
                "numel": tensor.numel(),
            }
        )

    if not tensors or a_count != b_count:
        raise ValueError(
            f"Invalid standard-LoRA state: A={a_count}, B={b_count}, tensors={len(tensors)}"
        )
    if b_abs_max != 0.0:
        raise ValueError(f"Standard-LoRA initial B must be exactly zero, got max={b_abs_max}")
    return {
        "canonical_sha256": digest.hexdigest(),
        "module_count": a_count,
        "tensor_count": len(tensors),
        "trainable_parameters": trainable_parameters,
        "b_abs_max": b_abs_max,
        "tensors": tensors,
    }


def _orthogonalize_lora_a_rows(model: torch.nn.Module) -> dict[str, Any]:
    """Orthogonalize each LoRA-A row space while preserving its Frobenius norm."""

    modules = []
    max_relative_norm_error = 0.0
    max_row_gram_error = 0.0
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if ".lora_A." not in name:
                continue
            original = parameter.detach().cpu().double()
            if original.ndim != 2 or original.shape[0] > original.shape[1]:
                raise ValueError(f"Cannot row-orthogonalize LoRA-A tensor {name}: {tuple(original.shape)}")
            original_norm = torch.linalg.vector_norm(original)
            q, _ = torch.linalg.qr(original.mT, mode="reduced")
            orthogonal = q.mT.mul(original_norm / torch.sqrt(torch.tensor(original.shape[0], dtype=torch.float64)))
            parameter.copy_(orthogonal.to(device=parameter.device, dtype=parameter.dtype))

            realized = parameter.detach().cpu().double()
            realized_norm = torch.linalg.vector_norm(realized)
            relative_norm_error = float((realized_norm - original_norm).abs() / original_norm.clamp_min(1e-30))
            row_scale = realized_norm.square() / realized.shape[0]
            gram = realized @ realized.mT
            target = torch.eye(realized.shape[0], dtype=torch.float64).mul(row_scale)
            gram_error = float(torch.linalg.vector_norm(gram - target) / row_scale.clamp_min(1e-30))
            max_relative_norm_error = max(max_relative_norm_error, relative_norm_error)
            max_row_gram_error = max(max_row_gram_error, gram_error)
            modules.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "pre_frobenius_norm": float(original_norm),
                    "post_frobenius_norm": float(realized_norm),
                    "relative_norm_error": relative_norm_error,
                    "normalized_row_gram_error": gram_error,
                }
            )
    if not modules:
        raise ValueError("No LoRA-A tensors found for random-orthogonal initialization")
    return {
        "module_count": len(modules),
        "max_relative_norm_error": max_relative_norm_error,
        "max_normalized_row_gram_error": max_row_gram_error,
        "modules": modules,
    }


def _normalize_lora_a_columns(model: torch.nn.Module) -> dict[str, Any]:
    """Apply NoRA-init by normalizing every input column along the rank axis."""

    modules = []
    max_column_norm_error = 0.0
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            if ".lora_A." not in name:
                continue
            if parameter.ndim != 2:
                raise ValueError(f"Cannot apply NoRA-init to LoRA-A tensor {name}: {tuple(parameter.shape)}")
            normalized = torch.nn.functional.normalize(parameter.float(), p=2.0, dim=0)
            parameter.copy_(normalized.to(device=parameter.device, dtype=parameter.dtype))

            column_norms = torch.linalg.vector_norm(parameter.detach().cpu().float(), dim=0)
            column_norm_error = float((column_norms - 1.0).abs().max().item())
            max_column_norm_error = max(max_column_norm_error, column_norm_error)
            modules.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "column_norm_error_max": column_norm_error,
                }
            )
    if not modules:
        raise ValueError("No LoRA-A tensors found for NoRA-init")
    return {
        "module_count": len(modules),
        "normalization_dimension": "rank",
        "max_column_norm_error": max_column_norm_error,
        "modules": modules,
    }


def _write_audit(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError(f"Standard-LoRA initialization audit already differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def create_standard_lora_model(
    model: torch.nn.Module,
    lora_config,
    *,
    init_seed: int | None = None,
    audit_path: str | Path | None = None,
    peer_audit_path: str | Path | None = None,
    peer_timeout_seconds: float = 600.0,
    init_mode: str = "standard",
) -> tuple[torch.nn.Module, dict[str, Any] | None]:
    """Construct PEFT LoRA under a scoped RNG and audit all initial A/B tensors."""

    from peft import __version__ as peft_version
    from peft import get_peft_model

    if init_mode not in {"standard", "random_orthogonal", "nora_init"}:
        raise ValueError(f"Unsupported LoRA initialization mode: {init_mode}")
    if init_mode in {"random_orthogonal", "nora_init"} and init_seed is None:
        raise ValueError(f"{init_mode} LoRA initialization requires lora_init_seed")
    if init_seed is None:
        if audit_path or peer_audit_path:
            raise ValueError("lora_init_seed is required when initialization audits are set")
        return get_peft_model(model, lora_config), None
    if init_seed < 0:
        raise ValueError(f"lora_init_seed must be non-negative, got {init_seed}")

    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(init_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(init_seed)
        peft_model = get_peft_model(model, lora_config)

    pre_orthogonal = _canonical_lora_state(peft_model)
    orthogonalization = None
    nora_normalization = None
    if init_mode == "random_orthogonal":
        orthogonalization = _orthogonalize_lora_a_rows(peft_model)
    elif init_mode == "nora_init":
        nora_normalization = _normalize_lora_a_columns(peft_model)
    local = _canonical_lora_state(peft_model)
    rank_hashes = [local["canonical_sha256"]]
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    world_size = torch.distributed.get_world_size() if distributed else 1
    if distributed:
        rank_hashes = [None] * world_size
        torch.distributed.all_gather_object(rank_hashes, local["canonical_sha256"])
    if len(set(rank_hashes)) != 1:
        raise ValueError(f"Standard-LoRA initial state differs across ranks: {rank_hashes}")

    audit = {
        "schema_version": 1,
        "status": "verified",
        "method": (
            "protocol_matched_standard_lora_initial_state_v1"
            if init_mode == "standard"
            else (
                "standard_seeded_a_then_row_qr_norm_matched_v1"
                if init_mode == "random_orthogonal"
                else "nora_init_rank_dimension_column_normalization_v1"
            )
        ),
        "init_mode": init_mode,
        "init_seed": init_seed,
        "peft_version": peft_version,
        "initializer": {
            "init_lora_weights": True,
            "lora_A": "torch.nn.init.kaiming_uniform_(weight, a=sqrt(5))",
            "lora_B": "torch.nn.init.zeros_(weight)",
        },
        "world_size": world_size,
        "rank_hashes": rank_hashes,
        "all_ranks_agree": True,
        "pre_orthogonal_canonical_sha256": pre_orthogonal["canonical_sha256"],
        "orthogonalization": orthogonalization,
        "nora_normalization": nora_normalization,
        **local,
    }
    if audit_path and rank == 0:
        own_path = Path(audit_path)
        _write_audit(own_path, audit)
        if peer_audit_path:
            peer_path = Path(peer_audit_path).expanduser().resolve()
            deadline = time.monotonic() + peer_timeout_seconds
            while not peer_path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for peer LoRA audit: {peer_path}")
                time.sleep(0.2)
            peer = json.loads(peer_path.read_text(encoding="utf-8"))
            peer_hash = peer.get("canonical_sha256")
            if peer_hash != local["canonical_sha256"]:
                raise ValueError(
                    "Standard-LoRA initial state differs across training seeds: "
                    f"own={local['canonical_sha256']} peer={peer_hash}"
                )
            audit["peer_audit_path"] = str(peer_path)
            audit["peer_canonical_sha256"] = peer_hash
            audit["cross_training_seed_match"] = True
            own_path.unlink()
            _write_audit(own_path, audit)
    if distributed:
        torch.distributed.barrier()
    return peft_model, audit
