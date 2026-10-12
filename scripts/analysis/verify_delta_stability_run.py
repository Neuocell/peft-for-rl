#!/usr/bin/env python3
"""Verify the initialization, snapshots, and resumable checkpoints for S0/O0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(args: argparse.Namespace) -> dict:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    expected_mode = "standard" if args.method == "s0" else "random_orthogonal"
    if audit.get("status") != "verified" or audit.get("init_mode") != expected_mode:
        raise ValueError(f"Initialization audit does not match {args.method}: {args.audit}")
    if audit.get("init_seed") != 42 or audit.get("module_count") != 196 or audit.get("b_abs_max") != 0.0:
        raise ValueError("Initialization audit violates seed/module/B-zero contract")
    if args.method == "o0":
        orthogonal = audit.get("orthogonalization") or {}
        if orthogonal.get("module_count") != 196:
            raise ValueError("O0 audit is missing 196 orthogonalized modules")
        if float(orthogonal.get("max_relative_norm_error", 1.0)) >= 1e-5:
            raise ValueError("O0 Frobenius norm preservation tolerance failed")
        if float(orthogonal.get("max_normalized_row_gram_error", 1.0)) >= 1e-4:
            raise ValueError("O0 row-orthogonality tolerance failed")
        if args.s0_audit:
            s0 = json.loads(args.s0_audit.read_text(encoding="utf-8"))
            if audit.get("pre_orthogonal_canonical_sha256") != s0.get("canonical_sha256"):
                raise ValueError("O0 did not start from the exact S0 standard random A/B state")

    snapshots = []
    for step in range(101):
        step_dir = args.snapshots / f"step_{step:03d}"
        tensor_file = step_dir / "adapter_model.safetensors"
        manifest_file = step_dir / "manifest.json"
        if not tensor_file.is_file() or not manifest_file.is_file():
            raise FileNotFoundError(f"Incomplete snapshot step {step}: {step_dir}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (
            manifest.get("trainer_step") != step
            or manifest.get("optimizer_update") != step * 4
            or manifest.get("rank") != 32
            or manifest.get("alpha") != 64.0
            or manifest.get("module_count") != 196
            or manifest.get("tensor_count") != 392
        ):
            raise ValueError(f"Snapshot manifest contract failed at step {step}")
        with safe_open(tensor_file, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if len(list(handle.keys())) != 392 or int(metadata["trainer_step"]) != step:
                raise ValueError(f"Snapshot tensor metadata failed at step {step}")
        snapshots.append({"step": step, "sha256": sha256(tensor_file), "bytes": tensor_file.stat().st_size})

    checkpoints = []
    for step in (0, 25, 50, 100):
        root = args.checkpoints / f"global_step_{step}"
        required = [root / "data.pt", root / "actor" / "peft_adapter" / "adapter_model.safetensors"]
        required += [root / "actor" / f"model_world_size_4_rank_{rank}.pt" for rank in range(4)]
        required += [root / "actor" / f"optim_world_size_4_rank_{rank}.pt" for rank in range(4)]
        required += [root / "actor" / f"extra_state_world_size_4_rank_{rank}.pt" for rank in range(4)]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
        if missing:
            raise FileNotFoundError(f"Incomplete formal checkpoint step {step}: {missing[0]}")
        checkpoints.append(
            {
                "step": step,
                "adapter_sha256": sha256(root / "actor" / "peft_adapter" / "adapter_model.safetensors"),
            }
        )
    latest = (args.checkpoints / "latest_checkpointed_iteration.txt").read_text().strip()
    if latest != "100":
        raise ValueError(f"Latest checkpoint marker is {latest}, expected 100")
    return {
        "schema_version": 1,
        "status": "verified",
        "method": args.method.upper(),
        "initialization_audit": str(args.audit.resolve()),
        "initialization_sha256": sha256(args.audit),
        "snapshot_count": len(snapshots),
        "snapshot_total_bytes": sum(row["bytes"] for row in snapshots),
        "snapshots": snapshots,
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("s0", "o0"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--s0-audit", type=Path)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "method", "snapshot_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
