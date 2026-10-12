#!/usr/bin/env python3
"""Verify the frozen RLVR protocol assets and allowed experiment variables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torchdata.stateful_dataloader.sampler import RandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    REPO_ROOT / "configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _read_indices(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [int(row["source_row_index"]) for row in csv.DictReader(handle)]


def _canonical_lr(raw: str) -> str:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid learning rate: {raw}") from exc
    allowed = {Decimal("1e-6"): "1e-6", Decimal("5e-6"): "5e-6", Decimal("1e-5"): "1e-5"}
    if value not in allowed:
        raise ValueError(f"LR must be one of {list(allowed.values())}, got {raw}")
    return allowed[value]


def verify(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "rlvr_dapo_math_1p5b_protocol_v1":
        raise ValueError(f"Unexpected protocol: {protocol.get('protocol_id')}")

    lr = _canonical_lr(args.learning_rate)
    methods = protocol["allowed_scientific_variations"]["method"]
    if args.method not in methods:
        raise ValueError(f"METHOD must be one of {methods}, got {args.method}")
    if args.rank <= 0:
        raise ValueError(f"LORA_RANK must be positive, got {args.rank}")
    if args.method == "nora_doft" and args.rank > 32:
        raise ValueError("nora_doft rank cannot exceed its frozen block size 32")

    controls = protocol["execution_controls"]
    max_steps = int(controls["max_training_steps"])
    snapshot = controls["evaluation_snapshot"]
    full_checkpoint = controls["full_checkpoint"]
    interval = int(snapshot["interval_steps"])
    if snapshot.get("resumable") is not False:
        raise ValueError("Evaluation snapshots must be explicitly marked non-resumable")
    if int(full_checkpoint["interval_steps"]) != 50:
        raise ValueError("Full resumable checkpoint interval must be 50 steps")
    if int(full_checkpoint["max_to_keep"]) != 2:
        raise ValueError("Exactly two full resumable checkpoints must be retained")
    if full_checkpoint.get("force_at_planned_terminal_step") is not True:
        raise ValueError("A full checkpoint must be forced at the planned terminal step")
    required_resume_contents = {"model", "optimizer", "extra", "dataloader"}
    if set(full_checkpoint["contents"]) != required_resume_contents:
        raise ValueError(
            "Full checkpoint contents must be model, optimizer, extra, and dataloader"
        )
    if controls.get("nfs_policy") != "post_run_archive_only_not_training_hot_path":
        raise ValueError("NFS must not be used in the synchronous training save path")
    if args.target_step <= 0 or args.target_step > max_steps or args.target_step % interval:
        raise ValueError(
            f"TARGET_STEP must be a positive multiple of {interval} and <= {max_steps}, "
            f"got {args.target_step}"
        )

    checked_assets: dict[str, dict[str, Any]] = {}
    for name, spec in protocol["assets"].items():
        path = _resolve(spec["path"]).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty protocol asset {name}: {path}")
        actual_hash = _sha256(path)
        if actual_hash != spec["sha256"]:
            raise ValueError(
                f"Protocol asset changed: {name} expected={spec['sha256']} actual={actual_hash}"
            )
        row_count = None
        if path.suffix == ".parquet":
            row_count = pq.ParquetFile(path).metadata.num_rows
            if row_count != int(spec["rows"]):
                raise ValueError(
                    f"Protocol row count changed: {name} expected={spec['rows']} actual={row_count}"
                )
        checked_assets[name] = {
            "path": str(path),
            "sha256": actual_hash,
            **({"rows": row_count} if row_count is not None else {}),
        }

    prefix_indices = _read_indices(Path(checked_assets["sampler_prefix_indices"]["path"]))
    source_rows = int(checked_assets["source_data"]["rows"])
    effective_rows = 64 * 270
    if len(prefix_indices) != effective_rows or len(set(prefix_indices)) != effective_rows:
        raise ValueError("Sampler-prefix indices are not exactly 17,280 unique rows")
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(range(source_rows), generator=generator)
    expected_prefix = list(iter(sampler))[:effective_rows]
    if prefix_indices != expected_prefix:
        raise ValueError(
            "Pinned sampler-prefix indices do not match the active torchdata RandomSampler(seed=42)"
        )

    return {
        "status": "verified",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "experiment_variables": {
            "learning_rate": lr,
            "method": args.method,
            "rank": args.rank,
        },
        "execution": {
            "target_step": args.target_step,
            "evaluation_snapshot_interval": interval,
            "evaluation_snapshot_resumable": False,
            "full_checkpoint_interval": int(full_checkpoint["interval_steps"]),
            "full_checkpoint_at_terminal_step": True,
            "full_checkpoints_to_keep": int(full_checkpoint["max_to_keep"]),
            "storage": "local_hot_storage",
            "nfs_policy": controls["nfs_policy"],
        },
        "data_order": {
            "source_rows": source_rows,
            "effective_rows_through_step_270": len(prefix_indices),
            "rows_not_consumed_through_step_270": source_rows - len(prefix_indices),
            "sampler": "torchdata.stateful_dataloader.sampler.RandomSampler",
            "shuffle": True,
            "data_seed": 42,
            "prefix_order_verified": True,
        },
        "assets": checked_assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--target-step", type=int, default=270)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        os.replace(temporary, output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
