#!/usr/bin/env python3
"""Save REAL CPU sufficient statistics at worker boundaries and export offline.

This optional observer does not change the gradient estimator or the queued
experiment's pinned source files. It runs on rank zero via the existing worker
RPC and wraps only that collector instance's end-of-step callback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_snapshot(collector, cache_dir, trainer_step=None, *, reserve_bytes=15 * 1024**3):
    """Call only between actor compute tasks or after a complete trainer step."""
    import torch
    from safetensors.torch import save_file

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    acc = collector.accumulator
    names = set(collector.module_shapes)
    count = collector.observations
    if not count or any(set(items) != names for items in (acc.sketches, acc.omegas, acc.counts, acc.energy_sums)):
        raise ValueError("Cannot cache an empty or incomplete REAL accumulator")
    if set(acc.counts.values()) != {count} or len(collector.prompt_ids) != count:
        raise ValueError("REAL cache requires a complete prompt-group boundary")
    if len(set(collector.prompt_ids)) != count:
        raise ValueError("REAL cache prompt IDs must be unique")
    for name in names:
        y, omega = acc.sketches[name], acc.omegas[name]
        if y.device.type != "cpu" or omega.device.type != "cpu":
            raise ValueError("Observer only saves existing CPU statistics")
        if y.shape != omega.shape or not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(omega).all()):
            raise ValueError(f"Invalid cached statistics for {name}")
    tag = f"step_{trainer_step:03d}" if trainer_step is not None else f"groups_{count:06d}"
    destination = root / tag
    if destination.exists():
        raise FileExistsError(f"REAL snapshot already exists: {destination}")
    size = sum(t.numel() * t.element_size() for t in acc.sketches.values())
    if shutil.disk_usage(root).free < reserve_bytes + 2 * size:
        raise OSError("Insufficient disk for cache while reserving training space")
    omega_path = root / "omega.safetensors"
    known_hash = getattr(collector, "_real_cache_omega_sha256", None)
    if omega_path.exists() and known_hash is None:
        raise FileExistsError("Refusing to mix a new collector with an existing cache")
    if not omega_path.exists():
        temporary = root / f".omega.{uuid.uuid4().hex}.tmp"
        save_file(acc.omegas, str(temporary))
        os.replace(temporary, omega_path)
    omega_hash = digest(omega_path)
    if known_hash is not None and known_hash != omega_hash:
        raise ValueError("Shared input sketch cache changed")
    collector._real_cache_omega_sha256 = omega_hash
    temporary = root / f".{tag}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        tensor_path = temporary / "y_sum.safetensors"
        save_file(acc.sketches, str(tensor_path))
        metadata = {
            "schema_version": 1, "status": "complete", "kind": "real_cumulative_second_moment_sketch",
            "created_at": datetime.now(timezone.utc).isoformat(), "trainer_step": trainer_step,
            "prompt_groups": count, "prompt_ids": list(collector.prompt_ids),
            "counts": dict(acc.counts), "energy_sums": dict(acc.energy_sums),
            "module_shapes": dict(collector.module_shapes), "rank": collector.rank,
            "input_width": collector.input_width, "output_width": collector.output_width,
            "seed": collector.seed, "scaling_ratio": collector.scaling_ratio,
            "relative_tolerance": acc.relative_tolerance,
            "source_model_path": collector.config.get("path"),
            "source_artifact_dir": str(collector.output_dir),
            "original_requested_steps": collector.target_steps,
            "original_requested_prompt_groups": collector.target_prompts,
            "omega_file": "../omega.safetensors", "omega_sha256": omega_hash,
            "y_file": "y_sum.safetensors", "y_sha256": digest(tensor_path),
            "y_bytes": size, "observer_sha256": digest(__file__),
            "reuse_scope": "same objective and sketch draws; exact saved prefix only; no per-group gradients",
        }
        write_json(temporary / "metadata.json", metadata)
        os.replace(temporary, destination)
        write_json(root / "latest.json", {"snapshot": str(destination), **{
            key: metadata[key] for key in ("trainer_step", "prompt_groups", "created_at")
        }})
        return {"snapshot": str(destination), "trainer_step": trainer_step, "prompt_groups": count,
                "y_bytes": size, "y_sha256": metadata["y_sha256"]}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def attach_worker_cache(worker, *, cache_dir, expected_output_dir, expected_stage_token, steps):
    """Install on a serial rank-zero actor via execute_with_func_generator."""
    if worker.rank != 0 or os.environ.get("REAL_PIPELINE_STAGE_TOKEN") != expected_stage_token:
        raise ValueError("Cache RPC addressed the wrong experiment/rank")
    collector = worker.actor._real_probe_collector
    if collector is None or str(collector.output_dir) != str(Path(expected_output_dir).resolve()):
        raise ValueError("Cache RPC addressed a different REAL collector")
    if getattr(collector, "_real_cache_attached", False):
        raise RuntimeError("A REAL cache observer is already attached")
    root = Path(cache_dir).resolve()
    initial = save_snapshot(collector, root)
    original = collector.finish_trainer_step
    selected_steps = frozenset(int(step) for step in steps)

    def finish(trainer_step):
        metrics = original(trainer_step)
        if trainer_step in selected_steps:
            try:
                saved = save_snapshot(collector, root, trainer_step)
                print(f"REAL cache saved: {json.dumps(saved)}", flush=True)
            except Exception as error:
                # A failed optional write must not abort the authorized probe.
                failure = {"trainer_step": trainer_step, "error": f"{type(error).__name__}: {error}"}
                try:
                    write_json(root / f"step_{trainer_step:03d}.error.json", failure)
                except Exception:
                    pass
                print(f"REAL cache write failed: {json.dumps(failure)}", flush=True)
        return metrics

    collector.finish_trainer_step = finish
    collector._real_cache_attached = True
    policy = {"status": "attached", "actor_pid": os.getpid(), "rank": worker.rank,
              "source_artifact_dir": str(collector.output_dir), "snapshot_steps": sorted(selected_steps),
              "initial": initial, "observer_sha256": digest(__file__),
              "earlier_prefixes_recoverable": False}
    write_json(root / "policy.json", policy)
    return policy


def export_snapshot(snapshot, output_dir, *, rank=None):
    """Recover an adapter basis without a model, rollouts, or backward passes."""
    from safetensors.torch import load_file
    from verl.utils.peft_real import RealProbeCollector, RealSubspaceAccumulator

    snapshot, output = Path(snapshot).resolve(), Path(output_dir).resolve()
    metadata = json.loads((snapshot / "metadata.json").read_text())
    if metadata.get("status") != "complete" or metadata.get("schema_version") != 1:
        raise ValueError("Incomplete/unsupported REAL snapshot")
    if output.exists():
        raise FileExistsError(f"Refusing existing export: {output}")
    for key in ("omega", "y"):
        if digest(snapshot / metadata[f"{key}_file"]) != metadata[f"{key}_sha256"]:
            raise ValueError(f"REAL cache {key} hash mismatch")
    count = metadata["prompt_groups"]
    if count <= 0 or set(metadata["counts"].values()) != {count} or len(metadata["prompt_ids"]) != count:
        raise ValueError("REAL cache prompt counts disagree")
    acc = RealSubspaceAccumulator(rank=metadata["rank"] if rank is None else rank,
                                 input_width=metadata["input_width"], seed=metadata["seed"],
                                 relative_tolerance=metadata["relative_tolerance"])
    acc.omegas = load_file(str(snapshot / metadata["omega_file"]))
    acc.sketches = load_file(str(snapshot / metadata["y_file"]))
    acc.counts, acc.energy_sums = metadata["counts"], metadata["energy_sums"]
    names = set(metadata["module_shapes"])
    if any(set(item) != names for item in (acc.omegas, acc.sketches, acc.counts, acc.energy_sums)):
        raise ValueError("REAL cache module coverage mismatch")
    bases, diagnostics = acc.snapshot()
    step = metadata["trainer_step"]
    exporter = SimpleNamespace(
        output_dir=output, rank=acc.rank, scaling_ratio=metadata["scaling_ratio"],
        module_shapes=metadata["module_shapes"], observations=count, prompt_ids=metadata["prompt_ids"],
        target_prompts=count if step is None else 0, target_steps=step,
        seed=metadata["seed"], config={"path": metadata["source_model_path"]},
        input_width=metadata["input_width"], output_width=metadata["output_width"], accumulator=acc,
    )
    # Export into a temporary sibling so a failed decomposition/export cannot
    # leave a partial artifact that looks ready for training.
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    exporter.output_dir = temporary
    try:
        RealProbeCollector._export(exporter, step, bases, diagnostics)
        for name in ("summary.json", "rank_map.json"):
            path = temporary / name
            value = json.loads(path.read_text())
            value.update(subspace_path=str(output / "subspaces.safetensors"),
                         replay_source=str(snapshot), replay_source_metadata_sha256=digest(snapshot / "metadata.json"))
            write_json(path, value)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"output_dir": str(output), "prompt_groups": count, "trainer_step": step, "rank": acc.rank}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    attach = commands.add_parser("attach")
    for key in ("address", "namespace", "actor", "cache-dir", "expected-output-dir", "expected-stage-token"):
        attach.add_argument(f"--{key}", required=True)
    attach.add_argument("--steps", nargs="+", type=int, default=[32, 48, 64])
    export = commands.add_parser("export")
    export.add_argument("--snapshot", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--rank", type=int)
    args = parser.parse_args()
    if args.command == "export":
        result = export_snapshot(args.snapshot, args.output_dir, rank=args.rank)
    else:
        import ray
        ray.init(address=args.address, namespace="real_cache_observer", log_to_driver=False)
        try:
            actor = ray.get_actor(args.actor, namespace=args.namespace)
            result = ray.get(actor.actor_rollout_execute_with_func_generator.remote(
                attach_worker_cache, cache_dir=args.cache_dir, expected_output_dir=args.expected_output_dir,
                expected_stage_token=args.expected_stage_token, steps=args.steps))
        finally:
            ray.shutdown()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
