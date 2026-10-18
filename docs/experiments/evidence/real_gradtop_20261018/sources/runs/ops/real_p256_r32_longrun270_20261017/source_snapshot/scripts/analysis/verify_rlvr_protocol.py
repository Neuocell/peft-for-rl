#!/usr/bin/env python3
"""Verify the frozen RLVR protocol assets and allowed experiment variables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


def validate_real_artifact(
    artifact_dir: Path, *, rank: int, alpha: int, model_weights: Path, definition: dict[str, Any]
) -> dict[str, Any]:
    """Check REAL observation provenance and the complete fixed-A initialization on CPU."""
    from safetensors import safe_open

    artifact_dir = artifact_dir.expanduser().resolve()
    files = {name: artifact_dir / name for name in ("rank_map.json", "summary.json", "subspaces.safetensors")}
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing REAL artifact {name}: {path}")
    allocation = json.loads(files["rank_map.json"].read_text(encoding="utf-8"))
    summary = json.loads(files["summary.json"].read_text(encoding="utf-8"))
    if allocation.get("method") != definition["artifact_method"] or allocation.get("freeze_a") is not True:
        raise ValueError("Expected a REAL fixed-A artifact")
    if allocation.get("schema_version") != 1:
        raise ValueError("Unsupported REAL artifact schema")
    if any(summary.get(key) != value for key, value in allocation.items()):
        raise ValueError("REAL summary and rank map disagree")
    ranks, alphas = allocation.get("rank_pattern"), allocation.get("alpha_pattern")
    if not isinstance(ranks, dict) or not ranks or not isinstance(alphas, dict) or set(alphas) != set(ranks):
        raise ValueError("REAL requires matching non-empty rank and alpha patterns")
    if any(type(value) is not int or value != rank for value in ranks.values()):
        raise ValueError(f"REAL must have uniform rank {rank}")
    if any(type(value) is not int or value != alpha for value in alphas.values()):
        raise ValueError(f"REAL must have uniform alpha {alpha}")
    scaling = allocation.get("constant_scaling")
    if isinstance(scaling, bool) or not isinstance(scaling, (float, int)) or not math.isclose(
        scaling, alpha / rank, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("REAL constant scaling must match alpha / rank")
    if Path(allocation.get("subspace_path", "")).expanduser().resolve() != files["subspaces.safetensors"]:
        raise ValueError("REAL rank map subspace_path does not match the supplied artifact directory")

    observation = definition["observation"]
    for key, expected in observation.items():
        if key == "sketch_width_multiple_of_rank":
            continue
        actual = summary.get(key)
        if actual != expected or (isinstance(expected, bool) and actual is not expected):
            raise ValueError(f"REAL observation {key} must be {expected!r}, got {actual!r}")
    if summary.get("probe_method") != "real":
        raise ValueError("REAL summary probe_method must be real")
    if summary.get("trainer_step") != observation["requested_observation_steps"]:
        raise ValueError("REAL observation did not complete the registered step budget")
    width = rank * observation["sketch_width_multiple_of_rank"]
    if summary.get("input_sketch_width") != width or summary.get("output_sketch_width") != width:
        raise ValueError(f"REAL input and output sketch widths must both equal {width}")
    if summary.get("distributed_observation") != "complete_prompt_group_replicated_on_data_parallel_ranks":
        raise ValueError("REAL requires complete prompt groups on all data parallel ranks")
    if Path(summary.get("source_model_path") or "").expanduser().resolve() != model_weights.resolve().parent:
        raise ValueError("REAL source model differs from the frozen protocol model")
    count = summary.get("prompt_groups")
    prompt_ids = summary.get("prompt_ids")
    if type(count) is not int or count <= 0 or count > observation["requested_observation_steps"] * 64:
        raise ValueError("REAL prompt_groups must be positive and within the observation budget")
    if not isinstance(prompt_ids, list) or any(not isinstance(item, str) for item in prompt_ids):
        raise ValueError("REAL prompt_ids must be a list of strings")
    if len(prompt_ids) != count or len(set(prompt_ids)) != count:
        raise ValueError("REAL prompt_ids must contain one unique ID per observed group")

    # Read tensor headers only; the 1.5B base model is never materialized here.
    with safe_open(str(model_weights), framework="pt", device="cpu") as base:
        expected_shapes = {
            name.removesuffix(".weight"): tuple(base.get_slice(name).get_shape())
            for name in base.keys()
            if name.startswith("model.layers.")
            and name.endswith(".weight")
            and len(base.get_slice(name).get_shape()) == 2
        }
    modules = summary.get("modules")
    if not expected_shapes or set(ranks) != set(expected_shapes):
        raise ValueError("REAL rank map does not cover every base-model linear module")
    if not isinstance(modules, dict) or set(modules) != set(ranks):
        raise ValueError("REAL module diagnostics have incomplete coverage")
    max_error = 0.0
    with safe_open(str(files["subspaces.safetensors"]), framework="pt", device="cpu") as bases:
        if set(bases.keys()) != set(ranks):
            raise ValueError("REAL tensors do not match rank-map module coverage")
        for name, (output_dim, input_dim) in expected_shapes.items():
            basis = bases.get_tensor(name)
            if tuple(basis.shape) != (rank, input_dim):
                raise ValueError(f"REAL basis shape mismatch for {name}")
            if not basis.is_floating_point() or not torch.isfinite(basis).all():
                raise ValueError(f"REAL basis must be finite floating-point values: {name}")
            basis = basis.float()
            error = float((basis @ basis.T - torch.eye(rank)).abs().max())
            if not math.isfinite(error) or error > 1e-4:
                raise ValueError(f"REAL basis is not orthonormal: {name} error={error}")
            max_error = max(max_error, error)
            module = modules[name]
            required = {
                "selected_rank": rank,
                "input_dimension": input_dim,
                "prompt_groups": count,
                "parameter_cost_per_rank": input_dim + output_dim,
                "b_only_parameter_cost_per_rank": output_dim,
            }
            if any(module.get(key) != value for key, value in required.items()):
                raise ValueError(f"REAL module dimensions, rank, or observation count disagree: {name}")
    trainable = sum(rank * shape[0] for shape in expected_shapes.values())
    if summary.get("formal_trainable_parameters") != trainable:
        raise ValueError("REAL fixed-A trainable parameter count does not match the base model")
    return {
        "method": allocation["method"],
        "directory": str(artifact_dir),
        "freeze_a": True,
        "initial_b": "zero",
        "module_count": len(ranks),
        "prompt_groups": count,
        "observation_steps": summary["trainer_step"],
        "formal_trainable_parameters": trainable,
        "orthogonality_error_max": max_error,
        "files": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in files.items()},
    }


def validate_gradtop_artifact(
    artifact_dir: Path, *, rank: int, alpha: int, model_weights: Path, definition: dict[str, Any]
) -> dict[str, Any]:
    """Validate the complete GradTop initialization and frozen observation budget on CPU."""
    from safetensors import safe_open

    artifact_dir = artifact_dir.expanduser().resolve()
    files = {name: artifact_dir / name for name in ("rank_map.json", "summary.json", "subspaces.safetensors")}
    for name, path in files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing GradTop artifact {name}: {path}")
    allocation = json.loads(files["rank_map.json"].read_text(encoding="utf-8"))
    summary = json.loads(files["summary.json"].read_text(encoding="utf-8"))
    if allocation.get("method") != definition["artifact_method"] or allocation.get("schema_version") != 1:
        raise ValueError("Expected a schema 1 GradTop energy artifact")
    if any(summary.get(key) != value for key, value in allocation.items()):
        raise ValueError("GradTop summary and rank map disagree")
    ranks, alphas = allocation.get("rank_pattern"), allocation.get("alpha_pattern")
    if not isinstance(ranks, dict) or not ranks or not isinstance(alphas, dict) or set(alphas) != set(ranks):
        raise ValueError("GradTop requires matching non-empty rank and alpha patterns")
    if any(type(value) is not int or value != rank for value in ranks.values()):
        raise ValueError(f"GradTop must have uniform rank {rank}")
    if any(type(value) is not int or value != alpha for value in alphas.values()):
        raise ValueError(f"GradTop must have uniform alpha {alpha}")
    scaling = allocation.get("constant_scaling")
    if isinstance(scaling, bool) or not isinstance(scaling, (float, int)) or not math.isclose(
        scaling, alpha / rank, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("GradTop constant scaling must match alpha / rank")
    if Path(allocation.get("subspace_path", "")).expanduser().resolve() != files["subspaces.safetensors"]:
        raise ValueError("GradTop rank map subspace_path does not match the supplied artifact directory")
    for key, expected in definition["observation"].items():
        actual = summary.get(key)
        if actual != expected or (type(expected) in (bool, int) and type(actual) is not type(expected)):
            raise ValueError(f"GradTop observation {key} must be {expected!r}, got {actual!r}")
    if summary.get("rank_bins") != [rank]:
        raise ValueError("GradTop rank_bins must contain only the requested uniform rank")
    if Path(summary.get("source_model_path") or "").expanduser().resolve() != model_weights.resolve().parent:
        raise ValueError("GradTop source model differs from the frozen protocol model")
    observations = definition["observation"]["observations"]
    metadata = summary.get("observation_metadata")
    if not isinstance(metadata, list) or len(metadata) != observations:
        raise ValueError("GradTop observation_metadata must cover every observation")
    for record in metadata:
        if not isinstance(record, dict) or record.get("prompt_groups") != 16 or record.get("responses") != 128:
            raise ValueError("GradTop observations require 16 complete prompt groups and 128 responses")
        valid_tokens = record.get("valid_tokens")
        if type(valid_tokens) is not int or not 0 < valid_tokens <= 128 * 8192:
            raise ValueError("GradTop observation valid_tokens must be positive and within the completion budget")
    prompt_groups = observations * 16
    if summary.get("observed_prompt_groups") != prompt_groups or summary.get("sampled_prompts") != prompt_groups:
        raise ValueError("GradTop prompt counts disagree with the frozen observation budget")
    if summary.get("sampled_rollouts") != prompt_groups * 8:
        raise ValueError("GradTop rollout count disagrees with the frozen observation budget")

    with safe_open(str(model_weights), framework="pt", device="cpu") as base:
        expected_shapes = {
            name.removesuffix(".weight"): tuple(base.get_slice(name).get_shape())
            for name in base.keys()
            if name.startswith("model.layers.")
            and name.endswith(".weight")
            and len(base.get_slice(name).get_shape()) == 2
        }
    modules = summary.get("modules")
    if not expected_shapes or set(ranks) != set(expected_shapes):
        raise ValueError("GradTop rank map does not cover every base-model linear module")
    if not isinstance(modules, dict) or set(modules) != set(ranks):
        raise ValueError("GradTop module diagnostics have incomplete coverage")
    max_error = 0.0
    completed_directions = 0
    with safe_open(str(files["subspaces.safetensors"]), framework="pt", device="cpu") as bases:
        if set(bases.keys()) != set(ranks):
            raise ValueError("GradTop tensors do not match rank-map module coverage")
        for name, (output_dim, input_dim) in expected_shapes.items():
            basis = bases.get_tensor(name)
            if tuple(basis.shape) != (rank, input_dim):
                raise ValueError(f"GradTop basis shape mismatch for {name}")
            if not basis.is_floating_point() or not torch.isfinite(basis).all():
                raise ValueError(f"GradTop basis must be finite floating-point values: {name}")
            basis = basis.float()
            error = float((basis @ basis.T - torch.eye(rank)).abs().max())
            if not math.isfinite(error) or error > 1e-4:
                raise ValueError(f"GradTop basis is not orthonormal: {name} error={error}")
            max_error = max(max_error, error)
            required = {
                "selected_rank": rank, "input_dimension": input_dim, "output_dimension": output_dim,
                "observations": observations, "parameter_cost_per_rank": input_dim + output_dim,
            }
            if not isinstance(modules[name], dict) or any(modules[name].get(key) != value for key, value in required.items()):
                raise ValueError(f"GradTop module dimensions, rank, or observation count disagree: {name}")
            supported = modules[name].get("sketch_supported_rank")
            completed = modules[name].get("basis_completion_rank")
            if type(supported) is not int or not 0 <= supported <= rank or type(completed) is not int or completed != rank - supported:
                raise ValueError(f"GradTop basis support and completion diagnostics disagree: {name}")
            expected_completion = "deterministic_coordinate_qr" if completed else "none"
            if modules[name].get("basis_completion_method") != expected_completion:
                raise ValueError(f"GradTop basis completion method disagrees: {name}")
            completed_directions += completed
    if summary.get("completed_basis_directions") != completed_directions:
        raise ValueError("GradTop completed_basis_directions disagrees with module diagnostics")
    trainable = sum(rank * sum(shape) for shape in expected_shapes.values())
    if summary.get("formal_trainable_parameters") != trainable:
        raise ValueError("GradTop trainable parameter count does not match the base model")
    return {
        "method": allocation["method"], "directory": str(artifact_dir), "freeze_a": False,
        "initial_b": "zero", "module_count": len(ranks), "observations": observations,
        "gradient_probe_variant": summary["gradient_probe_variant"], "optimizer_updates": 0,
        "trainer_step": summary["trainer_step"], "formal_trainable_parameters": trainable,
        "orthogonality_error_max": max_error, "completed_basis_directions": completed_directions,
        "files": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in files.items()},
    }


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
    artifact_dir = getattr(args, "real_artifact_dir", None)
    if args.method == "real" and artifact_dir is None:
        raise ValueError("METHOD=real requires --real-artifact-dir with a completed REAL observation")
    if args.method != "real" and artifact_dir is not None:
        raise ValueError("--real-artifact-dir is only supported with METHOD=real")

    gradtop_definition = protocol.get("method_definitions", {}).get(args.method, {})
    is_gradtop = args.method.startswith("gradtop")
    gradtop_dir = getattr(args, "gradtop_artifact_dir", None)
    preview_missing = getattr(args, "preview_missing_gradtop_artifact", False)
    if is_gradtop and gradtop_dir is None:
        raise ValueError("GradTop methods require --gradtop-artifact-dir with a completed observation")
    if not is_gradtop and (gradtop_dir is not None or preview_missing):
        raise ValueError("GradTop artifact options are only supported with GradTop methods")
    if is_gradtop and args.rank > 64:
        raise ValueError("GradTop rank cannot exceed the frozen sketch capacity 64")

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

    real_artifact = None
    if args.method == "real":
        real_artifact = validate_real_artifact(
            artifact_dir,
            rank=args.rank,
            alpha=int(protocol["training"]["alpha"]),
            model_weights=Path(checked_assets["base_model_weights"]["path"]),
            definition=protocol["method_definitions"]["real"],
        )

    gradtop_artifact = None
    if is_gradtop:
        missing = not any((gradtop_dir / name).exists() for name in ("rank_map.json", "summary.json", "subspaces.safetensors"))
        if preview_missing and missing:
            gradtop_artifact = {"status": "pending_probe", "directory": str(gradtop_dir.resolve())}
        else:
            gradtop_artifact = validate_gradtop_artifact(
                gradtop_dir, rank=args.rank, alpha=int(protocol["training"]["alpha"]),
                model_weights=Path(checked_assets["base_model_weights"]["path"]), definition=gradtop_definition,
            )

    return {
        "status": "preview_pending_gradtop_artifact" if gradtop_artifact and gradtop_artifact.get("status") == "pending_probe" else "verified",
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
        **({"real_artifact": real_artifact} if real_artifact is not None else {}),
        **({"gradtop_artifact": gradtop_artifact} if gradtop_artifact is not None else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--learning-rate", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--target-step", type=int, default=270)
    parser.add_argument("--real-artifact-dir", type=Path)
    parser.add_argument("--gradtop-artifact-dir", type=Path)
    parser.add_argument("--preview-missing-gradtop-artifact", action="store_true", help="Preview only; produces an explicitly unverified contract if the probe has not run")
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
