#!/usr/bin/env python3
"""Record immutable run provenance and reject mixing pre/post-fix RLVR trajectories."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

AFFECTED_METHODS = {"dora", "nora_dora", "nora_dora_continuous", "nora_doft"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exporter_version(path: Path) -> str:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CUSTOM_PEFT_ROLLOUT_EXPORT_VERSION"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise ValueError("Custom PEFT exporter has no version; refusing unverified runtime")


def make_identity(contract: dict, protocol: dict, repo: Path) -> dict:
    if contract.get("status") != "verified" or contract["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("A verified matching protocol contract is required")
    method = contract["experiment_variables"]["method"]
    identity = {
        "protocol_id": contract["protocol_id"],
        "experiment_variables": contract["experiment_variables"],
        "training": protocol["training"],
        "data_order": contract["data_order"],
        "assets": {
            name: {key: value for key, value in spec.items() if key != "path"}
            for name, spec in contract["assets"].items()
        },
    }
    artifact_key = None
    if method == "real":
        artifact_key = "real_artifact"
    elif method.startswith("gradtop"):
        artifact_key = "gradtop_artifact"
    if artifact_key is not None:
        artifact = contract.get(artifact_key, {})
        files = artifact.get("files", {})
        required = {"rank_map.json", "summary.json", "subspaces.safetensors"}
        if set(files) != required or any(not spec.get("sha256") for spec in files.values()):
            raise ValueError(f"Verified {artifact_key} file hashes are required for runtime provenance")
        # Bind initialization content, allowing identical artifacts to move directories.
        identity[artifact_key] = {name: spec["sha256"] for name, spec in files.items()}
    if method in AFFECTED_METHODS:
        identity["rollout_export_version"] = exporter_version(repo / "verl/utils/fsdp_utils.py")
        implementation = "peft_nora_doft.py" if method == "nora_doft" else "peft_nora_dora.py"
        identity["method_source_sha256"] = sha256(repo / "verl/utils" / implementation)
    if method in {"dora", "nora_dora", "nora_dora_continuous", "nora_init"}:
        identity["initialization_version"] = "peft_nora_reference_v2"
        identity["initialization_source_sha256"] = sha256(repo / "verl/utils/adapter_initialization.py")
        if method == "nora_init":
            identity["method_source_sha256"] = sha256(repo / "verl/utils/peft_lora_init.py")
    return identity


def validate_run(identity: dict, manifest: Path, checkpoints: Path, snapshots: Path, resume_mode: str) -> str:
    has_checkpoints = any(checkpoints.glob("global_step_*"))
    has_snapshots = any(snapshots.glob("step_*"))
    if resume_mode == "disable" and (has_checkpoints or has_snapshots):
        raise ValueError("Existing training artifacts: use a new EXP_NAME for a fresh run")
    if resume_mode == "auto" and has_snapshots and not has_checkpoints:
        raise ValueError("Only evaluation snapshots exist; they cannot resume optimizer/RNG/data state")
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("Runtime identity changed (data/config/exporter/method); use a new EXP_NAME")
        return previous["trajectory_origin"]
    if has_checkpoints:
        if identity["experiment_variables"]["method"] in AFFECTED_METHODS:
            raise ValueError("Legacy custom-PEFT checkpoint lacks corrected-exporter provenance; restart from base")
        raise ValueError(
            "Legacy checkpoint lacks runtime input provenance; verify and migrate its original "
            "data/config before resuming through the frozen-protocol launcher (retraining is not required)"
        )
    return "fresh_base"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--resume-mode", choices=("disable", "auto"), required=True)
    parser.add_argument("--record", action="store_true", help="Record an actual launch; omit for dry-run")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    protocol = json.loads(Path(contract["protocol_path"]).read_text(encoding="utf-8"))
    identity = make_identity(contract, protocol, REPO_ROOT)
    origin = validate_run(identity, args.manifest, args.checkpoints, args.snapshots, args.resume_mode)
    if args.record and not args.manifest.exists():
        sources = [
            "verl/utils/fsdp_utils.py",
            "verl/utils/peft_nora_dora.py",
            "verl/utils/peft_nora_doft.py",
            "verl/utils/adapter_initialization.py",
            "verl/utils/peft_lora_init.py",
            "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh",
            "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh",
        ]
        record = {
            "schema_version": 1,
            "record_type": "launch_provenance_not_training_completion",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "trajectory_origin": origin,
            "identity": identity,
            "initial_target_step": contract["execution"]["target_step"],
            "source_sha256": {name: sha256(REPO_ROOT / name) for name in sources},
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves the original provenance if two launches race.
        with args.manifest.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    version = identity.get("rollout_export_version", "method-specific standard export")
    print(f"Runtime provenance verified: {args.manifest}; export={version}; origin={origin}")


if __name__ == "__main__":
    main()
