#!/usr/bin/env python3
"""Create and verify immutable Phase-0.6/Phase-1 training contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PREPARATION_KINDS = ("phase1", "phase06")
METHODS_BY_KIND = {
    "phase1": ("I0", "I8", "I16", "I32"),
    "phase06": ("P0", "P2", "C0", "C2"),
}
IMMUTABLE_FIELDS_BY_KIND = {
    "phase1": (
        "source_validation_sha256",
        "replay_validation_sha256",
        "phase05_gate_sha256",
        "phase06_outcome_kind",
        "phase06_outcome_path",
        "phase06_outcome_sha256",
        "signal_candidate_sha256",
    ),
    "phase06": (
        "source_validation_sha256",
        "replay_validation_sha256",
        "gate_sha256",
    ),
}
CODE_PROVENANCE_ROOTS = {
    "verl": (".py", ".yaml", ".yml"),
    "examples/verl_train": (".py", ".sh", ".yaml", ".yml"),
    "tina_run/scripts/local/eval": (".py", ".sh", ".yaml", ".yml"),
}
CODE_PROVENANCE_FILES = (
    "scripts/analysis/aggregate_multiseed_full_bench.py",
    "scripts/analysis/compare_paired_full_bench.py",
    "scripts/analysis/phase1_training_contract.py",
    "scripts/analysis/prepare_phase1_signal_random_artifacts.py",
    "scripts/analysis/select_phase1_seed42_candidate.py",
    "scripts/analysis/verify_full_bench_run.py",
    "scripts/local/run_phase1_signal_random_confirmation.sh",
    "scripts/local/run_phase1_signal_random_step50_fullbench.sh",
    "scripts/local/start_full_gradient_uniform_r8_4gpu.sh",
    "scripts/local/start_phase1_signal_random_uniform_r32_4gpu.sh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _code_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    paths: set[Path] = set()
    for relative_root, suffixes in CODE_PROVENANCE_ROOTS.items():
        root = repository_root / relative_root
        if not root.is_dir():
            raise ValueError(f"Missing code-provenance directory: {root}")
        paths.update(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    for relative_path in CODE_PROVENANCE_FILES:
        path = repository_root / relative_path
        if not path.is_file():
            raise ValueError(f"Missing code-provenance file: {path}")
        paths.add(path)

    files = {
        path.relative_to(repository_root).as_posix(): _sha256(path)
        for path in sorted(paths)
    }
    digest = hashlib.sha256()
    for relative_path, file_hash in files.items():
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "file_count": len(files),
        "aggregate_sha256": digest.hexdigest(),
        "files": files,
    }


def _expected_contract(
    preparation_path: Path,
    method: str,
    training_seed: int,
    experiment_name: str,
    preparation_kind: str = "phase1",
) -> dict[str, Any]:
    if preparation_kind not in PREPARATION_KINDS:
        raise ValueError(f"Unknown training preparation kind: {preparation_kind}")
    label = "Phase-1" if preparation_kind == "phase1" else "Phase-0.6"
    preparation_path = preparation_path.expanduser().resolve()
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("status") != "ready":
        raise ValueError(f"{label} preparation is not ready")
    if method not in METHODS_BY_KIND[preparation_kind]:
        raise ValueError(f"{label} method is not allowed: {method}")
    allocation = preparation.get("allocations", {}).get(method)
    if not isinstance(allocation, dict):
        raise ValueError(f"{label} method is not prepared: {method}")
    if training_seed not in (42, 43, 44):
        raise ValueError(f"{label} training seed must be one of 42, 43 or 44")
    if not experiment_name:
        raise ValueError(f"{label} experiment name cannot be empty")
    contract = {
        "schema_version": 1,
        "method": f"{preparation_kind}_training_provenance_contract_v1",
        "preparation_kind": preparation_kind,
        "candidate_method": method,
        "training_seed": training_seed,
        "experiment_name": experiment_name,
        "preparation_path": str(preparation_path),
        "preparation_sha256": _sha256(preparation_path),
        "rank_map_sha256": allocation["rank_map_sha256"],
        "subspace_sha256": allocation["subspace_sha256"],
        "allocation_summary_sha256": allocation["allocation_summary_sha256"],
        "code_provenance": _code_provenance(),
        "training_configuration": {
            "advantage_estimator": "grpo",
            "learning_rate": 1e-6,
            "learning_rate_warmup_steps": 0,
            "learning_rate_scheduler": "constant",
            "weight_decay": 0.0,
            "prompt_batch_size": 64,
            "prompt_mini_batch_size": 16,
            "responses_per_prompt": 8,
            "ppo_epochs": 1,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "use_kl_in_reward": False,
            "use_kl_loss": False,
            "rollout_temperature": 1.0,
            "rollout_top_p": 1.0,
            "rollout_top_k": -1,
            "train_shuffle": True,
            "actor_shuffle": False,
            "total_steps": 50,
            "checkpoint_steps": [25, 50],
            "uniform_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "gradient_subspace_scaling": 2.0,
        },
    }
    for field in IMMUTABLE_FIELDS_BY_KIND[preparation_kind]:
        contract[field] = preparation[field]
    return contract


def create_contract(
    preparation_path: Path,
    method: str,
    training_seed: int,
    experiment_name: str,
    output_path: Path,
    preparation_kind: str = "phase1",
) -> dict[str, Any]:
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise ValueError(f"Refusing to overwrite training contract: {output_path}")
    contract = {
        **_expected_contract(
            preparation_path,
            method,
            training_seed,
            experiment_name,
            preparation_kind,
        ),
        "status": "prepared",
        "adapter_path": None,
        "adapter_sha256": None,
    }
    _write_json_atomic(output_path, contract)
    return contract


def finalize_contract(contract_path: Path, adapter_path: Path) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    adapter_path = adapter_path.expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preparation_kind = contract.get("preparation_kind")
    if preparation_kind not in PREPARATION_KINDS:
        raise ValueError("Training contract has an unknown preparation kind")
    label = "Phase-1" if preparation_kind == "phase1" else "Phase-0.6"
    if contract.get("status") != "prepared":
        raise ValueError(f"{label} contract is not awaiting finalization")
    if not adapter_path.is_file() or adapter_path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty {label} adapter: {adapter_path}")
    contract.update(
        {
            "status": "complete",
            "adapter_path": str(adapter_path),
            "adapter_sha256": _sha256(adapter_path),
        }
    )
    _write_json_atomic(contract_path, contract)
    return contract


def verify_contract(
    preparation_path: Path,
    method: str,
    training_seed: int,
    experiment_name: str,
    contract_path: Path,
    adapter_path: Path,
    preparation_kind: str = "phase1",
) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    adapter_path = adapter_path.expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = _expected_contract(
        preparation_path,
        method,
        training_seed,
        experiment_name,
        preparation_kind,
    )
    label = "Phase-1" if preparation_kind == "phase1" else "Phase-0.6"
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"{label} training contract changed: {key}")
    if contract.get("status") != "complete":
        raise ValueError(f"{label} training contract is not complete")
    if Path(contract.get("adapter_path", "")).expanduser().resolve() != adapter_path:
        raise ValueError(f"{label} training contract points to another adapter")
    if not adapter_path.is_file() or adapter_path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty {label} adapter: {adapter_path}")
    if contract.get("adapter_sha256") != _sha256(adapter_path):
        raise ValueError(f"{label} adapter SHA-256 changed after training")
    return {
        "status": "verified",
        "preparation_kind": preparation_kind,
        "candidate_method": method,
        "training_seed": training_seed,
        "experiment_name": experiment_name,
        "contract_path": str(contract_path),
        "adapter_path": str(adapter_path),
        "adapter_sha256": contract["adapter_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    verify = subparsers.add_parser("verify")
    for command in (create, verify):
        command.add_argument("--preparation", type=Path, required=True)
        command.add_argument(
            "--preparation-kind", choices=PREPARATION_KINDS, default="phase1"
        )
        command.add_argument(
            "--method",
            choices=tuple(
                sorted(
                    {
                        method
                        for methods in METHODS_BY_KIND.values()
                        for method in methods
                    }
                )
            ),
            required=True,
        )
        command.add_argument("--training-seed", type=int, required=True)
        command.add_argument("--experiment-name", required=True)
        command.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--adapter", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--contract", type=Path, required=True)
    finalize.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "create":
        result = create_contract(
            args.preparation,
            args.method,
            args.training_seed,
            args.experiment_name,
            args.contract,
            args.preparation_kind,
        )
    elif args.command == "finalize":
        result = finalize_contract(args.contract, args.adapter)
    else:
        result = verify_contract(
            args.preparation,
            args.method,
            args.training_seed,
            args.experiment_name,
            args.contract,
            args.adapter,
            args.preparation_kind,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
