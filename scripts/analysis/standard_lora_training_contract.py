#!/usr/bin/env python3
"""Create and verify protocol-matched standard-LoRA training contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import peft

from scripts.analysis.phase1_training_contract import _code_provenance
from scripts.analysis.verify_full_bench_run import verify_full_bench_run


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "peft_version": "0.19.1",
    "base_model_sha256": "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945",
    "training_data_sha256": "8e3c9314db8b83c61ab62a3dc85e0a704dcc5f3a9d404d236943f719095ce82f",
    "snapshot_sha256": "3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da",
    "module_count": 196,
    "trainable_parameters": 36929536,
    "init_seed": 42,
}
LOCAL_CODE_FILES = (
    "scripts/analysis/standard_lora_training_contract.py",
    "scripts/local/start_matched_standard_lora_r32_4gpu.sh",
    "scripts/local/run_matched_standard_lora_r32_pipeline.sh",
    "tests/test_standard_lora_init.py",
)
SOURCE_SPECS = (
    "runs/phase1-signal-random-v1/analysis/phase1_two_seed_paper_baseline_amendment.json",
    "runs/phase1-signal-random-v1/analysis/phase1_two_seed_standard_lora_init_seed_correction.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any], *, refuse_existing: bool = False) -> None:
    path = path.expanduser().resolve()
    if refuse_existing and path.exists():
        raise ValueError(f"Refusing to overwrite contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _local_code_provenance() -> dict[str, str]:
    result = {}
    for relative in LOCAL_CODE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"Missing matched-LoRA code file: {path}")
        result[relative] = _sha256(path)
    return result


def _peft_source_provenance() -> dict[str, str]:
    package_root = Path(peft.__file__).resolve().parent
    relative_paths = ("tuners/lora/config.py", "tuners/lora/layer.py")
    return {relative: _sha256(package_root / relative) for relative in relative_paths}


def _package_lock() -> dict[str, Any]:
    completed = subprocess.run(
        [os.sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = sorted(line.strip() for line in completed.stdout.splitlines() if line.strip())
    encoded = ("\n".join(packages) + "\n").encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "packages": packages}


def _expected_contract(
    *, experiment_name: str, training_seed: int, model: Path, train_file: Path,
    snapshot: Path, audit: Path,
) -> dict[str, Any]:
    if training_seed not in (42, 43):
        raise ValueError("Matched standard LoRA only permits training seeds 42 and 43")
    if not experiment_name:
        raise ValueError("Experiment name cannot be empty")
    model = model.expanduser().resolve()
    train_file = train_file.expanduser().resolve()
    snapshot = snapshot.expanduser().resolve()
    audit = audit.expanduser().resolve()
    model_weights = model / "model.safetensors"
    for label, path in (("model", model_weights), ("training data", train_file), ("snapshot", snapshot)):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty {label}: {path}")
    actual = {
        "base_model_sha256": _sha256(model_weights),
        "training_data_sha256": _sha256(train_file),
        "snapshot_sha256": _sha256(snapshot),
    }
    for key, value in actual.items():
        if value != EXPECTED[key]:
            raise ValueError(f"Pinned asset changed: {key} expected={EXPECTED[key]} actual={value}")
    if peft.__version__ != EXPECTED["peft_version"]:
        raise ValueError(f"Expected PEFT {EXPECTED['peft_version']}, got {peft.__version__}")
    source_specs = {
        relative: _sha256(REPO_ROOT / relative) for relative in SOURCE_SPECS
    }
    return {
        "schema_version": 1,
        "method": "protocol_matched_standard_lora_r32_training_contract_v1",
        "experiment_name": experiment_name,
        "training_seed": training_seed,
        "standard_lora_init_seed": EXPECTED["init_seed"],
        "execution_authorization": {
            "kind": "explicit_owner_baseline_completion_after_phase1_results",
            "date": "2026-10-05",
            "note": "Manual launch authorization; the earlier automatic gate activation condition was not satisfied.",
        },
        "model_path": str(model),
        "training_data_path": str(train_file),
        "benchmark_snapshot_path": str(snapshot),
        "initial_state_audit_path": str(audit),
        **actual,
        "source_specs": source_specs,
        "peft": {
            "version": peft.__version__,
            "source_sha256": _peft_source_provenance(),
            "init_lora_weights": True,
            "lora_A_initializer": "torch.nn.init.kaiming_uniform_(weight, a=sqrt(5))",
            "lora_B_initializer": "torch.nn.init.zeros_(weight)",
        },
        "training_configuration": {
            "peft_type": "lora",
            "target_modules": "all-linear",
            "expected_target_module_count": 196,
            "rank": 32,
            "alpha": 64,
            "scaling": 2.0,
            "dropout": 0.0,
            "bias": "none",
            "lora_freeze_a": False,
            "optimizer_steps": 50,
            "prompts_per_step": 64,
            "prompt_minibatch_size": 16,
            "responses_per_prompt": 8,
            "ppo_epochs": 1,
            "learning_rate": 1e-6,
            "learning_rate_warmup_steps": 0,
            "learning_rate_scheduler": "constant",
            "weight_decay": 0.0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "use_kl_in_reward": False,
            "use_kl_loss": False,
            "rollout_temperature": 1.0,
            "rollout_top_p": 1.0,
            "rollout_top_k": -1,
            "train_shuffle": True,
            "actor_shuffle": False,
            "data_seed": training_seed,
            "actor_data_loader_seed": training_seed,
            "rollout_seed": training_seed,
            "checkpoint_steps": [25, 50],
            "checkpoint_contents": ["model", "optimizer", "extra"],
            "resume_mode": "disable",
        },
        "evaluation_configuration": {
            "benchmarks": ["aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva"],
            "seed": 42,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 32768,
            "max_model_len": 34816,
            "records": 7248,
            "problems": 902,
        },
        "code_provenance": _code_provenance(),
        "matched_lora_code_provenance": _local_code_provenance(),
        "package_lock": _package_lock(),
    }


def create(args) -> dict[str, Any]:
    contract = _expected_contract(
        experiment_name=args.experiment_name,
        training_seed=args.training_seed,
        model=args.model,
        train_file=args.train_file,
        snapshot=args.snapshot,
        audit=args.audit,
    )
    contract.update({"status": "prepared", "initial_state": None, "adapter": None})
    _write(args.contract, contract, refuse_existing=True)
    return contract


def _verify_static(contract: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_contract(
        experiment_name=contract["experiment_name"],
        training_seed=int(contract["training_seed"]),
        model=Path(contract["model_path"]),
        train_file=Path(contract["training_data_path"]),
        snapshot=Path(contract["benchmark_snapshot_path"]),
        audit=Path(contract["initial_state_audit_path"]),
    )
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Standard-LoRA contract changed: {key}")
    return expected


def finalize(args) -> dict[str, Any]:
    path = args.contract.expanduser().resolve()
    contract = json.loads(path.read_text(encoding="utf-8"))
    _verify_static(contract)
    if contract.get("status") != "prepared":
        raise ValueError("Standard-LoRA contract is not awaiting finalization")
    audit_path = Path(contract["initial_state_audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_audit = {
        "status": "verified",
        "init_seed": EXPECTED["init_seed"],
        "peft_version": EXPECTED["peft_version"],
        "module_count": EXPECTED["module_count"],
        "trainable_parameters": EXPECTED["trainable_parameters"],
        "b_abs_max": 0.0,
        "all_ranks_agree": True,
        "cross_training_seed_match": True,
    }
    for key, value in required_audit.items():
        if audit.get(key) != value:
            raise ValueError(f"Standard-LoRA initialization audit failed: {key}")
    adapter = args.adapter.expanduser().resolve()
    if not adapter.is_file() or adapter.stat().st_size <= 0:
        raise ValueError(f"Missing or empty adapter: {adapter}")
    config = adapter.parent / "adapter_config.json"
    contract.update(
        {
            "status": "complete",
            "initial_state": {
                "audit_sha256": _sha256(audit_path),
                "canonical_sha256": audit["canonical_sha256"],
                "peer_canonical_sha256": audit["peer_canonical_sha256"],
                "module_count": audit["module_count"],
                "trainable_parameters": audit["trainable_parameters"],
            },
            "adapter": {
                "path": str(adapter),
                "sha256": _sha256(adapter),
                "config_path": str(config.resolve()),
                "config_sha256": _sha256(config),
            },
        }
    )
    _write(path, contract)
    return contract


def verify(args) -> dict[str, Any]:
    path = args.contract.expanduser().resolve()
    contract = json.loads(path.read_text(encoding="utf-8"))
    _verify_static(contract)
    if contract.get("status") != "complete":
        raise ValueError("Standard-LoRA contract is incomplete")
    adapter = args.adapter.expanduser().resolve()
    if contract["adapter"]["path"] != str(adapter) or contract["adapter"]["sha256"] != _sha256(adapter):
        raise ValueError("Standard-LoRA adapter changed after finalization")
    audit_path = Path(contract["initial_state_audit_path"])
    if contract["initial_state"]["audit_sha256"] != _sha256(audit_path):
        raise ValueError("Standard-LoRA initialization audit changed")
    return {
        "status": "verified",
        "experiment_name": contract["experiment_name"],
        "training_seed": contract["training_seed"],
        "initial_state_sha256": contract["initial_state"]["canonical_sha256"],
        "adapter_sha256": contract["adapter"]["sha256"],
    }


def postverify(args) -> dict[str, Any]:
    training = verify(args)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    evaluation = verify_full_bench_run(
        args.summary, args.records, args.eval_dir, args.eval_name,
        args.adapter.parent, Path(contract["model_path"]),
        Path(contract["benchmark_snapshot_path"]), contract["snapshot_sha256"],
    )
    result = {
        "schema_version": 1,
        "status": "verified",
        "method": "protocol_matched_standard_lora_postverification_v1",
        "training": training,
        "evaluation": evaluation,
        "artifacts": {
            "contract": {"path": str(args.contract.resolve()), "sha256": _sha256(args.contract)},
            "adapter": {"path": str(args.adapter.resolve()), "sha256": _sha256(args.adapter)},
            "records": {"path": str(args.records.resolve()), "sha256": _sha256(args.records)},
            "summary": {"path": str(args.summary.resolve()), "sha256": _sha256(args.summary)},
        },
    }
    _write(args.output, result, refuse_existing=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--experiment-name", required=True)
    create_parser.add_argument("--training-seed", type=int, required=True)
    create_parser.add_argument("--model", type=Path, required=True)
    create_parser.add_argument("--train-file", type=Path, required=True)
    create_parser.add_argument("--snapshot", type=Path, required=True)
    create_parser.add_argument("--audit", type=Path, required=True)
    create_parser.add_argument("--contract", type=Path, required=True)
    for name in ("finalize", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--adapter", type=Path, required=True)
    post = commands.add_parser("postverify")
    post.add_argument("--contract", type=Path, required=True)
    post.add_argument("--adapter", type=Path, required=True)
    post.add_argument("--summary", type=Path, required=True)
    post.add_argument("--records", type=Path, required=True)
    post.add_argument("--eval-dir", type=Path, required=True)
    post.add_argument("--eval-name", required=True)
    post.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create(args)
    elif args.command == "finalize":
        result = finalize(args)
    elif args.command == "verify":
        result = verify(args)
    else:
        result = postverify(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
