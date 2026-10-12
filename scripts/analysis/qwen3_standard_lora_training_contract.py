#!/usr/bin/env python3
"""Create and verify the fixed Qwen3-4B standard-LoRA baseline contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import peft

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.phase1_training_contract import _code_provenance
from scripts.analysis.verify_full_bench_run import verify_full_bench_run


MODEL_FILES = {
    "config.json": "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00003.safetensors": "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors": "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors": "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1",
    "model.safetensors.index.json": "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
EXPECTED = {
    "peft_version": "0.19.1",
    "training_data_sha256": "8e3c9314db8b83c61ab62a3dc85e0a704dcc5f3a9d404d236943f719095ce82f",
    "snapshot_sha256": "3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da",
    "module_count": 252,
    "trainable_parameters": 66060288,
    "init_seed": 42,
    "training_seed": 42,
}
LOCAL_CODE_FILES = (
    "scripts/analysis/qwen3_standard_lora_training_contract.py",
    "scripts/local/start_qwen3_4b_standard_lora_4gpu.sh",
    "scripts/local/run_qwen3_4b_standard_lora_smoke.sh",
    "scripts/local/run_qwen3_4b_standard_lora_pipeline.sh",
    "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh",
    "verl/utils/peft_lora_init.py",
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


def _asset_manifest(model: Path) -> dict[str, str]:
    model = model.expanduser().resolve()
    actual = {}
    for name, expected_hash in MODEL_FILES.items():
        path = model / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty Qwen3-4B model asset: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Pinned Qwen3-4B asset changed: {name} expected={expected_hash} actual={actual_hash}"
            )
        actual[name] = actual_hash
    return actual


def _local_code_provenance() -> dict[str, str]:
    result = {}
    for relative in LOCAL_CODE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"Missing Qwen baseline code file: {path}")
        result[relative] = _sha256(path)
    return result


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


def _peft_source_provenance() -> dict[str, str]:
    package_root = Path(peft.__file__).resolve().parent
    relative_paths = ("tuners/lora/config.py", "tuners/lora/layer.py")
    return {relative: _sha256(package_root / relative) for relative in relative_paths}


def _expected_contract(
    *, experiment_name: str, training_seed: int, model: Path, train_file: Path,
    snapshot: Path, audit: Path,
) -> dict[str, Any]:
    if training_seed != EXPECTED["training_seed"]:
        raise ValueError("The frozen Qwen3-4B baseline permits only training seed 42")
    if not experiment_name:
        raise ValueError("Experiment name cannot be empty")
    model = model.expanduser().resolve()
    train_file = train_file.expanduser().resolve()
    snapshot = snapshot.expanduser().resolve()
    audit = audit.expanduser().resolve()
    for label, path in (("training data", train_file), ("snapshot", snapshot)):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Missing or empty {label}: {path}")
    actual_data_hash = _sha256(train_file)
    actual_snapshot_hash = _sha256(snapshot)
    if actual_data_hash != EXPECTED["training_data_sha256"]:
        raise ValueError("Pinned training data changed")
    if actual_snapshot_hash != EXPECTED["snapshot_sha256"]:
        raise ValueError("Pinned benchmark snapshot changed")
    if peft.__version__ != EXPECTED["peft_version"]:
        raise ValueError(f"Expected PEFT {EXPECTED['peft_version']}, got {peft.__version__}")
    return {
        "schema_version": 1,
        "method": "qwen3_4b_standard_lora_r32_training_contract_v1",
        "status": "expected",
        "experiment_name": experiment_name,
        "training_seed": training_seed,
        "standard_lora_init_seed": EXPECTED["init_seed"],
        "execution_authorization": {
            "kind": "explicit_owner_request",
            "date": "2026-10-06",
            "request": "run one Qwen3-4B LoRA baseline",
        },
        "claim_boundary": (
            "One standard-LoRA baseline on Qwen3-4B in the existing math-RL protocol. "
            "It is not evidence for I8 transfer or multi-seed robustness."
        ),
        "model_path": str(model),
        "model_assets_sha256": _asset_manifest(model),
        "training_data_path": str(train_file),
        "training_data_sha256": actual_data_hash,
        "benchmark_snapshot_path": str(snapshot),
        "snapshot_sha256": actual_snapshot_hash,
        "initial_state_audit_path": str(audit),
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
            "target_module_suffixes": [
                "down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"
            ],
            "expected_target_module_count": EXPECTED["module_count"],
            "expected_trainable_parameters": EXPECTED["trainable_parameters"],
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
            "max_prompt_length": 1024,
            "max_response_length": 8192,
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
        "qwen_baseline_code_provenance": _local_code_provenance(),
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


def _verify_static(contract: dict[str, Any]) -> None:
    expected = _expected_contract(
        experiment_name=contract["experiment_name"],
        training_seed=int(contract["training_seed"]),
        model=Path(contract["model_path"]),
        train_file=Path(contract["training_data_path"]),
        snapshot=Path(contract["benchmark_snapshot_path"]),
        audit=Path(contract["initial_state_audit_path"]),
    )
    for key, value in expected.items():
        if key == "status":
            continue
        if contract.get(key) != value:
            raise ValueError(f"Qwen3-4B standard-LoRA contract changed: {key}")


def finalize(args) -> dict[str, Any]:
    path = args.contract.expanduser().resolve()
    contract = json.loads(path.read_text(encoding="utf-8"))
    _verify_static(contract)
    if contract.get("status") != "prepared":
        raise ValueError("Qwen3-4B standard-LoRA contract is not awaiting finalization")
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
    }
    for key, value in required_audit.items():
        if audit.get(key) != value:
            raise ValueError(f"Qwen3-4B initialization audit failed: {key}")
    adapter = args.adapter.expanduser().resolve()
    config = adapter.parent / "adapter_config.json"
    for label, artifact in (("adapter", adapter), ("adapter config", config)):
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"Missing or empty {label}: {artifact}")
    adapter_config = json.loads(config.read_text(encoding="utf-8"))
    if (
        adapter_config.get("r") != 32
        or adapter_config.get("lora_alpha") != 64
        or float(adapter_config.get("lora_dropout", -1)) != 0.0
        or len(adapter_config.get("target_modules", [])) != 7
    ):
        raise ValueError("Saved Qwen3-4B adapter configuration changed")
    contract.update(
        {
            "status": "complete",
            "initial_state": {
                "audit_sha256": _sha256(audit_path),
                "canonical_sha256": audit["canonical_sha256"],
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
        raise ValueError("Qwen3-4B standard-LoRA contract is incomplete")
    adapter = args.adapter.expanduser().resolve()
    if contract["adapter"]["path"] != str(adapter) or contract["adapter"]["sha256"] != _sha256(adapter):
        raise ValueError("Qwen3-4B adapter changed after finalization")
    audit_path = Path(contract["initial_state_audit_path"])
    if contract["initial_state"]["audit_sha256"] != _sha256(audit_path):
        raise ValueError("Qwen3-4B initialization audit changed")
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
        args.summary,
        args.records,
        args.eval_dir,
        args.eval_name,
        args.adapter.parent,
        Path(contract["model_path"]),
        Path(contract["benchmark_snapshot_path"]),
        contract["snapshot_sha256"],
    )
    result = {
        "schema_version": 1,
        "status": "verified",
        "method": "qwen3_4b_standard_lora_postverification_v1",
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
