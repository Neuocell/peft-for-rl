#!/usr/bin/env python
"""Merge a NoRA-DoRA-wrapped HF checkpoint into a plain HF CausalLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.utils.adapter_initialization import INIT_VERSIONS
from verl.utils.fsdp_utils import collect_nora_dora_full_params
from verl.utils.peft_nora_dora import (
    DEFAULT_LINEAR_TARGETS,
    apply_nora_dora_adapters,
    nora_dora_snapshot_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--wrapped_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--normalization", choices=["standard", "init", "continuous"])
    parser.add_argument("--eps", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initialization_version", choices=INIT_VERSIONS)
    parser.add_argument("--target_modules", default=",".join(sorted(DEFAULT_LINEAR_TARGETS)))
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def load_wrapped_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    adapter_file = model_dir / "adapter_model.safetensors"
    if adapter_file.exists():
        return load_file(str(adapter_file), device="cpu")
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return load_file(str(single_file), device="cpu")
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No safetensors checkpoint found under {model_dir}")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(str(model_dir / shard_name), device="cpu"))
    return state_dict


def resolve_adapter_config(wrapped_dir: Path, args: argparse.Namespace) -> dict:
    manifest_path = wrapped_dir / "manifest.json"
    method = json.loads(manifest_path.read_text())["method_config"] if manifest_path.exists() else {}
    defaults = {"rank": 8, "alpha": 8.0, "normalization": "init", "eps": 1e-6, "seed": 42}
    for key in (*defaults, "initialization_version"):
        supplied = getattr(args, key, None)
        if supplied is not None:
            if key in method and method[key] != supplied:
                raise ValueError(f"Snapshot {key}={method[key]!r} conflicts with requested {supplied!r}")
            # Absence of a version in an existing manifest is an explicit legacy
            # semantic; callers may not silently reinterpret it as reference v2.
            if key == "initialization_version" and manifest_path.exists() and key not in method:
                if supplied != "legacy_v1":
                    raise ValueError("Unversioned snapshot requires legacy_v1 normalization")
            method[key] = supplied
        elif key not in method and key in defaults:
            method[key] = defaults[key]
    config = nora_dora_snapshot_config(method)
    config["target_modules"] = args.target_modules
    return config


def main() -> None:
    args = parse_args()
    wrapped_dir = Path(args.wrapped_model)
    adapter_config = resolve_adapter_config(wrapped_dir, args)
    output_dir = Path(args.output_dir)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=config, torch_dtype=dtype, device_map=None
    )
    apply_nora_dora_adapters(
        model,
        adapter_config,
    )
    state_dict = load_wrapped_state_dict(wrapped_dir)
    is_adapter_snapshot = (wrapped_dir / "adapter_model.safetensors").is_file()
    incompatible = model.load_state_dict(state_dict, strict=not is_adapter_snapshot)
    if is_adapter_snapshot:
        expected = {name for name, _ in model.named_parameters() if "nora_dora_" in name}
        if set(state_dict) != expected or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incomplete or incompatible NoRA-DoRA evaluation snapshot: "
                f"expected={len(expected)} actual={len(state_dict)} "
                f"unexpected={incompatible.unexpected_keys}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_dir,
        state_dict=collect_nora_dora_full_params(model),
        safe_serialization=True,
    )
    tokenizer_source = wrapped_dir if (wrapped_dir / "tokenizer_config.json").exists() else args.base_model
    AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(output_dir)
    try:
        generation_config = GenerationConfig.from_pretrained(wrapped_dir)
    except OSError:
        try:
            generation_config = GenerationConfig.from_pretrained(args.base_model)
        except OSError:
            generation_config = None
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
