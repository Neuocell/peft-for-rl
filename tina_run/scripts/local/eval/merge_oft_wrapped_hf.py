#!/usr/bin/env python
"""Merge a PEFT OFT-wrapped HF checkpoint into a plain HF CausalLM model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import OFTConfig, TaskType, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from verl.utils.peft_oft_compat import patch_peft_oft_no_inplace_skew
except Exception:
    patch_peft_oft_no_inplace_skew = None

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--wrapped_model", required=True, help="HF dir saved from verl.model_merger with OFT PEFT keys")
    parser.add_argument("--output_dir", required=True, help="Plain HF model dir for vLLM")
    parser.add_argument("--oft_rank", type=int, default=0)
    parser.add_argument("--oft_block_size", type=int, default=32)
    parser.add_argument("--oft_dropout", type=float, default=0.0)
    parser.add_argument("--oft_coft", action="store_true")
    parser.add_argument("--oft_eps", type=float, default=6e-5)
    parser.add_argument("--oft_block_share", action="store_true")
    parser.add_argument("--oft_use_cayley_neumann", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oft_num_cayley_neumann_terms", type=int, default=5)
    parser.add_argument("--target_modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    wrapped_dir = Path(args.wrapped_model)
    output_dir = Path(args.output_dir)
    model_path = wrapped_dir / "model.safetensors"
    if not model_path.exists():
        raise SystemExit(f"Missing wrapped OFT safetensors: {model_path}")

    dtype = dtype_from_name(args.dtype)
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]

    config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=dtype, device_map=None)
    if patch_peft_oft_no_inplace_skew is not None:
        patch_peft_oft_no_inplace_skew()
    peft_config = OFTConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.oft_rank,
        oft_block_size=args.oft_block_size,
        module_dropout=args.oft_dropout,
        coft=args.oft_coft,
        eps=args.oft_eps,
        block_share=args.oft_block_share,
        use_cayley_neumann=args.oft_use_cayley_neumann,
        num_cayley_neumann_terms=args.oft_num_cayley_neumann_terms,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    state_dict = load_file(str(model_path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"OFT state dict mismatch: missing={missing}, unexpected={unexpected}")

    merged_model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(wrapped_dir)
    tokenizer.save_pretrained(output_dir)
    try:
        generation_config = GenerationConfig.from_pretrained(wrapped_dir)
    except OSError:
        generation_config = None
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
