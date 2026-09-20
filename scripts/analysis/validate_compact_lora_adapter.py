#!/usr/bin/env python3
"""Measure policy distortion between a source and physically compact adapter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--compact-adapter", type=Path, required=True)
    parser.add_argument("--diagnostic-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--num-sequences", type=int, default=48)
    parser.add_argument("--max-seq-tokens", type=int, default=8192)
    parser.add_argument("--tokens-per-sequence", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_diagnostic_helpers() -> Any:
    path = Path(__file__).with_name("diagnose_rank_pruning_policy_distortion.py")
    spec = importlib.util.spec_from_file_location("rank_pruning_diagnostic", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import diagnostic helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_logits(
    *,
    base_model_path: Path,
    adapter_path: Path,
    encoded: list[Any],
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    helpers: Any,
    label: str,
) -> list[torch.Tensor]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).to(device)
    model.eval()
    modules = helpers.canonicalize_lora_modules(model)
    if len(modules) != 196:
        raise ValueError(f"Expected 196 LoRA modules in {adapter_path}, found {len(modules)}")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters() if parameter.requires_grad):
        raise FloatingPointError(f"Non-finite trainable parameter in {adapter_path}")
    logits = helpers.collect_policy_logits(model.get_base_model(), encoded, device, label)
    del modules, model, base
    torch.cuda.empty_cache()
    return logits


def main() -> None:
    args = parse_args()
    helpers = load_diagnostic_helpers()
    summary = json.loads(args.diagnostic_summary.read_text(encoding="utf-8"))
    records_path = helpers.resolve_records_path(Path(summary["records"]))
    benchmarks = list(helpers.DEFAULT_BENCHMARKS)
    records = helpers.select_records(records_path, benchmarks, args.num_sequences, args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model.resolve())
    encoded = helpers.encode_records(records, tokenizer, args.max_seq_tokens, args.tokens_per_sequence)
    validation = [item for item in encoded if item.record.split == "validation"]
    device = torch.device(args.device)
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    source_logits = load_logits(
        base_model_path=args.base_model.resolve(),
        adapter_path=args.source_adapter.resolve(),
        encoded=validation,
        device=device,
        dtype=dtypes[args.model_dtype],
        attn_implementation=args.attn_implementation,
        helpers=helpers,
        label="source",
    )
    compact_logits = load_logits(
        base_model_path=args.base_model.resolve(),
        adapter_path=args.compact_adapter.resolve(),
        encoded=validation,
        device=device,
        dtype=dtypes[args.model_dtype],
        attn_implementation=args.attn_implementation,
        helpers=helpers,
        label="compact",
    )
    policy, rows = helpers.summarize_policy_comparison(
        validation,
        source_logits,
        compact_logits,
        "physical_compact_adapter",
        "source_step100_adapter",
    )
    compact_config = json.loads((args.compact_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    result = {
        "base_model": str(args.base_model.resolve()),
        "source_adapter": str(args.source_adapter.resolve()),
        "compact_adapter": str(args.compact_adapter.resolve()),
        "validation_sequences": len(validation),
        "validation_tokens": sum(int(item.target_ids.numel()) for item in validation),
        "rank_mean": sum(compact_config["rank_pattern"].values()) / len(compact_config["rank_pattern"]),
        "rank_range": [min(compact_config["rank_pattern"].values()), max(compact_config["rank_pattern"].values())],
        "scaling_values": sorted(
            {compact_config["alpha_pattern"][key] / rank for key, rank in compact_config["rank_pattern"].items()}
        ),
        "policy": policy,
        "sequences": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    global_metrics = policy["global"]
    print(
        "Compact adapter validation passed: "
        f"sequences={len(validation)}, tokens={result['validation_tokens']}, "
        f"KL_mean={global_metrics['forward_kl']['mean']:.6e}, "
        f"KL_p99={global_metrics['forward_kl']['p99']:.6e}, "
        f"target_logprob_MAE={global_metrics['target_logprob_mae']:.6e}, "
        f"top1_flip={global_metrics['top1_flip_ratio']:.6%}"
    )


if __name__ == "__main__":
    main()
