#!/usr/bin/env python3
"""Build a LoRA endpoint with base singular values restored.

For LoRA-targeted linear weights, this creates

    W_restored = U_rl @ diag(sigma(W_base)) @ Vh_rl

where W_rl = W_base + B @ A * alpha / r. All non-target tensors are copied from
the base model unchanged. This implements the spectrum-restoration intervention
used to test whether RLVR behavior depends on singular-value changes or mainly
on singular-frame changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--metadata-json", default="")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_adapter_config(adapter_dir: Path) -> dict:
    with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def base_key_from_lora_a_key(key: str) -> str:
    prefix = "base_model.model."
    if key.startswith(prefix):
        key = key[len(prefix) :]
    return key.replace(".lora_A.weight", ".weight")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compute_device = torch.device(args.device)
    out_dtype = torch_dtype(args.dtype)
    cfg = load_adapter_config(adapter_dir)
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])

    base_path = base_dir / "model.safetensors"
    adapter_path = adapter_dir / "adapter_model.safetensors"

    with safe_open(adapter_path, framework="pt", device="cpu") as af:
        lora_a_keys = sorted(k for k in af.keys() if k.endswith(".lora_A.weight"))
        target_base_keys = {base_key_from_lora_a_key(k): k for k in lora_a_keys}

    tensors = {}
    records = []
    with safe_open(base_path, framework="pt", device="cpu") as bf, safe_open(
        adapter_path, framework="pt", device="cpu"
    ) as af:
        base_keys = list(bf.keys())
        for idx, key in enumerate(base_keys, 1):
            w0_cpu = bf.get_tensor(key)
            if key not in target_base_keys:
                tensors[key] = w0_cpu.to(dtype=out_dtype).contiguous()
                continue

            a_key = target_base_keys[key]
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            w0 = w0_cpu.to(device=compute_device, dtype=torch.float32)
            a = af.get_tensor(a_key).to(device=compute_device, dtype=torch.float32)
            b = af.get_tensor(b_key).to(device=compute_device, dtype=torch.float32)
            delta = torch.matmul(b, a).mul_(scale)
            w_rl = w0 + delta

            # SVD of W_rl provides the learned frames; singular values are
            # replaced by the base values paired by descending rank index.
            u_rl, _, vh_rl = torch.linalg.svd(w_rl, full_matrices=False)
            sigma0 = torch.linalg.svdvals(w0)
            restored = torch.matmul(u_rl * sigma0.unsqueeze(0), vh_rl)

            sigma_diff_rl = torch.linalg.vector_norm(torch.linalg.svdvals(w_rl) - sigma0).item()
            sigma_diff_restored = torch.linalg.vector_norm(torch.linalg.svdvals(restored) - sigma0).item()
            delta_fro = torch.linalg.matrix_norm(delta, ord="fro").item()
            frame_delta_fro = torch.linalg.matrix_norm(restored - w0, ord="fro").item()
            records.append(
                {
                    "base_key": key,
                    "shape": list(w0.shape),
                    "delta_fro": delta_fro,
                    "frame_delta_fro": frame_delta_fro,
                    "sigma_diff_rl": sigma_diff_rl,
                    "sigma_diff_restored": sigma_diff_restored,
                }
            )
            tensors[key] = restored.detach().cpu().to(dtype=out_dtype).contiguous()
            print(
                f"[{idx:03d}/{len(base_keys):03d}] restored spectrum for {key} "
                f"sigma_diff_rl={sigma_diff_rl:.6e} sigma_diff_restored={sigma_diff_restored:.6e}",
                flush=True,
            )

            del w0, a, b, delta, w_rl, u_rl, vh_rl, sigma0, restored
            if compute_device.type == "cuda":
                torch.cuda.empty_cache()

    save_file(tensors, str(out_dir / "model.safetensors"))

    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
        "LICENSE",
    ]:
        src = base_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    metadata = {
        "base_model": str(base_dir),
        "adapter": str(adapter_dir),
        "intervention": "For LoRA target matrices, keep singular frames of W_base + Delta_LoRA and restore singular values of W_base.",
        "dtype": args.dtype,
        "lora_config": {
            "r": cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "target_modules": cfg.get("target_modules"),
        },
        "num_restored_matrices": len(records),
        "records": records,
    }
    meta_path = Path(args.metadata_json) if args.metadata_json else out_dir / "spectrum_restoration_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote model to {out_dir}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
