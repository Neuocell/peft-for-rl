#!/usr/bin/env python3
"""Build a same-rank LoRA adapter along a functional checkpoint chord.

The requested functional update is

    DeltaW(lambda) = (1 - lambda) * DeltaW_from + lambda * DeltaW_to.

Raw LoRA factors are not interpolated. Each module is instead projected into
an explicit orthonormal anchor A0 and saved as B(lambda) @ A0. This preserves
the source rank while making extrapolation beyond either endpoint possible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-adapter", type=Path, required=True)
    parser.add_argument("--to-adapter", type=Path, required=True)
    parser.add_argument("--anchor-subspaces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lambda-value", type=float, required=True)
    return parser.parse_args()


def adapter_module_key(a_key: str) -> str:
    return a_key.removesuffix(".lora_A.weight").removeprefix("base_model.model.")


def factored_squared_norm(left: torch.Tensor, right: torch.Tensor) -> float:
    value = torch.trace((left.mT @ left) @ (right @ right.mT))
    return max(float(value.item()), 0.0)


def project_chord_to_anchor(
    from_a: torch.Tensor,
    from_b: torch.Tensor,
    to_a: torch.Tensor,
    to_b: torch.Tensor,
    anchor_a: torch.Tensor,
    lambda_value: float,
) -> tuple[torch.Tensor, float, float]:
    work_dtype = torch.float64
    from_a = from_a.to(dtype=work_dtype)
    from_b = from_b.to(dtype=work_dtype)
    to_a = to_a.to(dtype=work_dtype)
    to_b = to_b.to(dtype=work_dtype)
    anchor_a = anchor_a.to(dtype=work_dtype)
    if from_a.shape != to_a.shape or from_b.shape != to_b.shape:
        raise ValueError(
            f"Endpoint factor shapes differ: {tuple(from_a.shape), tuple(from_b.shape)} vs "
            f"{tuple(to_a.shape), tuple(to_b.shape)}"
        )
    if anchor_a.shape != from_a.shape:
        raise ValueError(
            f"Anchor shape {tuple(anchor_a.shape)} does not match A shape {tuple(from_a.shape)}"
        )

    gram = anchor_a @ anchor_a.mT
    identity = torch.eye(anchor_a.shape[0], dtype=work_dtype)
    orthogonality_error = torch.linalg.matrix_norm(gram - identity, ord=2).item()
    if orthogonality_error > 1e-3:
        raise ValueError(f"Anchor rows are not orthonormal: spectral error={orthogonality_error:.3e}")

    from_projected_b = from_b @ (from_a @ anchor_a.mT)
    to_projected_b = to_b @ (to_a @ anchor_a.mT)
    chord_b = (1.0 - lambda_value) * from_projected_b + lambda_value * to_projected_b

    desired_left = torch.cat((from_b, to_b), dim=1)
    desired_right = torch.cat(
        ((1.0 - lambda_value) * from_a, lambda_value * to_a), dim=0
    )
    residual_left = torch.cat((from_b, to_b, chord_b), dim=1)
    residual_right = torch.cat(
        ((1.0 - lambda_value) * from_a, lambda_value * to_a, -anchor_a), dim=0
    )
    return (
        chord_b,
        factored_squared_norm(residual_left, residual_right),
        factored_squared_norm(desired_left, desired_right),
    )


def load_config(adapter: Path) -> dict[str, Any]:
    return json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_chord_adapter(
    from_adapter: Path,
    to_adapter: Path,
    anchor_subspaces: Path,
    output_dir: Path,
    lambda_value: float,
) -> dict[str, Any]:
    from_adapter = from_adapter.resolve()
    to_adapter = to_adapter.resolve()
    anchor_subspaces = anchor_subspaces.resolve()
    output_dir = output_dir.resolve()
    from_config = load_config(from_adapter)
    to_config = load_config(to_adapter)
    compatibility_fields = (
        "r",
        "lora_alpha",
        "rank_pattern",
        "alpha_pattern",
        "use_rslora",
        "target_modules",
    )
    for field in compatibility_fields:
        if from_config.get(field) != to_config.get(field):
            raise ValueError(f"Endpoint adapter configs differ in {field}")

    from_path = from_adapter / "adapter_model.safetensors"
    to_path = to_adapter / "adapter_model.safetensors"
    output_tensors: dict[str, torch.Tensor] = {}
    per_module: list[dict[str, Any]] = []
    global_residual_sq = 0.0
    global_desired_sq = 0.0
    with (
        safe_open(from_path, framework="pt", device="cpu") as from_handle,
        safe_open(to_path, framework="pt", device="cpu") as to_handle,
        safe_open(anchor_subspaces, framework="pt", device="cpu") as anchor_handle,
    ):
        from_keys = set(from_handle.keys())
        to_keys = set(to_handle.keys())
        if from_keys != to_keys:
            raise ValueError("Endpoint adapter tensor keys differ")
        a_keys = sorted(key for key in from_keys if key.endswith(".lora_A.weight"))
        if not a_keys:
            raise ValueError("No LoRA A tensors found")
        expected_keys = set(a_keys) | {key.replace(".lora_A.weight", ".lora_B.weight") for key in a_keys}
        if from_keys != expected_keys:
            extras = sorted(from_keys - expected_keys)
            raise ValueError(f"Unsupported non-LoRA tensors in adapter: {extras[:5]}")

        anchor_keys = set(anchor_handle.keys())
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            module_key = adapter_module_key(a_key)
            if module_key not in anchor_keys:
                raise KeyError(f"Missing anchor subspace for {module_key}")
            source_a = from_handle.get_tensor(a_key)
            source_b = from_handle.get_tensor(b_key)
            target_a = to_handle.get_tensor(a_key)
            target_b = to_handle.get_tensor(b_key)
            anchor_a = anchor_handle.get_tensor(module_key)
            chord_b, residual_sq, desired_sq = project_chord_to_anchor(
                source_a,
                source_b,
                target_a,
                target_b,
                anchor_a,
                lambda_value,
            )
            output_tensors[a_key] = anchor_a.to(dtype=source_a.dtype).contiguous()
            output_tensors[b_key] = chord_b.to(dtype=source_b.dtype).contiguous()
            global_residual_sq += residual_sq
            global_desired_sq += desired_sq
            per_module.append(
                {
                    "module": module_key,
                    "rank": int(anchor_a.shape[0]),
                    "relative_projection_residual": (
                        math.sqrt(residual_sq / desired_sq) if desired_sq > 0 else 0.0
                    ),
                }
            )

        metadata = from_handle.metadata()

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "adapter_model.safetensors"
    temporary_tensor_path = tensor_path.with_suffix(".safetensors.tmp")
    save_file(output_tensors, str(temporary_tensor_path), metadata=metadata)
    os.replace(temporary_tensor_path, tensor_path)
    shutil.copy2(from_adapter / "adapter_config.json", output_dir / "adapter_config.json")
    summary = {
        "method": "functional_lora_checkpoint_chord_projected_to_probe_anchor",
        "from_adapter": str(from_adapter),
        "to_adapter": str(to_adapter),
        "anchor_subspaces": str(anchor_subspaces),
        "lambda": lambda_value,
        "module_count": len(per_module),
        "global_relative_projection_residual": (
            math.sqrt(global_residual_sq / global_desired_sq) if global_desired_sq > 0 else 0.0
        ),
        "max_module_relative_projection_residual": max(
            row["relative_projection_residual"] for row in per_module
        ),
        "per_module": per_module,
    }
    write_json_atomic(output_dir / "chord_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_chord_adapter(
        args.from_adapter,
        args.to_adapter,
        args.anchor_subspaces,
        args.output_dir,
        args.lambda_value,
    )
    print(
        "Built functional LoRA chord adapter: "
        f"lambda={summary['lambda']}, modules={summary['module_count']}, "
        f"global_residual={summary['global_relative_projection_residual']:.6%}, "
        f"max_module_residual={summary['max_module_relative_projection_residual']:.6%}"
    )


if __name__ == "__main__":
    main()
