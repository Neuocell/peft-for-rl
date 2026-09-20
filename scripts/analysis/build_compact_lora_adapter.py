#!/usr/bin/env python3
"""Build a physically compact LoRA adapter from an offline rank map.

Each source update is factorized with reduced QR plus an SVD of the small
rank-by-rank core.  The output tensors have the selected physical rank; no
mask, gate, or frozen factor is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

LORA_KEY_RE = re.compile(r"^(.*)\.lora_([AB])\.weight$")
PEFT_MODULE_RE = re.compile(r"(model\.layers\.\d+\.(?:self_attn|mlp)\.[^.]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--diagnostic-summary", type=Path, required=True)
    parser.add_argument("--candidate", default="energy_90")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_module(name: str) -> str:
    match = PEFT_MODULE_RE.search(name)
    if match is None:
        raise ValueError(f"Unsupported LoRA module name: {name}")
    return match.group(1)


def load_rank_map(path: Path, candidate: str) -> tuple[dict[str, int], dict[str, Any]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    try:
        raw = summary["candidates"][candidate]["rank_pattern"]
    except KeyError as exc:
        raise ValueError(f"Candidate {candidate!r} is absent from {path}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Candidate {candidate!r} has no rank pattern")
    rank_map: dict[str, int] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid rank for {name}: {value!r}")
        key = normalized_module(name)
        if key in rank_map:
            raise ValueError(f"Duplicate normalized module: {key}")
        rank_map[key] = value
    return rank_map, summary


def main() -> None:
    args = parse_args()
    source = args.source_adapter.resolve()
    output = args.output_dir.resolve()
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete source adapter: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists (pass --overwrite to replace): {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    rank_map, diagnostic = load_rank_map(args.diagnostic_summary.resolve(), args.candidate)
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    source_weights = load_file(str(weights_path), device="cpu")
    factors: dict[str, dict[str, torch.Tensor]] = {}
    tensor_keys: dict[tuple[str, str], str] = {}
    passthrough: dict[str, torch.Tensor] = {}
    for key, tensor in source_weights.items():
        match = LORA_KEY_RE.match(key)
        if match is None:
            passthrough[key] = tensor
            continue
        module_name, factor = match.groups()
        module_key = normalized_module(module_name)
        factors.setdefault(module_key, {})[factor] = tensor
        tensor_keys[(module_key, factor)] = key

    if set(factors) != set(rank_map):
        missing = sorted(set(factors) - set(rank_map))
        extra = sorted(set(rank_map) - set(factors))
        raise ValueError(f"Rank-map coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")

    compact = dict(passthrough)
    module_report = {}
    maximum_reconstruction_error = 0.0
    for module_name in sorted(factors):
        if set(factors[module_name]) != {"A", "B"}:
            raise ValueError(f"Expected A/B tensors for {module_name}, got {sorted(factors[module_name])}")
        source_a = factors[module_name]["A"]
        source_b = factors[module_name]["B"]
        a = source_a.double()
        b = source_b.double()
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"Rank mismatch for {module_name}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        selected_rank = rank_map[module_name]
        source_rank = int(a.shape[0])
        if selected_rank > source_rank:
            raise ValueError(f"Selected rank {selected_rank} exceeds source rank {source_rank} for {module_name}")

        q_b, r_b = torch.linalg.qr(b, mode="reduced")
        q_a, r_a = torch.linalg.qr(a.mT, mode="reduced")
        u, singular, vh = torch.linalg.svd(r_b @ r_a.mT, full_matrices=False)
        compact_b64 = (q_b @ u[:, :selected_rank]) * singular[:selected_rank].unsqueeze(0)
        compact_a64 = vh[:selected_rank] @ q_a.mT
        target64 = compact_b64 @ compact_a64
        denominator = torch.linalg.norm(target64).clamp_min(torch.finfo(torch.float64).tiny)
        saved_a = compact_a64.to(source_a.dtype).contiguous()
        saved_b = compact_b64.to(source_b.dtype).contiguous()
        reconstruction_error = float(
            (torch.linalg.norm(saved_b.double() @ saved_a.double() - target64) / denominator).item()
        )
        maximum_reconstruction_error = max(maximum_reconstruction_error, reconstruction_error)
        compact[tensor_keys[(module_name, "A")]] = saved_a
        compact[tensor_keys[(module_name, "B")]] = saved_b
        total_energy = singular.square().sum().clamp_min(torch.finfo(torch.float64).tiny)
        module_report[module_name] = {
            "source_rank": source_rank,
            "selected_rank": selected_rank,
            "retained_frobenius_energy": float(singular[:selected_rank].square().sum().div(total_energy).item()),
            "saved_factor_reconstruction_relative_error": reconstruction_error,
        }

    scaling = float(source_config["lora_alpha"]) / int(source_config["r"])
    if not math.isclose(scaling, 2.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Expected source alpha/r=2, got {scaling:g}")
    alpha_pattern = {key: int(round(rank * scaling)) for key, rank in rank_map.items()}
    max_rank = max(rank_map.values())
    source_config.update(
        {
            "r": max_rank,
            "lora_alpha": int(round(max_rank * scaling)),
            "rank_pattern": rank_map,
            "alpha_pattern": alpha_pattern,
            "inference_mode": False,
        }
    )
    save_file(compact, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
    (output / "adapter_config.json").write_text(
        json.dumps(source_config, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )
    rank_config = {
        "rank_pattern": rank_map,
        "alpha_pattern": alpha_pattern,
        "constant_scaling": scaling,
        "source_adapter": str(source),
        "diagnostic_summary": str(args.diagnostic_summary.resolve()),
        "candidate": args.candidate,
    }
    (output / "rank_pattern.json").write_text(json.dumps(rank_config, indent=2) + "\n", encoding="utf-8")
    ranks = list(rank_map.values())
    report = {
        "source_adapter": str(source),
        "output_adapter": str(output),
        "candidate": args.candidate,
        "module_count": len(rank_map),
        "rank_mean": sum(ranks) / len(ranks),
        "rank_min": min(ranks),
        "rank_max": max_rank,
        "rank_histogram": dict(sorted(Counter(ranks).items())),
        "constant_scaling": scaling,
        "source_diagnostic_map": diagnostic["candidates"][args.candidate].get("map"),
        "max_saved_factor_reconstruction_relative_error": maximum_reconstruction_error,
        "modules": module_report,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Built compact LoRA adapter: "
        f"modules={len(rank_map)}, rank_mean={report['rank_mean']:.4f}, "
        f"rank_range=[{min(ranks)}, {max_rank}], alpha/r={scaling:g}, "
        f"max_reconstruction_error={maximum_reconstruction_error:.3e}, output={output}"
    )


if __name__ == "__main__":
    main()
