#!/usr/bin/env python3
"""Measure paper-style off-principal overlap for a merged LoRA checkpoint.

The diagnostic follows the coordinate-mask definition from
"The Path Not Taken: RLVR Provably Learns Off the Principals"
(arXiv:2511.08567):

* reconstruct a base weight with a rank-k SVD;
* mark the largest-magnitude coordinates of that reconstruction as principal;
* materialize W0 + DeltaW in bf16 and mark coordinates whose stored value changes;
* compare the changed-coordinate mask with principal and low-magnitude masks.

This is deliberately separate from principal-angle analysis.  A DeltaW can be
contained in a singular-vector span while still avoiding the elementwise
"principal weight" mask used by the paper, or vice versa.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--principal-ranks", default="8,16,32")
    parser.add_argument("--mask-densities", default="0.1,0.3,0.5")
    parser.add_argument("--module-filter", default="")
    parser.add_argument("--max-modules", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def base_key_from_a_key(key: str) -> str:
    key = key.removeprefix("base_model.model.")
    return key.removesuffix(".lora_A.weight") + ".weight"


def module_type(key: str) -> str:
    return key.removesuffix(".lora_A.weight").split(".")[-1]


def layer_index(key: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def layer_band(layer: int | None, layer_count: int) -> str:
    if layer is None:
        return "other"
    boundary = max(1, math.ceil(layer_count / 3))
    if layer < boundary:
        return "early"
    if layer < 2 * boundary:
        return "middle"
    return "late"


def rank_pattern_value(config: dict[str, Any], field: str, a_key: str, default: float) -> float:
    pattern = config.get(field) or {}
    module = a_key.removesuffix(".lora_A.weight").removeprefix("base_model.model.")
    if module in pattern:
        return float(pattern[module])
    matches = [float(value) for key, value in pattern.items() if module.endswith(key)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {field} pattern for {module}")
    return matches[0] if matches else float(default)


def density_mask(values: torch.Tensor, density: float, largest: bool) -> torch.Tensor:
    """Return an approximately exact-density mask without sorting the full tensor."""

    flat = values.reshape(-1)
    count = min(flat.numel(), max(1, int(round(density * flat.numel()))))
    kth = flat.numel() - count + 1 if largest else count
    threshold = torch.kthvalue(flat, kth).values
    return values >= threshold if largest else values <= threshold


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(values) if values else float("nan")


def aggregate(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    output = []
    for group_key, group in sorted(groups.items()):
        total = sum(int(row["numel"]) for row in group)
        changed = sum(int(row["changed_count"]) for row in group)
        principal = sum(int(row["principal_count"]) for row in group)
        low = sum(int(row["low_count"]) for row in group)
        principal_nonlow = sum(int(row["principal_nonlow_count"]) for row in group)
        delta_energy = sum(float(row["delta_energy"]) for row in group)

        item = {field: value for field, value in zip(group_fields, group_key, strict=True)}
        item.update(
            {
                "modules": len(group),
                "coordinates": total,
                "changed_density": changed / total,
                "principal_density": principal / total,
                "low_density": low / total,
                "principal_nonlow_density": principal_nonlow / total,
                "changed_principal_overlap": sum(int(row["changed_principal_count"]) for row in group)
                / max(changed, 1),
                "changed_low_overlap": sum(int(row["changed_low_count"]) for row in group) / max(changed, 1),
                "changed_principal_nonlow_overlap": sum(
                    int(row["changed_principal_nonlow_count"]) for row in group
                )
                / max(changed, 1),
                "delta_energy_in_principal": sum(float(row["delta_energy_principal"]) for row in group)
                / max(delta_energy, 1e-300),
                "delta_energy_in_low": sum(float(row["delta_energy_low"]) for row in group)
                / max(delta_energy, 1e-300),
                "delta_energy_in_principal_nonlow": sum(
                    float(row["delta_energy_principal_nonlow"]) for row in group
                )
                / max(delta_energy, 1e-300),
                "changed_principal_overlap_module_mean": finite_mean(
                    [float(row["changed_principal_overlap"]) for row in group]
                ),
                "changed_low_overlap_module_mean": finite_mean(
                    [float(row["changed_low_overlap"]) for row in group]
                ),
            }
        )
        for prefix in ("changed", "delta_energy"):
            for mask in ("principal", "low", "principal_nonlow"):
                overlap = float(item[f"{prefix}_{mask}_overlap"] if prefix == "changed" else item[f"{prefix}_in_{mask}"])
                density = float(item[f"{mask}_density"])
                item[f"{prefix}_{mask}_enrichment"] = overlap / density if density > 0 else float("nan")
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, result: dict[str, Any]) -> None:
    overall = result["overall"]
    rows_30 = [row for row in overall if math.isclose(float(row["mask_density_target"]), 0.3)]
    rows_kmax = [row for row in overall if int(row["principal_rank"]) == max(result["principal_ranks"])]
    lines = [
        "# Principal-init LoRA: Paper-aligned Off-principal Diagnostic",
        "",
        "## Scope",
        "",
        f"- Adapter: `{result['adapter']}`",
        f"- Modules: `{result['modules']}`",
        f"- Principal reconstruction ranks: `{result['principal_ranks']}`",
        f"- Mask density targets: `{result['mask_densities']}`",
        "- Update mask: compare `bf16(W0 + DeltaW)` with stored `bf16(W0)` coordinate by coordinate.",
        "- Principal mask: largest-magnitude coordinates of the rank-k SVD reconstruction of `W0`.",
        "- Low-magnitude mask: smallest-magnitude coordinates of `W0`.",
        "",
        "This is the coordinate-mask definition used by *The Path Not Taken: RLVR Provably Learns Off the "
        "Principals* (arXiv:2511.08567), not a principal-angle metric.",
        "",
        "## Rank Sweep At 30% Mask Density",
        "",
        "| Base reconstruction rank | bf16 changed density | Changed in principal | Enrichment vs random | "
        "Changed in low-mag | Low-mag enrichment | Delta energy in principal |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows_30:
        lines.append(
            f"| {row['principal_rank']} | {row['changed_density']:.3%} | "
            f"{row['changed_principal_overlap']:.3%} | {row['changed_principal_enrichment']:.3f}x | "
            f"{row['changed_low_overlap']:.3%} | {row['changed_low_enrichment']:.3f}x | "
            f"{row['delta_energy_in_principal']:.3%} |"
        )
    lines.extend(
        [
            "",
            f"## Density Sweep At Rank {max(result['principal_ranks'])}",
            "",
            "| Target density | Actual principal density | Changed in principal | Enrichment | "
            "Changed in principal/non-low | Its random density | Delta energy in principal |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows_kmax:
        lines.append(
            f"| {row['mask_density_target']:.0%} | {row['principal_density']:.3%} | "
            f"{row['changed_principal_overlap']:.3%} | {row['changed_principal_enrichment']:.3f}x | "
            f"{row['changed_principal_nonlow_overlap']:.3%} | {row['principal_nonlow_density']:.3%} | "
            f"{row['delta_energy_in_principal']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Reading The Numbers",
            "",
            "`Enrichment = overlap / mask density`. Values below 1 mean the realized bf16 update mask "
            "avoids that mask relative to random coordinates; values above 1 mean concentration. The continuous "
            "DeltaW-energy columns are reported separately because bf16 thresholding can hide small updates.",
            "",
            "A principal singular-vector span and the paper's principal-weight coordinate mask are different "
            "objects. Span containment therefore cannot by itself establish agreement or conflict with the "
            "paper's off-principal claim.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ranks = sorted({int(value) for value in args.principal_ranks.split(",") if value.strip()})
    densities = sorted({float(value) for value in args.mask_densities.split(",") if value.strip()})
    if not ranks or min(ranks) <= 0:
        raise ValueError("principal ranks must be positive")
    if not densities or min(densities) <= 0 or max(densities) >= 1:
        raise ValueError("mask densities must lie strictly between 0 and 1")

    config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_path = args.adapter / "adapter_model.safetensors"
    base_files = sorted(args.base_model.glob("*.safetensors"))
    if len(base_files) != 1:
        raise ValueError("This diagnostic currently expects one base-model safetensors file")
    filters = [value.strip() for value in args.module_filter.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []

    with safe_open(adapter_path, framework="pt", device="cpu") as adapter, safe_open(
        base_files[0], framework="pt", device="cpu"
    ) as base:
        a_keys = sorted(key for key in adapter.keys() if key.endswith(".lora_A.weight"))
        if filters:
            a_keys = [key for key in a_keys if any(value in key for value in filters)]
        if args.max_modules > 0:
            a_keys = a_keys[: args.max_modules]
        layers = [layer_index(key) for key in a_keys]
        layer_count = max((layer for layer in layers if layer is not None), default=-1) + 1

        for index, a_key in enumerate(a_keys, 1):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            base_key = base_key_from_a_key(a_key)
            w0_stored = base.get_tensor(base_key)
            w0 = w0_stored.to(device=device, dtype=torch.float32)
            a = adapter.get_tensor(a_key).to(device=device, dtype=torch.float32)
            b = adapter.get_tensor(b_key).to(device=device, dtype=torch.float32)
            rank = a.shape[0]
            alpha = rank_pattern_value(config, "alpha_pattern", a_key, config["lora_alpha"])
            scale = alpha / rank
            delta = (b @ a).mul_(scale)
            delta_sq = delta.square()
            delta_energy = float(delta_sq.sum().item())
            changed = w0.to(torch.bfloat16) != (w0 + delta).to(torch.bfloat16)
            changed_count = int(changed.sum().item())
            abs_w0 = w0.abs()
            low_masks = {density: density_mask(abs_w0, density, largest=False) for density in densities}

            u, singular, vh = torch.linalg.svd(w0, full_matrices=False)
            for principal_rank in ranks:
                if principal_rank > singular.numel():
                    raise ValueError(f"rank {principal_rank} exceeds shape of {base_key}")
                reconstruction = (u[:, :principal_rank] * singular[:principal_rank]) @ vh[:principal_rank]
                score = reconstruction.abs()
                for density in densities:
                    principal = density_mask(score, density, largest=True)
                    low = low_masks[density]
                    principal_nonlow = principal & ~low
                    principal_count = int(principal.sum().item())
                    low_count = int(low.sum().item())
                    principal_nonlow_count = int(principal_nonlow.sum().item())
                    changed_principal = int((changed & principal).sum().item())
                    changed_low = int((changed & low).sum().item())
                    changed_principal_nonlow = int((changed & principal_nonlow).sum().item())
                    rows.append(
                        {
                            "adapter_key": a_key.removesuffix(".lora_A.weight"),
                            "base_key": base_key,
                            "layer": layer_index(a_key),
                            "layer_band": layer_band(layer_index(a_key), layer_count),
                            "module_type": module_type(a_key),
                            "principal_rank": principal_rank,
                            "mask_density_target": density,
                            "numel": w0.numel(),
                            "changed_count": changed_count,
                            "changed_density": changed_count / w0.numel(),
                            "principal_count": principal_count,
                            "principal_density": principal_count / w0.numel(),
                            "low_count": low_count,
                            "low_density": low_count / w0.numel(),
                            "principal_nonlow_count": principal_nonlow_count,
                            "principal_nonlow_density": principal_nonlow_count / w0.numel(),
                            "changed_principal_count": changed_principal,
                            "changed_low_count": changed_low,
                            "changed_principal_nonlow_count": changed_principal_nonlow,
                            "changed_principal_overlap": changed_principal / changed_count
                            if changed_count
                            else float("nan"),
                            "changed_low_overlap": changed_low / changed_count if changed_count else float("nan"),
                            "changed_principal_nonlow_overlap": changed_principal_nonlow / changed_count
                            if changed_count
                            else float("nan"),
                            "delta_energy": delta_energy,
                            "delta_energy_principal": float(delta_sq[principal].sum().item()),
                            "delta_energy_low": float(delta_sq[low].sum().item()),
                            "delta_energy_principal_nonlow": float(delta_sq[principal_nonlow].sum().item()),
                        }
                    )
                del reconstruction, score

            print(
                f"[{index:03d}/{len(a_keys):03d}] {base_key} changed={changed_count / w0.numel():.3%}",
                flush=True,
            )
            del w0_stored, w0, a, b, delta, delta_sq, changed, abs_w0, low_masks, u, singular, vh
            if device.type == "cuda":
                torch.cuda.empty_cache()

    overall = aggregate(rows, ("principal_rank", "mask_density_target"))
    by_module = aggregate(rows, ("principal_rank", "mask_density_target", "module_type"))
    by_band = aggregate(rows, ("principal_rank", "mask_density_target", "layer_band"))
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "paper_metric_source": "https://arxiv.org/abs/2511.08567v1",
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "modules": len({row["adapter_key"] for row in rows}),
        "principal_ranks": ranks,
        "mask_densities": densities,
        "definitions": {
            "delta_w": "(alpha/r) * B @ A",
            "changed_mask": "bf16(W0 + DeltaW) != bf16(W0)",
            "principal_mask": "largest-magnitude coordinates of rank-k SVD reconstruction of W0",
            "low_mask": "smallest-magnitude coordinates of W0",
            "enrichment": "observed overlap divided by actual mask density; random reference is 1",
        },
        "overall": overall,
        "by_module_type": by_module,
        "by_layer_band": by_band,
    }
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_module.csv", rows)
    write_csv(out_dir / "overall.csv", overall)
    write_csv(out_dir / "by_module_type.csv", by_module)
    write_csv(out_dir / "by_layer_band.csv", by_band)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", result)
    print(f"Wrote off-principal diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
