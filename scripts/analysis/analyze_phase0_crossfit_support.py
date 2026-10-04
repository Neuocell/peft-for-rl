#!/usr/bin/env python3
"""Diagnose supported and completion directions in Phase-0 P2/P3 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DEFAULT_PREFIXES = (1, 2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--relative-threshold", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def supported_rank(
    spectrum: torch.Tensor, relative_threshold: float, eps: float
) -> int:
    threshold = max(float(spectrum.max().item()) * relative_threshold, eps)
    return int((spectrum > threshold).sum().item())


def correlation(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        return 0.0
    x = torch.tensor(first, dtype=torch.float64)
    y = torch.tensor(second, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float((x @ y / denominator).item())


def ordinal_ranks(values: list[float]) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(tensor, stable=True)
    ranks = torch.empty(len(values), dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks.tolist()


def correlations(calibration: list[float], audit: list[float]) -> dict[str, float]:
    return {
        "pearson": correlation(calibration, audit),
        "spearman_ordinal": correlation(
            ordinal_ranks(calibration), ordinal_ranks(audit)
        ),
    }


def quantiles(values: list[int]) -> dict[str, float]:
    ordered = torch.tensor(sorted(values), dtype=torch.float64)
    return {
        "min": float(ordered[0].item()),
        "q25": float(torch.quantile(ordered, 0.25).item()),
        "median": float(torch.quantile(ordered, 0.5).item()),
        "q75": float(torch.quantile(ordered, 0.75).item()),
        "max": float(ordered[-1].item()),
        "mean": float(ordered.mean().item()),
    }


def module_coordinates(name: str) -> tuple[int | None, str]:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    layer = int(match.group(1)) if match else None
    return layer, name.rsplit(".", 1)[-1]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def analyze(
    artifact_dir: Path, *, relative_threshold: float, eps: float
) -> dict[str, Any]:
    paths = {
        "p2_candidates": artifact_dir / "candidates_P2.safetensors",
        "p3_candidates": artifact_dir / "candidates_P3.safetensors",
        "p2_scores": artifact_dir / "atom_scores_P2.safetensors",
        "p3_scores": artifact_dir / "atom_scores_P3.safetensors",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase-0 artifacts: {missing}")

    p2_candidates = load_safetensors(paths["p2_candidates"])
    p3_candidates = load_safetensors(paths["p3_candidates"])
    p2_scores = load_safetensors(paths["p2_scores"])
    p3_scores = load_safetensors(paths["p3_scores"])
    modules = sorted(p2_candidates)
    if modules != sorted(p3_candidates):
        raise ValueError("P2 and P3 candidate module sets differ")

    records: dict[str, dict[str, Any]] = {}
    ranks: list[int] = []
    ranks_by_type: defaultdict[str, list[int]] = defaultdict(list)
    ranks_by_layer: defaultdict[str, list[int]] = defaultdict(list)
    p3_all_calibration: list[float] = []
    p3_all_audit: list[float] = []
    p3_supported_calibration: list[float] = []
    p3_supported_audit: list[float] = []
    p2_all_calibration: list[float] = []
    p2_all_audit: list[float] = []
    supported_p3_calibration = 0.0
    supported_p3_audit = 0.0
    completion_p3_calibration = 0.0
    completion_p3_audit = 0.0
    matched_p2_calibration = 0.0
    matched_p2_audit = 0.0
    p3_better_calibration = 0
    p3_better_audit = 0
    p3_better_both = 0

    prefix_totals = {
        prefix: {"calibration": 0.0, "audit": 0.0}
        for prefix in DEFAULT_PREFIXES
    }
    for name in modules:
        spectrum = p3_scores[f"{name}.spectrum"].float().clamp_min(0)
        rank = supported_rank(spectrum, relative_threshold, eps)
        ranks.append(rank)
        layer, module_type = module_coordinates(name)
        ranks_by_type[module_type].append(rank)
        ranks_by_layer[str(layer)].append(rank)

        p2_calibration = p2_scores[f"{name}.F"].double()
        p2_audit = p2_scores[f"{name}.audit_F"].double()
        p3_calibration = p3_scores[f"{name}.F"].double()
        p3_audit = p3_scores[f"{name}.audit_F"].double()
        p2_all_calibration.extend(p2_calibration.tolist())
        p2_all_audit.extend(p2_audit.tolist())
        p3_all_calibration.extend(p3_calibration.tolist())
        p3_all_audit.extend(p3_audit.tolist())
        p3_supported_calibration.extend(p3_calibration[:rank].tolist())
        p3_supported_audit.extend(p3_audit[:rank].tolist())

        p3_cal_supported = float(p3_calibration[:rank].sum().item())
        p3_audit_supported = float(p3_audit[:rank].sum().item())
        p3_cal_completion = float(p3_calibration[rank:].sum().item())
        p3_audit_completion = float(p3_audit[rank:].sum().item())
        p2_cal_matched = float(p2_calibration[:rank].sum().item())
        p2_audit_matched = float(p2_audit[:rank].sum().item())
        supported_p3_calibration += p3_cal_supported
        supported_p3_audit += p3_audit_supported
        completion_p3_calibration += p3_cal_completion
        completion_p3_audit += p3_audit_completion
        matched_p2_calibration += p2_cal_matched
        matched_p2_audit += p2_audit_matched

        calibration_better = rank > 0 and p3_cal_supported > p2_cal_matched
        audit_better = rank > 0 and p3_audit_supported > p2_audit_matched
        p3_better_calibration += int(calibration_better)
        p3_better_audit += int(audit_better)
        p3_better_both += int(calibration_better and audit_better)

        if rank:
            p3_supported_basis = p3_candidates[name][:rank].float()
            p2_basis = p2_candidates[name].float()
            p2_projection_fraction = float(
                (p3_supported_basis @ p2_basis.T).square().sum().item() / rank
            )
        else:
            p2_projection_fraction = 0.0
        spectrum_total = float(spectrum.sum().item())
        prefix_concentration = {}
        for prefix in DEFAULT_PREFIXES:
            count = min(prefix, rank)
            prefix_calibration = float(p3_calibration[:count].sum().item())
            prefix_audit = float(p3_audit[:count].sum().item())
            prefix_totals[prefix]["calibration"] += prefix_calibration
            prefix_totals[prefix]["audit"] += prefix_audit
            prefix_concentration[str(prefix)] = (
                float(spectrum[:count].sum().item()) / spectrum_total
                if spectrum_total > 0.0
                else 0.0
            )

        records[name] = {
            "layer": layer,
            "module_type": module_type,
            "p3_consensus_supported_rank": rank,
            "p3_spectrum_prefix_concentration": prefix_concentration,
            "p3_supported_projection_fraction_in_p2_subspace": p2_projection_fraction,
            "p3_supported_calibration_energy": p3_cal_supported,
            "p3_supported_audit_energy": p3_audit_supported,
            "p3_completion_calibration_energy": p3_cal_completion,
            "p3_completion_audit_energy": p3_audit_completion,
            "p2_matched_prefix_calibration_energy": p2_cal_matched,
            "p2_matched_prefix_audit_energy": p2_audit_matched,
            "p3_to_p2_matched_calibration_ratio": safe_ratio(
                p3_cal_supported, p2_cal_matched
            ),
            "p3_to_p2_matched_audit_ratio": safe_ratio(
                p3_audit_supported, p2_audit_matched
            ),
        }

    total_p3_calibration = supported_p3_calibration + completion_p3_calibration
    total_p3_audit = supported_p3_audit + completion_p3_audit
    return {
        "schema_version": 1,
        "analysis": "phase0_crossfit_support_posthoc_diagnostic",
        "artifact_dir": str(artifact_dir),
        "audit_usage": "diagnostic only; audit is not used to choose a hybrid candidate",
        "support_rule": {
            "relative_threshold": relative_threshold,
            "absolute_epsilon": eps,
            "formula": "spectrum > max(max_spectrum * relative_threshold, absolute_epsilon)",
        },
        "source_sha256": {key: sha256(path) for key, path in paths.items()},
        "module_count": len(modules),
        "p3_consensus_supported_rank": {
            **quantiles(ranks),
            "total": sum(ranks),
            "histogram": {
                str(value): ranks.count(value) for value in sorted(set(ranks))
            },
            "by_module_type": {
                key: quantiles(value) for key, value in sorted(ranks_by_type.items())
            },
            "by_layer": {
                key: quantiles(value)
                for key, value in sorted(
                    ranks_by_layer.items(), key=lambda item: int(item[0])
                )
            },
        },
        "held_out_energy": {
            "p3_supported_fraction": {
                "calibration": safe_ratio(
                    supported_p3_calibration, total_p3_calibration
                ),
                "audit": safe_ratio(supported_p3_audit, total_p3_audit),
            },
            "p3_supported_to_p2_matched_prefix_ratio": {
                "calibration": safe_ratio(
                    supported_p3_calibration, matched_p2_calibration
                ),
                "audit": safe_ratio(supported_p3_audit, matched_p2_audit),
            },
            "modules_with_p3_supported_energy_above_p2_matched_prefix": {
                "calibration": p3_better_calibration,
                "audit": p3_better_audit,
                "both": p3_better_both,
                "eligible": sum(rank > 0 for rank in ranks),
            },
            "supported_prefix_fraction_of_p3_supported_energy": {
                str(prefix): {
                    split: safe_ratio(value, total)
                    for split, value, total in (
                        (
                            "calibration",
                            totals["calibration"],
                            supported_p3_calibration,
                        ),
                        ("audit", totals["audit"], supported_p3_audit),
                    )
                }
                for prefix, totals in prefix_totals.items()
            },
        },
        "calibration_audit_atom_correlation": {
            "P2_all": correlations(p2_all_calibration, p2_all_audit),
            "P3_all": correlations(p3_all_calibration, p3_all_audit),
            "P3_supported": correlations(
                p3_supported_calibration, p3_supported_audit
            ),
        },
        "mean_p3_supported_projection_fraction_in_p2_subspace": statistics.mean(
            item["p3_supported_projection_fraction_in_p2_subspace"]
            for item in records.values()
            if item["p3_consensus_supported_rank"] > 0
        ),
        "modules": records,
    }


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else artifact_dir / "phase0_crossfit_support_analysis.json"
    )
    result = analyze(
        artifact_dir,
        relative_threshold=args.relative_threshold,
        eps=args.eps,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    summary = {
        "output": str(output),
        "supported_rank": result["p3_consensus_supported_rank"],
        "held_out_energy": result["held_out_energy"],
        "calibration_audit_atom_correlation": result[
            "calibration_audit_atom_correlation"
        ],
        "mean_p3_supported_projection_fraction_in_p2_subspace": result[
            "mean_p3_supported_projection_fraction_in_p2_subspace"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
