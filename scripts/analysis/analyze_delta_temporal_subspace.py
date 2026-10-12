#!/usr/bin/env python3
"""Analyze temporal subspaces of globally flattened functional LoRA updates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ATTENTION = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP = {"gate_proj", "up_proj", "down_proj"}
DEFAULT_HORIZONS = (1, 5, 10, 25, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0-snapshots", type=Path, required=True)
    parser.add_argument("--o0-snapshots", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--last-step", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def layer_index(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        raise ValueError(f"Cannot infer layer from {name}")
    return int(match.group(1))


def module_groups(name: str, max_layer: int) -> list[str]:
    kind = name.split(".")[-1]
    family = "attention" if kind in ATTENTION else "mlp" if kind in MLP else "other"
    layer = layer_index(name)
    boundary = max(1, math.ceil((max_layer + 1) / 3))
    depth = "early" if layer < boundary else "middle" if layer < 2 * boundary else "late"
    return ["all", f"family:{family}", f"depth:{depth}"]


def snapshot_paths(root: Path, last_step: int) -> list[Path]:
    paths = [root / f"step_{step:03d}" / "adapter_model.safetensors" for step in range(last_step + 1)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} snapshots; first={missing[0]}")
    return paths


def module_function_gram(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """Return <scale B_t A_t, scale B_s A_s> for every pair of times."""

    b_cross = torch.einsum("tor,sop->tsrp", b, b)
    a_cross = torch.einsum("tri,spi->tsrp", a, a)
    gram = torch.einsum("tsrp,tsrp->ts", b_cross, a_cross).mul(scale * scale)
    return (gram + gram.mT).mul(0.5)


def delta_gram(function_gram: torch.Tensor, horizon: int) -> torch.Tensor:
    """Gram of D(t,h)=DeltaW_(t+h)-DeltaW_t from the DeltaW time Gram."""

    gram = (
        function_gram[horizon:, horizon:]
        + function_gram[:-horizon, :-horizon]
        - function_gram[horizon:, :-horizon]
        - function_gram[:-horizon, horizon:]
    )
    return (gram + gram.mT).mul(0.5)


def positive_eigh(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(gram.double())
    values = values.flip(0)
    vectors = vectors.flip(1)
    if not values.numel():
        return values, vectors
    tolerance = max(float(values[0]), 0.0) * max(gram.shape) * torch.finfo(torch.float64).eps
    values = values.clamp_min(0)
    keep = values > tolerance
    return values[keep], vectors[:, keep]


def rank_at_energy(values: torch.Tensor, target: float) -> int:
    total = values.sum()
    if not values.numel() or float(total) <= 0:
        return 0
    return int(torch.searchsorted(values.cumsum(0) / total, target).item()) + 1


def effective_rank(values: torch.Tensor) -> float:
    total = values.sum()
    if not values.numel() or float(total) <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def adjacent_cosines(gram: torch.Tensor, horizon: int) -> torch.Tensor:
    count = gram.shape[0] - horizon
    if count <= 0:
        return torch.empty(0, dtype=torch.float64)
    first = torch.arange(count)
    second = first + horizon
    denominator = (gram[first, first].clamp_min(0) * gram[second, second].clamp_min(0)).sqrt()
    valid = denominator > 0
    return gram[first[valid], second[valid]] / denominator[valid]


def heldout_metrics(gram: torch.Tensor, horizon: int, cut: int) -> dict[str, Any] | None:
    if horizon > cut or 2 * cut > gram.shape[0] + horizon - 1:
        return None
    gram = gram.double()
    early = torch.arange(0, cut - horizon + 1)
    future = torch.arange(cut, 2 * cut - horizon + 1)
    early_gram = gram[early][:, early]
    future_gram = gram[future][:, future]
    cross = gram[early][:, future]
    early_values, early_vectors = positive_eigh(early_gram)
    future_values, future_vectors = positive_eigh(future_gram)
    rank90 = rank_at_energy(early_values, 0.90)
    rank95 = rank_at_energy(early_values, 0.95)
    future_rank95 = rank_at_energy(future_values, 0.95)
    future_energy = float(future_gram.trace())

    capture = float("nan")
    overlap = float("nan")
    if rank95 and future_energy > 0:
        basis_coefficients = cross.mT @ early_vectors[:, :rank95]
        projected = basis_coefficients.square() / early_values[:rank95]
        capture = float(projected.sum() / future_energy)
    if rank95 and future_rank95:
        normalized_cross = early_vectors[:, :rank95].mT @ cross @ future_vectors[:, :future_rank95]
        normalized_cross = normalized_cross / early_values[:rank95].sqrt().unsqueeze(1)
        normalized_cross = normalized_cross / future_values[:future_rank95].sqrt().unsqueeze(0)
        width = min(rank95, future_rank95)
        cosines = torch.linalg.svdvals(normalized_cross).clamp(0, 1)[:width]
        overlap = float(cosines.square().mean())
    return {
        "cut": cut,
        "early_windows": len(early),
        "future_windows": len(future),
        "early_effective_rank": effective_rank(early_values),
        "early_rank90": rank90,
        "early_rank95": rank95,
        "future_rank95": future_rank95,
        "future_capture_by_early_rank95": capture,
        "early_future_principal_overlap_95": overlap,
    }


def summarize_gram(
    method: str,
    group: str,
    weighting: str,
    horizon: int,
    gram: torch.Tensor,
) -> dict[str, Any]:
    values, _vectors = positive_eigh(gram)
    cosines = adjacent_cosines(gram, horizon)
    nonoverlap_gram = gram[::horizon, ::horizon]
    nonoverlap_values, _nonoverlap_vectors = positive_eigh(nonoverlap_gram)
    row: dict[str, Any] = {
        "method": method,
        "group": group,
        "weighting": weighting,
        "h": horizon,
        "windows": gram.shape[0],
        "total_energy": float(gram.trace()),
        "temporal_effective_rank": effective_rank(values),
        "temporal_rank90": rank_at_energy(values, 0.90),
        "temporal_rank95": rank_at_energy(values, 0.95),
        "adjacent_pairs": cosines.numel(),
        "adjacent_cosine_mean": float(cosines.mean()) if cosines.numel() else float("nan"),
        "adjacent_cosine_median": float(cosines.median()) if cosines.numel() else float("nan"),
        "nonoverlap_windows": nonoverlap_gram.shape[0],
        "nonoverlap_temporal_effective_rank": effective_rank(nonoverlap_values),
        "nonoverlap_temporal_rank90": rank_at_energy(nonoverlap_values, 0.90),
        "nonoverlap_temporal_rank95": rank_at_energy(nonoverlap_values, 0.95),
    }
    for cut in (25, 50):
        heldout = heldout_metrics(gram, horizon, cut)
        for field in (
            "early_windows",
            "future_windows",
            "early_effective_rank",
            "early_rank90",
            "early_rank95",
            "future_rank95",
            "future_capture_by_early_rank95",
            "early_future_principal_overlap_95",
        ):
            row[f"cut{cut}_{field}"] = heldout[field] if heldout else float("nan")
    return row


def analyze_method(
    method: str,
    root: Path,
    last_step: int,
    horizons: list[int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    paths = snapshot_paths(root, last_step)
    handles = [safe_open(path, framework="pt", device="cpu") for path in paths]
    try:
        keys = set(handles[0].keys())
        a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
        if len(a_keys) != 196:
            raise ValueError(f"Expected 196 LoRA modules, found {len(a_keys)} under {root}")
        scale = float(handles[0].metadata()["scaling"])
        max_layer = max(layer_index(key) for key in a_keys)
        raw: dict[tuple[str, int], torch.Tensor] = {}
        balanced: dict[int, torch.Tensor] = {}
        balanced_counts: defaultdict[int, int] = defaultdict(int)

        for module_index, a_key in enumerate(a_keys, start=1):
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            name = a_key.removesuffix(".lora_A.weight")
            a = torch.stack([handle.get_tensor(a_key) for handle in handles]).to(device)
            b = torch.stack([handle.get_tensor(b_key) for handle in handles]).to(device)
            function_gram = module_function_gram(a, b, scale).cpu().double()
            del a, b
            for horizon in horizons:
                gram = delta_gram(function_gram, horizon)
                for group in module_groups(name, max_layer):
                    raw.setdefault((group, horizon), torch.zeros_like(gram)).add_(gram)
                energy = float(gram.trace())
                if energy > 0:
                    balanced.setdefault(horizon, torch.zeros_like(gram)).add_(gram / energy)
                    balanced_counts[horizon] += 1
            del function_gram
            if module_index == 1 or module_index % 10 == 0 or module_index == len(a_keys):
                print(f"[{method}] modules {module_index}/{len(a_keys)}", flush=True)

        rows = []
        matrices: dict[str, torch.Tensor] = {}
        for (group, horizon), gram in sorted(raw.items()):
            rows.append(summarize_gram(method, group, "global_frobenius", horizon, gram))
            matrices[f"{method}|global_frobenius|{group}|h={horizon}"] = gram
        for horizon, gram in sorted(balanced.items()):
            gram = gram / balanced_counts[horizon]
            rows.append(summarize_gram(method, "all", "module_balanced", horizon, gram))
            matrices[f"{method}|module_balanced|all|h={horizon}"] = gram
        return rows, matrices
    finally:
        for handle in handles:
            handle.__exit__(None, None, None)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["group"] == "all" and row["weighting"] == "global_frobenius"
    ]
    metrics = (
        ("temporal_effective_rank", "Temporal effective rank"),
        ("temporal_rank95", "Temporal rank@95%"),
        ("adjacent_cosine_mean", "Adjacent interval cosine"),
        ("cut25_future_capture_by_early_rank95", "0-25 basis capture of 25-50"),
        ("cut50_future_capture_by_early_rank95", "0-50 basis capture of 50-100"),
        ("cut50_early_future_principal_overlap_95", "0-50 vs 50-100 overlap"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, metrics):
        for method, label in (("S0", "S0 standard"), ("O0", "O0 random-orthogonal")):
            method_rows = sorted((row for row in selected if row["method"] == method), key=lambda row: row["h"])
            axis.plot([row["h"] for row in method_rows], [row[field] for row in method_rows], marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel("h (trainer steps)")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.savefig(out_dir / "s0_o0_temporal_subspace.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    if any(horizon <= 0 or horizon > args.last_step for horizon in horizons):
        raise ValueError(f"Invalid horizons: {horizons}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    rows = []
    matrices = {}
    for method, root in (("S0", args.s0_snapshots), ("O0", args.o0_snapshots)):
        method_rows, method_matrices = analyze_method(method, root, args.last_step, horizons, device)
        rows.extend(method_rows)
        matrices.update(method_matrices)
    write_csv(args.out_dir / "temporal_subspace_summary.csv", rows)
    torch.save(matrices, args.out_dir / "temporal_gram_matrices.pt")
    plot_summary(rows, args.out_dir)
    artifact = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "exact_global_functional_delta_temporal_gram_v1",
        "definition": "d(t,h)=concat_l vec(DeltaW_l(t+h)-DeltaW_l(t)); K_ij=<d_i,d_j>",
        "horizons": horizons,
        "last_step": args.last_step,
        "splits": [
            {"cut": 25, "basis": "intervals contained in steps 0-25", "future": "intervals contained in steps 25-50"},
            {"cut": 50, "basis": "intervals contained in steps 0-50", "future": "intervals contained in steps 50-100"},
        ],
        "weightings": {
            "global_frobenius": "natural squared Frobenius energy of the globally flattened functional update",
            "module_balanced": "each module Gram normalized to unit total interval energy before averaging",
        },
        "rows": len(rows),
        "summary": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(artifact, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f"DELTA_TEMPORAL_SUBSPACE_COMPLETE {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
