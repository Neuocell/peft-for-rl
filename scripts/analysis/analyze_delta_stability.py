#!/usr/bin/env python3
"""Build multi-timescale functional DeltaW stability artifacts from LoRA snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DEFAULT_HORIZONS = (1, 5, 10, 25, 50)
ATTENTION = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP = {"gate_proj", "up_proj", "down_proj"}


@dataclass
class Spectrum:
    singular: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    @property
    def energy(self) -> float:
        return float(self.singular.square().sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0-snapshots", type=Path, required=True)
    parser.add_argument("--o0-snapshots", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--calibration-records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--last-step", type=int, default=100)
    parser.add_argument("--calibration-prompts", type=int, default=16)
    parser.add_argument("--tokens-per-prompt", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rank_at_energy(singular: torch.Tensor, target: float) -> int:
    # Individual SVDs are already ordered, but full-model summaries concatenate
    # spectra from many modules and must be sorted again before taking a rank.
    energy = singular.double().square().sort(descending=True).values
    total = energy.sum()
    if float(total) <= 0:
        return 0
    return int(torch.searchsorted(torch.cumsum(energy, 0) / total, target).item()) + 1


def effective_rank(singular: torch.Tensor) -> float:
    energy = singular.double().square()
    total = energy.sum()
    if float(total) <= 0:
        return 0.0
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def compact_svd(a0: torch.Tensor, b0: torch.Tensor, a1: torch.Tensor, b1: torch.Tensor, scale: float) -> Spectrum:
    """Exact SVD of scale * (B1 A1 - B0 A0) via a <=64 square core."""

    left_factor = torch.cat((b1.double(), -b0.double()), dim=1)
    right_factor = torch.cat((a1.double(), a0.double()), dim=0)
    q_left, r_left = torch.linalg.qr(left_factor, mode="reduced")
    q_right, r_right = torch.linalg.qr(right_factor.mT, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(r_left @ r_right.mT, full_matrices=False)
    tolerance = torch.finfo(singular.dtype).eps * max(left_factor.shape[0], right_factor.shape[1])
    keep = singular > (singular.max() * tolerance if singular.numel() else 0)
    singular = singular[keep].mul(abs(scale))
    return Spectrum(
        singular=singular,
        left=(q_left @ u_core[:, keep]).float(),
        right=(q_right @ vh_core.mT[:, keep]).float(),
    )


def principal_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    width = min(first.shape[1], second.shape[1])
    if width == 0:
        return float("nan")
    cosines = torch.linalg.svdvals(first.mT.double() @ second.double())[:width]
    return float(cosines.square().mean())


def cross_capture(previous: Spectrum, future: Spectrum, target: float) -> dict[str, float]:
    k_previous = rank_at_energy(previous.singular, target)
    k_future = rank_at_energy(future.singular, target)
    if not k_previous or not k_future or future.energy <= 0:
        return {key: float("nan") for key in ("left_overlap", "right_overlap", "left", "right", "full")}
    up = previous.left[:, :k_previous].double()
    vp = previous.right[:, :k_previous].double()
    uf = future.left.double()
    vf = future.right.double()
    squared = future.singular.double().square()
    denominator = squared.sum()
    left_projection = up.mT @ uf
    right_projection = vp.mT @ vf
    core = left_projection @ torch.diag(future.singular.double()) @ right_projection.mT
    return {
        "left_overlap": principal_overlap(up, uf[:, :k_future]),
        "right_overlap": principal_overlap(vp, vf[:, :k_future]),
        "left": float((left_projection.square().sum(0) * squared).sum() / denominator),
        "right": float((right_projection.square().sum(0) * squared).sum() / denominator),
        "full": float(core.square().sum() / denominator),
    }


def activation_capture(x: torch.Tensor, previous: Spectrum, future: Spectrum, target: float) -> dict[str, float]:
    k = rank_at_energy(previous.singular, target)
    if not k:
        return {"left": float("nan"), "right": float("nan"), "full": float("nan"), "total": 0.0}
    x = x.double()
    up = previous.left[:, :k].double()
    vp = previous.right[:, :k].double()
    uf = future.left.double()
    vf = future.right.double()
    singular = future.singular.double()
    future_coefficients = (x @ vf) * singular
    total = future_coefficients.square().sum()
    if float(total) <= 0:
        return {"left": float("nan"), "right": float("nan"), "full": float("nan"), "total": 0.0}
    output_to_previous = uf.mT @ up
    right_to_future = vp.mT @ vf
    right_coefficients = (x @ vp) @ right_to_future
    right_coefficients = right_coefficients * singular
    return {
        "left": float((future_coefficients @ output_to_previous).square().sum() / total),
        "right": float(right_coefficients.square().sum() / total),
        "full": float((right_coefficients @ output_to_previous).square().sum() / total),
        "total": float(total),
    }


def layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def module_type(name: str) -> str:
    return name.split(".")[-1]


def module_family(name: str) -> str:
    kind = module_type(name)
    if kind in ATTENTION:
        return "attention"
    if kind in MLP:
        return "mlp"
    return "other"


def layer_band(layer: int | None, max_layer: int) -> str:
    if layer is None:
        return "other"
    boundary = max(1, math.ceil((max_layer + 1) / 3))
    return "early" if layer < boundary else "middle" if layer < 2 * boundary else "late"


def snapshot_paths(root: Path, last_step: int) -> list[Path]:
    paths = [root / f"step_{step:03d}" / "adapter_model.safetensors" for step in range(last_step + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} snapshots; first={missing[0]}")
    return paths


def fixed_prompts(records: Path, count: int, seed: int) -> tuple[list[str], dict[str, Any]]:
    unique: dict[tuple[str, str], str] = {}
    with records.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt = row.get("prompt")
            if prompt:
                key = (str(row.get("benchmark", "")), str(row.get("id", row.get("problem_index", ""))))
                unique.setdefault(key, str(prompt))
    ranked = sorted(
        unique.items(),
        key=lambda item: hashlib.sha256(f"{seed}\0{item[0][0]}\0{item[0][1]}".encode()).digest(),
    )[:count]
    if len(ranked) < count:
        raise ValueError(f"Requested {count} fixed prompts, found {len(ranked)}")
    prompts = [prompt for _key, prompt in ranked]
    return prompts, {
        "source": str(records.resolve()),
        "source_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        "seed": seed,
        "keys": [{"benchmark": key[0], "id": key[1]} for key, _prompt in ranked],
        "prompts_sha256": hashlib.sha256("\0".join(prompts).encode()).hexdigest(),
    }


def collect_activations(
    base_model: Path,
    prompts: list[str],
    module_names: list[str],
    tokens_per_prompt: int,
    max_prompt_tokens: int,
    device: str,
    dtype_name: str,
    attn_implementation: str,
) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtypes[dtype_name],
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    named = dict(model.named_modules())
    activations: dict[str, list[torch.Tensor]] = defaultdict(list)
    hooks = []

    for adapter_name in module_names:
        suffix = adapter_name.removeprefix("base_model.model.")
        matches = [(name, module) for name, module in named.items() if name.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"Cannot resolve {adapter_name}: {len(matches)} matches")

        def make_hook(name: str):
            def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                hidden = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                count = min(tokens_per_prompt, hidden.shape[0])
                positions = torch.linspace(0, hidden.shape[0] - 1, count, device=hidden.device).round().long()
                activations[name].append(hidden.index_select(0, positions).to(dtype=torch.float32, device="cpu"))

            return hook

        hooks.append(matches[0][1].register_forward_pre_hook(make_hook(adapter_name)))
    try:
        with torch.inference_mode():
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_tokens)
                tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {name: torch.cat(values, dim=0) for name, values in activations.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def weighted_mean(rows: list[dict[str, Any]], field: str, weight_field: str = "future_energy") -> float:
    usable = [(float(row[field]), float(row[weight_field])) for row in rows if math.isfinite(float(row[field]))]
    denominator = sum(weight for _value, weight in usable)
    return sum(value * weight for value, weight in usable) / denominator if denominator else float("nan")


def aggregate(method: str, interval_rows: list[dict[str, Any]], adjacency_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    dimensions = (("all", lambda row: True),)
    families = sorted({str(row["family"]) for row in interval_rows})
    bands = sorted({str(row["layer_band"]) for row in interval_rows})
    dimensions += tuple((f"family:{value}", lambda row, value=value: row["family"] == value) for value in families)
    dimensions += tuple((f"depth:{value}", lambda row, value=value: row["layer_band"] == value) for value in bands)
    for horizon in sorted({int(row["h"]) for row in interval_rows}):
        for group, predicate in dimensions:
            intervals = [row for row in interval_rows if int(row["h"]) == horizon and predicate(row)]
            adjacent = [row for row in adjacency_rows if int(row["h"]) == horizon and predicate(row)]
            if not intervals:
                continue
            pooled_singular = torch.tensor(
                [value for row in intervals for value in json.loads(str(row["singular_values"]))], dtype=torch.float64
            )
            per_time_global = []
            for start in sorted({int(row["t"]) for row in intervals}):
                singular = torch.tensor(
                    [
                        value
                        for row in intervals
                        if int(row["t"]) == start
                        for value in json.loads(str(row["singular_values"]))
                    ],
                    dtype=torch.float64,
                )
                per_time_global.append(
                    {
                        "effective_rank": effective_rank(singular),
                        "rank90": rank_at_energy(singular, 0.90),
                        "rank95": rank_at_energy(singular, 0.95),
                    }
                )
            result = {
                "method": method,
                "h": horizon,
                "group": group,
                "module_intervals": len(intervals),
                "adjacent_pairs": len(adjacent),
                "effective_rank_global_pooled": effective_rank(pooled_singular),
                "rank90_global_pooled": rank_at_energy(pooled_singular, 0.90),
                "rank95_global_pooled": rank_at_energy(pooled_singular, 0.95),
                "effective_rank_global_mean": statistics.fmean(row["effective_rank"] for row in per_time_global),
                "rank90_global_mean": statistics.fmean(row["rank90"] for row in per_time_global),
                "rank95_global_mean": statistics.fmean(row["rank95"] for row in per_time_global),
                "effective_rank_module_mean": statistics.fmean(float(row["effective_rank"]) for row in intervals),
                "rank90_module_mean": statistics.fmean(float(row["rank90"]) for row in intervals),
                "rank95_module_mean": statistics.fmean(float(row["rank95"]) for row in intervals),
            }
            for field in (
                "left_overlap_95",
                "right_overlap_95",
                "left_capture_95",
                "right_capture_95",
                "full_capture_95",
                "activation_left_capture_95",
                "activation_right_capture_95",
                "activation_full_capture_95",
            ):
                weight_field = "activation_future_energy" if field.startswith("activation_") else "future_energy"
                result[field] = weighted_mean(adjacent, field, weight_field) if adjacent else float("nan")
            output.append(result)
    return output


def analyze_method(
    method: str,
    root: Path,
    last_step: int,
    horizons: list[int],
    activations: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = snapshot_paths(root, last_step)
    handles = [safe_open(path, framework="pt", device="cpu") for path in paths]
    try:
        keys = set(handles[0].keys())
        a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
        if len(a_keys) != 196:
            raise ValueError(f"Expected 196 LoRA modules, found {len(a_keys)} under {root}")
        metadata = handles[0].metadata()
        scale = float(metadata["scaling"])
        max_layer = max(filter(lambda value: value is not None, (layer_index(key) for key in a_keys)))
        interval_rows: list[dict[str, Any]] = []
        adjacency_rows: list[dict[str, Any]] = []
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            name = a_key.removesuffix(".lora_A.weight")
            layer = layer_index(name)
            common = {
                "method": method,
                "module": name,
                "module_type": module_type(name),
                "family": module_family(name),
                "layer": layer,
                "layer_band": layer_band(layer, max_layer),
            }
            factors = [(handle.get_tensor(a_key), handle.get_tensor(b_key)) for handle in handles]
            for horizon in horizons:
                spectra: dict[int, Spectrum] = {}
                for start in range(0, last_step - horizon + 1):
                    end = start + horizon
                    a0, b0 = factors[start]
                    a1, b1 = factors[end]
                    spectrum = compact_svd(a0, b0, a1, b1, scale)
                    spectra[start] = spectrum
                    interval_rows.append(
                        {
                            **common,
                            "t": start,
                            "end": end,
                            "h": horizon,
                            "frobenius_norm": math.sqrt(spectrum.energy),
                            "effective_rank": effective_rank(spectrum.singular),
                            "rank90": rank_at_energy(spectrum.singular, 0.90),
                            "rank95": rank_at_energy(spectrum.singular, 0.95),
                            "singular_values": json.dumps(spectrum.singular.tolist(), separators=(",", ":")),
                        }
                    )
                for start in range(0, last_step - 2 * horizon + 1):
                    previous = spectra[start]
                    future = spectra[start + horizon]
                    captures = cross_capture(previous, future, 0.95)
                    activation = activation_capture(activations[name], previous, future, 0.95)
                    adjacency_rows.append(
                        {
                            **common,
                            "t": start,
                            "middle": start + horizon,
                            "end": start + 2 * horizon,
                            "h": horizon,
                            "future_energy": future.energy,
                            "left_overlap_95": captures["left_overlap"],
                            "right_overlap_95": captures["right_overlap"],
                            "left_capture_95": captures["left"],
                            "right_capture_95": captures["right"],
                            "full_capture_95": captures["full"],
                            "activation_left_capture_95": activation["left"],
                            "activation_right_capture_95": activation["right"],
                            "activation_full_capture_95": activation["full"],
                            "activation_future_energy": activation["total"],
                        }
                    )
        return interval_rows, adjacency_rows, aggregate(method, interval_rows, adjacency_rows)
    finally:
        for handle in handles:
            handle.__exit__(None, None, None)


def plot_comparison(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = [row for row in rows if row["group"] == "all"]
    metrics = (
        ("full_capture_95", "Functional cross-capture@95"),
        ("right_overlap_95", "Input principal overlap@95"),
        ("left_overlap_95", "Output principal overlap@95"),
        ("effective_rank_module_mean", "Mean effective rank"),
        ("activation_full_capture_95", "Activation-weighted full capture@95"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, metrics):
        for method, label in (("S0", "S0 standard"), ("O0", "O0 random-orthogonal")):
            selected = sorted((row for row in all_rows if row["method"] == method), key=lambda row: row["h"])
            axis.plot([row["h"] for row in selected], [row[field] for row in selected], marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel("h (trainer steps)")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right")
    fig.savefig(out_dir / "s0_o0_multiscale_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    if any(value <= 0 or value > args.last_step for value in horizons):
        raise ValueError(f"Invalid horizons for last step {args.last_step}: {horizons}")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    first_snapshot = snapshot_paths(args.s0_snapshots, args.last_step)[0]
    with safe_open(first_snapshot, framework="pt", device="cpu") as handle:
        module_names = sorted(key.removesuffix(".lora_A.weight") for key in handle.keys() if key.endswith(".lora_A.weight"))
    prompts, prompt_manifest = fixed_prompts(args.calibration_records, args.calibration_prompts, args.seed)
    activations = collect_activations(
        args.base_model.resolve(),
        prompts,
        module_names,
        args.tokens_per_prompt,
        args.max_prompt_tokens,
        args.device,
        args.model_dtype,
        args.attn_implementation,
    )
    if set(activations) != set(module_names):
        raise ValueError(f"Activation coverage mismatch: got={len(activations)} expected={len(module_names)}")

    all_intervals = []
    all_adjacency = []
    all_summary = []
    for method, root in (("S0", args.s0_snapshots), ("O0", args.o0_snapshots)):
        intervals, adjacency, summary = analyze_method(method, root, args.last_step, horizons, activations)
        all_intervals.extend(intervals)
        all_adjacency.extend(adjacency)
        all_summary.extend(summary)

    write_csv(args.out_dir / "module_interval_spectra.csv", all_intervals)
    write_csv(args.out_dir / "adjacent_period_stability.csv", all_adjacency)
    write_csv(args.out_dir / "group_summary.csv", all_summary)
    plot_comparison(all_summary, args.out_dir)
    artifact = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "exact_low_rank_delta_w_multiscale_stability_v1",
        "definition": "D(t,h) = DeltaW_(t+h) - DeltaW_t; DeltaW = (alpha/r) B A",
        "horizons": horizons,
        "last_step": args.last_step,
        "calibration": {
            **prompt_manifest,
            "prompt_count": len(prompts),
            "tokens_per_prompt_per_module": args.tokens_per_prompt,
            "activation_policy": "frozen base model; identical prompts and token positions for S0/O0",
        },
        "rows": {
            "module_interval_spectra": len(all_intervals),
            "adjacent_period_stability": len(all_adjacency),
            "group_summary": len(all_summary),
        },
        "summary": all_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(artifact, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
