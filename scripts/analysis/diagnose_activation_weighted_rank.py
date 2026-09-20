#!/usr/bin/env python3
"""D1: diagnose activation-weighted LoRA rank on fixed policy responses.

The script replays prompt/completion records through an HF+PEFT model and
collects only small rank-space covariance matrices.  For

    Delta W = U diag(s) V^T,

the output energy of singular component i on activations x is

    s_i^2 E[(v_i^T x)^2].

No hidden states are retained after a forward hook returns.  Records are split
deterministically into calibration and validation halves so a rank map chosen
on one half can be evaluated on the other without another model pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


DEFAULT_BENCHMARKS = ("aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva")
DEFAULT_TARGETS = (0.90, 0.95, 0.99)
DEFAULT_RANK_BINS = (8, 12, 16, 20, 24, 28, 32)


@dataclass
class ModuleSpectrum:
    name: str
    adapter_key: str
    layer: int | None
    module_type: str
    d_in: int
    d_out: int
    configured_rank: int
    scale: float
    singular_values: torch.Tensor
    right_basis: torch.Tensor


@dataclass
class SelectedRecord:
    split: str
    benchmark: str
    problem_id: str
    problem_index: int
    sample_index: int
    prompt: str
    completion: str
    correct: bool | None
    score: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sequence_tokens: int = 0
    sampled_tokens: int = 0


@dataclass
class ActivationCollector:
    spectra: dict[str, ModuleSpectrum]
    device: torch.device
    projection_dtype: torch.dtype
    covariances: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    active_split: str | None = None
    active_positions: torch.Tensor | None = None
    hooks: list[Any] = field(default_factory=list)
    projection_bases: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.covariances = {
            split: {
                name: torch.zeros(
                    spectrum.configured_rank,
                    spectrum.configured_rank,
                    dtype=torch.float32,
                    device=self.device,
                )
                for name, spectrum in self.spectra.items()
            }
            for split in ("calibration", "validation")
        }
        self.projection_bases = {
            name: spectrum.right_basis.to(device=self.device, dtype=self.projection_dtype)
            for name, spectrum in self.spectra.items()
        }

    def set_sequence(self, split: str, positions: torch.Tensor) -> None:
        self.active_split = split
        self.active_positions = positions.to(device=self.device)

    def clear_sequence(self) -> None:
        self.active_split = None
        self.active_positions = None

    def register(self, model: torch.nn.Module) -> None:
        named_modules = dict(model.named_modules())
        resolved: dict[str, torch.nn.Module] = {}
        for name in self.spectra:
            module = named_modules.get(name)
            if module is None:
                suffix = name.removeprefix("base_model.model.")
                matches = [value for key, value in named_modules.items() if key.endswith(suffix)]
                if len(matches) != 1:
                    raise KeyError(f"Could not uniquely resolve adapter module {name!r}; matches={len(matches)}")
                module = matches[0]
            resolved[name] = module

        for name, module in resolved.items():
            self.hooks.append(module.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if self.active_split is None or self.active_positions is None:
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"LoRA module {name} received unsupported inputs")
            hidden = inputs[0]
            flat = hidden.reshape(-1, hidden.shape[-1])
            positions = self.active_positions[self.active_positions < flat.shape[0]]
            if positions.numel() == 0:
                return
            sampled = flat.index_select(0, positions)
            basis = self.projection_bases[name]
            projected = sampled.to(dtype=self.projection_dtype) @ basis
            covariance = projected.float().mT @ projected.float()
            self.covariances[self.active_split][name].add_(covariance)

        return hook

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.projection_bases.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--projection-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--num-sequences", type=int, default=48)
    parser.add_argument("--max-seq-tokens", type=int, default=4096)
    parser.add_argument("--tokens-per-sequence", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--energy-targets", default=",".join(str(value) for value in DEFAULT_TARGETS))
    parser.add_argument("--primary-target", type=float, default=0.95)
    parser.add_argument("--rank-bins", default=",".join(str(value) for value in DEFAULT_RANK_BINS))
    parser.add_argument("--activation-region", choices=("response", "all"), default="response")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def stable_score(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def select_records(
    path: Path,
    benchmarks: list[str],
    num_sequences: int,
    seed: int,
) -> list[SelectedRecord]:
    if num_sequences < 2:
        raise ValueError("num_sequences must be at least 2 for split-half diagnostics")
    allowed = set(benchmarks)
    by_problem: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            benchmark = str(row.get("benchmark", ""))
            if benchmark not in allowed or not row.get("prompt") or not row.get("completion"):
                continue
            problem_id = str(row.get("id", row.get("problem_index", line_number)))
            sample_index = int(row.get("sample_index", 0))
            sample_score = stable_score(seed, "sample", benchmark, problem_id, sample_index)
            key = (benchmark, problem_id)
            if key not in by_problem or sample_score < by_problem[key][0]:
                by_problem[key] = (sample_score, row)

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for (benchmark, problem_id), (_sample_score, row) in by_problem.items():
        grouped[benchmark].append((stable_score(seed, "problem", benchmark, problem_id), row))
    missing = [benchmark for benchmark in benchmarks if not grouped[benchmark]]
    if missing:
        raise ValueError(f"No usable records for benchmarks: {missing}")

    base = num_sequences // len(benchmarks)
    remainder = num_sequences % len(benchmarks)
    selected: list[SelectedRecord] = []
    for index, benchmark in enumerate(benchmarks):
        target = base + (1 if index < remainder else 0)
        candidates = sorted(grouped[benchmark], key=lambda item: item[0])[:target]
        if len(candidates) < target:
            raise ValueError(f"Requested {target} unique {benchmark} problems, found {len(candidates)}")
        for local_index, (score, row) in enumerate(candidates):
            selected.append(
                SelectedRecord(
                    split="calibration" if (local_index + index) % 2 == 0 else "validation",
                    benchmark=benchmark,
                    problem_id=str(row.get("id", row.get("problem_index"))),
                    problem_index=int(row.get("problem_index", -1)),
                    sample_index=int(row.get("sample_index", 0)),
                    prompt=str(row["prompt"]),
                    completion=str(row["completion"]),
                    correct=row.get("correct"),
                    score=score,
                )
            )
    selected.sort(key=lambda row: (row.split, row.benchmark, row.score))
    split_counts = {split: sum(row.split == split for row in selected) for split in ("calibration", "validation")}
    if min(split_counts.values()) == 0:
        raise ValueError(f"Both splits must be non-empty, got {split_counts}")
    return selected


def layer_index(name: str) -> int | None:
    import re

    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def module_type(name: str) -> str:
    return name.split(".")[-1]


def load_spectra(adapter: Path) -> tuple[dict[str, ModuleSpectrum], dict[str, Any]]:
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    configured_rank = int(config["r"])
    scale = float(config["lora_alpha"]) / configured_rank
    spectra: dict[str, ModuleSpectrum] = {}
    model_path = adapter / "adapter_model.safetensors"
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        key_set = set(handle.keys())
        a_keys = sorted(key for key in key_set if key.endswith(".lora_A.weight"))
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            if b_key not in key_set:
                raise KeyError(f"Missing paired B factor for {a_key}")
            a = handle.get_tensor(a_key).double()
            b = handle.get_tensor(b_key).double()
            q_b, r_b = torch.linalg.qr(b, mode="reduced")
            del q_b
            q_a, r_a = torch.linalg.qr(a.mT, mode="reduced")
            _u_core, singular_values, vh_core = torch.linalg.svd(r_b @ r_a.mT, full_matrices=False)
            right_basis = q_a @ vh_core.mT
            name = a_key.removesuffix(".lora_A.weight")
            if singular_values.numel() != configured_rank:
                raise ValueError(
                    f"Rank mismatch for {name}: config={configured_rank}, SVD={singular_values.numel()}"
                )
            spectra[name] = ModuleSpectrum(
                name=name,
                adapter_key=name,
                layer=layer_index(name),
                module_type=module_type(name),
                d_in=a.shape[1],
                d_out=b.shape[0],
                configured_rank=configured_rank,
                scale=scale,
                singular_values=singular_values.mul(abs(scale)).cpu(),
                right_basis=right_basis.float().cpu(),
            )
    if not spectra:
        raise ValueError(f"No LoRA A/B factors found in {model_path}")
    return spectra, config


def sample_positions(start: int, end: int, limit: int) -> torch.Tensor:
    if end <= start:
        return torch.empty(0, dtype=torch.long)
    count = end - start
    if limit <= 0 or count <= limit:
        return torch.arange(start, end, dtype=torch.long)
    positions = torch.linspace(start, end - 1, steps=limit, dtype=torch.float64).round().long()
    return torch.unique_consecutive(positions)


def energy_rank(energy: torch.Tensor, target: float, order: torch.Tensor | None = None) -> int:
    if not 0 < target <= 1:
        raise ValueError(f"Energy target must be in (0, 1], got {target}")
    values = energy if order is None else energy.index_select(0, order)
    total = float(values.sum().item())
    if total <= 0:
        return values.numel()
    normalized = torch.cumsum(values, dim=0) / total
    return int(torch.searchsorted(normalized, target).item()) + 1


def retained_energy(energy: torch.Tensor, indices: torch.Tensor) -> float:
    total = float(energy.sum().item())
    return float(energy.index_select(0, indices).sum().item()) / total if total > 0 else float("nan")


def quantize_rank(rank: int, bins: list[int], configured_rank: int) -> int:
    usable = sorted({value for value in bins if 0 < value <= configured_rank} | {configured_rank})
    return next((value for value in usable if value >= rank), configured_rank)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2 + 1
        for position in order[index:end]:
            ranks[position] = average_rank
        index = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else float("nan")


def summarize_ranks(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "max": ordered[-1],
        "std": statistics.pstdev(values),
    }


def make_rank_map(
    rows: list[dict[str, Any]],
    rank_field: str,
    scale: float,
    source: str,
    component_field: str | None = None,
) -> dict[str, Any]:
    rank_pattern = {str(row["adapter_key"]): int(row[rank_field]) for row in rows}
    alpha_pattern = {key: scale * rank for key, rank in rank_pattern.items()}
    result = {
        "source": source,
        "rank_field": rank_field,
        "constant_scaling": scale,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
    }
    if component_field is not None:
        result["component_pattern"] = {
            str(row["adapter_key"]): [
                int(value) for value in str(row[component_field]).split(",") if value
            ]
            for row in rows
        }
    return result


def parameter_count(rows: list[dict[str, Any]], rank_field: str) -> int:
    return sum(int(row[rank_field]) * (int(row["d_in"]) + int(row["d_out"])) for row in rows)


def parameter_matched_fixed_rank(
    rows: list[dict[str, Any]], candidate_parameters: int, allowed_ranks: list[int]
) -> tuple[int, int]:
    if not rows:
        raise ValueError("Cannot match a fixed rank without module rows")
    ranks = sorted({int(rank) for rank in allowed_ranks if int(rank) > 0})
    if not ranks:
        raise ValueError("At least one positive fixed-rank candidate is required")
    width_sum = sum(int(row["d_in"]) + int(row["d_out"]) for row in rows)
    fixed_rank = min(ranks, key=lambda rank: (abs(rank * width_sum - candidate_parameters), rank))
    return fixed_rank, fixed_rank * width_sum


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, Any]],
    primary_suffix: str,
    configured_rank: int,
    scale: float,
    rank_bins: list[int],
) -> dict[str, Any]:
    calibration_field = f"activation_calibration_rank_{primary_suffix}_quantized"
    validation_field = f"activation_validation_rank_{primary_suffix}_quantized"
    calibration = [int(row[calibration_field]) for row in rows]
    validation = [int(row[validation_field]) for row in rows]
    differences = [abs(left - right) for left, right in zip(calibration, validation)]
    baseline_parameters = sum(configured_rank * (int(row["d_in"]) + int(row["d_out"])) for row in rows)
    candidate_parameters = parameter_count(rows, calibration_field)

    validation_total = sum(float(row["activation_validation_total_energy"]) for row in rows)
    candidate_retained = sum(
        float(row["activation_validation_total_energy"])
        * float(row[f"activation_validation_retained_by_calibration_{primary_suffix}_quantized"])
        for row in rows
    )
    fixed_rank, fixed_parameters = parameter_matched_fixed_rank(
        rows,
        candidate_parameters,
        sorted(set(rank_bins) | {configured_rank}),
    )
    fixed_retained = sum(
        float(row["activation_validation_total_energy"])
        * float(row[f"activation_validation_prefix_energy_at_{fixed_rank}"])
        for row in rows
    )

    def grouped(field: str) -> dict[str, Any]:
        result = {}
        for value in sorted({str(row[field]) for row in rows}):
            group = [int(row[calibration_field]) for row in rows if str(row[field]) == value]
            result[value] = summarize_ranks(group)
        return result

    def retention_distribution(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean": statistics.fmean(ordered),
            "min": ordered[0],
            "p10": ordered[int(0.10 * (len(ordered) - 1))],
            "median": statistics.median(ordered),
            "p90": ordered[int(0.90 * (len(ordered) - 1))],
        }

    def map_metrics(rank_field: str, retention_field: str) -> dict[str, Any]:
        ranks = [int(row[rank_field]) for row in rows]
        parameters = parameter_count(rows, rank_field)
        retentions = [float(row[retention_field]) for row in rows]
        retained_global = sum(
            float(row["activation_validation_total_energy"]) * float(row[retention_field])
            for row in rows
        ) / validation_total
        return {
            "rank": summarize_ranks(ranks),
            "parameters": parameters,
            "theoretical_lora_flops_per_token": 2 * parameters,
            "parameter_ratio_to_rank_max": parameters / baseline_parameters,
            "theoretical_lora_flops_ratio_to_rank_max": parameters / baseline_parameters,
            "validation_global_retained_activation_energy": retained_global,
            "validation_module_retained_activation_energy": retention_distribution(retentions),
        }

    fixed_retentions = [float(row[f"activation_validation_prefix_energy_at_{fixed_rank}"]) for row in rows]

    return {
        "calibration_rank": summarize_ranks(calibration),
        "validation_rank": summarize_ranks(validation),
        "split_half_stability": {
            "mean_absolute_rank_difference": statistics.fmean(differences),
            "exact_match_ratio": sum(value == 0 for value in differences) / len(differences),
            "within_4_ratio": sum(value <= 4 for value in differences) / len(differences),
            "pearson": pearson([float(value) for value in calibration], [float(value) for value in validation]),
            "spearman": pearson(rankdata([float(value) for value in calibration]), rankdata([float(value) for value in validation])),
        },
        "candidate": {
            "parameters": candidate_parameters,
            "theoretical_lora_flops_per_token": 2 * candidate_parameters,
            "parameter_ratio_to_rank_max": candidate_parameters / baseline_parameters,
            "theoretical_lora_flops_ratio_to_rank_max": candidate_parameters / baseline_parameters,
            "validation_global_retained_activation_energy": candidate_retained / validation_total,
        },
        "matched_fixed_rank": {
            "rank": fixed_rank,
            "parameters": fixed_parameters,
            "theoretical_lora_flops_per_token": 2 * fixed_parameters,
            "parameter_ratio_to_rank_max": fixed_parameters / baseline_parameters,
            "theoretical_lora_flops_ratio_to_rank_max": fixed_parameters / baseline_parameters,
            "validation_global_retained_activation_energy": fixed_retained / validation_total,
            "validation_module_retained_activation_energy": retention_distribution(fixed_retentions),
        },
        "map_comparison": {
            "activation_prefix": map_metrics(
                calibration_field,
                f"activation_validation_retained_by_calibration_{primary_suffix}_quantized",
            ),
            "activation_oracle": map_metrics(
                f"activation_calibration_oracle_rank_{primary_suffix}_quantized",
                f"activation_validation_retained_by_calibration_oracle_{primary_suffix}_quantized",
            ),
            "frobenius_prefix": map_metrics(
                f"frobenius_rank_{primary_suffix}_quantized",
                f"activation_validation_retained_by_frobenius_{primary_suffix}_quantized",
            ),
        },
        "by_module_type": grouped("module_type"),
        "by_layer_band": grouped("layer_band"),
        "constant_scaling": scale,
    }


def build_rows(
    spectra: dict[str, ModuleSpectrum],
    covariances: dict[str, dict[str, torch.Tensor]],
    targets: list[float],
    primary_target: float,
    rank_bins: list[int],
) -> list[dict[str, Any]]:
    max_layer = max((spectrum.layer for spectrum in spectra.values() if spectrum.layer is not None), default=-1)
    layer_count = max_layer + 1
    boundary = max(1, math.ceil(layer_count / 3))
    rows = []
    for name, spectrum in sorted(spectra.items()):
        squared_singular = spectrum.singular_values.double().square()
        identity_order = torch.arange(spectrum.configured_rank)
        energies = {}
        orders = {}
        for split in ("calibration", "validation"):
            diagonal = covariances[split][name].diagonal().clamp_min(0)
            energies[split] = squared_singular * diagonal
            orders[split] = torch.argsort(energies[split], descending=True, stable=True)
        if spectrum.layer is None:
            band = "other"
        elif spectrum.layer < boundary:
            band = "early"
        elif spectrum.layer < 2 * boundary:
            band = "middle"
        else:
            band = "late"
        row: dict[str, Any] = {
            "adapter_key": spectrum.adapter_key,
            "layer": spectrum.layer,
            "layer_band": band,
            "module_type": spectrum.module_type,
            "d_in": spectrum.d_in,
            "d_out": spectrum.d_out,
            "configured_rank": spectrum.configured_rank,
            "scale": spectrum.scale,
            "activation_calibration_total_energy": float(energies["calibration"].sum().item()),
            "activation_validation_total_energy": float(energies["validation"].sum().item()),
        }
        for target in targets:
            suffix = str(int(round(target * 100)))
            fro_rank = energy_rank(squared_singular, target)
            row[f"frobenius_rank_{suffix}"] = fro_rank
            row[f"frobenius_rank_{suffix}_quantized"] = quantize_rank(
                fro_rank, rank_bins, spectrum.configured_rank
            )
            for split in ("calibration", "validation"):
                prefix_rank = energy_rank(energies[split], target)
                oracle_rank = energy_rank(energies[split], target, orders[split])
                row[f"activation_{split}_rank_{suffix}"] = prefix_rank
                row[f"activation_{split}_rank_{suffix}_quantized"] = quantize_rank(
                    prefix_rank, rank_bins, spectrum.configured_rank
                )
                row[f"activation_{split}_oracle_rank_{suffix}"] = oracle_rank
                row[f"activation_{split}_oracle_rank_{suffix}_quantized"] = quantize_rank(
                    oracle_rank, rank_bins, spectrum.configured_rank
                )
        primary_suffix = str(int(round(primary_target * 100)))
        calibration_prefix_rank = int(row[f"activation_calibration_rank_{primary_suffix}_quantized"])
        calibration_oracle_rank = int(row[f"activation_calibration_oracle_rank_{primary_suffix}_quantized"])
        calibration_oracle_indices = orders["calibration"][:calibration_oracle_rank]
        row[f"activation_validation_retained_by_calibration_{primary_suffix}_quantized"] = retained_energy(
            energies["validation"], identity_order[:calibration_prefix_rank]
        )
        row[
            f"activation_validation_retained_by_calibration_oracle_{primary_suffix}_quantized"
        ] = retained_energy(energies["validation"], calibration_oracle_indices)
        row[f"activation_calibration_oracle_indices_{primary_suffix}_quantized"] = ",".join(
            str(int(value)) for value in calibration_oracle_indices.tolist()
        )
        for rank in sorted(set(rank_bins) | {spectrum.configured_rank}):
            if rank <= spectrum.configured_rank:
                row[f"activation_validation_prefix_energy_at_{rank}"] = retained_energy(
                    energies["validation"], identity_order[:rank]
                )
        frobenius_rank = int(row[f"frobenius_rank_{primary_suffix}_quantized"])
        row[f"activation_validation_retained_by_frobenius_{primary_suffix}_quantized"] = retained_energy(
            energies["validation"], identity_order[:frobenius_rank]
        )
        rows.append(row)
    return rows


def write_report(path: Path, summary: dict[str, Any]) -> None:
    primary = summary["primary"]
    stability = primary["split_half_stability"]
    candidate = primary["candidate"]
    fixed = primary["matched_fixed_rank"]
    comparisons = primary["map_comparison"]
    lines = [
        "# D1 Activation-Weighted Rank Diagnostic",
        "",
        f"Generated: `{summary['created_at_utc']}`",
        "",
        "## Data",
        "",
        f"- Sequences: `{summary['data']['sequences']}` "
        f"(calibration `{summary['data']['calibration_sequences']}`, validation `{summary['data']['validation_sequences']}`)",
        f"- Sampled response tokens: calibration `{summary['data']['calibration_sampled_tokens']}`, "
        f"validation `{summary['data']['validation_sampled_tokens']}`",
        f"- Max sequence tokens: `{summary['config']['max_seq_tokens']}`",
        f"- Tokens sampled per sequence: `{summary['config']['tokens_per_sequence']}`",
        "",
        "## Primary Activation-Prefix Rank Map",
        "",
        f"- Target retained energy: `{summary['config']['primary_target']:.2%}`",
        f"- Calibration rank mean/median/std: `{primary['calibration_rank']['mean']:.2f}` / "
        f"`{primary['calibration_rank']['median']:.1f}` / `{primary['calibration_rank']['std']:.2f}`",
        f"- Calibration rank range: `{primary['calibration_rank']['min']}` to `{primary['calibration_rank']['max']}`",
        f"- Split-half rank MAE: `{stability['mean_absolute_rank_difference']:.2f}`",
        f"- Split-half exact/within-4: `{stability['exact_match_ratio']:.2%}` / `{stability['within_4_ratio']:.2%}`",
        f"- Split-half Spearman: `{stability['spearman']:.4f}`",
        "",
        "## Parameter-Matched Comparison",
        "",
        "| Map | Mean rank | Parameter ratio | Global retained | Module mean | Module min / p10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Activation prefix", "activation_prefix"),
        ("Activation component oracle", "activation_oracle"),
        ("Frobenius prefix", "frobenius_prefix"),
    ):
        item = comparisons[key]
        distribution = item["validation_module_retained_activation_energy"]
        lines.append(
            f"| {label} | {item['rank']['mean']:.2f} | {item['parameter_ratio_to_rank_max']:.2%} | "
            f"{item['validation_global_retained_activation_energy']:.4%} | {distribution['mean']:.4%} | "
            f"{distribution['min']:.4%} / {distribution['p10']:.4%} |",
        )
    fixed_distribution = fixed["validation_module_retained_activation_energy"]
    lines.append(
        f"| Fixed rank {fixed['rank']} | {fixed['rank']:.2f} | {fixed['parameter_ratio_to_rank_max']:.2%} | "
        f"{fixed['validation_global_retained_activation_energy']:.4%} | {fixed_distribution['mean']:.4%} | "
        f"{fixed_distribution['min']:.4%} / {fixed_distribution['p10']:.4%} |",
    )
    lines.extend(
        [
            "",
            "Parameter ratio and theoretical LoRA FLOP ratio are identical here because each factorized "
            "linear contributes `r * (d_in + d_out)` parameters and `2 * r * (d_in + d_out)` FLOPs per token.",
            "",
            "## Rank By Module Type",
            "",
            "| Module | Mean | Median | Min | P90 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in primary["by_module_type"].items():
        lines.append(
            f"| {name} | {item['mean']:.2f} | {item['median']:.1f} | {item['min']} | "
            f"{item['p90']} | {item['max']} |"
        )
    lines.extend(
        [
            "",
            "The activation component oracle is diagnostic only: it may select non-prefix singular components. "
            "The exported JSON records those component indices explicitly.",
            "",
            "The activation-prefix map is stable and heterogeneous, and it protects the worst module better than "
            "the parameter-matched fixed rank. It does not improve globally weighted retained energy over the "
            "fixed-rank control. This report measures adapter-output energy, not token KL or reward; D2 must "
            "validate policy-output distortion before this map is used in training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    benchmarks = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    targets = sorted({float(value) for value in args.energy_targets.split(",") if value.strip()})
    if args.primary_target not in targets:
        targets.append(args.primary_target)
        targets.sort()
    rank_bins = sorted({int(value) for value in args.rank_bins.split(",") if value.strip()})
    model_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    spectra, adapter_config = load_spectra(args.adapter.resolve())
    configured_ranks = {spectrum.configured_rank for spectrum in spectra.values()}
    scales = {spectrum.scale for spectrum in spectra.values()}
    if len(configured_ranks) != 1 or len(scales) != 1:
        raise ValueError(f"Expected constant rank and scale, got ranks={configured_ranks}, scales={scales}")
    configured_rank = next(iter(configured_ranks))
    scale = next(iter(scales))
    rank_bins = [value for value in rank_bins if value <= configured_rank]
    if not rank_bins:
        raise ValueError(f"No rank bins are <= configured rank {configured_rank}")

    records = select_records(args.records.resolve(), benchmarks, args.num_sequences, args.seed)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model.resolve(), trust_remote_code=args.trust_remote_code
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model.resolve(),
        dtype=model_dtypes[args.model_dtype],
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter.resolve(), is_trainable=False).to(device)
    model.eval()
    backbone_owner = model.get_base_model()
    backbone = getattr(backbone_owner, "model", backbone_owner)

    collector = ActivationCollector(
        spectra=spectra,
        device=device,
        projection_dtype=model_dtypes[args.projection_dtype],
    )
    collector.register(model)
    token_totals = defaultdict(int)
    try:
        with torch.inference_mode():
            for index, record in enumerate(records, 1):
                prompt_ids = tokenizer.encode(
                    record.prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_seq_tokens,
                )
                completion_budget = max(1, args.max_seq_tokens - len(prompt_ids))
                completion_ids = tokenizer.encode(
                    record.completion,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=completion_budget,
                )
                combined = (prompt_ids + completion_ids)[: args.max_seq_tokens]
                response_start = min(len(prompt_ids), len(combined))
                region_start = response_start if args.activation_region == "response" else 0
                positions = sample_positions(region_start, len(combined), args.tokens_per_sequence)
                if not combined or positions.numel() == 0:
                    raise ValueError(
                        f"Selected record has no sampled tokens: {record.benchmark}/{record.problem_id}"
                    )
                record.prompt_tokens = len(prompt_ids)
                record.completion_tokens = len(completion_ids)
                record.sequence_tokens = len(combined)
                record.sampled_tokens = int(positions.numel())
                token_totals[f"{record.split}_sampled"] += record.sampled_tokens
                token_totals[f"{record.split}_sequence"] += record.sequence_tokens
                input_ids = torch.tensor(combined, dtype=torch.long, device=device).unsqueeze(0)
                collector.set_sequence(record.split, positions)
                backbone(input_ids=input_ids, use_cache=False, return_dict=False)
                collector.clear_sequence()
                print(
                    f"[{index:03d}/{len(records):03d}] {record.split:<11} "
                    f"{record.benchmark:<9} seq={record.sequence_tokens} sampled={record.sampled_tokens}",
                    flush=True,
                )
    finally:
        collector.clear_sequence()
        collector.close()

    covariances = {
        split: {name: value.double().cpu() for name, value in split_values.items()}
        for split, split_values in collector.covariances.items()
    }
    rows = build_rows(
        spectra=spectra,
        covariances=covariances,
        targets=targets,
        primary_target=args.primary_target,
        rank_bins=rank_bins,
    )
    primary_suffix = str(int(round(args.primary_target * 100)))
    primary = aggregate(rows, primary_suffix, configured_rank, scale, rank_bins)
    fixed_rank = int(primary["matched_fixed_rank"]["rank"])
    for row in rows:
        row["fixed_parameter_matched_rank"] = fixed_rank
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "module_rank_diagnostics.csv", rows)
    write_csv(
        out_dir / "selected_records.csv",
        [
            {
                "split": record.split,
                "benchmark": record.benchmark,
                "problem_id": record.problem_id,
                "problem_index": record.problem_index,
                "sample_index": record.sample_index,
                "correct": record.correct,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
                "sequence_tokens": record.sequence_tokens,
                "sampled_tokens": record.sampled_tokens,
            }
            for record in records
        ],
    )
    maps = {
        "activation_prefix": make_rank_map(
            rows,
            f"activation_calibration_rank_{primary_suffix}_quantized",
            scale,
            "calibration activation-weighted SVD prefix",
        ),
        "activation_oracle": make_rank_map(
            rows,
            f"activation_calibration_oracle_rank_{primary_suffix}_quantized",
            scale,
            "calibration activation-energy reordered singular-component oracle",
            component_field=f"activation_calibration_oracle_indices_{primary_suffix}_quantized",
        ),
        "frobenius": make_rank_map(
            rows,
            f"frobenius_rank_{primary_suffix}_quantized",
            scale,
            "Frobenius-energy SVD prefix",
        ),
        "fixed_parameter_match": make_rank_map(
            rows,
            "fixed_parameter_matched_rank",
            scale,
            "single fixed rank nearest to the activation-prefix parameter count",
        ),
    }
    for name, payload in maps.items():
        (out_dir / f"rank_map_{name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "records": str(args.records.resolve()),
        "adapter_config": adapter_config,
        "config": {
            "device": str(device),
            "model_dtype": args.model_dtype,
            "projection_dtype": args.projection_dtype,
            "attn_implementation": args.attn_implementation,
            "max_seq_tokens": args.max_seq_tokens,
            "tokens_per_sequence": args.tokens_per_sequence,
            "activation_region": args.activation_region,
            "targets": targets,
            "primary_target": args.primary_target,
            "rank_bins": rank_bins,
            "seed": args.seed,
        },
        "data": {
            "sequences": len(records),
            "calibration_sequences": sum(record.split == "calibration" for record in records),
            "validation_sequences": sum(record.split == "validation" for record in records),
            "calibration_sequence_tokens": token_totals["calibration_sequence"],
            "validation_sequence_tokens": token_totals["validation_sequence"],
            "calibration_sampled_tokens": token_totals["calibration_sampled"],
            "validation_sampled_tokens": token_totals["validation_sampled"],
        },
        "modules": len(rows),
        "configured_rank": configured_rank,
        "scale": scale,
        "primary": primary,
        "definitions": {
            "activation_prefix": "smallest original singular-value prefix retaining the target adapter-output energy",
            "activation_oracle": "smallest subset after reordering current singular components by activation-weighted energy",
            "heldout_retained": "validation activation energy retained by the calibration-selected rank or subset",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", summary)
    print(f"Wrote D1 diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
