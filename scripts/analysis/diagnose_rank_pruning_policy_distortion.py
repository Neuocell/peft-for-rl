#!/usr/bin/env python3
"""Diagnose offline low-order LoRA pruning on fixed policy responses.

For each LoRA update ``Delta W = scale * B @ A``, this script computes an
exact reduced SVD without materializing Delta W.  Response-token activations
from a calibration split select a per-module prefix rank.  The selected map is
then checked on a held-out split in two ways:

1. retained adapter-output energy; and
2. policy distortion after all modules are truncated simultaneously.

For policy evaluation, A stays at its checkpoint value and B is replaced by
the unique projected factor whose product with A is the requested top-k
matrix.  This avoids the BF16 roundoff shift caused by an otherwise equivalent
rotation of both factors. Tensor shapes and PEFT scaling stay fixed; this is an
offline diagnostic and does not alter checkpoint files or imply that A should
be frozen during training.
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
import torch.nn.functional as F

DEFAULT_BENCHMARKS = ("aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva")
DEFAULT_TARGETS = (0.90, 0.95, 0.98)
DEFAULT_RANK_BINS = (8, 12, 16, 20, 24, 28, 32)


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


@dataclass
class EncodedRecord:
    record: SelectedRecord
    input_ids: torch.Tensor
    activation_positions: torch.Tensor
    predictor_positions: torch.Tensor
    target_ids: torch.Tensor


@dataclass
class CanonicalModule:
    name: str
    layer: int | None
    module_type: str
    d_in: int
    d_out: int
    actual_rank: int
    scale: float
    singular_values: torch.Tensor
    signed_singular_values: torch.Tensor
    left_basis: torch.Tensor
    right_basis: torch.Tensor
    original_a: torch.Tensor
    original_b: torch.Tensor
    projected_b_left: torch.Tensor
    projected_b_right: torch.Tensor
    reconstruction_relative_error: float
    runtime_module: torch.nn.Module | None = None


@dataclass
class ActivationCollector:
    modules: dict[str, CanonicalModule]
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
                name: torch.zeros(item.actual_rank, item.actual_rank, dtype=torch.float32, device=self.device)
                for name, item in self.modules.items()
            }
            for split in ("calibration", "validation")
        }
        self.projection_bases = {
            name: item.right_basis.to(device=self.device, dtype=self.projection_dtype)
            for name, item in self.modules.items()
        }

    def register(self) -> None:
        for name, item in self.modules.items():
            if item.runtime_module is None:
                raise RuntimeError(f"Runtime module not resolved for {name}")
            self.hooks.append(item.runtime_module.register_forward_pre_hook(self._make_hook(name)))

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
            projected = sampled.to(self.projection_dtype) @ self.projection_bases[name]
            self.covariances[self.active_split][name].add_(projected.float().mT @ projected.float())

        return hook

    def set_sequence(self, split: str, positions: torch.Tensor) -> None:
        self.active_split = split
        self.active_positions = positions.to(self.device)

    def clear_sequence(self) -> None:
        self.active_split = None
        self.active_positions = None

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.projection_bases.clear()


@dataclass
class IdealTopKOverride:
    modules: dict[str, CanonicalModule]
    ranks: dict[str, int]
    hooks: list[Any] = field(default_factory=list)
    factors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def register(self) -> None:
        for name, item in self.modules.items():
            rank = int(self.ranks[name])
            if rank == item.actual_rank:
                continue
            if item.runtime_module is None:
                raise RuntimeError(f"Runtime module missing for {name}")
            slot = adapter_slot(item.runtime_module)
            weight = item.runtime_module.lora_A[slot].weight
            self.factors[name] = (
                item.left_basis[:, :rank].to(device=weight.device, dtype=weight.dtype),
                item.signed_singular_values[:rank].to(device=weight.device, dtype=weight.dtype),
                item.right_basis[:, :rank].to(device=weight.device, dtype=weight.dtype),
            )
            self.hooks.append(item.runtime_module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], _output: torch.Tensor) -> torch.Tensor:
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"LoRA module {name} received unsupported inputs")
            x = inputs[0]
            base = module.base_layer(x)
            left, singular, right = self.factors[name]
            projected = x.to(right.dtype) @ right
            topk_delta = (projected * singular) @ left.mT
            return (base + topk_delta).to(base.dtype)

        return hook

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.factors.clear()


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
    parser.add_argument("--max-seq-tokens", type=int, default=8192)
    parser.add_argument("--tokens-per-sequence", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--targets", default=",".join(str(value) for value in DEFAULT_TARGETS))
    parser.add_argument("--rank-bins", default=",".join(str(value) for value in DEFAULT_RANK_BINS))
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def stable_score(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def resolve_records_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates = sorted((path / "records").glob("*.jsonl")) if (path / "records").is_dir() else []
    if not candidates:
        candidates = sorted(path.glob("*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one merged JSONL under {path}, found {len(candidates)}")
    return candidates[0]


def select_records(path: Path, benchmarks: list[str], num_sequences: int, seed: int) -> list[SelectedRecord]:
    if num_sequences < 2:
        raise ValueError("num_sequences must be at least 2")
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
    for (benchmark, problem_id), (_score, row) in by_problem.items():
        grouped[benchmark].append((stable_score(seed, "problem", benchmark, problem_id), row))
    missing = [benchmark for benchmark in benchmarks if not grouped[benchmark]]
    if missing:
        raise ValueError(f"No usable records for benchmarks: {missing}")

    base = num_sequences // len(benchmarks)
    remainder = num_sequences % len(benchmarks)
    selected: list[SelectedRecord] = []
    for benchmark_index, benchmark in enumerate(benchmarks):
        target = base + (1 if benchmark_index < remainder else 0)
        candidates = sorted(grouped[benchmark], key=lambda item: item[0])[:target]
        if len(candidates) < target:
            raise ValueError(f"Requested {target} unique {benchmark} problems, found {len(candidates)}")
        for local_index, (score, row) in enumerate(candidates):
            selected.append(
                SelectedRecord(
                    split="calibration" if (local_index + benchmark_index) % 2 == 0 else "validation",
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
    counts = {split: sum(row.split == split for row in selected) for split in ("calibration", "validation")}
    if min(counts.values()) == 0:
        raise ValueError(f"Both splits must be non-empty, got {counts}")
    return selected


def sample_positions(start: int, end: int, limit: int) -> torch.Tensor:
    if end <= start:
        return torch.empty(0, dtype=torch.long)
    count = end - start
    if limit <= 0 or count <= limit:
        return torch.arange(start, end, dtype=torch.long)
    return torch.unique_consecutive(torch.linspace(start, end - 1, steps=limit, dtype=torch.float64).round().long())


def encode_records(
    records: list[SelectedRecord], tokenizer: Any, max_tokens: int, sample_limit: int
) -> list[EncodedRecord]:
    encoded = []
    for record in records:
        prompt_ids = tokenizer.encode(record.prompt, add_special_tokens=False, truncation=True, max_length=max_tokens)
        completion_budget = max(1, max_tokens - len(prompt_ids))
        completion_ids = tokenizer.encode(
            record.completion, add_special_tokens=False, truncation=True, max_length=completion_budget
        )
        combined = (prompt_ids + completion_ids)[:max_tokens]
        response_start = min(len(prompt_ids), len(combined))
        target_positions = sample_positions(max(1, response_start), len(combined), sample_limit)
        if not combined or target_positions.numel() == 0:
            raise ValueError(f"No response target tokens for {record.benchmark}/{record.problem_id}")
        input_ids = torch.tensor(combined, dtype=torch.long)
        encoded.append(
            EncodedRecord(
                record=record,
                input_ids=input_ids,
                activation_positions=target_positions.clone(),
                predictor_positions=target_positions - 1,
                target_ids=input_ids.index_select(0, target_positions),
            )
        )
    return encoded


def layer_index(name: str) -> int | None:
    import re

    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def resolve_runtime_module(name: str, named_modules: dict[str, torch.nn.Module]) -> torch.nn.Module:
    if name in named_modules:
        return named_modules[name]
    suffix = name.removeprefix("base_model.model.")
    matches = [module for key, module in named_modules.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Could not uniquely resolve {name!r}; matches={len(matches)}")
    return matches[0]


def adapter_slot(module: torch.nn.Module) -> str:
    keys = list(module.lora_A.keys())
    if len(keys) != 1:
        raise ValueError(f"Expected one loaded adapter, found {keys}")
    return keys[0]


def canonicalize_lora_modules(model: torch.nn.Module) -> dict[str, CanonicalModule]:
    named_modules = dict(model.named_modules())
    lora_modules = {
        name: module
        for name, module in named_modules.items()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and len(module.lora_A) == 1
    }
    if not lora_modules:
        raise ValueError("No PEFT LoRA modules found in loaded model")

    result = {}
    for name, module in sorted(lora_modules.items()):
        slot = adapter_slot(module)
        a = module.lora_A[slot].weight.detach().double().cpu()
        b = module.lora_B[slot].weight.detach().double().cpu()
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"A/B rank mismatch for {name}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        q_b, r_b = torch.linalg.qr(b, mode="reduced")
        q_a, r_a = torch.linalg.qr(a.mT, mode="reduced")
        u_core, singular_values, vh_core = torch.linalg.svd(r_b @ r_a.mT, full_matrices=False)
        canonical_b = q_b @ u_core @ torch.diag(singular_values)
        canonical_a = vh_core @ q_a.mT
        # A = R_a^T Q_a^T. For a top-k Delta W, keeping the original A gives
        # B_k = Q_b U_k S_k V_k^T (R_a^T)^-1.  Store the two small/factorized
        # sides so each candidate can be written without materializing Delta W.
        projected_b_right = torch.linalg.solve(r_a, vh_core.mT).mT
        denominator = torch.linalg.norm(b @ a).clamp_min(torch.finfo(torch.float64).tiny)
        error = float((torch.linalg.norm(canonical_b @ canonical_a - b @ a) / denominator).item())
        scale = float(module.scaling[slot])
        result[name] = CanonicalModule(
            name=name,
            layer=layer_index(name),
            module_type=name.split(".")[-1],
            d_in=int(a.shape[1]),
            d_out=int(b.shape[0]),
            actual_rank=int(a.shape[0]),
            scale=scale,
            singular_values=singular_values.mul(abs(scale)).float(),
            signed_singular_values=singular_values.mul(scale).float(),
            left_basis=(q_b @ u_core).float(),
            right_basis=(q_a @ vh_core.mT).float(),
            original_a=a.float(),
            original_b=b.float(),
            projected_b_left=canonical_b.float(),
            projected_b_right=projected_b_right.float(),
            reconstruction_relative_error=error,
            runtime_module=module,
        )
    return result


def write_rank_map(
    modules: dict[str, CanonicalModule],
    ranks: dict[str, int],
    *,
    exact_full_restore: bool = True,
) -> None:
    with torch.no_grad():
        for name, item in modules.items():
            if item.runtime_module is None:
                raise RuntimeError(f"Runtime module missing for {name}")
            slot = adapter_slot(item.runtime_module)
            a_weight = item.runtime_module.lora_A[slot].weight
            b_weight = item.runtime_module.lora_B[slot].weight
            rank = int(ranks[name])
            if not 0 < rank <= item.actual_rank:
                raise ValueError(f"Invalid rank {rank} for {name} with rank {item.actual_rank}")
            a_weight.copy_(item.original_a.to(device=a_weight.device, dtype=a_weight.dtype))
            if rank == item.actual_rank and exact_full_restore:
                # Exact restoration makes the full-rank control a true zero-
                # distortion baseline, independent of factorization rounding.
                b_new = item.original_b
            else:
                b_new = item.projected_b_left[:, :rank] @ item.projected_b_right[:rank, :]
            b_weight.copy_(b_new.to(device=b_weight.device, dtype=b_weight.dtype))


def energy_rank(energy: torch.Tensor, target: float) -> int:
    if not 0 < target <= 1:
        raise ValueError(f"Energy target must be in (0, 1], got {target}")
    total = energy.sum()
    if float(total.item()) <= 0:
        return int(energy.numel())
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cumulative, torch.tensor(target, dtype=cumulative.dtype)).item()) + 1


def quantize_rank(rank: int, bins: list[int], actual_rank: int) -> int:
    usable = sorted({value for value in bins if 0 < value <= actual_rank} | {actual_rank})
    return next((value for value in usable if value >= rank), actual_rank)


def build_candidate_maps(
    modules: dict[str, CanonicalModule],
    covariances: dict[str, dict[str, torch.Tensor]],
    targets: list[float],
    rank_bins: list[int],
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    maps = {target_key(target): {} for target in targets}
    rows = []
    for name, item in modules.items():
        singular_sq = item.singular_values.double().square()
        energies = {}
        for split in ("calibration", "validation"):
            projected_variance = covariances[split][name].double().diagonal().clamp_min(0)
            energies[split] = singular_sq * projected_variance
        row: dict[str, Any] = {
            "module": name,
            "layer": item.layer,
            "module_type": item.module_type,
            "d_in": item.d_in,
            "d_out": item.d_out,
            "actual_rank": item.actual_rank,
            "scale": item.scale,
            "canonical_reconstruction_relative_error": item.reconstruction_relative_error,
            "calibration_total_activation_energy": float(energies["calibration"].sum().item()),
            "validation_total_activation_energy": float(energies["validation"].sum().item()),
        }
        for target in targets:
            key = target_key(target)
            raw_rank = energy_rank(energies["calibration"], target)
            rank = quantize_rank(raw_rank, rank_bins, item.actual_rank)
            maps[key][name] = rank
            validation_total = float(energies["validation"].sum().item())
            retained = (
                float(energies["validation"][:rank].sum().item()) / validation_total
                if validation_total > 0
                else math.nan
            )
            row[f"raw_rank_{key}"] = raw_rank
            row[f"rank_{key}"] = rank
            row[f"validation_retained_{key}"] = retained
        rows.append(row)
    return maps, rows


def target_key(target: float) -> str:
    return f"energy_{int(round(target * 100)):02d}"


def decoder_backbone(causal_model: torch.nn.Module) -> torch.nn.Module:
    decoder = getattr(causal_model, "model", None)
    if decoder is None:
        raise AttributeError(f"Cannot find decoder backbone on {type(causal_model).__name__}")
    return decoder


def policy_logits(
    causal_model: torch.nn.Module,
    input_ids: torch.Tensor,
    predictor_positions: torch.Tensor,
) -> torch.Tensor:
    positions = predictor_positions.to(device=input_ids.device)
    outputs = causal_model(
        input_ids=input_ids.unsqueeze(0),
        use_cache=False,
        return_dict=True,
        logits_to_keep=positions,
    )
    logits = outputs.logits.squeeze(0)
    if logits.ndim != 2 or logits.shape[0] != positions.numel():
        raise RuntimeError(f"logits_to_keep returned {tuple(logits.shape)}, expected ({positions.numel()}, vocab)")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite policy logits")
    return logits


def collect_activations(
    decoder: torch.nn.Module,
    collector: ActivationCollector,
    encoded: list[EncodedRecord],
    device: torch.device,
) -> None:
    collector.register()
    try:
        with torch.inference_mode():
            for index, item in enumerate(encoded, 1):
                collector.set_sequence(item.record.split, item.activation_positions)
                decoder(input_ids=item.input_ids.to(device).unsqueeze(0), use_cache=False, return_dict=False)
                collector.clear_sequence()
                print(
                    f"[activation {index:03d}/{len(encoded):03d}] {item.record.split:<11} "
                    f"{item.record.benchmark:<9} seq={item.input_ids.numel()} sampled={item.target_ids.numel()}",
                    flush=True,
                )
    finally:
        collector.clear_sequence()
        collector.close()


def collect_policy_logits(
    causal_model: torch.nn.Module,
    validation: list[EncodedRecord],
    device: torch.device,
    label: str,
) -> list[torch.Tensor]:
    result = []
    with torch.inference_mode():
        for index, item in enumerate(validation, 1):
            logits = policy_logits(
                causal_model,
                item.input_ids.to(device),
                item.predictor_positions,
            )
            # Preserve the model's output precision. Quantizing this cache would
            # create an artificial KL floor when candidates remain in FP32.
            result.append(logits.float().cpu())
            print(f"[{label} {index:03d}/{len(validation):03d}] {item.record.benchmark:<9}", flush=True)
    return result


def tensor_distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    if values.numel() == 0:
        return {key: math.nan for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def policy_metrics(baseline: torch.Tensor, candidate: torch.Tensor, target_ids: torch.Tensor) -> dict[str, Any]:
    baseline = baseline.float()
    candidate = candidate.float()
    if baseline.shape != candidate.shape:
        raise ValueError(f"Policy logit shape mismatch: {tuple(baseline.shape)} vs {tuple(candidate.shape)}")
    baseline_log_probs = F.log_softmax(baseline, dim=-1)
    candidate_log_probs = F.log_softmax(candidate, dim=-1)
    baseline_probs = baseline_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    forward_kl = (baseline_probs * (baseline_log_probs - candidate_log_probs)).sum(dim=-1).clamp_min(0)
    reverse_kl = (candidate_probs * (candidate_log_probs - baseline_log_probs)).sum(dim=-1).clamp_min(0)
    gather_ids = target_ids.long().unsqueeze(-1)
    target_delta = (candidate_log_probs.gather(-1, gather_ids) - baseline_log_probs.gather(-1, gather_ids)).squeeze(-1)
    top1_flip = baseline.argmax(dim=-1).ne(candidate.argmax(dim=-1))
    return {
        "forward_kl": forward_kl.cpu(),
        "reverse_kl": reverse_kl.cpu(),
        "target_delta": target_delta.cpu(),
        "target_abs_delta": target_delta.abs().cpu(),
        "top1_flip": top1_flip.cpu(),
    }


def summarize_policy_comparison(
    validation: list[EncodedRecord],
    reference_logits: list[torch.Tensor],
    candidate_logits: list[torch.Tensor],
    label: str,
    reference_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    token_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
    by_benchmark: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    sequence_rows = []
    iterator = zip(validation, reference_logits, candidate_logits, strict=True)
    for item, reference, candidate in iterator:
        values = policy_metrics(reference, candidate, item.target_ids)
        for metric, tensor in values.items():
            token_metrics[metric].append(tensor)
            by_benchmark[item.record.benchmark][metric].append(tensor)
        sequence_rows.append(
            {
                "candidate": label,
                "reference": reference_label,
                "benchmark": item.record.benchmark,
                "problem_id": item.record.problem_id,
                "sample_index": item.record.sample_index,
                "tokens": int(item.target_ids.numel()),
                "forward_kl_mean": float(values["forward_kl"].mean().item()),
                "target_logprob_abs_delta_mean": float(values["target_abs_delta"].mean().item()),
                "top1_flip_ratio": float(values["top1_flip"].float().mean().item()),
            }
        )

    def summarize(group: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
        merged = {key: torch.cat(values) for key, values in group.items()}
        return {
            "tokens": int(merged["forward_kl"].numel()),
            "forward_kl": tensor_distribution(merged["forward_kl"]),
            "reverse_kl": tensor_distribution(merged["reverse_kl"]),
            "target_logprob_mae": float(merged["target_abs_delta"].mean().item()),
            "target_logprob_delta_mean": float(merged["target_delta"].mean().item()),
            "target_logprob_abs_delta": tensor_distribution(merged["target_abs_delta"]),
            "top1_flip_ratio": float(merged["top1_flip"].float().mean().item()),
        }

    return {
        "global": summarize(token_metrics),
        "by_benchmark": {benchmark: summarize(values) for benchmark, values in sorted(by_benchmark.items())},
    }, sequence_rows


def map_statistics(
    modules: dict[str, CanonicalModule],
    ranks: dict[str, int],
    rows_by_name: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    rank_values = list(ranks.values())
    adapter_before = sum(item.actual_rank * (item.d_in + item.d_out) for item in modules.values())
    adapter_after = sum(ranks[name] * (item.d_in + item.d_out) for name, item in modules.items())
    b_before = sum(item.actual_rank * item.d_out for item in modules.values())
    b_after = sum(ranks[name] * item.d_out for name, item in modules.items())
    weighted_total = sum(float(row["validation_total_activation_energy"]) for row in rows_by_name.values())
    weighted_retained = sum(
        float(row["validation_total_activation_energy"]) * float(row[f"validation_retained_{key}"])
        for row in rows_by_name.values()
    )
    retention = sorted(float(row[f"validation_retained_{key}"]) for row in rows_by_name.values())
    return {
        "rank": {
            "mean": statistics.fmean(rank_values),
            "median": statistics.median(rank_values),
            "min": min(rank_values),
            "max": max(rank_values),
        },
        "adapter_parameters": adapter_after,
        "adapter_parameter_ratio": adapter_after / adapter_before,
        "b_only_parameters": b_after,
        "b_only_parameter_ratio": b_after / b_before,
        "validation_global_retained_activation_energy": weighted_retained / weighted_total,
        "validation_module_retained_activation_energy": {
            "mean": statistics.fmean(retention),
            "min": retention[0],
            "p10": retention[int(0.10 * (len(retention) - 1))],
            "median": statistics.median(retention),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Low-Order Rank Pruning Policy Distortion",
        "",
        f"Generated: `{summary['created_at_utc']}`",
        "",
        "## Integrity Checks",
        "",
        f"- Modules: `{summary['modules']}`",
        f"- Actual rank histogram: `{summary['actual_rank_histogram']}`",
        f"- Max FP64 canonical reconstruction relative error: "
        f"`{summary['integrity']['max_canonical_reconstruction_relative_error']:.3e}`",
        f"- Full-rank rewrite forward KL mean / max: "
        f"`{summary['full_rank_rewrite']['global']['forward_kl']['mean']:.3e}` / "
        f"`{summary['full_rank_rewrite']['global']['forward_kl']['max']:.3e}`",
        f"- Projected full-rank control KL mean / max: "
        f"`{summary['projected_full_rank_control']['global']['forward_kl']['mean']:.3e}` / "
        f"`{summary['projected_full_rank_control']['global']['forward_kl']['max']:.3e}`",
        "",
        "## Candidate Maps",
        "",
        "| Target | Mean rank | Adapter params | Held-out energy | Ideal KL mean / p99 | "
        "Ideal logp MAE | Ideal top-1 flip | Projected controlled KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in summary["candidates"].items():
        rank = item["map"]["rank"]
        ideal = item["ideal_policy_vs_original"]["global"]
        controlled = item["policy_vs_projected_full"]["global"]
        lines.append(
            f"| {item['target']:.0%} | {rank['mean']:.2f} | {item['map']['adapter_parameter_ratio']:.2%} | "
            f"{item['map']['validation_global_retained_activation_energy']:.4%} | "
            f"{ideal['forward_kl']['mean']:.3e} / {ideal['forward_kl']['p99']:.3e} | "
            f"{ideal['target_logprob_mae']:.3e} | {ideal['top1_flip_ratio']:.3%} | "
            f"{controlled['forward_kl']['mean']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Ranks are selected only from calibration response-token activations. Held-out energy and policy "
            "metrics use validation records. The highest energy target serves as the conservative control. "
            "Ideal policy metrics directly evaluate base(x) + top-k Delta-W(x), without rotating A/B. The "
            "projected control preserves checkpoint A and projects B; it measures factorized deployment "
            "behavior, not the optimization dynamics of frozen-A training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    targets = sorted({float(value) for value in args.targets.split(",") if value.strip()})
    rank_bins = sorted({int(value) for value in args.rank_bins.split(",") if value.strip()})
    benchmarks = [value.strip() for value in args.benchmarks.split(",") if value.strip()]
    if not targets or any(not 0 < value <= 1 for value in targets):
        raise ValueError(f"Targets must be in (0, 1], got {targets}")
    if not rank_bins or any(value <= 0 for value in rank_bins):
        raise ValueError(f"Rank bins must be positive, got {rank_bins}")
    if not benchmarks:
        raise ValueError("At least one benchmark is required")
    records_path = resolve_records_path(args.records)
    records = select_records(records_path, benchmarks, args.num_sequences, args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model.resolve(), trust_remote_code=args.trust_remote_code)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model.resolve(),
        dtype=dtypes[args.model_dtype],
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model = PeftModel.from_pretrained(base_model, args.adapter.resolve(), is_trainable=False).to(device)
    model.eval()
    causal_model = model.get_base_model()
    decoder = decoder_backbone(causal_model)
    modules = canonicalize_lora_modules(model)
    encoded = encode_records(records, tokenizer, args.max_seq_tokens, args.tokens_per_sequence)

    collector = ActivationCollector(modules, device, dtypes[args.projection_dtype])
    collect_activations(decoder, collector, encoded, device)
    covariances = {
        split: {name: tensor.double().cpu() for name, tensor in values.items()}
        for split, values in collector.covariances.items()
    }
    maps, module_rows = build_candidate_maps(modules, covariances, targets, rank_bins)
    rows_by_name = {str(row["module"]): row for row in module_rows}

    validation = [item for item in encoded if item.record.split == "validation"]
    baseline_logits = collect_policy_logits(causal_model, validation, device, "original_baseline")
    full_map = {name: item.actual_rank for name, item in modules.items()}
    write_rank_map(modules, full_map)
    full_logits = collect_policy_logits(causal_model, validation, device, "full_rank_rewrite")
    full_policy, sequence_rows = summarize_policy_comparison(
        validation,
        baseline_logits,
        full_logits,
        "full_rank_rewrite",
        "original_checkpoint",
    )
    del full_logits
    write_rank_map(modules, full_map, exact_full_restore=False)
    projected_full_logits = collect_policy_logits(causal_model, validation, device, "projected_full_rank_control")
    projected_full_policy, projected_full_rows = summarize_policy_comparison(
        validation,
        baseline_logits,
        projected_full_logits,
        "projected_full_rank_control",
        "original_checkpoint",
    )
    sequence_rows.extend(projected_full_rows)

    candidates = {}
    for target in targets:
        key = target_key(target)
        write_rank_map(modules, maps[key], exact_full_restore=False)
        candidate_logits = collect_policy_logits(causal_model, validation, device, key)
        policy, candidate_rows = summarize_policy_comparison(
            validation,
            baseline_logits,
            candidate_logits,
            key,
            "original_checkpoint",
        )
        controlled_policy, controlled_rows = summarize_policy_comparison(
            validation,
            projected_full_logits,
            candidate_logits,
            key,
            "projected_full_rank_control",
        )
        sequence_rows.extend(candidate_rows)
        sequence_rows.extend(controlled_rows)
        write_rank_map(modules, full_map)
        override = IdealTopKOverride(modules, maps[key])
        override.register()
        try:
            ideal_logits = collect_policy_logits(causal_model, validation, device, f"{key}_ideal")
        finally:
            override.close()
        ideal_policy, ideal_rows = summarize_policy_comparison(
            validation,
            baseline_logits,
            ideal_logits,
            f"{key}_ideal",
            "original_checkpoint",
        )
        sequence_rows.extend(ideal_rows)
        candidates[key] = {
            "target": target,
            "map": map_statistics(modules, maps[key], rows_by_name, key),
            "policy": policy,
            "policy_vs_projected_full": controlled_policy,
            "ideal_policy_vs_original": ideal_policy,
            "rank_pattern": maps[key],
        }
        del candidate_logits
        del ideal_logits
    write_rank_map(modules, full_map)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "module_pruning_diagnostics.csv", module_rows)
    write_csv(out_dir / "policy_distortion_by_sequence.csv", sequence_rows)
    write_csv(
        out_dir / "selected_records.csv",
        [
            {
                "split": item.record.split,
                "benchmark": item.record.benchmark,
                "problem_id": item.record.problem_id,
                "problem_index": item.record.problem_index,
                "sample_index": item.record.sample_index,
                "correct": item.record.correct,
                "sequence_tokens": int(item.input_ids.numel()),
                "sampled_target_tokens": int(item.target_ids.numel()),
            }
            for item in encoded
        ],
    )
    rank_histogram: dict[str, int] = defaultdict(int)
    for item in modules.values():
        rank_histogram[str(item.actual_rank)] += 1
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "records": str(records_path),
        "config": {
            "device": str(device),
            "model_dtype": args.model_dtype,
            "projection_dtype": args.projection_dtype,
            "attn_implementation": args.attn_implementation,
            "num_sequences": args.num_sequences,
            "max_seq_tokens": args.max_seq_tokens,
            "tokens_per_sequence": args.tokens_per_sequence,
            "targets": targets,
            "rank_bins": rank_bins,
            "seed": args.seed,
        },
        "modules": len(modules),
        "actual_rank_histogram": dict(sorted(rank_histogram.items(), key=lambda item: int(item[0]))),
        "integrity": {
            "max_canonical_reconstruction_relative_error": max(
                item.reconstruction_relative_error for item in modules.values()
            ),
        },
        "data": {
            "calibration_sequences": sum(item.record.split == "calibration" for item in encoded),
            "validation_sequences": len(validation),
            "calibration_sampled_tokens": sum(
                item.target_ids.numel() for item in encoded if item.record.split == "calibration"
            ),
            "validation_sampled_tokens": sum(item.target_ids.numel() for item in validation),
        },
        "full_rank_rewrite": full_policy,
        "projected_full_rank_control": projected_full_policy,
        "candidates": candidates,
        "definitions": {
            "forward_kl": "KL(pi_full_original || pi_truncated) at sampled response target positions",
            "controlled_forward_kl": (
                "KL(pi_projected_full || pi_projected_top_k); both policies use the same projected-B "
                "parameterization so the comparison isolates removed low-order components"
            ),
            "ideal_forward_kl": (
                "KL(pi_original || pi_ideal_top_k), where each pruned module directly computes "
                "base(x) + top-k Delta-W(x) and unpruned modules retain their original forward path"
            ),
            "rank_selection": (
                "smallest quantized singular-value prefix retaining calibration "
                "activation-weighted adapter-output energy"
            ),
            "heldout_energy": (
                "validation response-token adapter-output energy retained by the calibration-selected prefix"
            ),
            "offline_parameterization": (
                "checkpoint A is preserved and B is projected so B@A equals the top-k Delta-W; "
                "no frozen-A training claim is made"
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", summary)
    print(f"Wrote pruning policy diagnostic to {out_dir}")


if __name__ == "__main__":
    main()
