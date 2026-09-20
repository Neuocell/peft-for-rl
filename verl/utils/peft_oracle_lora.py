"""Load and validate per-layer LoRA ranks derived from a reference update."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ATTENTION_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_EXPLICIT_KEY_RE = re.compile(r"(model\.layers\.(\d+)\.(self_attn|mlp)\.([^.]+))$")


@dataclass(frozen=True)
class OracleLoraPatterns:
    rank_pattern: dict[str, int]
    alpha_pattern: dict[str, int]
    rank_sum: int
    rank_min: int
    rank_max: int
    scaling_ratio: float

    @property
    def module_count(self) -> int:
        return len(self.rank_pattern)

    @property
    def rank_mean(self) -> float:
        return self.rank_sum / self.module_count


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return value


def _validate_global_scaling(
    scaling_ratio: Any, *, base_rank: int, base_alpha: int, source: Path
) -> tuple[float, int, int]:
    if (
        isinstance(scaling_ratio, bool)
        or not isinstance(scaling_ratio, (int, float))
        or scaling_ratio <= 0
    ):
        raise ValueError(f"LoRA scaling ratio must be positive in {source}, got {scaling_ratio!r}")
    ratio = float(scaling_ratio)
    checked_rank = _positive_int(base_rank, field="base_rank")
    checked_alpha = _positive_int(base_alpha, field="base_alpha")
    global_ratio = checked_alpha / checked_rank
    if not math.isclose(global_ratio, ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Global LoRA scaling must match rank-pattern scaling for vLLM: "
            f"base_alpha/base_rank={global_ratio:g}, pattern={ratio:g}"
        )
    return ratio, checked_rank, checked_alpha


def _load_explicit_patterns(
    raw: dict[str, Any], *, path: Path, base_rank: int, base_alpha: int
) -> OracleLoraPatterns:
    """Validate explicit PEFT patterns, including D1 diagnostic output."""

    raw_ranks = raw.get("rank_pattern")
    raw_alphas = raw.get("alpha_pattern")
    if not isinstance(raw_ranks, dict) or not raw_ranks:
        raise ValueError(f"rank_pattern must be a non-empty object in {path}")
    if not isinstance(raw_alphas, dict) or set(raw_alphas) != set(raw_ranks):
        raise ValueError(f"alpha_pattern must cover exactly the rank_pattern keys in {path}")
    scaling_ratio, base_rank, _ = _validate_global_scaling(
        raw.get("constant_scaling"),
        base_rank=base_rank,
        base_alpha=base_alpha,
        source=path,
    )

    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    coverage: dict[int, set[str]] = {}
    for source_key, raw_rank in raw_ranks.items():
        if not isinstance(source_key, str):
            raise ValueError(f"rank_pattern keys must be strings in {path}")
        match = _EXPLICIT_KEY_RE.search(source_key)
        if match is None:
            raise ValueError(f"Unsupported explicit rank-pattern key in {path}: {source_key!r}")
        key, layer_text, block, module = match.groups()
        if key in rank_pattern:
            raise ValueError(f"Duplicate normalized rank-pattern key in {path}: {key}")
        expected_block = "self_attn" if module in _ATTENTION_MODULES else "mlp"
        if block != expected_block:
            raise ValueError(f"Module {module} must be under {expected_block}, got {block}")
        rank = _positive_int(raw_rank, field=f"rank_pattern.{source_key}")
        raw_alpha = raw_alphas[source_key]
        if isinstance(raw_alpha, bool) or not isinstance(raw_alpha, (int, float)):
            raise ValueError(f"alpha_pattern.{source_key} must be numeric, got {raw_alpha!r}")
        alpha = round(float(raw_alpha))
        if not math.isclose(float(raw_alpha), alpha, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"alpha_pattern.{source_key} must be integral, got {raw_alpha!r}")
        if not math.isclose(alpha / rank, scaling_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Per-module scaling mismatch for {source_key}: alpha/r={alpha / rank:g}, "
                f"expected {scaling_ratio:g}"
            )
        rank_pattern[key] = rank
        alpha_pattern[key] = alpha
        coverage.setdefault(int(layer_text), set()).add(module)

    layers = sorted(coverage)
    if layers != list(range(layers[-1] + 1)):
        raise ValueError(f"Explicit rank pattern layers must be contiguous from zero, got {layers}")
    expected_modules = coverage[0]
    for layer in layers:
        if coverage[layer] != expected_modules:
            raise ValueError(
                f"Explicit rank pattern layer {layer} coverage differs from layer 0: "
                f"expected={sorted(expected_modules)}, actual={sorted(coverage[layer])}"
            )
    ranks = list(rank_pattern.values())
    if base_rank < max(ranks):
        raise ValueError(f"Global lora_rank={base_rank} is below pattern maximum rank={max(ranks)}")
    return OracleLoraPatterns(
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        rank_sum=sum(ranks),
        rank_min=min(ranks),
        rank_max=max(ranks),
        scaling_ratio=scaling_ratio,
    )


def load_oracle_lora_patterns(
    config_path: str | Path,
    *,
    base_rank: int,
    base_alpha: int,
) -> OracleLoraPatterns:
    """Turn the compact layer/module table into PEFT rank/alpha patterns.

    vLLM 0.11 applies the global ``lora_alpha / r`` scaling to every loaded
    tensor and ignores PEFT's alpha_pattern. Requiring the global and per-module
    ratios to agree keeps actor and rollout logits equivalent.
    """

    path = Path(config_path).expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Oracle LoRA rank config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Oracle LoRA JSON at {path}: {exc}") from exc

    if "rank_pattern" in raw or "alpha_pattern" in raw:
        return _load_explicit_patterns(raw, path=path, base_rank=base_rank, base_alpha=base_alpha)
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported oracle rank schema_version in {path}: {raw.get('schema_version')!r}")

    layer_count = _positive_int(raw.get("model", {}).get("num_hidden_layers"), field="model.num_hidden_layers")
    modules = raw.get("model", {}).get("target_modules")
    if not isinstance(modules, list) or not modules or any(not isinstance(item, str) for item in modules):
        raise ValueError("model.target_modules must be a non-empty list of module names")
    if len(modules) != len(set(modules)):
        raise ValueError("model.target_modules contains duplicates")

    scaling_ratio, base_rank, _ = _validate_global_scaling(
        raw.get("lora_scaling_ratio"),
        base_rank=base_rank,
        base_alpha=base_alpha,
        source=path,
    )

    ranks_by_layer = raw.get("ranks_by_layer")
    if not isinstance(ranks_by_layer, dict):
        raise ValueError("ranks_by_layer must be an object")
    expected_layers = {str(layer) for layer in range(layer_count)}
    actual_layers = set(ranks_by_layer)
    if actual_layers != expected_layers:
        missing = sorted(expected_layers - actual_layers, key=int)
        extra = sorted(actual_layers - expected_layers)
        raise ValueError(f"ranks_by_layer coverage mismatch: missing={missing}, extra={extra}")

    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    for layer in range(layer_count):
        layer_ranks = ranks_by_layer[str(layer)]
        if not isinstance(layer_ranks, dict) or set(layer_ranks) != set(modules):
            raise ValueError(f"Layer {layer} must contain exactly these modules: {modules}")
        for module in modules:
            rank = _positive_int(layer_ranks[module], field=f"ranks_by_layer.{layer}.{module}")
            alpha_float = rank * scaling_ratio
            alpha = round(alpha_float)
            if not math.isclose(alpha_float, alpha, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Layer {layer} module {module} gives non-integral lora_alpha={alpha_float}; "
                    "PEFT LoraConfig requires integer alpha values"
                )
            block = "self_attn" if module in _ATTENTION_MODULES else "mlp"
            key = f"model.layers.{layer}.{block}.{module}"
            rank_pattern[key] = rank
            alpha_pattern[key] = alpha

    ranks = list(rank_pattern.values())
    rank_max = max(ranks)
    if base_rank < rank_max:
        raise ValueError(f"Global lora_rank={base_rank} is below oracle maximum rank={rank_max}")

    return OracleLoraPatterns(
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        rank_sum=sum(ranks),
        rank_min=min(ranks),
        rank_max=rank_max,
        scaling_ratio=scaling_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize an oracle LoRA rank config")
    parser.add_argument("config", type=Path)
    parser.add_argument("--base-rank", type=int, required=True)
    parser.add_argument("--base-alpha", type=int, required=True)
    args = parser.parse_args()
    patterns = load_oracle_lora_patterns(args.config, base_rank=args.base_rank, base_alpha=args.base_alpha)
    print(
        "Oracle LoRA rank config validated: "
        f"modules={patterns.module_count}, rank_sum={patterns.rank_sum}, "
        f"rank_mean={patterns.rank_mean:.4f}, rank_range=[{patterns.rank_min}, {patterns.rank_max}], "
        f"alpha/r={patterns.scaling_ratio:g}"
    )


if __name__ == "__main__":
    main()
