from __future__ import annotations

import json
from pathlib import Path

import pytest

from verl.utils.peft_oracle_lora import load_oracle_lora_patterns

REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_CONFIG = REPO_ROOT / "examples/verl_train/config/oracle_justrl_stable_rank_ceil_1p5b.json"


def test_checked_in_oracle_rank_config_has_full_coverage_and_expected_budget():
    patterns = load_oracle_lora_patterns(ORACLE_CONFIG, base_rank=44, base_alpha=88)

    assert patterns.module_count == 28 * 7
    assert patterns.rank_sum == 3111
    assert patterns.rank_mean == pytest.approx(15.872448979591837)
    assert (patterns.rank_min, patterns.rank_max) == (3, 44)
    assert patterns.scaling_ratio == 2.0
    assert patterns.rank_pattern["model.layers.0.self_attn.q_proj"] == 12
    assert patterns.alpha_pattern["model.layers.4.mlp.gate_proj"] == 88
    assert patterns.rank_pattern["model.layers.27.mlp.down_proj"] == 3


def test_oracle_rank_config_rejects_rollout_scaling_mismatch():
    with pytest.raises(ValueError, match="Global LoRA scaling must match"):
        load_oracle_lora_patterns(ORACLE_CONFIG, base_rank=44, base_alpha=64)


def test_oracle_rank_config_rejects_incomplete_layer(tmp_path: Path):
    config = json.loads(ORACLE_CONFIG.read_text(encoding="utf-8"))
    del config["ranks_by_layer"]["27"]["down_proj"]
    bad_config = tmp_path / "bad.json"
    bad_config.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Layer 27 must contain exactly"):
        load_oracle_lora_patterns(bad_config, base_rank=44, base_alpha=88)


def test_d1_explicit_rank_map_is_normalized_and_validated(tmp_path: Path):
    config = {
        "source": "D1 activation prefix",
        "constant_scaling": 2.0,
        "rank_pattern": {
            "base_model.model.model.layers.0.self_attn.q_proj": 8,
            "base_model.model.model.layers.0.mlp.up_proj": 12,
            "base_model.model.model.layers.1.self_attn.q_proj": 16,
            "base_model.model.model.layers.1.mlp.up_proj": 20,
        },
        "alpha_pattern": {
            "base_model.model.model.layers.0.self_attn.q_proj": 16.0,
            "base_model.model.model.layers.0.mlp.up_proj": 24.0,
            "base_model.model.model.layers.1.self_attn.q_proj": 32.0,
            "base_model.model.model.layers.1.mlp.up_proj": 40.0,
        },
    }
    path = tmp_path / "d1-map.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    patterns = load_oracle_lora_patterns(path, base_rank=32, base_alpha=64)
    assert patterns.rank_pattern == {
        "model.layers.0.self_attn.q_proj": 8,
        "model.layers.0.mlp.up_proj": 12,
        "model.layers.1.self_attn.q_proj": 16,
        "model.layers.1.mlp.up_proj": 20,
    }
    assert patterns.alpha_pattern["model.layers.1.mlp.up_proj"] == 40
    assert patterns.rank_mean == 14
    assert patterns.scaling_ratio == 2.0


def test_oracle_launcher_stays_in_review_mode_by_default():
    launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_oracle_rank_lora_1p5b_4gpu_8k.sh").read_text(
        encoding="utf-8"
    )
    base_launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh").read_text(
        encoding="utf-8"
    )

    assert 'export DRY_RUN="${DRY_RUN:-1}"' in launcher
    assert 'export LORA_RANK="${LORA_RANK:-44}"' in launcher
    assert 'export LORA_ALPHA="${LORA_ALPHA:-88}"' in launcher
    assert "actor_rollout_ref.model.lora_rank_pattern_path=${LORA_RANK_PATTERN_PATH:-null}" in base_launcher
