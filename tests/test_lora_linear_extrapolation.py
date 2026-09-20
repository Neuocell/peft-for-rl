from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/analysis/diagnose_lora_linear_extrapolation.py"
SPEC = importlib.util.spec_from_file_location("diagnose_lora_linear_extrapolation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def dense(terms, scale: float) -> torch.Tensor:
    return scale * sum(coefficient * b @ a for coefficient, a, b in terms)


def test_factored_linear_combination_matches_dense_matrix() -> None:
    torch.manual_seed(0)
    terms = [
        (0.5, torch.randn(2, 5), torch.randn(4, 2)),
        (-1.25, torch.randn(3, 5), torch.randn(4, 3)),
    ]
    factored = MODULE.linear_combination(terms, scale=2.0)
    expected = dense(terms, scale=2.0)

    assert torch.allclose(factored.left @ factored.right, expected)
    assert MODULE.squared_norm(factored) == pytest.approx(expected.square().sum().item())


def test_constant_velocity_prediction_has_zero_error_on_linear_path() -> None:
    torch.manual_seed(1)
    a = torch.randn(2, 5, dtype=torch.float64)
    b0 = torch.randn(4, 2, dtype=torch.float64)
    velocity = torch.randn(4, 2, dtype=torch.float64)
    factors = {
        10: (a, b0 + 1.0 * velocity),
        20: (a, b0 + 2.0 * velocity),
        35: (a, b0 + 3.5 * velocity),
    }
    ratio = (35 - 20) / (20 - 10)
    a_from, b_from = factors[10]
    a_anchor, b_anchor = factors[20]
    a_target, b_target = factors[35]
    error = MODULE.linear_combination(
        [
            (ratio, a_from, b_from),
            (-(1.0 + ratio), a_anchor, b_anchor),
            (1.0, a_target, b_target),
        ],
        scale=2.0,
    )

    assert MODULE.squared_norm(error) == pytest.approx(0.0, abs=1e-12)


def test_compact_svd_matches_dense_singular_values() -> None:
    torch.manual_seed(2)
    terms = [
        (1.0, torch.randn(2, 6, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)),
        (-1.0, torch.randn(2, 6, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)),
    ]
    factored = MODULE.linear_combination(terms, scale=1.5)
    actual = MODULE.compact_svd(factored).singular
    expected = torch.linalg.svdvals(dense(terms, scale=1.5))

    assert actual.tolist() == pytest.approx(expected[: actual.numel()].tolist(), abs=1e-10)
    assert expected[actual.numel() :].square().sum().item() == pytest.approx(0.0, abs=1e-20)


def test_radial_aggregation_recovers_linear_growth_from_base() -> None:
    row = {
        "from_step": 10,
        "anchor_step": 20,
        "target_step": 30,
        "from_sq": 1.0,
        "anchor_sq": 4.0,
        "target_sq": 9.0,
        "anchor_target_inner": 6.0,
        "previous_move_sq": 1.0,
        "next_move_sq": 1.0,
        "move_inner": 1.0,
        "prediction_error_sq": 0.0,
        "predicted_target_sq": 9.0,
        "predicted_target_inner": 9.0,
        "schedule_ratio": 1.0,
    }

    result = MODULE.aggregate_group([row], "all")

    assert result["origin_schedule_factor"] == pytest.approx(1.5)
    assert result["origin_schedule_error_over_target"] == pytest.approx(0.0)
    assert result["norm_trend_factor"] == pytest.approx(1.5)
    assert result["norm_trend_error_over_target"] == pytest.approx(0.0)
    assert result["oracle_radial_factor"] == pytest.approx(1.5)
    assert result["oracle_radial_error_over_target"] == pytest.approx(0.0)
