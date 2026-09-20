import math
import sys
from pathlib import Path

import torch


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from diagnose_lora_direction_by_distance import (  # noqa: E402
    chord_metrics,
    chord_vector,
    radial_metrics,
)


def test_radial_metrics_detect_pure_scaling() -> None:
    gram = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    metrics = radial_metrics(gram, 0, 1)

    assert math.isclose(metrics["cosine"], 1.0)
    assert math.isclose(metrics["norm_ratio"], 2.0)
    assert math.isclose(metrics["oracle_radial_factor"], 2.0)
    assert math.isclose(metrics["oracle_radial_residual_over_late"], 0.0)


def test_radial_metrics_separate_rotation_from_growth() -> None:
    # Early=(1, 0), late=(1, 1): the best scalar preserves only one component.
    gram = torch.tensor([[1.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    metrics = radial_metrics(gram, 0, 1)

    assert math.isclose(metrics["cosine"], 1.0 / math.sqrt(2.0))
    assert math.isclose(metrics["oracle_radial_factor"], 1.0)
    assert math.isclose(
        metrics["oracle_radial_residual_over_late"], 1.0 / math.sqrt(2.0)
    )


def test_chord_metrics_use_functional_differences() -> None:
    # Three mutually orthogonal checkpoint updates. The two adjacent chords
    # have inner product -1 and squared norm 2.
    gram = torch.eye(3, dtype=torch.float64)
    left = chord_vector(3, 0, 1)
    right = chord_vector(3, 1, 2)
    metrics = chord_metrics(gram, left, right)

    assert math.isclose(metrics["cosine"], -0.5)
    assert math.isclose(metrics["best_scalar_left_to_right"], -0.5)
    assert math.isclose(
        metrics["best_scalar_residual_over_right"], math.sqrt(3.0) / 2.0
    )
