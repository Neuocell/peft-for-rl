from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/analysis/analyze_delta_stability.py"
SPEC = importlib.util.spec_from_file_location("analyze_delta_stability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_compact_svd_reconstructs_dense_delta() -> None:
    torch.manual_seed(11)
    a0, a1 = torch.randn(3, 7), torch.randn(3, 7)
    b0, b1 = torch.randn(5, 3), torch.randn(5, 3)
    spectrum = MODULE.compact_svd(a0, b0, a1, b1, 2.0)
    reconstructed = (spectrum.left.double() * spectrum.singular) @ spectrum.right.double().mT
    expected = 2.0 * (b1.double() @ a1.double() - b0.double() @ a0.double())
    assert torch.allclose(reconstructed, expected, rtol=1e-5, atol=1e-6)


def test_rank_and_effective_rank() -> None:
    singular = torch.tensor([3.0, 1.0])
    assert MODULE.rank_at_energy(singular, 0.90) == 1
    assert MODULE.rank_at_energy(singular, 0.95) == 2
    assert 1.0 < MODULE.effective_rank(singular) < 2.0


def test_rank_at_energy_sorts_concatenated_module_spectra() -> None:
    singular = torch.tensor([3.0, 1.0, 4.0, 2.0])
    assert MODULE.rank_at_energy(singular, 0.80) == 2


def test_identical_subspaces_have_perfect_cross_capture() -> None:
    torch.manual_seed(3)
    left, _ = torch.linalg.qr(torch.randn(8, 3))
    right, _ = torch.linalg.qr(torch.randn(9, 3))
    spectrum = MODULE.Spectrum(torch.tensor([4.0, 2.0, 1.0]), left, right)
    metrics = MODULE.cross_capture(spectrum, spectrum, 0.95)
    assert abs(metrics["left_overlap"] - 1.0) < 1e-6
    assert abs(metrics["right_overlap"] - 1.0) < 1e-6
    expected_capture = 20.0 / 21.0
    assert all(abs(metrics[key] - expected_capture) < 1e-6 for key in ("left", "right", "full"))


def test_activation_capture_identical_subspaces_is_one() -> None:
    torch.manual_seed(5)
    left, _ = torch.linalg.qr(torch.randn(8, 3))
    right, _ = torch.linalg.qr(torch.randn(9, 3))
    spectrum = MODULE.Spectrum(torch.tensor([4.0, 2.0, 1.0]), left, right)
    metrics = MODULE.activation_capture(torch.randn(21, 9), spectrum, spectrum, 1.0)
    assert abs(metrics["left"] - 1.0) < 1e-6
    assert abs(metrics["right"] - 1.0) < 1e-6
    assert abs(metrics["full"] - 1.0) < 1e-6
