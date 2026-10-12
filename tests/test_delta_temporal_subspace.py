from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/analysis/analyze_delta_temporal_subspace.py"
SPEC = importlib.util.spec_from_file_location("analyze_delta_temporal_subspace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_module_function_gram_matches_dense_delta_w() -> None:
    torch.manual_seed(5)
    a = torch.randn(6, 3, 7)
    b = torch.randn(6, 5, 3)
    gram = MODULE.module_function_gram(a, b, 2.0)
    dense = torch.stack([2.0 * b[t] @ a[t] for t in range(6)]).flatten(1)
    assert torch.allclose(gram, dense @ dense.mT, rtol=1e-5, atol=2e-4)


def test_delta_gram_matches_dense_intervals() -> None:
    torch.manual_seed(7)
    dense = torch.randn(8, 13)
    interval = dense[3:] - dense[:-3]
    gram = MODULE.delta_gram(dense @ dense.mT, 3)
    assert torch.allclose(gram, interval @ interval.mT, rtol=1e-6, atol=1e-6)


def test_heldout_capture_is_one_for_shared_subspace() -> None:
    torch.manual_seed(9)
    coefficients = torch.randn(21, 3)
    basis, _ = torch.linalg.qr(torch.randn(17, 3))
    updates = coefficients @ basis.mT
    metrics = MODULE.heldout_metrics(updates @ updates.mT, horizon=1, cut=10)
    assert metrics is not None
    assert abs(metrics["future_capture_by_early_rank95"] - 1.0) < 1e-6
    assert abs(metrics["early_future_principal_overlap_95"] - 1.0) < 1e-6


def test_nonoverlap_summary_uses_stride_horizon() -> None:
    gram = torch.eye(10, dtype=torch.float64)
    row = MODULE.summarize_gram("S0", "all", "global_frobenius", 5, gram)
    assert row["nonoverlap_windows"] == 2
    assert row["nonoverlap_temporal_rank95"] == 2
