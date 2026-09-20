from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/analysis/build_lora_chord_adapter.py"
SPEC = importlib.util.spec_from_file_location("build_lora_chord_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("lambda_value", [0.0, 0.5, 1.0, 1.25])
def test_projection_matches_exact_shared_anchor_chord(lambda_value: float) -> None:
    torch.manual_seed(0)
    anchor, _ = torch.linalg.qr(torch.randn(7, 3), mode="reduced")
    anchor = anchor.mT.contiguous()
    from_b = torch.randn(5, 3)
    to_b = torch.randn(5, 3)

    chord_b, residual_sq, desired_sq = MODULE.project_chord_to_anchor(
        anchor,
        from_b,
        anchor,
        to_b,
        anchor,
        lambda_value,
    )
    actual = chord_b @ anchor.double()
    expected = (1.0 - lambda_value) * (from_b @ anchor) + lambda_value * (to_b @ anchor)

    assert torch.allclose(actual, expected.to(dtype=actual.dtype), atol=1e-10)
    assert residual_sq == pytest.approx(0.0, abs=1e-12)
    assert desired_sq == pytest.approx(expected.double().square().sum().item())


def test_projection_reports_off_anchor_residual() -> None:
    torch.manual_seed(1)
    anchor, _ = torch.linalg.qr(torch.randn(8, 2), mode="reduced")
    anchor = anchor.mT.contiguous()
    from_a = anchor + 0.05 * torch.randn_like(anchor)
    to_a = anchor + 0.05 * torch.randn_like(anchor)
    from_b = torch.randn(4, 2)
    to_b = torch.randn(4, 2)

    _, residual_sq, desired_sq = MODULE.project_chord_to_anchor(
        from_a,
        from_b,
        to_a,
        to_b,
        anchor,
        0.5,
    )

    assert residual_sq > 0
    assert desired_sq > residual_sq
