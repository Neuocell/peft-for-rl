from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/analysis/diagnose_rlpo_init_subspace.py"
SPEC = importlib.util.spec_from_file_location("diagnose_rlpo_init_subspace", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_base_weight_key_matches_hf_checkpoint_names() -> None:
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert MODULE.base_weight_key(key) == "model.layers.0.self_attn.q_proj.weight"


def test_subspace_metrics_distinguish_containment_and_rotation() -> None:
    reference = torch.eye(4)[:, :2]
    contained = torch.eye(4)[:, :1]
    orthogonal = torch.eye(4)[:, 2:3]
    assert MODULE.subspace_metrics(reference, contained)["overlap"] == pytest.approx(1.0)
    assert MODULE.subspace_metrics(reference, orthogonal)["overlap"] == pytest.approx(0.0)


def test_compact_delta_right_space_and_energy_capture() -> None:
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    delta = MODULE.compact_delta_svd(a, b, scale=2.0)
    initial = torch.eye(3)[:, :2]
    assert delta.singular.tolist() == pytest.approx([4.0, 2.0])
    assert MODULE.energy_capture(initial, delta) == pytest.approx(1.0)
    assert MODULE.energy_capture(initial[:, :1], delta) == pytest.approx(0.8)
    assert MODULE.subspace_metrics(initial, delta.right)["overlap"] == pytest.approx(1.0)


def test_normalized_a_orthogonality_error() -> None:
    assert MODULE.a_orthogonality_error(torch.eye(3)) == pytest.approx(0.0)
    assert MODULE.a_orthogonality_error(2.0 * torch.eye(3)) == pytest.approx(3.0)


def test_delta_path_decomposition_uses_initial_a_and_exact_function_update() -> None:
    a_initial = torch.eye(2)
    a = torch.tensor([[1.1, 0.0], [0.0, 0.9]])
    b = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    metrics = MODULE.delta_path_decomposition(a, b, a_initial, scale=2.0)
    dense_full = 2.0 * b @ a
    dense_initial_path = 2.0 * b @ a_initial
    dense_drift_path = 2.0 * b @ (a - a_initial)
    assert torch.allclose(dense_full, dense_initial_path + dense_drift_path)
    assert metrics["delta_from_initial_a_fro"] == pytest.approx(dense_initial_path.norm().item())
    assert metrics["delta_from_a_drift_fro"] == pytest.approx(dense_drift_path.norm().item())
    assert metrics["delta_from_a_drift_relative_fro"] == pytest.approx(
        dense_drift_path.norm().item() / dense_full.norm().item()
    )
