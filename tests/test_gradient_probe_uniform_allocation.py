from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.analysis.build_gradient_probe_uniform_allocation import (
    build_gradient_probe_uniform_allocation,
)


def test_legacy_gradient_probe_can_be_truncated_to_equal_rank(tmp_path: Path) -> None:
    artifact = tmp_path / "probe"
    artifact.mkdir()
    bases = {
        "model.layers.0.self_attn.q_proj": torch.eye(5)[:4],
        "model.layers.0.mlp.up_proj": torch.linalg.qr(torch.randn(7, 4)).Q.T,
    }
    save_file(bases, artifact / "subspaces.safetensors")
    modules = {
        name: {
            "selected_rank": 4,
            "parameter_cost_per_rank": 11 + index,
        }
        for index, name in enumerate(bases)
    }
    (artifact / "summary.json").write_text(
        json.dumps(
            {
                "probe_method": "energy",
                "constant_scaling": 2.0,
                "modules": modules,
            }
        ),
        encoding="utf-8",
    )

    result = build_gradient_probe_uniform_allocation(
        artifact, tmp_path / "uniform", uniform_rank=2
    )
    assert result["trainable_parameters"] == 2 * (11 + 12)
    assert result["active_rank_mean"] == pytest.approx(2.0)
    assert result["constant_scaling"] == pytest.approx(2.0)
    assert all(alpha == 4 for alpha in result["alpha_pattern"].values())
    assert result["structure"] == {
        "active_rank_mean": pytest.approx(2.0),
        "active_rank_min": 2,
        "active_rank_max": 2,
        "rank_by_family": {"q_proj": pytest.approx(2.0), "up_proj": pytest.approx(2.0)},
        "rank_by_layer_segment": {"low": pytest.approx(2.0)},
        "calibration_energy_capture": pytest.approx(0.0),
        "u_score_capture": pytest.approx(0.0),
        "capture_metrics_available": False,
    }
    written = json.loads(
        (tmp_path / "uniform/allocation_summary.json").read_text(encoding="utf-8")
    )
    assert written["structure"]["active_rank_mean"] == pytest.approx(2.0)
    with safe_open(tmp_path / "uniform/subspaces.safetensors", framework="pt") as tensors:
        assert all(tuple(tensors.get_tensor(name).shape) == (2, bases[name].shape[1]) for name in bases)


def test_legacy_uniform_builder_rejects_artifact_without_parameter_costs(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "probe"
    artifact.mkdir()
    name = "model.layers.0.self_attn.q_proj"
    save_file({name: torch.eye(3)}, artifact / "subspaces.safetensors")
    (artifact / "summary.json").write_text(
        json.dumps(
            {
                "probe_method": "energy",
                "constant_scaling": 2.0,
                "modules": {name: {"selected_rank": 3}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parameter_cost_per_rank"):
        build_gradient_probe_uniform_allocation(
            artifact, tmp_path / "uniform", uniform_rank=2
        )
