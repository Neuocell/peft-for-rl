from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/analysis/diagnose_activation_weighted_rank.py"
SPEC = importlib.util.spec_from_file_location("diagnose_activation_weighted_rank", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_activation_energy_matches_dense_delta_output() -> None:
    torch.manual_seed(7)
    a = torch.randn(6, 11, dtype=torch.float64)
    b = torch.randn(13, 6, dtype=torch.float64)
    x = torch.randn(37, 11, dtype=torch.float64)

    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.mT, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(r_b @ r_a.mT, full_matrices=False)
    u = q_b @ u_core
    v = q_a @ vh_core.mT
    projected = x @ v
    component_energy = singular.square() * projected.square().sum(dim=0)

    dense = x @ (b @ a).mT
    assert torch.allclose(component_energy.sum(), dense.square().sum(), rtol=1e-10, atol=1e-10)
    k = 3
    truncated = (u[:, :k] * singular[:k]) @ v[:, :k].mT
    retained = x @ truncated.mT
    assert torch.allclose(component_energy[:k].sum(), retained.square().sum(), rtol=1e-10, atol=1e-10)


def test_rank_selection_quantization_and_retention() -> None:
    energy = torch.tensor([6.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    assert MODULE.energy_rank(energy, 0.80) == 2
    order = torch.tensor([1, 0, 2, 3])
    assert MODULE.energy_rank(energy, 0.80, order) == 2
    assert MODULE.quantize_rank(5, [4, 8, 12], 12) == 8
    assert MODULE.quantize_rank(12, [4, 8], 12) == 12
    assert MODULE.retained_energy(energy, torch.tensor([0, 2])) == 0.7


def test_sample_positions_is_bounded_and_deterministic() -> None:
    positions = MODULE.sample_positions(10, 110, 8)
    assert positions.tolist() == sorted(set(positions.tolist()))
    assert len(positions) == 8
    assert int(positions.min()) >= 10
    assert int(positions.max()) < 110


def test_parameter_matched_fixed_rank_uses_module_widths_and_lower_tie() -> None:
    rows = [
        {"d_in": 10, "d_out": 20},
        {"d_in": 30, "d_out": 40},
    ]
    assert MODULE.parameter_matched_fixed_rank(rows, 1800, [8, 16, 20, 24, 32]) == (16, 1600)
    assert MODULE.parameter_matched_fixed_rank(rows, 1800, [16, 20]) == (16, 1600)


def test_rank_map_exports_constant_scaling_and_oracle_components() -> None:
    rows = [
        {"adapter_key": "layer.q_proj", "rank": 2, "indices": "3,1"},
        {"adapter_key": "layer.v_proj", "rank": 3, "indices": "0,2,1"},
    ]
    rank_map = MODULE.make_rank_map(
        rows,
        rank_field="rank",
        scale=2.0,
        source="test",
        component_field="indices",
    )
    assert rank_map["rank_pattern"] == {"layer.q_proj": 2, "layer.v_proj": 3}
    assert rank_map["alpha_pattern"] == {"layer.q_proj": 4.0, "layer.v_proj": 6.0}
    assert rank_map["component_pattern"] == {
        "layer.q_proj": [3, 1],
        "layer.v_proj": [0, 2, 1],
    }


def test_aggregate_compares_all_maps_and_fixed_rank() -> None:
    rows = []
    for index, calibration_rank in enumerate((8, 16)):
        rows.append(
            {
                "adapter_key": f"layer.{index}.q_proj",
                "module_type": "q_proj",
                "layer_band": "early",
                "d_in": 10,
                "d_out": 10,
                "activation_validation_total_energy": 1.0,
                "activation_calibration_rank_95_quantized": calibration_rank,
                "activation_validation_rank_95_quantized": calibration_rank,
                "activation_calibration_oracle_rank_95_quantized": 8,
                "frobenius_rank_95_quantized": 16,
                "activation_validation_retained_by_calibration_95_quantized": 0.95,
                "activation_validation_retained_by_calibration_oracle_95_quantized": 0.96,
                "activation_validation_retained_by_frobenius_95_quantized": 0.98,
                "activation_validation_prefix_energy_at_8": 0.85,
                "activation_validation_prefix_energy_at_16": 0.97,
                "activation_validation_prefix_energy_at_32": 1.0,
            }
        )
    summary = MODULE.aggregate(rows, "95", configured_rank=32, scale=2.0, rank_bins=[8, 16, 32])
    assert set(summary["map_comparison"]) == {
        "activation_prefix",
        "activation_oracle",
        "frobenius_prefix",
    }
    assert summary["matched_fixed_rank"]["rank"] == 8
    assert summary["map_comparison"]["activation_prefix"]["rank"]["mean"] == 12
    assert summary["map_comparison"]["activation_oracle"]["parameters"] == 320
    assert summary["map_comparison"]["frobenius_prefix"]["theoretical_lora_flops_per_token"] == 1280
