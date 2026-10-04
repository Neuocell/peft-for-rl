from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.analysis.audit_phase05_artifact import (
    FORMAL_CACHE_KEYS,
    audit_phase05_artifact,
)
from scripts.analysis.audit_phase06_artifact import audit_phase06_artifact
from scripts.analysis.build_full_gradient_uniform_allocation import (
    build_adaptive_allocation,
    build_uniform_allocation,
)
from scripts.analysis.evaluate_phase05_hybrid_gate import (
    evaluate as evaluate_phase05_gate,
)
from scripts.analysis.evaluate_phase06_crossfit_gate import (
    evaluate as evaluate_phase06_gate,
)
from scripts.analysis.prepare_phase05_training_artifacts import (
    prepare_phase05_training_artifacts,
    verify_prepared_phase05_allocation,
)
from scripts.analysis.prepare_phase06_training_artifacts import (
    prepare_phase06_training_artifacts,
    verify_prepared_phase06_allocation,
)
from scripts.analysis.prepare_phase1_signal_random_artifacts import (
    prepare_phase1_signal_random_artifacts,
    verify_prepared_phase1_allocation,
)
from scripts.analysis.record_phase06_invalidation import record_phase06_invalidation
from scripts.analysis.phase1_training_contract import (
    create_contract as create_phase1_training_contract,
    finalize_contract as finalize_phase1_training_contract,
    verify_contract as verify_phase1_training_contract,
)
from verl.utils.full_gradient_rl_probe import (
    FullGradientRLProbeCollector,
    Phase0GradientDiagnosticsCollector,
    Phase06CrossfitReplayCollector,
    WindowedAdamConsensusCollector,
    build_probe_token_mask,
    configure_full_gradient_probe_parameters,
    covariance_effective_ranks,
    covariance_sketch,
    deterministic_response_offset,
    deterministic_rollout_halves,
    principal_subspace_overlap,
    probe_token_mask_active,
    probe_token_keep_count,
    positive_spectrum_rank,
    signal_random_hybrid_basis,
    stability_supported_hybrid_basis,
    unbiased_single_response_scale,
    validate_full_gradient_probe_artifact,
    virtual_adam_update,
)
from verl.workers.actor.dp_actor import gradient_subspace_allocation_metrics
from verl.utils.peft_gradient_subspace import apply_gradient_subspace_initialization


class TinyPolicy(torch.nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(width, width, bias=False)
        layer.mlp = torch.nn.Module()
        layer.mlp.up_proj = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.model.layers[0].self_attn.q_proj(inputs)
        hidden = torch.tanh(self.model.layers[0].mlp.up_proj(hidden))
        return self.output(hidden)


def test_stability_supported_hybrid_ignores_zero_spectrum_completion() -> None:
    stable = torch.eye(5)[:4]
    stable_values = torch.tensor([4.0, 0.0, 0.0, 0.0])
    fallback = torch.eye(5)[2:5]
    fallback_values = torch.tensor([3.0, 2.0, 1.0])

    basis, values, selected = stability_supported_hybrid_basis(
        stable,
        stable_values,
        fallback,
        fallback_values,
        3,
        stable_directions=3,
        seed=11,
    )

    assert positive_spectrum_rank(stable_values) == 1
    assert selected == 1
    assert basis.shape == (3, 5)
    assert values.tolist() == pytest.approx([4.0, 3.0, 2.0])
    assert basis @ basis.T == pytest.approx(torch.eye(3), abs=1e-6)
    assert torch.linalg.vector_norm(basis @ stable[1]) == pytest.approx(0.0, abs=1e-6)


def test_stability_supported_hybrid_residualizes_fallback_deterministically() -> None:
    stable = torch.tensor([[1.0, 1.0, 0.0, 0.0]]) / 2**0.5
    fallback = torch.eye(4)[:3]
    stable_values = torch.tensor([2.0])
    fallback_values = torch.tensor([3.0, 2.0, 1.0])
    kwargs = {
        "rank": 3,
        "stable_directions": 1,
        "seed": 29,
    }

    first = stability_supported_hybrid_basis(
        stable, stable_values, fallback, fallback_values, **kwargs
    )
    second = stability_supported_hybrid_basis(
        stable, stable_values, fallback, fallback_values, **kwargs
    )
    basis, _, selected = first

    assert selected == 1
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert basis @ basis.T == pytest.approx(torch.eye(3), abs=1e-6)
    assert torch.linalg.vector_norm(basis @ stable[0]) == pytest.approx(1.0, abs=1e-6)
    expected_span = torch.cat((stable, fallback), dim=0)
    assert torch.linalg.matrix_rank(basis) == 3
    assert torch.linalg.matrix_rank(expected_span) == 3


def test_signal_random_hybrid_preserves_signal_and_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(5)
    signal = torch.linalg.qr(
        torch.randn((12, 8), generator=generator), mode="reduced"
    ).Q.T

    random_only = signal_random_hybrid_basis(signal, 8, signal_directions=0, seed=17)
    mixed = signal_random_hybrid_basis(signal, 8, signal_directions=4, seed=17)
    repeated = signal_random_hybrid_basis(signal, 8, signal_directions=4, seed=17)
    signal_only = signal_random_hybrid_basis(signal, 8, signal_directions=8, seed=17)

    for basis in (random_only, mixed, signal_only):
        assert basis @ basis.T == pytest.approx(torch.eye(8), abs=1e-5)
    assert torch.equal(mixed, repeated)
    assert torch.equal(mixed[:4], signal[:4])
    assert torch.equal(signal_only, signal)
    assert float((mixed[4:] @ signal[:4].T).abs().max()) < 1e-5


def test_generic_gradient_subspace_allocation_metrics_do_not_require_spar_structure() -> (
    None
):
    metrics = gradient_subspace_allocation_metrics(
        {
            "trainable_parameters": 1234,
            "rank_pattern": {"layer.a": 32, "layer.b": 32},
        }
    )

    assert metrics == {
        "actor/active_parameter_count": 1234.0,
        "gradient_subspace/rank_mean": 32.0,
        "gradient_subspace/rank_min": 32.0,
        "gradient_subspace/rank_max": 32.0,
    }


def test_phase1_training_contract_rejects_changed_adapter(tmp_path: Path) -> None:
    allocation_dir = tmp_path / "I8_uniform_r32"
    allocation_dir.mkdir()
    preparation_path = tmp_path / "phase1_training_preparation.json"
    preparation_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "source_validation_sha256": "source",
                "replay_validation_sha256": "replay",
                "phase05_gate_sha256": "gate05",
                "phase06_outcome_kind": "formal_no_go_gate",
                "phase06_outcome_path": "/tmp/phase06_gate_decision.json",
                "phase06_outcome_sha256": "gate06",
                "signal_candidate_sha256": "signal",
                "allocations": {
                    "I8": {
                        "rank_map_sha256": "rank-map",
                        "subspace_sha256": "subspace",
                        "allocation_summary_sha256": "allocation",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract_path = tmp_path / "contract.json"
    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_bytes(b"adapter-v1")
    created = create_phase1_training_contract(
        preparation_path, "I8", 42, "phase1_i8_seed42", contract_path
    )
    assert created["status"] == "prepared"
    assert created["code_provenance"]["algorithm"] == "sha256"
    assert created["code_provenance"]["file_count"] > 0
    assert len(created["code_provenance"]["aggregate_sha256"]) == 64
    completed = finalize_phase1_training_contract(contract_path, adapter_path)
    assert completed["status"] == "complete"
    verified = verify_phase1_training_contract(
        preparation_path,
        "I8",
        42,
        "phase1_i8_seed42",
        contract_path,
        adapter_path,
    )
    assert verified["status"] == "verified"

    tampered_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    tampered_contract["code_provenance"]["aggregate_sha256"] = "0" * 64
    contract_path.write_text(json.dumps(tampered_contract) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed: code_provenance"):
        verify_phase1_training_contract(
            preparation_path,
            "I8",
            42,
            "phase1_i8_seed42",
            contract_path,
            adapter_path,
        )
    contract_path.write_text(json.dumps(completed) + "\n", encoding="utf-8")

    adapter_path.write_bytes(b"adapter-v2")
    with pytest.raises(ValueError, match="adapter SHA-256 changed"):
        verify_phase1_training_contract(
            preparation_path,
            "I8",
            42,
            "phase1_i8_seed42",
            contract_path,
            adapter_path,
        )


def test_phase06_training_contract_locks_gate_and_allocation(tmp_path: Path) -> None:
    preparation_path = tmp_path / "phase06_training_preparation.json"
    preparation_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "source_validation_sha256": "source",
                "replay_validation_sha256": "replay",
                "gate_sha256": "gate06",
                "allocations": {
                    "C0": {
                        "rank_map_sha256": "rank-map",
                        "subspace_sha256": "subspace",
                        "allocation_summary_sha256": "allocation",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contract_path = tmp_path / "phase06_contract.json"
    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_bytes(b"phase06-adapter")
    created = create_phase1_training_contract(
        preparation_path,
        "C0",
        42,
        "phase06_c0_seed42",
        contract_path,
        preparation_kind="phase06",
    )
    assert created["method"] == "phase06_training_provenance_contract_v1"
    assert created["gate_sha256"] == "gate06"
    finalize_phase1_training_contract(contract_path, adapter_path)
    verified = verify_phase1_training_contract(
        preparation_path,
        "C0",
        42,
        "phase06_c0_seed42",
        contract_path,
        adapter_path,
        preparation_kind="phase06",
    )
    assert verified["preparation_kind"] == "phase06"

    tampered = json.loads(contract_path.read_text(encoding="utf-8"))
    tampered["gate_sha256"] = "another-gate"
    contract_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed: gate_sha256"):
        verify_phase1_training_contract(
            preparation_path,
            "C0",
            42,
            "phase06_c0_seed42",
            contract_path,
            adapter_path,
            preparation_kind="phase06",
        )


def test_phase05_gate_selects_on_calibration_then_checks_audit(tmp_path: Path) -> None:
    shared = {"module": torch.eye(2)}
    shared_scores = {"module.F": torch.tensor([1.0, 0.5])}
    for stem, tensors in (("candidates", shared), ("atom_scores", shared_scores)):
        save_file(tensors, tmp_path / f"{stem}_P2.safetensors")
        save_file(tensors, tmp_path / f"{stem}_H0.safetensors")
    diagnostics = {
        "P2": {"calibration_capture": 0.55, "audit_capture": 0.54},
        "H0": {
            "calibration_capture": 0.55,
            "audit_capture": 0.54,
            "prompt_split_overlap": {"mean_cosine": 0.72},
        },
    }
    for method, calibration, audit in (
        ("H2", 0.5520, 0.5415),
        ("H4", 0.5524, 0.5430),
        ("H8", 0.5510, 0.5420),
        ("H16", 0.54, 0.55),
    ):
        diagnostics[method] = {
            "calibration_capture": calibration,
            "audit_capture": audit,
            "prompt_split_overlap": {"mean_cosine": 0.72},
        }
    (tmp_path / "probe_summary.json").write_text(
        json.dumps({"diagnostics": diagnostics}), encoding="utf-8"
    )

    result = evaluate_phase05_gate(
        tmp_path,
        capture_tolerance=0.002,
        overlap_tolerance=0.02,
        tie_tolerance=5e-4,
        minimum_improvement=0.001,
    )

    assert result["calibration_selected_method"] == "H2"
    assert result["decision"] == "go"
    assert "H16" not in result["calibration_admissible_methods"]


def test_phase05_training_preparation_skips_preregistered_no_go(
    tmp_path: Path,
) -> None:
    gate = {
        "decision": "no-go",
        "calibration_selected_method": "H2",
    }
    (tmp_path / "phase05_gate_decision.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )

    result = prepare_phase05_training_artifacts(tmp_path)

    assert result["status"] == "skipped_no_go"
    assert result["allocations"] == {}
    persisted = json.loads(
        (tmp_path / "phase05_training_preparation.json").read_text(encoding="utf-8")
    )
    assert persisted == result


def test_phase05_training_preparation_builds_only_baseline_and_selected_hybrid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = {
        "decision": "go",
        "calibration_selected_method": "H4",
    }
    (tmp_path / "phase05_gate_decision.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )
    calls: list[tuple[str, int]] = []

    def fake_build(
        artifact_dir: Path,
        output_dir: Path,
        *,
        candidate_method: str,
        uniform_rank: int,
    ) -> dict:
        assert artifact_dir == tmp_path.resolve()
        calls.append((candidate_method, uniform_rank))
        output_dir.mkdir(parents=True)
        (output_dir / "rank_map.json").write_text("{}", encoding="utf-8")
        (output_dir / "subspaces.safetensors").write_bytes(b"subspace")
        (output_dir / "allocation_summary.json").write_text("{}", encoding="utf-8")
        return {
            "modules": {"module": {"rank": 32}},
            "trainable_parameters": 64,
        }

    monkeypatch.setattr(
        "scripts.analysis.prepare_phase05_training_artifacts.build_uniform_allocation",
        fake_build,
    )

    result = prepare_phase05_training_artifacts(tmp_path)

    assert calls == [("H0", 32), ("H4", 32)]
    assert result["status"] == "ready"
    assert set(result["allocations"]) == {"H0", "H4"}


def test_phase05_training_preparation_rejects_unknown_gate_state(
    tmp_path: Path,
) -> None:
    (tmp_path / "phase05_gate_decision.json").write_text(
        json.dumps({"decision": "pending"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unknown Phase-0.5 gate decision"):
        prepare_phase05_training_artifacts(tmp_path)


def test_phase05_prepared_allocation_is_revalidated_before_training(
    tmp_path: Path,
) -> None:
    gate_path = tmp_path / "phase05_gate_decision.json"
    gate_path.write_text(
        json.dumps({"decision": "go", "calibration_selected_method": "H2"}),
        encoding="utf-8",
    )
    (tmp_path / "artifact_validation.json").write_text(
        json.dumps({"status": "valid"}), encoding="utf-8"
    )
    allocation_dir = tmp_path / "training_allocations/H0_uniform_r32"
    allocation_dir.mkdir(parents=True)
    subspace_path = allocation_dir / "subspaces.safetensors"
    save_file({"module": torch.eye(2)}, subspace_path)
    rank_map_path = allocation_dir / "rank_map.json"
    rank_map_path.write_text(
        json.dumps(
            {
                "rank_pattern": {"module": 2},
                "alpha_pattern": {"module": 4},
                "constant_scaling": 2.0,
                "subspace_path": str(subspace_path.resolve()),
            }
        ),
        encoding="utf-8",
    )
    allocation_summary_path = allocation_dir / "allocation_summary.json"
    allocation_summary_path.write_text(
        json.dumps({"candidate_method": "H0", "trainable_parameters": 8}),
        encoding="utf-8",
    )

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    preparation = {
        "status": "ready",
        "gate_sha256": sha256(gate_path),
        "allocations": {
            "H0": {
                "rank_map_path": str(rank_map_path),
                "rank_map_sha256": sha256(rank_map_path),
                "subspace_path": str(subspace_path),
                "subspace_sha256": sha256(subspace_path),
                "allocation_summary_path": str(allocation_summary_path),
                "allocation_summary_sha256": sha256(allocation_summary_path),
                "trainable_parameters": 8,
            }
        },
    }
    (tmp_path / "phase05_training_preparation.json").write_text(
        json.dumps(preparation), encoding="utf-8"
    )

    result = verify_prepared_phase05_allocation(
        tmp_path, "H0", expected_modules=1, expected_rank=2
    )

    assert result["status"] == "verified"
    assert result["orthogonality_error_max"] == pytest.approx(0.0)


def test_phase06_gate_selects_on_calibration_and_keeps_audit_untouched(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    replay = tmp_path / "replay"
    source.mkdir()
    replay.mkdir()
    p3_candidates = {"module": torch.eye(3)[:2]}
    c3_candidates = {"module": torch.diag(torch.tensor([-1.0, 1.0, 1.0]))[:2]}
    scores = {"module.F": torch.tensor([0.6, 0.4])}
    save_file(p3_candidates, source / "candidates_P3.safetensors")
    save_file(c3_candidates, replay / "candidates_C3.safetensors")
    save_file(scores, source / "atom_scores_P3.safetensors")
    save_file(scores, replay / "atom_scores_C3.safetensors")
    baseline_overlap = {"mean_cosine": 0.72}
    source_summary = {
        "diagnostics": {
            "P0": {
                "calibration_capture": 0.55,
                "audit_capture": 0.54,
                "prompt_split_overlap": baseline_overlap,
            },
            "P2": {
                "calibration_capture": 0.56,
                "audit_capture": 0.55,
                "prompt_split_overlap": baseline_overlap,
            },
            "P3": {
                "calibration_capture": 0.40,
                "audit_capture": 0.39,
            },
        }
    }
    replay_summary = {
        "selection_uses_audit": False,
        "source_prompt_ids_sha256": "same",
        "replayed_prompt_ids_sha256": "same",
        "diagnostics": {
            "C0": {
                "calibration_capture": 0.5520,
                "audit_capture": 0.5415,
                "prompt_split_overlap": baseline_overlap,
            },
            "C2": {
                "calibration_capture": 0.5624,
                "audit_capture": 0.5480,
                "prompt_split_overlap": baseline_overlap,
            },
            "C3": {
                "calibration_capture": 0.40,
                "audit_capture": 0.39,
                "prompt_split_overlap": baseline_overlap,
            },
        },
    }
    (source / "probe_summary.json").write_text(
        json.dumps(source_summary), encoding="utf-8"
    )
    (replay / "probe_summary.json").write_text(
        json.dumps(replay_summary), encoding="utf-8"
    )

    result = evaluate_phase06_gate(source, replay)

    assert result["calibration_selected_method"] == "C2"
    assert result["decision"] == "no-go"
    assert result["candidates"]["C2"]["audit_delta_vs_standard"] < 0
    assert result["selection_uses_audit"] is False


def test_top_surprisal_token_mask_has_exact_budget_and_preserves_suffix() -> None:
    log_prob = torch.tensor([[-1.0, -5.0, -3.0, -2.0, -0.5, -0.25]], requires_grad=True)
    response_mask = torch.ones_like(log_prob)
    mask = build_probe_token_mask(
        log_prob,
        response_mask,
        mode="top_surprisal",
        keep_ratio=0.5,
        min_keep=0,
        final_tokens=2,
    )
    assert mask.tolist() == [[0.0, 1.0, 0.0, 0.0, 1.0, 1.0]]
    assert mask.requires_grad is False
    assert int(mask.sum().item()) == probe_token_keep_count(
        6, keep_ratio=0.5, min_keep=0, final_tokens=2
    )


def test_top_surprisal_token_mask_handles_padding_ties_and_minimum() -> None:
    log_prob = torch.full((2, 6), -2.0, requires_grad=True)
    response_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.float32
    )
    first = build_probe_token_mask(
        log_prob,
        response_mask,
        mode="top_surprisal",
        keep_ratio=0.25,
        min_keep=2,
        final_tokens=0,
    )
    second = build_probe_token_mask(
        log_prob,
        response_mask,
        mode="top_surprisal",
        keep_ratio=0.25,
        min_keep=2,
        final_tokens=0,
    )
    assert torch.equal(first, second)
    assert first.tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6]


def test_probe_token_mask_validates_configuration() -> None:
    with pytest.raises(ValueError, match="keep_ratio"):
        probe_token_keep_count(10, keep_ratio=0.0, min_keep=0, final_tokens=0)
    with pytest.raises(ValueError, match="Unknown"):
        build_probe_token_mask(
            torch.zeros(1, 2),
            torch.ones(1, 2),
            mode="unknown",
            keep_ratio=0.5,
            min_keep=0,
            final_tokens=0,
        )


def test_density_matched_random_mask_is_keyed_and_reproducible() -> None:
    log_prob = torch.zeros(1, 12)
    response_mask = torch.ones_like(log_prob)
    kwargs = {
        "mode": "random",
        "keep_ratio": 0.5,
        "min_keep": 0,
        "final_tokens": 0,
        "seed": 42,
    }
    first = build_probe_token_mask(
        log_prob, response_mask, sample_key="p0:r0", **kwargs
    )
    repeated = build_probe_token_mask(
        log_prob, response_mask, sample_key="p0:r0", **kwargs
    )
    other = build_probe_token_mask(
        log_prob, response_mask, sample_key="p0:r1", **kwargs
    )
    assert torch.equal(first, repeated)
    assert int(first.sum().item()) == 6
    assert not torch.equal(first, other)


def test_advantage_entropy_stable_band_excludes_extreme_surprisal() -> None:
    log_prob = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -100.0]])
    response_mask = torch.ones_like(log_prob)
    entropy = torch.tensor([[1.0, 5.0, 4.0, 3.0, 100.0]])
    advantages = torch.full_like(log_prob, 2.0)
    mask = build_probe_token_mask(
        log_prob,
        response_mask,
        mode="advantage_entropy_stable_band",
        keep_ratio=0.4,
        min_keep=0,
        final_tokens=0,
        entropy=entropy,
        advantages=advantages,
        surprisal_upper_quantile=0.8,
    )
    assert mask.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]


def test_crossfit_halves_and_overlap_diagnostics_are_deterministic() -> None:
    splits = deterministic_rollout_halves(42, "prompt-3", 8, 3)
    assert splits == deterministic_rollout_halves(42, "prompt-3", 8, 3)
    assert len(set(splits)) == 3
    for first, second in splits:
        assert len(first) == len(second) == 4
        assert set(first).isdisjoint(second)
        assert set(first) | set(second) == set(range(8))

    identity = torch.eye(4)[:2]
    orthogonal = torch.eye(4)[2:]
    same = principal_subspace_overlap(identity, identity)
    disjoint = principal_subspace_overlap(identity, orthogonal)
    assert same["mean_cosine"] == pytest.approx(1.0)
    assert disjoint["mean_cosine"] == pytest.approx(0.0)

    ranks = covariance_effective_ranks(torch.tensor([4.0, 1.0, 0.0]))
    assert ranks["stable_rank"] == pytest.approx(1.25)
    assert 1.0 < ranks["entropy_rank"] < 2.0
    assert ranks["positive_rank"] == 2


def test_probe_token_mask_scope_can_mask_selection_but_not_audit() -> None:
    assert probe_token_mask_active(
        mode="top_surprisal", scope="discovery_calibration", phase="discovery"
    )
    assert probe_token_mask_active(
        mode="top_surprisal", scope="discovery_calibration", phase="calibration"
    )
    assert not probe_token_mask_active(
        mode="top_surprisal", scope="discovery_calibration", phase="audit"
    )
    assert not probe_token_mask_active(
        mode="none", scope="discovery_calibration", phase="calibration"
    )


def test_uncentered_second_moment_retains_shared_gradient_direction() -> None:
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    omega = torch.eye(2)
    second_moment_y = 2 * gradient.T @ gradient @ omega
    centered = covariance_sketch(
        second_moment_y,
        gradient,
        omega,
        2,
        estimator="centered_population_covariance",
    )
    uncentered = covariance_sketch(
        second_moment_y,
        gradient,
        omega,
        2,
        estimator="uncentered_second_moment",
    )
    assert torch.allclose(centered, torch.zeros_like(centered))
    assert torch.allclose(uncentered, gradient.T @ gradient)


def _probe_config(path: Path) -> dict:
    return {
        "target_modules": "all-linear",
        "gradient_probe_output_dir": str(path),
        "gradient_probe_seed": 17,
        "full_gradient_probe_rank": 2,
        "full_gradient_probe_sketch_width": 4,
        "full_gradient_probe_svd_oversample": 2,
        "full_gradient_probe_svd_niter": 1,
        "full_gradient_probe_svd_device": "cpu",
        "full_gradient_probe_discovery_prompts": 2,
        "full_gradient_probe_calibration_prompts": 2,
        "full_gradient_probe_audit_prompts": 1,
        "full_gradient_probe_clip_factor": 2.5,
        "full_gradient_probe_confidence_z": 1.0,
        "full_gradient_probe_covariance_estimator": "centered_population_covariance",
        "full_gradient_probe_token_mask_scope": "legacy",
    }


def _windowed_probe_config(path: Path) -> dict:
    config = _probe_config(path)
    config.update(
        {
            "full_gradient_probe_mode": "windowed_adam_consensus",
            "full_gradient_probe_window_prompts": 2,
            "full_gradient_probe_discovery_windows": 3,
            "full_gradient_probe_calibration_windows": 2,
            "full_gradient_probe_audit_windows": 2,
            "full_gradient_probe_local_atoms": 1,
            "full_gradient_probe_adam_beta1": 0.5,
            "full_gradient_probe_adam_beta2": 0.75,
            "full_gradient_probe_adam_eps": 1e-6,
            "full_gradient_probe_future_lcb_z": 1.0,
        }
    )
    return config


def _phase0_probe_config(path: Path) -> dict:
    config = _probe_config(path)
    config.update(
        {
            "full_gradient_probe_mode": "phase0_diagnostics",
            "full_gradient_probe_crossfit_splits": 1,
            "full_gradient_probe_token_keep_ratio": 0.5,
            "full_gradient_probe_token_min_keep": 0,
            "full_gradient_probe_token_keep_final": 1,
            "full_gradient_probe_stable_surprisal_quantile": 0.9,
        }
    )
    return config


def test_phase0_collector_exports_required_diagnostics(tmp_path: Path) -> None:
    torch.manual_seed(23)
    model = TinyPolicy()
    config = _phase0_probe_config(tmp_path)
    configure_full_gradient_probe_parameters(model, config)
    collector = Phase0GradientDiagnosticsCollector(model, config)

    def cache_batch() -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.arange(12).reshape(2, 6),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
            "position_ids": torch.arange(6).repeat(2, 1),
            "responses": torch.arange(12).reshape(2, 6),
            "response_mask": torch.ones(2, 6),
            "advantages": torch.ones(2, 6),
            "old_log_probs": torch.zeros(2, 6),
            "token_level_scores": torch.ones(2, 6),
        }

    for prompt_index in range(2):
        prompt_id = f"discovery-{prompt_index}"
        collector.cache_prompt_group(
            phase="discovery", prompt_id=prompt_id, batch=cache_batch()
        )
        for selector in ("P0", "P1", "P2"):
            model.zero_grad(set_to_none=True)
            model(torch.randn(3, 6)).square().mean().backward()
            collector.capture_standard_discovery(selector)
        collector.begin_crossfit_prompt(prompt_id, 2)
        for response_offset in range(2):
            model.zero_grad(set_to_none=True)
            model(torch.randn(3, 6)).square().mean().backward()
            collector.capture_crossfit_response(response_offset, selected_tokens=3)
        collector.finish_crossfit_prompt()
        collector.complete_discovery_prompt(
            prompt_id=prompt_id,
            response_count=2,
            response_tokens=12,
            advantage_rms=1.0,
        )

    for split, count in (("calibration", 2), ("audit", 1)):
        for index in range(count):
            prompt_id = f"{split}-{index}"
            collector.cache_prompt_group(
                phase=split, prompt_id=prompt_id, batch=cache_batch()
            )
            model.zero_grad(set_to_none=True)
            model(torch.randn(3, 6)).square().mean().backward()
            collector.capture_held_out(
                split=split,
                prompt_id=prompt_id,
                response_count=2,
                response_tokens=12,
                advantage_rms=1.0,
            )

    assert collector.ready
    summary = json.loads((tmp_path / "probe_summary.json").read_text())
    assert summary["schema_version"] == 4
    assert summary["shared_rollouts_across_selectors"] is True
    assert summary["selection_uses_audit"] is False
    assert len(summary["rollout_cache"]) == 5
    assert Path(summary["rollout_cache"][0]["path"]).is_file()
    assert set(summary["selectors"]) == {"P0", "P1", "P2", "P3"}
    assert summary["uncentered_controls"] == [
        "P0_uncentered",
        "P1_uncentered",
        "P2_uncentered",
    ]
    assert summary["stability_supported_hybrids"]["methods"] == ["H0", "H2"]
    assert {"H0", "H2"}.issubset(summary["candidate_methods"])
    for method in summary["candidate_methods"]:
        item = summary["diagnostics"][method]
        assert torch.isfinite(torch.tensor(item["audit_capture"]))
        assert "capture_gap" in item
        assert "prompt_split_overlap" in item
        assert "effective_rank" in item
    assert "cross_half_overlap" in summary["diagnostics"]["P3"]
    assert "response_split_overlap" in summary["diagnostics"]["P3"]
    with (
        safe_open(
            tmp_path / "candidates_P2.safetensors", framework="pt", device="cpu"
        ) as p2_file,
        safe_open(
            tmp_path / "candidates_H0.safetensors", framework="pt", device="cpu"
        ) as h0_file,
    ):
        for name in p2_file.keys():
            assert torch.equal(p2_file.get_tensor(name), h0_file.get_tensor(name))
    validation = validate_full_gradient_probe_artifact(tmp_path, candidate_method="P3")
    assert validation["module_count"] == 2
    formal_validation = audit_phase05_artifact(
        tmp_path,
        expected_discovery=2,
        expected_calibration=2,
        expected_audit=1,
        expected_modules=2,
        expected_rank=2,
        expected_methods=tuple(summary["candidate_methods"]),
        required_cache_keys=FORMAL_CACHE_KEYS,
    )
    assert formal_validation["status"] == "valid"
    assert formal_validation["rollout_cache"]["files"] == 5
    phase0_uniform = build_uniform_allocation(
        tmp_path,
        tmp_path / "phase0-h0-uniform-r2",
        candidate_method="H0",
        uniform_rank=2,
    )
    assert phase0_uniform["candidate_method"] == "H0"
    assert {item["rank"] for item in phase0_uniform["modules"].values()} == {2}
    (tmp_path / "phase05_gate_decision.json").write_text(
        json.dumps({"decision": "no-go"}) + "\n", encoding="utf-8"
    )

    replay_dir = tmp_path / "phase06-replay"
    replay_config = _phase0_probe_config(replay_dir)
    replay_config.update(
        {
            "full_gradient_probe_mode": "phase06_crossfit_replay",
            "full_gradient_probe_replay_source_dir": str(tmp_path),
        }
    )
    replay = Phase06CrossfitReplayCollector(model, replay_config)
    for method in ("C0", "C2", "C3"):
        replay.start_replay_method(method)
        for prompt_index in range(2):
            prompt_id = f"discovery-{prompt_index}"
            replay.begin_crossfit_prompt(prompt_id, 2)
            for response_offset in range(2):
                model.zero_grad(set_to_none=True)
                inputs = torch.full((3, 6), 0.1 + prompt_index + response_offset)
                model(inputs).square().mean().backward()
                replay.capture_crossfit_response(response_offset, selected_tokens=3)
            replay.finish_crossfit_prompt()
            replay.complete_replay_discovery_prompt(
                prompt_id=prompt_id,
                response_count=2,
                response_tokens=12,
                advantage_rms=1.0,
            )
        replay.finish_replay_method()
        assert replay.cross == []

    replay.prepare_held_out_replay()
    for split, count in (("calibration", 2), ("audit", 1)):
        for index in range(count):
            model.zero_grad(set_to_none=True)
            model(torch.full((3, 6), 1.5 + index)).square().mean().backward()
            replay.capture_held_out(
                split=split,
                prompt_id=f"{split}-{index}",
                response_count=2,
                response_tokens=12,
                advantage_rms=1.0,
            )

    assert replay.ready
    replay_summary = json.loads(
        (replay_dir / "probe_summary.json").read_text(encoding="utf-8")
    )
    assert replay_summary["candidate_methods"] == ["C0", "C2", "C3"]
    assert (
        replay_summary["source_prompt_ids_sha256"]
        == replay_summary["replayed_prompt_ids_sha256"]
    )
    for method in replay_summary["candidate_methods"]:
        replay_validation = validate_full_gradient_probe_artifact(
            replay_dir, candidate_method=method
        )
        assert replay_validation["module_count"] == 2
    replay_audit = audit_phase06_artifact(
        tmp_path,
        replay_dir,
        expected_discovery=2,
        expected_calibration=2,
        expected_audit=1,
        expected_modules=2,
        expected_rank=2,
    )
    assert replay_audit["status"] == "valid"
    assert replay_audit["prompt_ids_exactly_replayed"] is True
    assert replay_audit["cache_manifest_exactly_replayed"] is True
    replay_summary_path = replay_dir / "probe_summary.json"
    tampered_replay_summary = dict(replay_summary)
    tampered_replay_summary["replay_method_masks"] = {
        **replay_summary["replay_method_masks"],
        "C2": "none",
    }
    replay_summary_path.write_text(
        json.dumps(tampered_replay_summary) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="method/mask mapping differs"):
        audit_phase06_artifact(
            tmp_path,
            replay_dir,
            expected_discovery=2,
            expected_calibration=2,
            expected_audit=1,
            expected_modules=2,
            expected_rank=2,
        )
    replay_summary_path.write_text(json.dumps(replay_summary) + "\n", encoding="utf-8")
    audit_phase06_artifact(
        tmp_path,
        replay_dir,
        expected_discovery=2,
        expected_calibration=2,
        expected_audit=1,
        expected_modules=2,
        expected_rank=2,
    )
    (replay_dir / "phase06_gate_decision.json").write_text(
        json.dumps({"decision": "go", "calibration_selected_method": "C0"}) + "\n",
        encoding="utf-8",
    )
    training_preparation = prepare_phase06_training_artifacts(
        tmp_path, replay_dir, uniform_rank=2
    )
    assert training_preparation["status"] == "ready"
    assert set(training_preparation["allocations"]) == {"P0", "C0"}
    for method in ("P0", "C0"):
        verified = verify_prepared_phase06_allocation(
            tmp_path,
            replay_dir,
            method,
            expected_modules=2,
            expected_rank=2,
        )
        assert verified["status"] == "verified"
        assert verified["matched_standard"] == "P0"
    (replay_dir / "phase06_gate_decision.json").write_text(
        json.dumps({"decision": "no-go"}) + "\n", encoding="utf-8"
    )
    phase1 = prepare_phase1_signal_random_artifacts(
        tmp_path,
        replay_dir,
        tmp_path / "phase1",
        rank=2,
        signal_counts=(0, 1, 2),
    )
    assert phase1["status"] == "ready"
    assert set(phase1["allocations"]) == {"I0", "I1", "I2"}
    for method in ("I0", "I1", "I2"):
        verified = verify_prepared_phase1_allocation(
            tmp_path / "phase1",
            method,
            expected_modules=2,
            expected_rank=2,
            expected_signal_counts=(0, 1, 2),
        )
        assert verified["status"] == "verified"
        assert verified["signal_directions"] == int(method[1:])
    with (
        safe_open(
            tmp_path / "candidates_P2.safetensors", framework="pt", device="cpu"
        ) as signal_file,
        safe_open(
            tmp_path / "phase1/I2_uniform_r2/subspaces.safetensors",
            framework="pt",
            device="cpu",
        ) as endpoint_file,
    ):
        for name in signal_file.keys():
            assert torch.equal(
                signal_file.get_tensor(name), endpoint_file.get_tensor(name)
            )

    invalid_replay = tmp_path / "phase06-invalid-replay"
    shutil.copytree(replay_dir, invalid_replay)
    (invalid_replay / "phase06_gate_decision.json").unlink()
    invalid_summary_path = invalid_replay / "probe_summary.json"
    invalid_summary = json.loads(invalid_summary_path.read_text(encoding="utf-8"))
    for method in ("C0", "C2"):
        invalid_summary["diagnostics"][method]["calibration_capture"] = 0.0
        invalid_summary["diagnostics"][method]["audit_capture"] = 0.0
        invalid_summary["diagnostics"][method]["prompt_split_overlap"][
            "mean_cosine"
        ] = 0.0
    invalid_summary_path.write_text(
        json.dumps(invalid_summary) + "\n", encoding="utf-8"
    )
    c3_path = invalid_replay / "candidates_C3.safetensors"
    with safe_open(c3_path, framework="pt", device="cpu") as c3_file:
        c3_tensors = {key: c3_file.get_tensor(key) for key in c3_file.keys()}
    first_key = sorted(c3_tensors)[0]
    c3_tensors[first_key] = c3_tensors[first_key].flip(0).contiguous()
    save_file(c3_tensors, c3_path)

    invalidation = record_phase06_invalidation(tmp_path, invalid_replay)
    assert invalidation["status"] == "invalid"
    assert invalidation["decision"] == "unavailable"
    assert invalidation["diagnosis"] == "crossfit_estimator_failure"
    assert invalidation["phase06_training_authorized"] is False
    assert invalidation["phase1_fallback"]["authorized"] is True
    invalid_phase1_root = tmp_path / "phase1-invalid-replay-fallback"
    invalid_phase1 = prepare_phase1_signal_random_artifacts(
        tmp_path,
        invalid_replay,
        invalid_phase1_root,
        rank=2,
        signal_counts=(0, 1, 2),
    )
    assert invalid_phase1["phase06_outcome_kind"] == "invalid_replay_fallback"
    verified = verify_prepared_phase1_allocation(
        invalid_phase1_root,
        "I1",
        expected_modules=2,
        expected_rank=2,
        expected_signal_counts=(0, 1, 2),
    )
    assert verified["status"] == "verified"
    invalidation_path = invalid_replay / "phase06_invalidation.json"
    tampered_invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    tampered_invalidation["phase1_fallback"]["authorized"] = False
    invalidation_path.write_text(
        json.dumps(tampered_invalidation) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after preparation"):
        verify_prepared_phase1_allocation(
            invalid_phase1_root,
            "I1",
            expected_modules=2,
            expected_rank=2,
            expected_signal_counts=(0, 1, 2),
        )

    preparation_path = tmp_path / "phase1/phase1_training_preparation.json"
    tampered = json.loads(preparation_path.read_text(encoding="utf-8"))
    tampered["complement_seed"] = 43
    preparation_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complement seed changed"):
        verify_prepared_phase1_allocation(
            tmp_path / "phase1",
            "I1",
            expected_modules=2,
            expected_rank=2,
            expected_signal_counts=(0, 1, 2),
        )


def test_single_response_estimator_is_unbiased_and_deterministic() -> None:
    response_losses = torch.tensor([-0.5, 0.25, 1.5, -0.75])
    full_group_loss = response_losses.sum()
    estimates = torch.stack(
        [
            response_losses[index]
            * unbiased_single_response_scale(len(response_losses), 1)
            for index in range(len(response_losses))
        ]
    )
    assert estimates.mean() == pytest.approx(float(full_group_loss))
    first = deterministic_response_offset(42, "prompt-7", 3, 8)
    assert first == deterministic_response_offset(42, "prompt-7", 3, 8)
    assert 0 <= first < 8


def test_virtual_adam_uses_bias_correction() -> None:
    first = torch.zeros(2)
    second = torch.zeros(2)
    gradient = torch.tensor([2.0, -4.0])
    update = virtual_adam_update(
        first,
        second,
        gradient,
        step=1,
        beta1=0.5,
        beta2=0.75,
        eps=0.0,
    )
    assert torch.allclose(update, torch.tensor([1.0, -1.0]))


def test_windowed_adam_consensus_exports_cross_fit_candidates(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = TinyPolicy()
    config = _windowed_probe_config(tmp_path)
    configure_full_gradient_probe_parameters(model, config)
    collector = WindowedAdamConsensusCollector(model, config)

    total_windows = 7
    for window in range(total_windows):
        model.zero_grad(set_to_none=True)
        inputs = torch.randn(4, 6) + window / 10
        loss = model(inputs).square().mean() * (1.0 + window / 5)
        loss.backward()
        collector.capture_window(
            loss=float(loss.item()),
            prompt_ids=[f"prompt-{window}-0", f"prompt-{window}-1"],
            response_counts=[8, 8],
            response_tokens=[24 + window, 28 + window],
            selected_response_offsets=[window % 8, (window + 1) % 8],
            advantage_rms=[0.5 + window, 0.75 + window],
        )

    assert collector.ready
    summary = json.loads((tmp_path / "probe_summary.json").read_text())
    assert summary["schema_version"] == 2
    assert summary["selection_uses_audit"] is False
    assert summary["held_out_scoring_space"] == "raw_full_policy_gradient"
    assert len(summary["discovery_prompt_ids"]) == 6
    assert len(summary["calibration_prompt_ids"]) == 4
    assert len(summary["audit_prompt_ids"]) == 4
    assert summary["prompt_splits_disjoint"] is True

    for method in ("raw_momentum", "adam_update", "consensus_hybrid"):
        validation = validate_full_gradient_probe_artifact(
            tmp_path, candidate_method=method
        )
        assert validation["module_count"] == 2
        with safe_open(
            tmp_path / f"atom_scores_{method}.safetensors",
            framework="pt",
            device="cpu",
        ) as score_file:
            for name in summary["modules"]:
                p_lcb = score_file.get_tensor(f"{name}.P_lcb")
                recurrence = score_file.get_tensor(f"{name}.recurrence")
                utility = score_file.get_tensor(f"{name}.U")
                assert torch.allclose(utility, recurrence * p_lcb.clamp_min(0))
                assert torch.isfinite(score_file.get_tensor(f"{name}.audit_U")).all()

    allocation = build_uniform_allocation(
        tmp_path,
        tmp_path / "windowed-allocation",
        candidate_method="consensus_hybrid",
        uniform_rank=1,
        selection_utility="future_lcb",
    )
    assert allocation["selection_utility"] == "future_lcb"
    assert allocation["constant_scaling"] == pytest.approx(2.0)


def test_full_gradient_probe_exports_three_cross_fit_candidate_sets(
    tmp_path: Path,
) -> None:
    torch.manual_seed(4)
    model = TinyPolicy()
    config = _probe_config(tmp_path)
    setup = configure_full_gradient_probe_parameters(model, config)
    assert setup["num_layers"] == 2
    assert model.output.weight.requires_grad is False
    collector = FullGradientRLProbeCollector(model, config)

    for index in range(5):
        model.zero_grad(set_to_none=True)
        inputs = torch.randn(4, 6) + index / 5
        signed_target = torch.randn(4, 6)
        loss = ((model(inputs) - signed_target) * (1 if index % 2 else -1)).mean()
        loss.backward()
        collector.capture_group(
            loss=float(loss.item()),
            prompt_id=f"prompt-{index}",
            response_count=2,
            response_tokens=20 + index,
            advantage_rms=0.5 + index,
        )

    assert collector.ready
    summary = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))
    assert summary["prompt_splits_disjoint"] is True
    assert summary["discovery_prompt_ids"] == ["prompt-0", "prompt-1"]
    assert summary["calibration_prompt_ids"] == ["prompt-2", "prompt-3"]
    assert summary["audit_prompt_ids"] == ["prompt-4"]
    assert summary["covariance_estimator"] == "centered_population_covariance"

    for method in ("mean", "covariance", "hybrid"):
        validation = validate_full_gradient_probe_artifact(
            tmp_path, candidate_method=method
        )
        assert validation["module_count"] == 2
        with safe_open(
            tmp_path / f"atom_scores_{method}.safetensors", framework="pt", device="cpu"
        ) as scores:
            name = next(iter(summary["modules"]))
            gain = scores.get_tensor(f"{name}.gain")
            gain_lcb = scores.get_tensor(f"{name}.gain_lcb")
            utility = scores.get_tensor(f"{name}.U")
            assert torch.allclose(utility, gain_lcb.clamp_min(0))
            assert torch.isfinite(gain).all()
            assert torch.isfinite(scores.get_tensor(f"{name}.audit_gain")).all()

    score_path = tmp_path / "atom_scores_mean.safetensors"
    with safe_open(score_path, framework="pt", device="cpu") as score_file:
        tensors = {key: score_file.get_tensor(key) for key in score_file.keys()}
    name = next(iter(summary["modules"]))
    tensors[f"{name}.audit_F"][0] = float("nan")
    save_file(tensors, score_path)
    with pytest.raises(ValueError, match="audit_F"):
        validate_full_gradient_probe_artifact(tmp_path, candidate_method="mean")


def test_full_gradient_uniform_allocation_uses_lcb_and_scaling_two(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    model = TinyPolicy()
    config = _probe_config(artifact)
    configure_full_gradient_probe_parameters(model, config)
    collector = FullGradientRLProbeCollector(model, config)
    torch.manual_seed(9)
    for index in range(5):
        model.zero_grad(set_to_none=True)
        output = model(torch.randn(3, 6))
        loss = output.square().mean() * (index + 1)
        loss.backward()
        collector.capture_group(
            loss=float(loss.item()),
            prompt_id=str(index),
            response_count=2,
            response_tokens=12,
            advantage_rms=1.0,
        )

    result = build_uniform_allocation(
        artifact,
        tmp_path / "allocation",
        candidate_method="covariance",
        uniform_rank=1,
    )
    assert result["candidate_method"] == "covariance"
    assert result["structure"]["active_rank_mean"] == pytest.approx(1.0)
    assert result["constant_scaling"] == pytest.approx(2.0)
    rank_map = json.loads(
        (tmp_path / "allocation/rank_map.json").read_text(encoding="utf-8")
    )
    assert all(rank == 1 for rank in rank_map["rank_pattern"].values())
    assert all(alpha == 2 for alpha in rank_map["alpha_pattern"].values())

    uniform_r2 = build_uniform_allocation(
        artifact,
        tmp_path / "uniform-r2",
        candidate_method="covariance",
        uniform_rank=2,
    )
    adaptive = build_adaptive_allocation(
        artifact,
        tmp_path / "adaptive",
        candidate_method="covariance",
        uniform_rank=2,
        r_min=1,
    )
    assert adaptive["allocation_mode"] == "adaptive"
    assert adaptive["budget_respected"] is True
    assert adaptive["trainable_parameters"] <= uniform_r2["trainable_parameters"]
    assert adaptive["constant_scaling"] == pytest.approx(2.0)
    assert all(item["rank"] >= 1 for item in adaptive["modules"].values())
    assert all(item["rank"] <= 4 for item in adaptive["modules"].values())
    assert sum(item["rank"] for item in adaptive["modules"].values()) == 4
    assert all(
        len(item["atom_indices"]) == item["rank"]
        for item in adaptive["modules"].values()
    )

    stable_energy = build_adaptive_allocation(
        artifact,
        tmp_path / "adaptive-stable-energy",
        candidate_method="covariance",
        uniform_rank=2,
        r_min=1,
        adaptive_utility="stable_energy",
    )
    assert stable_energy["adaptive_utility"] == "stable_energy"
    assert stable_energy["budget_respected"] is True
    assert stable_energy["trainable_parameters"] <= uniform_r2["trainable_parameters"]

    matched_uniform = build_uniform_allocation(
        artifact,
        tmp_path / "uniform-stable-energy",
        candidate_method="covariance",
        uniform_rank=1,
        selection_utility="stable_energy",
    )
    assert matched_uniform["selection_utility"] == "stable_energy"
    assert matched_uniform["trainable_parameters"] == result["trainable_parameters"]
    with safe_open(
        artifact / "atom_scores_covariance.safetensors", framework="pt", device="cpu"
    ) as score_file:
        for name, item in matched_uniform["modules"].items():
            utility = score_file.get_tensor(f"{name}.P") * score_file.get_tensor(
                f"{name}.R"
            )
            assert item["atom_indices"] == [int(utility.argmax().item())]


def test_full_gradient_probe_integration_is_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    actor = (root / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    worker = (root / "verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
    assert "_build_full_gradient_probe_batch" in trainer
    assert "run_full_gradient_probe" in actor
    assert 'self._peft_type == "full_gradient_probe"' in worker
    assert (root / "scripts/local/start_full_gradient_rl_probe_4gpu.sh").is_file()
    assert (root / "scripts/local/start_full_gradient_uniform_r8_4gpu.sh").is_file()
    assert (root / "scripts/local/start_full_gradient_adaptive_eqr8_4gpu.sh").is_file()
    phase06_launcher = root / "scripts/local/start_phase06_crossfit_replay_4gpu.sh"
    phase06_postprocess = root / "scripts/local/postprocess_phase06_crossfit_replay.sh"
    phase06_continuation = root / "scripts/local/continue_phase05_to_phase06.sh"
    phase06_train = root / "scripts/local/start_phase06_crossfit_uniform_r32_4gpu.sh"
    phase06_fullbench = root / "scripts/local/run_phase06_crossfit_step50_fullbench.sh"
    assert phase06_launcher.is_file()
    assert phase06_postprocess.is_file()
    assert phase06_continuation.is_file()
    assert phase06_train.is_file()
    assert phase06_fullbench.is_file()
    assert (
        "FULL_GRADIENT_PROBE_MODE=phase06_crossfit_replay"
        in phase06_launcher.read_text()
    )
    assert "evaluate_phase06_crossfit_gate.py" in phase06_postprocess.read_text()
    assert "phase05_gate_decision.json" in phase06_continuation.read_text()
    assert 'probe_mode == "phase06_crossfit_replay"' in trainer
    phase06_train_text = phase06_train.read_text()
    for fixed_setting in (
        "export LR=1e-6",
        "export LR_WARMUP_STEPS=0",
        "export LR_SCHEDULER_TYPE=constant",
        "export WEIGHT_DECAY=0",
        "export PPO_EPOCHS=1",
        "export CLIP_RATIO_LOW=0.2",
        "export CLIP_RATIO_HIGH=0.28",
        "export USE_KL_LOSS=False",
        "export TEMPERATURE=1.0",
        "export TOP_P=1.0",
        "export TOP_K=-1",
        "export TRAIN_SHUFFLE=True",
        "export ACTOR_SHUFFLE=False",
    ):
        assert fixed_setting in phase06_train_text
    phase06_pipeline = phase06_fullbench.read_text()
    for contract_action in ("create", "finalize", "verify"):
        assert f"phase1_training_contract.py {contract_action}" in phase06_pipeline
    assert "--preparation-kind phase06" in phase06_pipeline
    assert "BENCHMARK_SNAPSHOT_SHA256" in phase06_pipeline
    assert "EVAL_RECORDS" in phase06_pipeline
    assert "verify_full_bench_run.py" in phase06_pipeline
    for eval_setting in (
        "SEED=42",
        "TEMPERATURE=0.6",
        "TOP_P=0.95",
        "MAX_NEW_TOKENS=32768",
        "MAX_MODEL_LEN=34816",
        "SAMPLES_SMALL=32",
        "SAMPLES_LARGE=4",
    ):
        assert eval_setting in phase06_pipeline
    phase1_train = root / "scripts/local/start_phase1_signal_random_uniform_r32_4gpu.sh"
    phase1_fullbench = (
        root / "scripts/local/run_phase1_signal_random_step50_fullbench.sh"
    )
    phase1_confirmation = (
        root / "scripts/local/run_phase1_signal_random_confirmation.sh"
    )
    phase1_contract = root / "scripts/analysis/phase1_training_contract.py"
    assert phase1_train.is_file()
    assert phase1_fullbench.is_file()
    assert phase1_confirmation.is_file()
    assert phase1_contract.is_file()
    assert "--verify-method" in phase1_train.read_text()
    phase1_launcher = phase1_train.read_text()
    for fixed_setting in (
        "export LR=1e-6",
        "export PPO_EPOCHS=1",
        "export TEMPERATURE=1.0",
        "export TOP_P=1.0",
        "export TOP_K=-1",
    ):
        assert fixed_setting in phase1_launcher
    assert (
        "/tmp/ray-p1-${METHOD_LOWER}-s${TRAIN_SEED}-r${PHASE1_RUN_REVISION}"
        in phase1_launcher
    )
    phase1_pipeline = phase1_fullbench.read_text()
    for contract_action in ("create", "finalize", "verify"):
        assert f"phase1_training_contract.py {contract_action}" in phase1_pipeline
    assert "BENCHMARK_SNAPSHOT_SHA256" in phase1_pipeline
    assert "--preparation-kind phase1" in phase1_pipeline
    assert "verify_full_bench_run.py" in phase1_pipeline
    assert "select_phase1_seed42_candidate.py" in phase1_confirmation.read_text()
    assert "aggregate_multiseed_full_bench.py" in phase1_confirmation.read_text()
    assert "flock -n 9" in phase1_confirmation.read_text()
    windowed_probe = root / "scripts/local/start_windowed_adam_consensus_probe_4gpu.sh"
    windowed_train = (
        root / "scripts/local/start_windowed_consensus_uniform_r8_50_4gpu.sh"
    )
    assert windowed_probe.is_file()
    assert windowed_train.is_file()
    probe_launcher = windowed_probe.read_text()
    train_launcher = windowed_train.read_text()
    for launcher in (probe_launcher, train_launcher):
        assert "export TRAIN_PROMPT_BSZ=64" in launcher
        assert "export TRAIN_PROMPT_MINI_BSZ=16" in launcher
        assert "export N_RESP_PER_PROMPT=8" in launcher
    assert "FULL_GRADIENT_PROBE_MODE=windowed_adam_consensus" in probe_launcher
    assert "export LORA_RANK=8" in train_launcher
    assert "export LORA_ALPHA=16" in train_launcher
    uniform_launcher = (
        root / "scripts/local/start_full_gradient_uniform_r8_4gpu.sh"
    ).read_text()
    assert 'export LORA_RANK="${LORA_RANK:-8}"' in uniform_launcher
    assert 'export LORA_ALPHA="${LORA_ALPHA:-16}"' in uniform_launcher
    adaptive_launcher = (
        root / "scripts/local/start_full_gradient_adaptive_eqr8_4gpu.sh"
    ).read_text()
    assert "export LORA_RANK=32" in adaptive_launcher
    assert "export LORA_ALPHA=64" in adaptive_launcher

    uncentered_launcher = (
        root / "scripts/local/run_token_mask_uncentered_covariance_adaptive_270.sh"
    ).read_text()
    assert (
        "FULL_GRADIENT_PROBE_COVARIANCE_ESTIMATOR=uncentered_second_moment"
        in uncentered_launcher
    )
    assert (
        "FULL_GRADIENT_PROBE_TOKEN_MASK_SCOPE=discovery_calibration"
        in uncentered_launcher
    )
    assert "TOTAL_TRAINING_STEPS=270" in uncentered_launcher
    assert "SAVE_CONTENTS=\"['model','optimizer','extra']\"" in uncentered_launcher
    assert "RESUME_MODE=auto" in uncentered_launcher


def test_heterogeneous_probe_lora_saves_restores_and_merges(tmp_path: Path) -> None:
    torch.manual_seed(13)
    base = TinyPolicy()
    restored_base = copy.deepcopy(base)
    model = get_peft_model(
        base,
        LoraConfig(
            r=1,
            lora_alpha=2,
            target_modules=["q_proj", "up_proj"],
            rank_pattern={"q_proj": 1, "up_proj": 2},
            alpha_pattern={"q_proj": 2, "up_proj": 4},
        ),
    )
    subspace_path = tmp_path / "subspaces.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj": torch.eye(6)[:1].contiguous(),
            "model.layers.0.mlp.up_proj": torch.eye(6)[:2].contiguous(),
        },
        subspace_path,
    )
    apply_gradient_subspace_initialization(
        model, SimpleNamespace(gradient_subspace_path=str(subspace_path))
    )
    inputs = torch.randn(3, 6)
    with model.disable_adapter():
        base_output = model(inputs).detach()
    assert torch.equal(model(inputs), base_output)
    layers = [module for module in model.modules() if hasattr(module, "lora_A")]
    assert sorted(module.r["default"] for module in layers) == [1, 2]
    assert all(module.scaling["default"] == 2 for module in layers)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    model(inputs).square().mean().backward()
    optimizer.step()
    trained_output = model(inputs).detach()
    assert not torch.equal(trained_output, base_output)

    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir)
    restored = PeftModel.from_pretrained(restored_base, adapter_dir)
    assert torch.allclose(restored(inputs), trained_output, atol=1e-6)
    merged = restored.merge_and_unload()
    assert torch.allclose(merged(inputs), trained_output, atol=1e-6)
