from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.local import real_experiment_eval as evaluation


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_controller_loads_exact_support_and_restores_import_state(tmp_path, monkeypatch):
    support = tmp_path / "step150_control.py"
    support.write_text("VALUE = 'pinned'\n")
    controller = tmp_path / "evaluate.py"
    controller.write_text("from step150_control import VALUE\nNAME = 'baseline'\n")
    other = SimpleNamespace(VALUE="wrong preexisting module")
    monkeypatch.setitem(sys.modules, "step150_control", other)
    original_path = sys.path.copy()
    loaded = evaluation._load_controller({
        "eval_controller_path": str(controller), "eval_controller_sha256": _digest(controller),
        "eval_support_path": str(support), "eval_support_sha256": _digest(support),
        "experiment": "real",
    })
    assert loaded.VALUE == "pinned"
    assert loaded.NAME == "real"
    assert sys.modules["step150_control"] is other
    assert sys.path == original_path


@pytest.mark.parametrize("changed", ["controller", "support"])
def test_changed_pinned_code_is_rejected_before_execution(tmp_path, changed):
    support = tmp_path / "step150_control.py"
    support.write_text("raise RuntimeError('must not execute')\n")
    controller = tmp_path / "evaluate.py"
    controller.write_text("raise RuntimeError('must not execute')\n")
    plan = {
        "eval_controller_path": str(controller), "eval_controller_sha256": _digest(controller),
        "eval_support_path": str(support), "eval_support_sha256": _digest(support), "experiment": "real",
    }
    (controller if changed == "controller" else support).write_text("# modified\n")
    with pytest.raises(ValueError, match="Pinned evaluation code changed"):
        evaluation._load_controller(plan)


def _adapter(tmp_path: Path) -> tuple[Path, str]:
    adapter = tmp_path / "checkpoint/global_step_50/actor/peft_adapter"
    adapter.mkdir(parents=True)
    base_model = str(tmp_path / "base")
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 32, "lora_alpha": 64, "lora_dropout": 0.0,
        "base_model_name_or_path": base_model, "rank_pattern": {"q_proj": 32},
    }))
    save_file({
        "base_model.model.q_proj.lora_A.weight": np.zeros((32, 48), dtype=np.float32),
        "base_model.model.q_proj.lora_B.weight": np.zeros((64, 32), dtype=np.float32),
    }, adapter / "adapter_model.safetensors")
    return adapter, base_model


def test_adapter_requires_paired_frozen_basis_export(tmp_path):
    adapter, base_model = _adapter(tmp_path)
    evaluation._check_adapter(adapter, base_model)
    save_file({"base_model.model.q_proj.lora_B.weight": np.zeros((64, 32), dtype=np.float32)},
              adapter / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="complete, paired A and B"):
        evaluation._check_adapter(adapter, base_model)


def test_adapter_rejects_hidden_rank_override(tmp_path):
    adapter, base_model = _adapter(tmp_path)
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["rank_pattern"]["q_proj"] = 16
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="nonuniform rank_pattern"):
        evaluation._check_adapter(adapter, base_model)


def test_unverified_baseline_refuses_launch(tmp_path, monkeypatch):
    adapter, base_model = _adapter(tmp_path)
    Path(base_model).mkdir()
    weights = Path(base_model) / "model.safetensors"
    weights.write_bytes(b"base model fixture")
    snapshot = tmp_path / "snapshot.jsonl"
    snapshot.write_text("snapshot fixture")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "verification.json").write_text(json.dumps({"status": "running", "num_samples": 7248}))
    monkeypatch.setattr(evaluation, "_load_controller", lambda plan: SimpleNamespace())
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="has not completed verification"):
        evaluation.evaluate_real({
            "target_step": 50, "base_model": base_model, "base_model_weights_sha256": _digest(weights),
            "snapshot": str(snapshot), "snapshot_sha256": _digest(snapshot),
            "baseline_eval_dir": str(baseline),
        }, lambda *args, **kwargs: None)


def test_preflight_manifest_failure_never_starts_gpu_evaluation(tmp_path, monkeypatch):
    adapter, base_model = _adapter(tmp_path)
    Path(base_model).mkdir()
    weights = Path(base_model) / "model.safetensors"
    weights.write_bytes(b"base model fixture")
    snapshot = tmp_path / "snapshot.jsonl"
    snapshot.write_text("snapshot fixture")
    calls = []

    def invalid_manifests(*args):
        raise ValueError("preflight does not match benchmark protocol")

    controller = SimpleNamespace(
        write_json=lambda path, value: Path(path).write_text(json.dumps(value)),
        environment=lambda plan, job, dry: {"DRY_RUN_NO_MODEL": "1" if dry else "0"},
        check_manifests=invalid_manifests,
        run_job=lambda *args: pytest.fail("must not launch model after bad preflight"),
    )
    monkeypatch.setattr(evaluation, "_load_controller", lambda plan: controller)
    monkeypatch.setattr(evaluation, "_verify_baseline", lambda *args: ({}, tmp_path / "baseline.jsonl"))
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *args, **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="preflight does not match"):
        evaluation.evaluate_real({
            "target_step": 50, "base_model": base_model, "base_model_weights_sha256": _digest(weights),
            "snapshot": str(snapshot), "snapshot_sha256": _digest(snapshot), "root": str(tmp_path),
            "checkpoint_root": str(tmp_path / "checkpoint"), "eval_output_dir": str(tmp_path / "eval"),
            "job_dir": str(tmp_path / "job"), "experiment": "real_test",
        }, lambda *args, **kwargs: None)
    assert len(calls) == 1
    assert calls[0]["env"]["DRY_RUN_NO_MODEL"] == "1"


def test_completed_evaluation_writes_real_specific_paired_report(tmp_path, monkeypatch):
    adapter, base_model = _adapter(tmp_path)
    Path(base_model).mkdir()
    weights = Path(base_model) / "model.safetensors"
    weights.write_bytes(b"base model fixture")
    snapshot = tmp_path / "snapshot.jsonl"
    snapshot.write_text("snapshot fixture")
    baseline_name = "standard_lora_step50_fullbench_32768_seed42"
    sizes = {"aime24": 30, "aime25": 30, "amc23": 40, "hmmt_feb": 30, "math500": 500, "minerva": 272}

    def records(name, candidate):
        rows = []
        for benchmark, size in sizes.items():
            count = 4 if benchmark in {"math500", "minerva"} else 32
            for problem in range(size):
                for sample in range(count):
                    rows.append(json.dumps({
                        "checkpoint": name, "benchmark": benchmark, "problem_index": problem,
                        "sample_index": sample, "id": str(problem), "problem": f"Question {problem}",
                        "gold_answer": "1", "samples_per_problem": count,
                        "correct": candidate or sample % 2 == 0,
                    }) + "\n")
        return "".join(rows)

    def summary(name, accuracy):
        return {
            "checkpoint": name, "num_samples": 7248, "num_benchmarks": 6,
            "macro_avg_at_k_over_benchmarks": accuracy, "macro_pass_at_k_over_benchmarks": 1.0,
            "parse_rate_overall": 1.0, "hit_max_rate_overall": 0.0, "length_mean_overall": 100.0,
            "benchmarks": [{"benchmark": name, "macro_avg_at_k": accuracy, "pass_at_k": 1.0,
                            "parse_rate": 1.0, "hit_max_rate": 0.0, "length_mean": 100.0}
                           for name in sizes],
        }

    baseline_path = tmp_path / "baseline_records.jsonl"
    baseline_path.write_text(records(baseline_name, False))
    baseline_summary = summary(baseline_name, 0.5)
    calls = []

    def run_job(plan, job, status):
        calls.append("model_evaluation")
        output = Path(job["output"])
        (output / "records").mkdir(parents=True)
        (output / "records" / f"{job['evaluation_name']}.jsonl").write_text(records(job["evaluation_name"], True))
        (output / "verification.json").write_text(json.dumps({"status": "verified", "num_samples": 7248}))
        return summary(job["evaluation_name"], 1.0)

    controller = SimpleNamespace(
        write_json=lambda path, value: Path(path).write_text(json.dumps(value)),
        environment=lambda plan, job, dry: {"DRY_RUN_NO_MODEL": "1" if dry else "0"},
        check_manifests=lambda *args: calls.append("preflight_verified"),
        run_job=run_job,
    )
    monkeypatch.setattr(evaluation, "_load_controller", lambda plan: controller)
    monkeypatch.setattr(evaluation, "_verify_baseline", lambda *args: (baseline_summary, baseline_path))
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *args, **kwargs: calls.append("preflight"))
    result = evaluation.evaluate_real({
        "target_step": 50, "base_model": base_model, "base_model_weights_sha256": _digest(weights),
        "snapshot": str(snapshot), "snapshot_sha256": _digest(snapshot),
        "root": str(Path(__file__).resolve().parents[1]), "checkpoint_root": str(tmp_path / "checkpoint"),
        "eval_output_dir": str(tmp_path / "eval"), "job_dir": str(tmp_path / "job"), "experiment": "real_test",
    }, lambda *args, **kwargs: None)
    assert calls == ["preflight", "preflight_verified", "model_evaluation"]
    assert result["status"] == "verified_real_step50_comparison"
    assert result["delta_macro_avg_at_k"] == pytest.approx(0.5)
    report = json.loads(Path(result["paired_report"]).read_text())
    assert report["num_samples"] == 7248
    assert report["bootstrap_samples"] == 10_000
    assert report["record_keys_exactly_matched"] is True
    metrics = json.loads(Path(result["summary_comparison"]).read_text())
    assert metrics["interpretation"] == "single_seed_screen_only"
    assert "next_experiment" not in metrics
    assert "heldout_gate" not in metrics
