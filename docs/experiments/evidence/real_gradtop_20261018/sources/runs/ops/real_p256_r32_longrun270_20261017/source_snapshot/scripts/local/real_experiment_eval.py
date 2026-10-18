#!/usr/bin/env python3
"""Evaluate a completed REAL step-50 checkpoint against the matched LoRA run.

The queue controller calls ``evaluate_real(plan, status)`` only after training
has exited and released the GPUs. The pinned evaluation controller supplies
sample validation and cleanup restricted to its own process group.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load evaluation helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_controller(plan: dict[str, Any]):
    controller_path = Path(plan["eval_controller_path"]).resolve()
    support_path = Path(plan["eval_support_path"]).resolve()
    if support_path != controller_path.parent / "step150_control.py":
        raise ValueError("Pinned evaluation support must be the controller's step150_control.py sibling")
    for path, expected in (
        (controller_path, plan["eval_controller_sha256"]),
        (support_path, plan["eval_support_sha256"]),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"Pinned evaluation code changed: {path}")
    # The legacy controller uses an unqualified sibling import. Explicitly load
    # that exact pinned file, even if another caller imported a same-name module.
    previous = sys.modules.get("step150_control")
    sys.path.insert(0, str(controller_path.parent))
    try:
        sys.modules["step150_control"] = _load_file("step150_control", support_path)
        module = _load_file("real_pinned_fullbench_controller", controller_path)
    finally:
        sys.path.pop(0)
        if previous is None:
            sys.modules.pop("step150_control", None)
        else:
            sys.modules["step150_control"] = previous
    module.NAME = plan["experiment"]
    return module


def _verify_baseline(plan: dict[str, Any], controller) -> tuple[dict, Path]:
    output = Path(plan["baseline_eval_dir"]).resolve()
    recorded = json.loads((output / "verification.json").read_text())
    if recorded.get("status") != "verified" or recorded.get("num_samples") != 7248:
        raise ValueError("Matched step-50 baseline has not completed verification")
    summaries = list((output / "summary").glob("*.json"))
    if len(summaries) != 1:
        raise ValueError("Expected exactly one matched baseline summary")
    summary = json.loads(summaries[0].read_text())
    name = summary["checkpoint"]
    if "_step50_" not in name:
        raise ValueError("REAL step50 must be compared against the step50 baseline")
    adapter = Path(recorded["adapter_path"])
    job = {
        "adapter": str(adapter),
        "adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "evaluation_name": name,
        "output": str(output),
    }
    # Re-run the read-only verifier. controller.verify also writes a baseline
    # report, so use its underlying public verifier directly here.
    verifier = _load_file(
        "real_fullbench_verifier", Path(plan["root"]) / "scripts/analysis/verify_full_bench_run.py"
    )
    verifier.verify_full_bench_run(
        summaries[0], output / "records" / f"{name}.jsonl", output, name,
        adapter, Path(plan["base_model"]), Path(plan["snapshot"]), plan["snapshot_sha256"],
    )
    controller.check_assets(plan, job)
    return summary, output / "records" / f"{name}.jsonl"


def _check_adapter(adapter: Path, base_model: str) -> None:
    from safetensors import safe_open

    config = json.loads((adapter / "adapter_config.json").read_text())
    if (config.get("peft_type"), config.get("r"), config.get("lora_alpha"), config.get("lora_dropout")) != (
        "LORA", 32, 64, 0.0
    ):
        raise ValueError("REAL evaluation requires a standard rank32/alpha64/dropout0 LoRA adapter")
    if Path(config.get("base_model_name_or_path", "")).resolve() != Path(base_model).resolve():
        raise ValueError("REAL adapter references a different base model")
    for field, expected in (("rank_pattern", 32), ("alpha_pattern", 64)):
        if any(value != expected for value in config.get(field, {}).values()):
            raise ValueError(f"REAL adapter has a nonuniform {field}")
    with safe_open(adapter / "adapter_model.safetensors", framework="numpy") as handle:
        keys = set(handle.keys())
        a_names = {key.removesuffix(".lora_A.weight") for key in keys if key.endswith(".lora_A.weight")}
        b_names = {key.removesuffix(".lora_B.weight") for key in keys if key.endswith(".lora_B.weight")}
        if not a_names or a_names != b_names or len(keys) != 2 * len(a_names):
            raise ValueError("REAL adapter must contain complete, paired A and B tensors")
        for name in a_names:
            a_shape = handle.get_slice(f"{name}.lora_A.weight").get_shape()
            b_shape = handle.get_slice(f"{name}.lora_B.weight").get_shape()
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != 32 or b_shape[1] != 32:
                raise ValueError(f"Invalid rank32 REAL adapter tensor shapes: {name}")


def evaluate_real(plan: dict[str, Any], status: Callable[..., None]) -> dict[str, Any]:
    """Run one verified six-benchmark evaluation and save paired comparisons.

    Required plan keys are root, python, experiment, target_step, base_model,
    base_model_weights_sha256, snapshot, snapshot_sha256, checkpoint_root,
    baseline_eval_dir, eval_output_dir, job_dir, eval_controller_path,
    eval_controller_sha256, eval_support_path, and eval_support_sha256.
    """
    if plan["target_step"] != 50:
        raise ValueError("This experiment is registered for a matched step50 comparison")
    controller = _load_controller(plan)
    if _sha256(plan["snapshot"]) != plan["snapshot_sha256"]:
        raise ValueError("Benchmark snapshot changed")
    if _sha256(Path(plan["base_model"]) / "model.safetensors") != plan["base_model_weights_sha256"]:
        raise ValueError("Base model weights changed")
    baseline, baseline_records = _verify_baseline(plan, controller)
    adapter = Path(plan["checkpoint_root"]) / "global_step_50/actor/peft_adapter"
    _check_adapter(adapter, plan["base_model"])
    output = Path(plan["eval_output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing existing REAL evaluation output: {output}")
    job_dir = Path(plan["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    name = f"{plan['experiment']}_step50_fullbench_32768_seed42"
    job = {
        "step": 50, "evaluation_name": name, "adapter": str(adapter),
        "adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256(adapter / "adapter_config.json"),
        "output": str(output), "preflight_output": str(job_dir / "evaluation_preflight"),
    }
    controller.write_json(job_dir / "evaluation_job.json", job)
    status("preflighting_real_evaluation", evaluation_name=name)
    wrapper = Path(plan["root"]) / "tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
    subprocess.run(
        ["bash", str(wrapper)], env=controller.environment(plan, job, dry=True),
        cwd=plan["root"], check=True,
    )
    controller.check_manifests(plan, job, job["preflight_output"])
    candidate = controller.run_job(plan, job, status)
    verification = json.loads((output / "verification.json").read_text())
    if verification.get("status") != "verified" or verification.get("num_samples") != 7248:
        raise ValueError("REAL six-benchmark evaluation did not verify all 7248 samples")

    status("comparing_real_to_standard_lora", baseline=baseline["checkpoint"], candidate=name)
    paired = _load_file("real_paired_comparison", Path(plan["root"]) / "scripts/analysis/compare_paired_full_bench.py")
    report = paired.compare_paired_records(
        baseline_records, output / "records" / f"{name}.jsonl", bootstrap_samples=10_000, seed=42
    )
    paired_path = job_dir / "paired_comparison.json"
    controller.write_json(paired_path, report)
    paired_path.with_suffix(".md").write_text(paired._markdown(report), encoding="utf-8")

    metrics = _load_file("real_summary_comparison", Path(plan["root"]) / "scripts/analysis/compare_full_bench_and_select_next.py")
    raw_metrics = metrics.compare(candidate, baseline, match_margin=0.0)
    # That older helper embeds a dominant-atoms/rank8 decision policy; retain
    # only its generic metrics for this separate REAL experiment.
    metric_report = {
        key: value for key, value in raw_metrics.items()
        if key in {"candidate", "baseline", "benchmarks"}
        or key.startswith(("candidate_", "baseline_", "delta_"))
    }
    metric_report.update(interpretation="single_seed_screen_only", training_step=50)
    metric_path = job_dir / "summary_comparison.json"
    controller.write_json(metric_path, metric_report)
    result = {
        "status": "verified_real_step50_comparison", "evaluation_name": name,
        "num_samples": 7248, "summary": str(output / "summary" / f"{name}.json"),
        "verification": str(output / "verification.json"), "paired_report": str(paired_path),
        "summary_comparison": str(metric_path), "baseline": baseline["checkpoint"],
        "candidate_macro_avg_at_k": candidate["macro_avg_at_k_over_benchmarks"],
        "baseline_macro_avg_at_k": baseline["macro_avg_at_k_over_benchmarks"],
        "delta_macro_avg_at_k": report["macro_delta_avg_at_k"],
        "ci95_low": report["ci95_low"], "ci95_high": report["ci95_high"],
    }
    controller.write_json(job_dir / "evaluation_result.json", result)
    return result
