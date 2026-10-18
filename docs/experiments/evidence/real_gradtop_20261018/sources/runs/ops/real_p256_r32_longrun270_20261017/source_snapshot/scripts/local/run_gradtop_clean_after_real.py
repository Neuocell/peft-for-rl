#!/usr/bin/env python3
"""Queue a fresh protocol GradTop-r32 probe/train/evaluation after REAL finishes.

--check only reads inputs; an absent probe artifact is explicitly a preview.
The plan's byte hash is supplied separately and rechecked before every stage.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local import run_real_after_eval as queue
from scripts.local.real_experiment_eval import _check_adapter, _load_controller, _load_file

MIN_FREE_BYTES = 15 * 1024**3
MIN_EVAL_FREE_BYTES = 2 * 1024**3


def read_json(path):
    return json.loads(Path(path).read_text())


def assert_plan_unchanged(path, expected):
    if queue.sha256(path) != expected:
        raise RuntimeError("Queued plan changed; review and register a new queue")


def validate_plan(plan):
    if plan.get("method", "gradtop") != "gradtop" or plan["target_step"] != 50:
        raise ValueError("This queue permits only fresh METHOD=gradtop, target_step=50")
    if plan.get("evaluations") != [50, 10]:
        raise ValueError("Evaluations must be [50, 10], with step50 primary")
    root = Path(plan["root"]).resolve()
    expected = {
        "checkpoint_root": root / "runs/ckpts/verl/peft-for-rl" / plan["experiment"],
        "snapshot_root": root / "runs/snapshots/rlvr_protocol" / plan["experiment"],
    }
    for key, path in expected.items():
        if Path(plan[key]).resolve() != path:
            raise ValueError(f"Unexpected {key}: launcher uses {path}")
    if Path(__file__).resolve().as_posix() not in plan["code_sha256"]:
        raise ValueError("Queue script must be frozen in code_sha256")
    if str(Path(plan["protocol_file"]).resolve()) not in plan["code_sha256"]:
        raise ValueError("Protocol must be frozen in code_sha256")
    if Path(plan["run_dir"]).resolve() != root / "runs/gt32_1015":
        raise ValueError("Probe requires the registered short runs/gt32_1015 directory")
    if set(plan["comparison_eval_dirs"]) != {"lora", "real64", "real32"}:
        raise ValueError("Expected all three matched step50 comparisons")
    original = predecessor_plan(plan)
    required_locks = {str(Path(original["job_dir"]) / "controller.lock"),
                      *original.get("run_locks", []), *original.get("prerequisite_locks", [])}
    if not required_locks.issubset(plan.get("resource_locks", [])):
        raise ValueError("Resource locks must include every frozen predecessor/shared lock")


def evaluation_output(plan, step):
    name = f"{plan['experiment']}_step{step}_fullbench_32768_seed42"
    return Path(plan["root"]) / "runs/outputs/full_bench_eval" / name


def require_clean_outputs(plan):
    paths = [Path(plan[key]) for key in ("artifact_dir", "run_dir", "checkpoint_root", "snapshot_root")]
    paths += [evaluation_output(plan, step) for step in plan["evaluations"]]
    paths += [Path(plan["root"]) / "runs/protocol_contracts" / f"{plan['experiment']}.json"]
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Refusing existing experiment output: {path}")


def protocol_check(plan, preview=False):
    verifier = _load_file("gradtop_queue_protocol_verifier", Path(plan["root"]) / "scripts/analysis/verify_rlvr_protocol.py")
    return verifier.verify(argparse.Namespace(
        protocol=Path(plan["protocol_file"]), learning_rate="1e-5", method="gradtop", rank=32,
        target_step=50, real_artifact_dir=None, gradtop_artifact_dir=Path(plan["artifact_dir"]),
        preview_missing_gradtop_artifact=preview,
    ))


def verify_eval_assets(plan):
    _load_controller(plan)
    if queue.sha256(plan["snapshot"]) != plan["snapshot_sha256"]:
        raise RuntimeError("Benchmark snapshot changed")
    if queue.sha256(Path(plan["base_model"]) / "model.safetensors") != plan["base_model_weights_sha256"]:
        raise RuntimeError("Base model weights changed")


def verified_evaluation(plan, output, *, expected_step=None, expected_name=None, adapter=None):
    """Reverify records and manifests without rewriting an existing report."""
    output = Path(output)
    report = read_json(output / "verification.json")
    if report.get("status") != "verified" or report.get("num_samples") != 7248:
        raise RuntimeError(f"Benchmark verification failed: {output}")
    summaries = list((output / "summary").glob("*.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected exactly one benchmark summary: {output}")
    summary = read_json(summaries[0])
    name = summary["checkpoint"]
    if expected_name is not None and name != expected_name:
        raise RuntimeError("Benchmark belongs to another experiment/checkpoint")
    if expected_step is not None and f"_step{expected_step}_" not in name:
        raise RuntimeError("Benchmark is not the requested training step")
    recorded_adapter = Path(report["adapter_path"])
    if adapter is not None and recorded_adapter.resolve() != Path(adapter).resolve():
        raise RuntimeError("Benchmark references another adapter")
    _check_adapter(recorded_adapter, plan["base_model"])
    verifier = _load_file("gradtop_queue_fullbench_verifier", Path(plan["root"]) / "scripts/analysis/verify_full_bench_run.py")
    records = output / "records" / f"{name}.jsonl"
    verifier.verify_full_bench_run(
        summaries[0], records, output, name, recorded_adapter,
        Path(plan["base_model"]), Path(plan["snapshot"]), plan["snapshot_sha256"],
    )
    return summary, records


def predecessor_plan(plan):
    predecessor = plan["predecessor"]
    assert_plan_unchanged(predecessor["plan"], predecessor["plan_sha256"])
    original = read_json(predecessor["plan"])
    if original["target_step"] != 150:
        raise RuntimeError("Predecessor must be the registered REAL continuation to150")
    if Path(plan["predecessor"]["status"]).resolve() != Path(original["job_dir"]).resolve() / "status.json":
        raise RuntimeError("Predecessor status is not the frozen controller's own status file")
    return original


def validate_complete_checkpoint(plan, step):
    if not queue.training_complete({**plan, "target_step": step}):
        raise RuntimeError(f"Predecessor requires a complete resumable checkpoint{step}")
    if queue.sha256(plan["eval_support_path"]) != plan["eval_support_sha256"]:
        raise RuntimeError("Pinned checkpoint validator changed")
    support = _load_file("gradtop_queue_checkpoint_validator", Path(plan["eval_support_path"]))
    report = support.validate_checkpoint(plan, step)
    import torch
    optimizer = torch.load(Path(plan["checkpoint_root"]) / f"global_step_{step}/actor/optim_world_size_4_rank_0.pt",
                           map_location="cpu", weights_only=False, mmap=True)
    updates = {int(item["step"]) for item in optimizer["state"].values() if "step" in item}
    if updates != {4 * step}:
        raise RuntimeError(f"Expected{4 * step} optimizer updates at checkpoint{step}; found {updates}")
    return {**report, "logical_optimizer_updates": 4 * step}


def process_alive(identity):
    process = queue.process_info(identity["pid"])
    return process is not None and process["state"] != "Z" and process["start_ticks"] == identity["start_ticks"]


def predecessor_state(plan):
    """Classify completion; stale artifacts never override a running/failed owner."""
    original = predecessor_plan(plan)
    state = read_json(plan["predecessor"]["status"])
    if state.get("experiment") != original["experiment"]:
        raise RuntimeError("Predecessor status belongs to another experiment")
    identity = plan["predecessor"]["process"]
    if (state.get("controller_pid"), state.get("controller_start_ticks")) != (identity["pid"], identity["start_ticks"]):
        raise RuntimeError("Predecessor status owner differs from the frozen process identity")
    if process_alive(identity):
        return "waiting"
    if state.get("status") == "complete":
        if not {100, 150}.issubset(state.get("completed_evaluations", [])):
            raise RuntimeError("Predecessor completion omits required evaluations100/150")
        validate_complete_checkpoint(original, 150)
        for step in (100, 150):
            verified_evaluation(original, evaluation_output(original, step), expected_step=step,
                                expected_name=evaluation_output(original, step).name)
        return "complete"
    raise RuntimeError(
        f"Predecessor exited without verified completion: {state.get('status')}: {state.get('error', '')}"
    )


def wait_resources(plan, status, minimum_free_bytes=MIN_FREE_BYTES):
    while True:
        queue.wait_idle(status)
        free = shutil.disk_usage(plan["root"]).free
        if free >= minimum_free_bytes:
            return
        status("waiting_for_disk_space", free_bytes=free, required_bytes=minimum_free_bytes,
               preserves_existing_files=True)
        time.sleep(30)


def verify_snapshot_compatibility(snapshot, checkpoint, *, equal_values=False):
    """Snapshot serialization can differ while every adapter tensor is identical."""
    import numpy as np
    from safetensors import safe_open
    with safe_open(snapshot, framework="numpy") as source, safe_open(checkpoint, framework="numpy") as target:
        keys = set(source.keys())
        if not keys or keys != set(target.keys()):
            raise RuntimeError("Snapshot and checkpoint adapter tensor names differ")
        for name in keys:
            if source.get_slice(name).get_shape() != target.get_slice(name).get_shape():
                raise RuntimeError(f"Snapshot and checkpoint adapter shapes differ: {name}")
            if equal_values and not np.array_equal(source.get_tensor(name), target.get_tensor(name)):
                raise RuntimeError(f"Step50 snapshot differs from full checkpoint: {name}")
    return {"tensor_count": len(keys), "equal_values": equal_values}


def evaluation_adapter(plan, step):
    if step == 50:
        if not queue.training_complete(plan):
            raise RuntimeError("Step50 evaluation requires the full resumable checkpoint")
        adapter = Path(plan["checkpoint_root"]) / "global_step_50/actor/peft_adapter"
    elif step == 10:
        source = Path(plan["snapshot_root"]) / "step_010"
        manifest = read_json(source / "manifest.json")
        expected = {"artifact_kind": "evaluation_snapshot", "trainer_step": 10,
                    "rank": 32, "alpha": 64, "module_count": 196,
                    "snapshot": "adapter_model.safetensors", "tensor_count": 392}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Step10 snapshot manifest is inconsistent")
        # Evaluation snapshots contain weights/manifest, but no PEFT config.
        # Export a separate adapter; never mutate the training snapshot.
        config_source = Path(plan["checkpoint_root"]) / "global_step_50/actor/peft_adapter/adapter_config.json"
        weights_source = source / "adapter_model.safetensors"
        full_weights = config_source.parent / "adapter_model.safetensors"
        snapshot50 = Path(plan["snapshot_root"]) / "step_050/adapter_model.safetensors"
        verify_snapshot_compatibility(snapshot50, full_weights, equal_values=True)
        compatibility = verify_snapshot_compatibility(weights_source, full_weights)
        adapter = Path(plan["job_dir"]) / "evaluation_adapters/step_010"
        adapter.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_source, adapter / "adapter_config.json")
        shutil.copy2(weights_source, adapter / "adapter_model.safetensors")
        _check_adapter(adapter, plan["base_model"])
        provenance = {
            "trainer_step": 10, "artifact_kind": "evaluation_adapter_export",
            "source_snapshot": str(source), "snapshot_manifest_sha256": queue.sha256(source / "manifest.json"),
            "source_weights_sha256": queue.sha256(weights_source), "source_config": str(config_source),
            "source_config_sha256": queue.sha256(config_source),
            "export_weights_sha256": queue.sha256(adapter / "adapter_model.safetensors"),
            "export_config_sha256": queue.sha256(adapter / "adapter_config.json"),
            "step50_snapshot_equals_full_checkpoint": True, "compatibility": compatibility,
        }
        if (provenance["source_weights_sha256"] != provenance["export_weights_sha256"]
                or provenance["source_config_sha256"] != provenance["export_config_sha256"]):
            raise RuntimeError("Step10 adapter export changed source weights/config")
        queue.write_json(adapter / "export_provenance.json", provenance)
    else:
        raise ValueError("Only registered evaluations50/10 are supported")
    _check_adapter(adapter, plan["base_model"])
    return adapter


def run_evaluation(plan, step, adapter, status, *, workspace, guard):
    guard()
    controller = _load_controller(plan)
    output = evaluation_output(plan, step)
    name = output.name
    if output.exists():
        raise FileExistsError(f"Refusing existing evaluation output: {output}")
    workspace = Path(workspace)
    job = {"step": step, "evaluation_name": name, "adapter": str(adapter),
           "adapter_sha256": queue.sha256(adapter / "adapter_model.safetensors"),
           "adapter_config_sha256": queue.sha256(adapter / "adapter_config.json"),
           "output": str(output), "preflight_output": str(workspace / f"evaluation_step{step}_preflight")}
    queue.write_json(workspace / f"evaluation_step{step}_job.json", job)
    wait_resources(plan, status, MIN_EVAL_FREE_BYTES)
    guard()
    subprocess.run(["bash", str(Path(plan["root"]) / "tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh")],
                   env=controller.environment(plan, job, dry=True), cwd=plan["root"], check=True)
    controller.check_manifests(plan, job, job["preflight_output"])
    wait_resources(plan, status, MIN_EVAL_FREE_BYTES)
    guard()
    result = controller.run_job(plan, job, status)
    verified_evaluation(plan, output, expected_step=step, expected_name=name, adapter=adapter)
    return result


def environment(plan, phase):
    # Do not inherit stale PEFT/probe/warm-start experiment switches from the caller.
    env = queue.common_environment(plan)
    allowed = {
        "PATH", "CONDA_PREFIX", "PYTHON_BIN", "PYTHONPATH", "PYTHONUNBUFFERED",
        "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "CUDA_VISIBLE_DEVICES", "RUNTIME_ROOT", "DRY_RUN",
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HOME", "USER", "LOGNAME", "SHELL", "LANG",
        "TMPDIR", "TMP", "TEMP", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH",
    }
    env = {key: value for key, value in env.items() if key in allowed or key.startswith(("LC_", "NCCL_"))}
    env.update(METHOD="gradtop", PHASE=phase, EXP_NAME=plan["experiment"],
               GRADTOP_ARTIFACT_DIR=plan["artifact_dir"], GRADTOP_RUN_DIR=plan["run_dir"],
               PROTOCOL_FILE=plan["protocol_file"], LORA_RANK="32", LR="1e-5", TARGET_STEP="50",
               RESUME_MODE="disable", RAY_TEMP_BASE="/tmp/gt32_1015", RAY_NUM_CPUS="20",
               LOG_FILE=str(Path(plan["job_dir"]) / f"{phase}_metrics.log"),
               PYTHONDONTWRITEBYTECODE="1")
    return env


def probe_complete(plan):
    directory = Path(plan["artifact_dir"])
    if not all((directory / name).is_file() for name in ("rank_map.json", "summary.json", "subspaces.safetensors")):
        return False
    summary = read_json(directory / "summary.json")
    return (summary.get("observations") == 16 and summary.get("optimizer_updates") == 0)


def run_training_stage(plan, name, command, env, completed, status, guard):
    while True:
        wait_resources(plan, status)
        guard()
        try:
            queue.run_stage(plan, name, command, env, completed, status)
            return
        except RuntimeError as error:
            # The shared helper rechecks disk after waiting for GPUs. Retry only
            # its pre-launch disk refusal; never retry a failed child or outputs.
            if str(error).startswith("Less than 15 GiB free before REAL stage") and not (Path(plan["job_dir"]) / f"{name}.log").exists():
                status("waiting_for_disk_space", preserves_existing_files=True)
                time.sleep(30)
                continue
            raise


def compare_results(plan, status):
    candidate, candidate_records = verified_evaluation(plan, evaluation_output(plan, 50), expected_step=50)
    paired = _load_file("gradtop_queue_paired_comparison", Path(plan["root"]) / "scripts/analysis/compare_paired_full_bench.py")
    metrics = _load_file("gradtop_queue_metric_comparison", Path(plan["root"]) / "scripts/analysis/compare_full_bench_and_select_next.py")
    results = {}
    for label, directory in plan["comparison_eval_dirs"].items():
        baseline, baseline_records = verified_evaluation(plan, directory, expected_step=50)
        report = paired.compare_paired_records(baseline_records, candidate_records, bootstrap_samples=10_000, seed=42)
        path = Path(plan["job_dir"]) / f"paired_vs_{label}_step50.json"
        queue.write_json(path, report)
        path.with_suffix(".md").write_text(paired._markdown(report))
        raw = metrics.compare(candidate, baseline, match_margin=0.0)
        comparison = {key: value for key, value in raw.items() if key in {"candidate", "baseline", "benchmarks"}
                      or key.startswith(("candidate_", "baseline_", "delta_"))}
        comparison.update(training_step=50, interpretation="single_seed_screen_only")
        queue.write_json(Path(plan["job_dir"]) / f"summary_vs_{label}_step50.json", comparison)
        results[label] = {"paired_report": str(path), "macro_delta_avg_at_k": report["macro_delta_avg_at_k"],
                          "ci95_low": report["ci95_low"], "ci95_high": report["ci95_high"]}
    status("comparisons_complete", comparisons=results)
    return results


def execute(plan, status, guard):
    guard()
    queue.verify_assets(plan)
    verify_eval_assets(plan)
    require_clean_outputs(plan)
    for directory in plan["comparison_eval_dirs"].values():
        verified_evaluation(plan, directory, expected_step=50)
    launch = ["bash", str(Path(plan["root"]) / "scripts/local/run_gradtop_candidates_4gpu.sh")]
    run_training_stage(plan, "probe", launch, environment(plan, "probe"), lambda: probe_complete(plan), status, guard)
    report = protocol_check(plan)
    if report.get("status") != "verified":
        raise RuntimeError("Completed probe did not pass full protocol validation")
    queue.write_json(Path(plan["job_dir"]) / "probe_artifact_verification.json", report)
    run_training_stage(plan, "train", launch, environment(plan, "train"), lambda: queue.training_complete(plan), status, guard)
    checkpoint_report = validate_complete_checkpoint(plan, 50)
    queue.write_json(Path(plan["job_dir"]) / "step50_checkpoint_verification.json", checkpoint_report)
    results = {}
    for step in plan["evaluations"]:
        wait_resources(plan, status, MIN_EVAL_FREE_BYTES)
        adapter = evaluation_adapter(plan, step)
        results[str(step)] = run_evaluation(plan, step, adapter, status, workspace=plan["job_dir"], guard=guard)
        status(f"evaluation_step{step}_complete", completed_evaluations=list(map(int, results)))
    comparisons = compare_results(plan, status)
    queue.write_json(Path(plan["job_dir"]) / "evaluation_result.json", {
        "status": "verified_gradtop_step50_and10", "primary_step": 50, "steps": [50, 10],
        "num_samples_per_checkpoint": 7248, "comparisons": comparisons,
        "macro_avg_at_k": {step: result["macro_avg_at_k_over_benchmarks"] for step, result in results.items()},
    })
    status("complete", completed_evaluations=[50, 10], auto_resume=False)


def check(plan):
    validate_plan(plan)
    queue.verify_code(plan)
    queue.verify_assets(plan)
    verify_eval_assets(plan)
    require_clean_outputs(plan)
    report = protocol_check(plan, preview=True)
    for directory in plan["comparison_eval_dirs"].values():
        verified_evaluation(plan, directory, expected_step=50)
    return {"status": "read_only_preview", "protocol_status": report["status"],
            "artifact_validation": "pending_probe" if report["status"] != "verified" else "verified",
            "predecessor": predecessor_state(plan), "free_bytes": shutil.disk_usage(plan["root"]).free,
            "gpu_pids": queue.gpu_pids(), "starts_workloads": False, "waits": False,
            "minimum_training_free_bytes": MIN_FREE_BYTES, "minimum_eval_free_bytes": MIN_EVAL_FREE_BYTES}


def acquire_resource_locks(stack, plan, status):
    paths = sorted(set(plan.get("resource_locks", plan.get("prerequisite_locks", []))))
    if not paths:
        raise ValueError("Shared resource locks must be explicitly pinned in the plan")
    while True:
        attempt = ExitStack()
        try:
            for filename in paths:
                handle = attempt.enter_context(Path(filename).open("a+"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            attempt.close()
            status("waiting_for_resource_locks", locks=paths)
            time.sleep(30)
        else:
            stack.enter_context(attempt)
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert_plan_unchanged(args.plan, args.plan_sha256)
    plan = read_json(args.plan)
    validate_plan(plan)

    def guard():
        assert_plan_unchanged(args.plan, args.plan_sha256)
        queue.verify_code(plan)

    guard()
    if args.check:
        print(json.dumps(check(plan), ensure_ascii=False, indent=2))
        return
    job_dir = Path(plan["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((job_dir / "controller.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (job_dir / "status.json").exists():
            raise RuntimeError("Queue already registered; refusing duplicate/restart")
        require_clean_outputs(plan)
        state = {"experiment": plan["experiment"], "plan": str(args.plan), "plan_sha256": args.plan_sha256,
                 "controller_pid": os.getpid(), "controller_start_ticks": queue.process_info(os.getpid())["start_ticks"],
                 "started_at": queue.now(), "completed_evaluations": []}

        def status(stage, **details):
            state.update(status=stage, updated_at=queue.now(), **details)
            queue.write_json(job_dir / "status.json", state)
            print(f"[{queue.now()}] {stage}: {json.dumps(details, ensure_ascii=False)}", flush=True)

        try:
            while predecessor_state(plan) == "waiting":
                guard()
                status("waiting_for_real150_and_evaluations", predecessor=plan["predecessor"]["status"])
                time.sleep(30)
            acquire_resource_locks(stack, plan, status)
            guard()
            if predecessor_state(plan) != "complete":
                raise RuntimeError("Predecessor changed while acquiring resource locks")
            execute(plan, status, guard)
        except BaseException as error:
            status("error", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
