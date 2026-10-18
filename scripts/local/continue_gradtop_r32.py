#!/usr/bin/env python3
"""Continue the registered GradTop run50→100→150, evaluating each new endpoint.

No new probe, initialization artifact, experiment name, or scientific setting is
introduced. --check is read-only, including the mutable launcher contract.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import math
import re
import subprocess
import shutil
import sys
import time
import uuid

sys.dont_write_bytecode = True
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.local import run_gradtop_clean_after_real as clean
from scripts.local import run_real_after_eval as queue
from scripts.local.real_experiment_eval import _check_adapter, _load_file
from scripts.local.run_real_cached_after_eval import BENCHMARKS, weighted_summary


STAGES = ((50, 100), (100, 150))
DISK_POLICY = {"checkpoint_growth_margin": 1.10, "snapshot_growth_margin": 1.10,
               "headroom_gib": 2.0, "credit_verified_retention": True}


def source_plan(plan):
    source = plan["source_run"]
    clean.assert_plan_unchanged(source["plan"], source["plan_sha256"])
    return clean.read_json(source["plan"])


def validate_plan(plan):
    if plan.get("method") != "gradtop" or plan.get("target_step") != 150 or plan.get("evaluations") != [100, 150]:
        raise ValueError("Only the registered GradTop50→100→150 continuation is supported")
    if plan["execution"].get("disk_budget") != DISK_POLICY:
        raise ValueError("Continuation requires the explicit measured disk-budget policy")
    original = source_plan(plan)
    for key in ("root", "python", "experiment", "protocol_file", "artifact_dir", "run_dir",
                "checkpoint_root", "snapshot_root", "base_model", "base_model_weights_sha256",
                "snapshot", "snapshot_sha256", "eval_controller_path", "eval_controller_sha256",
                "eval_support_path", "eval_support_sha256"):
        if plan[key] != original[key]:
            raise ValueError(f"Continuation changes the original experiment setting: {key}")
    if original.get("method") != "gradtop" or original.get("target_step") != 50:
        raise ValueError("Source run must be the completed clean GradTop-r32 step50 run")
    root = Path(plan["root"])
    contract = root / "runs/protocol_contracts" / f"{plan['experiment']}.json"
    if Path(plan["source_contract"]["path"]).resolve() != contract.resolve():
        raise ValueError("Source contract is not the original launcher's contract")
    if str(contract) in plan["code_sha256"]:
        raise ValueError("Mutable launcher contract must not be pinned as immutable code")
    for path in (Path(__file__).resolve(), Path(plan["protocol_file"]).resolve(),
                 root / "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
                 root / "verl/utils/checkpoint/checkpoint_manager.py"):
        if str(path) not in plan["code_sha256"]:
            raise ValueError(f"Required implementation/protocol pin missing: {path}")
    required_locks = {str(Path(original["job_dir"]) / "controller.lock"), *original["resource_locks"]}
    if not required_locks.issubset(plan["resource_locks"]):
        raise ValueError("Continuation must hold all source and shared resource locks")
    if set(plan["preserved_eval_dirs"]) != {"10", "50"}:
        raise ValueError("Both existing step10 and step50 evaluations must be retained")
    for step in (10, 50):
        if Path(plan["preserved_eval_dirs"][str(step)]) != clean.evaluation_output(plan, step):
            raise ValueError("Preserved evaluation is not from this experiment")
    if set(plan["comparison_eval_dirs"]) != {"100", "150"}:
        raise ValueError("Declare matched comparisons at both new endpoints")
    if any(set(paths) != {"lora", "real32"} for paths in plan["comparison_eval_dirs"].values()):
        raise ValueError("Both LoRA and REAL32 comparisons are required at each endpoint")


def source_ready(plan):
    original = source_plan(plan)
    dependency = plan["source_run"]
    if Path(dependency["status"]).resolve() != Path(original["job_dir"]).resolve() / "status.json":
        raise RuntimeError("Source status is not the original controller's own file")
    if dependency.get("status_sha256"):
        clean.assert_plan_unchanged(dependency["status"], dependency["status_sha256"])
    state = clean.read_json(dependency["status"])
    identity = dependency["process"]
    if (state.get("experiment"), state.get("controller_pid"), state.get("controller_start_ticks")) != (
        plan["experiment"], identity["pid"], identity["start_ticks"]
    ):
        raise RuntimeError("Source controller identity changed")
    if clean.process_alive(identity):
        return False
    if state.get("status") != "complete" or set(state.get("completed_evaluations", [])) != {10, 50}:
        raise RuntimeError("Source controller exited without verified step10/50 completion")
    return True


def ensure_latest(plan, step):
    marker = Path(plan["checkpoint_root"]) / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text().strip() != str(step):
        raise RuntimeError(f"Resume requires latest checkpoint exactly{step}")
    for path in Path(plan["checkpoint_root"]).glob("global_step_*"):
        suffix = path.name.removeprefix("global_step_")
        if suffix.isdigit() and int(suffix) > step:
            raise RuntimeError(f"Unexpected future checkpoint already exists: {path}")


def require_new_endpoints(plan):
    for step in (100, 150):
        paths = [clean.evaluation_output(plan, step),
                 Path(plan["checkpoint_root"]) / f"global_step_{step}"]
        for path in paths:
            if path.exists():
                raise FileExistsError(f"Refusing an existing continuation endpoint: {path}")
    for path in Path(plan["snapshot_root"]).glob("step_*"):
        if path.name.removeprefix("step_").isdigit() and int(path.name.removeprefix("step_")) > 50:
            raise FileExistsError(f"Refusing an existing post-step50 snapshot: {path}")


def protocol_report(plan, target):
    verifier = _load_file("gradtop_continuation_protocol_verifier", Path(plan["root"]) / "scripts/analysis/verify_rlvr_protocol.py")
    report = verifier.verify(argparse.Namespace(
        protocol=Path(plan["protocol_file"]), learning_rate="1e-5", method="gradtop", rank=32,
        target_step=target, real_artifact_dir=None, gradtop_artifact_dir=Path(plan["artifact_dir"]),
        preview_missing_gradtop_artifact=False,
    ))
    if report.get("status") != "verified":
        raise RuntimeError("Continuation requires fully verified original probe artifacts")
    return report


def contract_identity(payload):
    """Ignore only the bounded CPU-recomputed orthogonality diagnostic.

    CPU reduction threads can change this diagnostic by ~1e-6 for the exact
    same three artifact hashes. All artifact bytes/configuration remain part of
    the identity; the verifier's existing finite <=1e-4 bound still applies.
    """
    value = copy.deepcopy(payload)
    diagnostic = value["gradtop_artifact"].pop("orthogonality_error_max")
    if (isinstance(diagnostic, bool) or not isinstance(diagnostic, (float, int))
            or not math.isfinite(diagnostic) or diagnostic < 0 or diagnostic > 1e-4):
        raise RuntimeError("Artifact orthogonality diagnostic violates the protocol bound")
    if set(value["gradtop_artifact"]["files"]) != {"rank_map.json", "summary.json", "subspaces.safetensors"}:
        raise RuntimeError("Contract must identify all three original probe artifacts")
    return value


def assert_contract_extension(previous, current, source_step, target_step):
    left, right = contract_identity(previous), contract_identity(current)
    if left["execution"].pop("target_step") != source_step or right["execution"].pop("target_step") != target_step:
        raise RuntimeError("Unexpected source or target contract training step")
    if left != right:
        raise RuntimeError("Continuation contract changed more than target_step")


def verify_comparisons(plan):
    for step, paths in plan["comparison_eval_dirs"].items():
        for directory in paths.values():
            clean.verified_evaluation(plan, directory, expected_step=int(step))


def check(plan):
    """Read-only validation; never invoke the launcher even in DRY_RUN mode."""
    validate_plan(plan)
    queue.verify_code(plan)
    queue.verify_assets(plan)
    for path, expected in plan.get("source_artifact_sha256", {}).items():
        clean.assert_plan_unchanged(path, expected)
    clean.verify_eval_assets(plan)
    if not source_ready(plan):
        raise RuntimeError("Original GradTop controller is still running")
    ensure_latest(plan, 50)
    require_new_endpoints(plan)
    report = clean.validate_complete_checkpoint(plan, 50)
    original = plan["source_contract"]
    clean.assert_plan_unchanged(original["path"], original["sha256"])
    original_contract = clean.read_json(original["path"])
    for target in (100, 150):
        assert_contract_extension(original_contract, protocol_report(plan, target), 50, target)
    for step, output in plan["preserved_eval_dirs"].items():
        clean.verified_evaluation(plan, output, expected_step=int(step), expected_name=Path(output).name)
    verify_comparisons(plan)
    return {"status": "verified_read_only_continuation", "resume_step": 50, "target_steps": [100, 150],
            "logical_optimizer_updates": report["logical_optimizer_updates"],
            "gpu_pids": queue.gpu_pids(), "free_bytes": shutil.disk_usage(plan["root"]).free,
            "training_disk_budget": stage_disk_budget(plan, 50, 100), "minimum_eval_free_bytes": clean.MIN_EVAL_FREE_BYTES,
            "starts_workloads": False, "waits": False, "launcher_contract_unmodified": True}


def exclusive_json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def preserve_source(plan):
    job = Path(plan["job_dir"])
    contract = plan["source_contract"]
    clean.assert_plan_unchanged(contract["path"], contract["sha256"])
    contract_copy = job / "contract_step50_original.json"
    with contract_copy.open("xb") as output:
        output.write(Path(contract["path"]).read_bytes())
    if queue.sha256(contract_copy) != contract["sha256"]:
        raise RuntimeError("Source contract copy did not preserve its bytes")
    source = Path(plan["checkpoint_root"]) / "global_step_50/actor/peft_adapter"
    backup = job / "step50_adapter_backup"
    backup.mkdir(exist_ok=False)
    files = {}
    for item in source.iterdir():
        if not item.is_file():
            raise RuntimeError(f"Unexpected nested adapter artifact: {item}")
        digest = queue.sha256(item)
        shutil.copy2(item, backup / item.name)
        if queue.sha256(backup / item.name) != digest:
            raise RuntimeError("Step50 adapter backup differs from its source")
        files[item.name] = digest
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
        raise RuntimeError("Step50 adapter backup is incomplete")
    exclusive_json(job / "source_preservation.json", {
        "source_contract": contract, "contract_backup": str(contract_copy),
        "step50_adapter_source": str(source), "step50_adapter_backup": str(backup),
        "step50_adapter_sha256": files,
        "restoration_scope": "evaluation_adapter_only_not_full_checkpoint",
    })


def restore_step50_adapter(plan):
    record = clean.read_json(Path(plan["job_dir"]) / "source_preservation.json")
    backup, source = Path(record["step50_adapter_backup"]), Path(record["step50_adapter_source"])
    for name, expected in record["step50_adapter_sha256"].items():
        stored, destination = backup / name, source / name
        if queue.sha256(stored) != expected:
            raise RuntimeError("Preserved step50 adapter changed")
        if destination.exists():
            if queue.sha256(destination) != expected:
                raise RuntimeError("Existing step50 adapter changed; refusing overwrite")
            continue
        source.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output, stored.open("rb") as handle:
            shutil.copyfileobj(handle, output)
        if queue.sha256(destination) != expected:
            raise RuntimeError("Restored step50 adapter hash mismatch")


def training_environment(plan, source_step, target):
    if (source_step, target) not in STAGES:
        raise ValueError("Only50→100 and100→150 stages are registered")
    env = clean.environment(plan, "train")
    env.update(TARGET_STEP=str(target), RESUME_MODE="auto", RAY_NUM_CPUS="20",
               RAY_TEMP_BASE=plan["execution"]["ray_train_temp_base"],
               LOG_FILE=str(Path(plan["job_dir"]) / f"train_to{target}_metrics.log"),
               LORA_INIT_AUDIT_PATH=str(Path(plan["job_dir"]) / f"initialization_to{target}.json"))
    return env


def directory_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def stage_disk_budget(plan, source_step, target):
    """Budget the additional peak bytes, crediting only verified pre-save retention."""
    if (source_step, target) not in STAGES:
        raise ValueError("Unregistered disk-budget stage")
    checkpoint = Path(plan["checkpoint_root"]) / f"global_step_{source_step}"
    snapshot = Path(plan["snapshot_root"]) / f"step_{source_step:03d}"
    device = Path(plan["root"]).stat().st_dev
    if checkpoint.stat().st_dev != device or snapshot.stat().st_dev != device:
        raise RuntimeError("Measured hot-storage budget requires checkpoint/snapshot on the root filesystem")
    checkpoint_bytes, snapshot_bytes = directory_bytes(checkpoint), directory_bytes(snapshot)
    if not checkpoint_bytes or not snapshot_bytes:
        raise RuntimeError("Cannot budget an empty source checkpoint/snapshot")
    # The pinned FSDP manager discovers existing actors and prunes oldest before
    # opening the new checkpoint. Only the150 stage has two older checkpoints.
    credit, restore = 0, 0
    if target == 150:
        contract = protocol_report(plan, target)
        if contract["execution"]["full_checkpoints_to_keep"] != 2:
            raise RuntimeError("Disk credit requires the frozen retention of two checkpoints")
        actor50 = Path(plan["checkpoint_root"]) / "global_step_50/actor"
        if actor50.is_dir() and not actor50.is_symlink() and actor50.stat().st_dev == device:
            for item in actor50.rglob("*"):
                if item.is_file() and not item.is_symlink():
                    stat = item.stat()
                    if stat.st_dev == device and stat.st_nlink == 1:
                        credit += stat.st_blocks * 512
            restore = directory_bytes(actor50 / "peft_adapter")
    new_checkpoint = math.ceil(checkpoint_bytes * DISK_POLICY["checkpoint_growth_margin"])
    snapshots = math.ceil(((target - source_step) // 10) * snapshot_bytes * DISK_POLICY["snapshot_growth_margin"])
    headroom = int(DISK_POLICY["headroom_gib"] * 1024**3)
    required = snapshots + headroom + max(0, new_checkpoint + restore - credit)
    return {"source_step": source_step, "target_step": target, "policy": DISK_POLICY,
            "measured_checkpoint_bytes": checkpoint_bytes, "measured_snapshot_bytes": snapshot_bytes,
            "projected_checkpoint_bytes": new_checkpoint, "projected_snapshot_bytes": snapshots,
            "verified_pre_save_retention_credit_bytes": credit, "restored_adapter_bytes": restore,
            "headroom_bytes": headroom, "required_free_bytes": required,
            "retention_performed_by_pinned_trainer_only": True}


def run_continuation_stage(plan, stage, command, env, completed, status, guard, budget):
    """Same owned-token cleanup as the shared runner, with an audited byte budget."""
    clean.wait_resources(plan, status, budget["required_free_bytes"])
    guard()
    token = uuid.uuid4().hex
    env = {**env, "REAL_PIPELINE_STAGE_TOKEN": token}
    log = Path(plan["job_dir"]) / f"{stage}.log"
    started, ready_at, cleaned = time.monotonic(), None, False
    with log.open("x") as output:
        child = subprocess.Popen(command, cwd=plan["root"], env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status(stage, stage_pid=child.pid, stage_token=token, stage_log=str(log), disk_budget=budget)
            while child.poll() is None:
                if completed() and ready_at is None:
                    ready_at = time.monotonic()
                    status(f"{stage}_output_ready")
                if ready_at is not None and time.monotonic() - ready_at > 90:
                    cleaned = True
                    queue.cleanup_stage(token)
                    child.wait(timeout=15)
                    break
                status(stage, latest_logged_step=queue.recent_step(log), elapsed_seconds=round(time.monotonic() - started))
                time.sleep(30)
            if not completed():
                raise RuntimeError(f"Continuation stage exited{child.returncode} without complete outputs: {log}")
            if child.returncode != 0 and not cleaned:
                raise RuntimeError(f"Continuation stage exited{child.returncode}: {log}")
        finally:
            queue.cleanup_stage(token)
    exclusive_json(Path(plan["job_dir"]) / f"{stage}_execution.json", {
        "status": "completed", "finished_at": queue.now(), "wall_seconds": time.monotonic() - started,
        "process_returncode": child.returncode, "command": command, "log": str(log),
        "cleaned_after_complete_output": cleaned, "disk_budget": budget,
    })
    status(f"{stage}_completed", stage_pid=None)


def verify_resume_log(log, checkpoint, *, required=False):
    if not Path(log).is_file():
        if required:
            raise RuntimeError("Missing continuation training log")
        return False
    with Path(log).open("rb") as handle:
        text = handle.read().decode("utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if "Training from scratch" in text or "No dataloader state found" in text:
        raise RuntimeError("Continuation attempted to restart model/data from scratch")
    paths = re.findall(r"Load from checkpoint folder:\s*(\S+)", text)
    if paths and any(Path(path).resolve() != Path(checkpoint).resolve() for path in paths):
        raise RuntimeError("Training loaded a different resume checkpoint")
    evidence = bool(paths) and all(f"Loaded {kind} from " in text for kind in ("model", "optimizer", "rng", "lr_scheduler"))
    if required and not evidence:
        raise RuntimeError("Training log does not confirm full model/optimizer/RNG/scheduler resume")
    return evidence


def metrics_row(plan, output, step):
    summary, records = clean.verified_evaluation(plan, output, expected_step=step)
    weighted = weighted_summary(output)
    return {**weighted, "step": step, "logical_optimizer_updates": 4 * step, "records": str(records),
            "macro_avg_at_k": summary["macro_avg_at_k_over_benchmarks"],
            "macro_pass_at_k": summary["macro_pass_at_k_over_benchmarks"],
            "parse_rate": summary["parse_rate_overall"], "hit_max_rate": summary["hit_max_rate_overall"],
            "mean_length": summary["length_mean_overall"]}


def write_results(plan, completed, comparisons):
    rows = {int(step): metrics_row(plan, output, int(step)) for step, output in plan["preserved_eval_dirs"].items()}
    rows.update({step: metrics_row(plan, output, step) for step, output in completed.items()})
    baselines = {}
    for step in completed:
        baselines[step] = {label: metrics_row(plan, directory, step)
                           for label, directory in plan["comparison_eval_dirs"][str(step)].items()}
    payload = {"experiment": plan["experiment"], "method": "gradtop", "probe_reused": True,
               "avg_definition": "response-count weighted: sum(num_samples * avg_at_k) / 7248",
               "macro_definition": "equal mean over the six benchmark Avg@k values",
               "steps": rows, "matched_baselines": baselines, "paired_macro_comparisons": comparisons,
               "training_seed_limit": "single_seed42"}
    queue.write_json(Path(plan["job_dir"]) / "results.json", payload)
    lines = ["GradTop r32 continuation; original step10/50 evaluations retained.", "",
             "Avg. is weighted by response count. Each task cell is Avg@k / Pass@k (%).",
             "Trainer step counts rollout iterations; logical optimizer updates =4 × step.", "",
             "| Method | Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    display = []
    for step, row in sorted(rows.items()):
        display.append(("GradTop", step, row))
        display.extend((label, step, baseline) for label, baseline in baselines.get(step, {}).items())
    for label, step, row in display:
        cells = [f"{100 * row['benchmarks'][key]['avg_at_k']:.1f} / {100 * row['benchmarks'][key]['pass_at_k']:.1f}"
                 for key in BENCHMARKS]
        lines.append(f"| {label} | {step} | {100 * row['weighted_avg']:.1f} | " + " | ".join(cells) + " |")
    lines += ["", "| Method | Step | Macro Avg@k | Macro Pass@k | Parse | Hit max | Mean length |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, step, row in display:
        lines.append(f"| {label} | {step} | {100*row['macro_avg_at_k']:.4f}% | {100*row['macro_pass_at_k']:.4f}% | "
                     f"{100*row['parse_rate']:.2f}% | {100*row['hit_max_rate']:.2f}% | {row['mean_length']:.1f} |")
    lines += ["", "| Step | Reference | Weighted Avg. delta (pp) | Macro Avg@k delta (pp) | Macro95% paired CI (pp) |",
              "| --- | --- | --- | --- | --- |"]
    for step, reports in comparisons.items():
        for label, report in reports.items():
            lines.append(f"| {step} | {label} | {100*report['weighted_avg_delta']:+.2f} | "
                         f"{100*report['macro_delta_avg_at_k']:+.3f} | "
                         f"[{100*report['ci95_low']:+.3f}, {100*report['ci95_high']:+.3f}] |")
    lines += ["", "Paired confidence intervals apply to Macro Avg@k, not the response-weighted Avg. Training uses one seed.", ""]
    (Path(plan["job_dir"]) / "results.md").write_text("\n".join(lines))


def compare_endpoint(plan, step):
    paired = _load_file("gradtop_continuation_paired", Path(plan["root"]) / "scripts/analysis/compare_paired_full_bench.py")
    candidate = metrics_row(plan, clean.evaluation_output(plan, step), step)
    reports = {}
    references = {**plan["comparison_eval_dirs"][str(step)], "own_step50": plan["preserved_eval_dirs"]["50"]}
    for label, directory in references.items():
        baseline = metrics_row(plan, directory, 50 if label == "own_step50" else step)
        report = paired.compare_paired_records(Path(baseline["records"]), Path(candidate["records"]),
                                              bootstrap_samples=10_000, seed=42)
        report["weighted_avg_delta"] = candidate["weighted_avg"] - baseline["weighted_avg"]
        report["weighted_avg_delta_has_no_ci"] = True
        report["comparison_type"] = "within_run_vs_step50" if label == "own_step50" else "matched_training_step"
        output = Path(plan["job_dir"]) / f"paired_step{step}_vs_{label}.json"
        exclusive_json(output, report)
        output.with_suffix(".md").write_text(paired._markdown(report))
        reports[label] = {"path": str(output), "weighted_avg_delta": report["weighted_avg_delta"],
                          **{key: report[key] for key in ("macro_delta_avg_at_k", "ci95_low", "ci95_high")}}
    return reports


def execute(plan, status, guard):
    guard()
    check(plan)
    preserve_source(plan)
    completed, comparisons = {}, {}
    write_results(plan, completed, comparisons)
    previous_contract = clean.read_json(Path(plan["job_dir"]) / "contract_step50_original.json")
    launch = ["bash", str(Path(plan["root"]) / "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh")]
    for source_step, target in STAGES:
        budget = stage_disk_budget(plan, source_step, target)
        exclusive_json(Path(plan["job_dir"]) / f"disk_budget_to{target}.json", budget)
        clean.wait_resources(plan, status, budget["required_free_bytes"])
        guard()
        ensure_latest(plan, source_step)
        resume_report = clean.validate_complete_checkpoint(plan, source_step)
        exclusive_json(Path(plan["job_dir"]) / f"resume_step{source_step}_verification.json", resume_report)
        expected_contract = protocol_report(plan, target)
        assert_contract_extension(previous_contract, expected_contract, source_step, target)
        exclusive_json(Path(plan["job_dir"]) / f"expected_contract_step{target}.json", expected_contract)
        env = training_environment(plan, source_step, target)
        stage_plan = {**plan, "target_step": target}
        log = Path(plan["job_dir"]) / f"train_to{target}.log"
        checkpoint = Path(plan["checkpoint_root"]) / f"global_step_{source_step}"

        def launch_guard():
            guard()
            ensure_latest(plan, source_step)

        def stage_complete():
            verify_resume_log(log, checkpoint)
            complete = queue.training_complete(stage_plan)
            if complete:
                verify_resume_log(log, checkpoint, required=True)
            return complete

        status(f"resuming_step{source_step}_to{target}", resume_from=str(checkpoint), target_step=target)
        try:
            run_continuation_stage(stage_plan, f"train_to{target}", launch, env, stage_complete, status, launch_guard, budget)
        finally:
            restore_step50_adapter(plan)
        report = clean.validate_complete_checkpoint(plan, target)
        exclusive_json(Path(plan["job_dir"]) / f"checkpoint_step{target}_verification.json", report)
        actual_contract = clean.read_json(plan["source_contract"]["path"])
        assert_contract_extension(expected_contract, actual_contract, target, target)
        exclusive_json(Path(plan["job_dir"]) / f"contract_step{target}.json", actual_contract)
        previous_contract = actual_contract
        adapter = Path(plan["checkpoint_root"]) / f"global_step_{target}/actor/peft_adapter"
        _check_adapter(adapter, plan["base_model"])
        clean.run_evaluation(plan, target, adapter, status, workspace=plan["job_dir"], guard=guard)
        completed[target] = str(clean.evaluation_output(plan, target))
        comparisons[str(target)] = compare_endpoint(plan, target)
        write_results(plan, completed, comparisons)
        status(f"evaluation_step{target}_complete", completed_evaluations=[10, 50, *completed], current_step=None)
    status("complete", completed_evaluations=[10, 50, 100, 150], results=str(Path(plan["job_dir"]) / "results.md"), auto_resume=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    clean.assert_plan_unchanged(args.plan, args.plan_sha256)
    plan = clean.read_json(args.plan)

    def guard():
        clean.assert_plan_unchanged(args.plan, args.plan_sha256)
        queue.verify_code(plan)
        for path, expected in plan.get("source_artifact_sha256", {}).items():
            clean.assert_plan_unchanged(path, expected)

    guard()
    if args.check:
        print(json.dumps(check(plan), ensure_ascii=False, indent=2))
        return
    validate_plan(plan)
    job = Path(plan["job_dir"])
    job.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((job / "controller.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (job / "status.json").exists():
            raise RuntimeError("Continuation already registered; refusing duplicate/restart")
        state = {"experiment": plan["experiment"], "controller_pid": os.getpid(),
                 "controller_start_ticks": queue.process_info(os.getpid())["start_ticks"],
                 "plan": str(args.plan), "plan_sha256": args.plan_sha256, "started_at": queue.now(),
                 "target_step": 150, "completed_evaluations": [10, 50]}

        def status(stage, **details):
            state.update(status=stage, updated_at=queue.now(), **details)
            queue.write_json(job / "status.json", state)
            print(f"[{queue.now()}] {stage}: {json.dumps(details, ensure_ascii=False)}", flush=True)

        try:
            while not source_ready(plan):
                guard()
                status("waiting_for_source_controller_exit")
                time.sleep(30)
            clean.acquire_resource_locks(stack, plan, status)
            guard()
            execute(plan, status, guard)
        except BaseException as error:
            status("error", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
