#!/usr/bin/env python3
"""Resume REAL probe32 from step50 through150, then evaluate100 and150."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts.local import run_real_after_eval as queue
from scripts.local.run_real_cached_after_eval import training_environment, weighted_summary, BENCHMARKS
from scripts.local.real_experiment_eval import _load_controller, _load_file, _check_adapter


def validate_resume(plan, step):
    marker = Path(plan["checkpoint_root"]) / "latest_checkpointed_iteration.txt"
    if marker.read_text().strip() != str(step):
        raise RuntimeError(f"Expected latest complete checkpoint {step}")
    validator = _load_file("real_resume_validator", Path(plan["eval_support_path"]))
    report = validator.validate_checkpoint(plan, step)
    import torch
    optimizer = torch.load(Path(report["checkpoint"]) / "actor/optim_world_size_4_rank_0.pt",
                           map_location="cpu", weights_only=False, mmap=True)
    updates = {int(value["step"]) for value in optimizer["state"].values() if "step" in value}
    if updates != {4 * step}:
        raise RuntimeError(f"Unexpected optimizer update counts: {updates}")
    report.update(logical_optimizer_updates=4 * step)
    return report


def restore_step50_adapter(plan):
    backup = Path(plan["step50_adapter_backup"])
    target = Path(plan["checkpoint_root"]) / "global_step_50/actor/peft_adapter"
    for source in backup.iterdir():
        if not source.is_file():
            continue
        output = target / source.name
        if output.exists():
            if queue.sha256(output) != queue.sha256(source):
                raise RuntimeError(f"Step50 adapter unexpectedly changed: {output}")
        else:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)


def evaluate_step(plan, step, status):
    queue.verify_code(plan)
    controller = _load_controller(plan)
    adapter = Path(plan["checkpoint_root"]) / f"global_step_{step}/actor/peft_adapter"
    _check_adapter(adapter, plan["base_model"])
    name = f"{plan['experiment']}_step{step}_fullbench_32768_seed42"
    output = Path(plan["root"]) / "runs/outputs/full_bench_eval" / name
    job = {"step": step, "evaluation_name": name, "adapter": str(adapter),
           "adapter_sha256": queue.sha256(adapter / "adapter_model.safetensors"),
           "adapter_config_sha256": queue.sha256(adapter / "adapter_config.json"),
           "output": str(output),
           "preflight_output": str(Path(plan["job_dir"]) / f"evaluation_step{step}_preflight")}
    queue.write_json(Path(plan["job_dir"]) / f"evaluation_step{step}_job.json", job)
    subprocess.run(["bash", str(Path(plan["root"]) / "tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh")],
                   env=controller.environment(plan, job, dry=True), cwd=plan["root"], check=True)
    controller.check_manifests(plan, job, job["preflight_output"])
    controller.run_job(plan, job, status)
    verification = json.loads((output / "verification.json").read_text())
    if verification.get("status") != "verified" or verification.get("num_samples") != 7248:
        raise RuntimeError("Evaluation did not verify all 7248 responses")
    return str(output)


def write_results(plan, completed):
    rows = {50: weighted_summary(plan["step50_eval_dir"])}
    for step, directory in completed.items():
        rows[step] = weighted_summary(directory)
    payload = {"avg_definition": "response-count weighted; correct responses / 7248",
               "observation_steps": 32, "effective_prompt_groups": 735, "steps": rows,
               "delta_vs_step50_pp": {s: 100 * (r["weighted_avg"] - rows[50]["weighted_avg"])
                                      for s, r in rows.items()}}
    job = Path(plan["job_dir"])
    queue.write_json(job / "results.json", payload)
    lines = ["REAL probe32 continuation. Step = trainer step; optimizer updates = 4 × Step.", "",
             "Avg. is response-count weighted. Task cells: Avg@k / Pass@k (%).", "",
             "| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for step, row in rows.items():
        cells = [f"{100*row['benchmarks'][k]['avg_at_k']:.1f} / {100*row['benchmarks'][k]['pass_at_k']:.1f}"
                 for k in BENCHMARKS]
        lines.append(f"| {step} | {100*row['weighted_avg']:.1f} | " + " | ".join(cells) + " |")
    (job / "results.md").write_text("\n".join(lines) + "\n")


def preflight(plan, log):
    queue.verify_code(plan)
    report = validate_resume(plan, 50)
    queue.write_json(Path(plan["job_dir"]) / "resume_step50_verification.json", report)
    env = training_environment(plan)
    env["RESUME_MODE"] = "auto"
    launch = ["bash", str(Path(plan["root"]) / "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh")]
    with Path(log).open("x") as handle:
        subprocess.run(launch, env={**env, "DRY_RUN": "1"}, cwd=plan["root"],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    return launch, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    job = Path(plan["job_dir"])
    if args.check:
        preflight(plan, job / "registration_preflight.log")
        print("Verified step50 optimizer/RNG/scheduler/dataloader and continuation dry run", flush=True)
        return
    with ExitStack() as stack:
        for path in [job / "controller.lock", *map(Path, plan["run_locks"])]:
            lock = stack.enter_context(path.open("a+"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (job / "status.json").exists():
            raise RuntimeError("Continuation already registered; inspect state before restarting")
        state = {"experiment": plan["experiment"], "controller_pid": os.getpid(),
                 "controller_start_ticks": queue.process_info(os.getpid())["start_ticks"],
                 "started_at": queue.now(), "plan": str(args.plan), "resume_step": 50,
                 "target_step": 150, "completed_evaluations": [50]}

        def status(stage, **details):
            state.update(status=stage, updated_at=queue.now(), **details)
            queue.write_json(job / "status.json", state)
            print(f"[{queue.now()}] {stage}: {json.dumps(details)}", flush=True)

        try:
            status("validating_resume")
            queue.wait_idle(status)
            launch, env = preflight(plan, job / "training_preflight.log")
            try:
                queue.run_stage(plan, "train", launch, env, lambda: queue.training_complete(plan), status)
            finally:
                restore_step50_adapter(plan)
            report = validate_resume(plan, 150)
            queue.write_json(job / "step150_checkpoint_verification.json", report)
            validator = _load_file("real_resume_validator100", Path(plan["eval_support_path"]))
            queue.write_json(job / "step100_checkpoint_verification.json", validator.validate_checkpoint(plan, 100))
            completed = {}
            write_results(plan, completed)
            for step in (100, 150):
                queue.wait_idle(status)
                completed[step] = evaluate_step(plan, step, status)
                write_results(plan, completed)
                status(f"evaluation_step{step}_complete", completed_evaluations=[50, *completed])
            status("complete", results=str(job / "results.md"), auto_resume=False)
        except BaseException as error:
            status("error", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
