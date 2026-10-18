#!/usr/bin/env python3
"""Train/evaluate a saved REAL prefix after a verified predecessor finishes."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.local import run_real_after_eval as queue
from scripts.local.real_experiment_eval import evaluate_real


def predecessor_ready(plan):
    dependency = plan["predecessor"]
    state = json.loads(Path(dependency["status"]).read_text())
    if state.get("experiment") != dependency["experiment"]:
        raise RuntimeError("Predecessor experiment mismatch")
    if state.get("status") == "error":
        raise RuntimeError(f"Predecessor failed: {state.get('error')}")
    identity = dependency["process"]
    process = queue.process_info(identity["pid"])
    alive = (process is not None and process["state"] != "Z"
             and process["start_ticks"] == identity["start_ticks"])
    if state.get("status") != "complete":
        if not alive:
            raise RuntimeError("Predecessor exited without verified completion")
        return False
    result = json.loads(Path(dependency["result"]).read_text())
    expected = f"{dependency['experiment']}_step50_fullbench_32768_seed42"
    if (result.get("status") != "verified_real_step50_comparison"
            or result.get("evaluation_name") != expected or result.get("num_samples") != 7248):
        raise RuntimeError("Predecessor evaluation result is not verified")
    verification = json.loads((Path(dependency["eval_output_dir"]) / "verification.json").read_text())
    if verification.get("status") != "verified" or verification.get("num_samples") != 7248:
        raise RuntimeError("Predecessor benchmark verification failed")
    return not alive


def training_environment(plan):
    env = queue.common_environment(plan)
    env.update(
        METHOD="real", REAL_ARTIFACT_DIR=plan["artifact_dir"], EXP_NAME=plan["experiment"],
        PROTOCOL_FILE=plan["protocol_file"], LR="1e-5", LORA_RANK="32",
        TARGET_STEP=str(plan["target_step"]), RAY_NUM_CPUS="20",
        RAY_TEMP_BASE=plan["execution"]["ray_train_temp_base"],
        LOG_FILE=str(Path(plan["job_dir"]) / "train_metrics.log"),
    )
    return env


def preflight(plan, filename):
    queue.verify_code(plan)
    metadata = json.loads((Path(plan["artifact_dir"]) / "summary.json").read_text())
    if (metadata.get("trainer_step") != plan["observation"]["steps"]
            or metadata.get("replay_source") != plan["cache_snapshot"]
            or metadata.get("optimizer_updates") != 0):
        raise RuntimeError("Replayed artifact does not match the registered prefix")
    launch = ["bash", str(Path(plan["root"]) / "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh")]
    with Path(filename).open("x") as handle:
        subprocess.run(launch, env={**training_environment(plan), "DRY_RUN": "1"},
                       cwd=plan["root"], stdout=handle, stderr=subprocess.STDOUT, check=True)
    return launch


BENCHMARKS = {"aime24": 960, "aime25": 960, "amc23": 1280,
              "hmmt_feb": 960, "math500": 2000, "minerva": 1088}


def weighted_summary(directory):
    files = list((Path(directory) / "summary").glob("*.json"))
    if len(files) != 1:
        raise ValueError("Expected one evaluation summary")
    data = json.loads(files[0].read_text())
    rows = {row["benchmark"]: row for row in data["benchmarks"]}
    if len(data["benchmarks"]) != 6 or {k: r["num_samples"] for k, r in rows.items()} != BENCHMARKS:
        raise ValueError("Unexpected benchmark sample counts")
    weighted = sum(rows[k]["avg_at_k"] * n for k, n in BENCHMARKS.items()) / 7248
    return {"checkpoint": data["checkpoint"], "weighted_avg": weighted, "num_samples": 7248,
            "benchmarks": rows}


def write_comparison(plan):
    rows = [("LoRA", weighted_summary(plan["baseline_eval_dir"])),
            ("REAL probe64", weighted_summary(plan["predecessor"]["eval_output_dir"])),
            ("REAL probe32", weighted_summary(plan["eval_output_dir"]))]
    directory = Path(plan["job_dir"])
    queue.write_json(directory / "probe_budget_comparison.json", {
        "training_step": 50, "optimizer_updates": 200,
        "avg_definition": "sum(num_samples * avg_at_k) / sum(num_samples)",
        "runs": dict(rows),
        "probe32_minus_probe64_pp": 100 * (rows[2][1]["weighted_avg"] - rows[1][1]["weighted_avg"]),
    })
    lines = ["All runs: training Step 50 (200 optimizer updates). Avg. is weighted by response count.", "",
             "| Method | Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, row in rows:
        cells = [f"{100 * row['benchmarks'][k]['avg_at_k']:.1f} / {100 * row['benchmarks'][k]['pass_at_k']:.1f}"
                 for k in BENCHMARKS]
        lines.append(f"| {label} | 50 | {100 * row['weighted_avg']:.1f} | " + " | ".join(cells) + " |")
    (directory / "probe_budget_comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    directory = Path(plan["job_dir"])
    queue.verify_code(plan)
    if args.check:
        preflight(plan, directory / "registration_preflight.log")
        print(json.dumps({"status": "preflight_passed", "predecessor_ready": predecessor_ready(plan)}))
        return
    with ExitStack() as stack:
        lock = stack.enter_context((directory / "controller.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "status.json").exists():
            raise RuntimeError("Queue already registered; inspect existing state before restarting")
        state = {"experiment": plan["experiment"], "plan": str(args.plan),
                 "controller_pid": os.getpid(), "controller_start_ticks": queue.process_info(os.getpid())["start_ticks"],
                 "started_at": queue.now()}

        def status(stage, **details):
            state.update(status=stage, updated_at=queue.now(), **details)
            queue.write_json(directory / "status.json", state)
            print(f"[{queue.now()}] {stage}: {json.dumps(details)}", flush=True)

        try:
            while not predecessor_ready(plan):
                status("waiting_for_real64_evaluation", prerequisite=plan["predecessor"]["status"])
                time.sleep(30)
            for path in [plan["predecessor"]["lock"], *plan["prerequisite_locks"]]:
                handle = stack.enter_context(Path(path).open("a+"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            queue.wait_idle(status)
            queue.verify_assets(plan)
            if list(Path(plan["checkpoint_root"]).glob("global_step_*")) or Path(plan["eval_output_dir"]).exists():
                raise RuntimeError("Refusing existing training or evaluation output")
            status("training_preflight")
            launch = preflight(plan, directory / "training_preflight.log")
            queue.run_stage(plan, "train", launch, training_environment(plan),
                            lambda: queue.training_complete(plan), status)
            queue.wait_idle(status)
            queue.verify_code(plan)
            result = evaluate_real(plan, status)
            write_comparison(plan)
            status("complete", result=result, auto_resume=False,
                   weighted_comparison=str(directory / "probe_budget_comparison.md"))
        except BaseException as error:
            status("error", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
