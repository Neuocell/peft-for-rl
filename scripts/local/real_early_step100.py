#!/usr/bin/env python3
"""Take over an active REAL continuation, evaluate100, then resume150."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.local import run_real_after_eval as queue
from scripts.local.continue_real_probe32 import (
    validate_resume, restore_step50_adapter, evaluate_step, write_results,
)
from scripts.local.run_real_cached_after_eval import training_environment


def same_live_process(identity):
    current = queue.process_info(identity["pid"])
    return bool(current and current["state"] != "Z" and current["start_ticks"] == identity["start_ticks"])


def validate_takeover(plan):
    takeover = plan["takeover"]
    if not same_live_process(takeover["controller"]) or not same_live_process(takeover["training"]):
        raise RuntimeError("Original controller/training process missing or PID reused")
    state = json.loads(Path(takeover["status_path"]).read_text())
    if (state.get("experiment") != plan["experiment"] or state.get("status") != "train"
            or state.get("stage_token") != takeover["stage_token"]
            or state.get("controller_pid") != takeover["controller"]["pid"]
            or state.get("stage_pid") != takeover["training"]["pid"]):
        raise RuntimeError("Original active stage identity changed")
    environment = Path(f"/proc/{takeover['training']['pid']}/environ").read_bytes().split(b"\0")
    if f"REAL_PIPELINE_STAGE_TOKEN={takeover['stage_token']}".encode() not in environment:
        raise RuntimeError("Training process token mismatch")
    marker = int((Path(plan["checkpoint_root"]) / "latest_checkpointed_iteration.txt").read_text())
    if marker not in (50, 100):
        raise RuntimeError(f"Cannot intercept step100 from checkpoint {marker}")


def prune_step50_training_state(plan):
    # Only after the successor checkpoint is fully validated and evaluated.
    validate_resume(plan, 100)
    old = Path(plan["checkpoint_root"]) / "global_step_50"
    files = [old / "data.pt", *(old / "actor").glob("*_world_size_4_rank_*.pt")]
    records = [{"path": str(p), "bytes": p.stat().st_size} for p in files if p.is_file()]
    report = Path(plan["job_dir"]) / "step50_training_state_cleanup.json"
    queue.write_json(report, {"status": "planned", "files": records, "resume_step": 100})
    for record in records:
        Path(record["path"]).unlink()
    queue.write_json(report, {"status": "complete", "files": records, "resume_step": 100,
                             "freed_bytes": sum(r["bytes"] for r in records)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    queue.verify_code(plan)
    validate_takeover(plan)
    if args.check:
        print("Code hashes, original controller, training identity and stage token verified")
        return
    job = Path(plan["job_dir"])
    takeover = plan["takeover"]
    with ExitStack() as stack:
        lock = stack.enter_context((job / "controller.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (job / "status.json").exists():
            raise RuntimeError("Early evaluation already registered")
        state = {"experiment": plan["experiment"], "controller_pid": os.getpid(),
                 "controller_start_ticks": queue.process_info(os.getpid())["start_ticks"],
                 "started_at": queue.now(), "plan": str(args.plan), "target_step": 150,
                 "completed_evaluations": [50]}

        def status(stage, **details):
            state.update(status=stage, updated_at=queue.now(), **details)
            queue.write_json(job / "status.json", state)
            print(f"[{queue.now()}] {stage}: {json.dumps(details)}", flush=True)

        try:
            # The original training stage is in its own session. Kill only the
            # old Python controller, bypassing its finally-block worker cleanup.
            # The existing trainer keeps running; this process assumes ownership.
            validate_takeover(plan)
            os.kill(takeover["controller"]["pid"], signal.SIGKILL)
            deadline = time.monotonic() + 15
            while same_live_process(takeover["controller"]):
                if time.monotonic() > deadline:
                    raise RuntimeError("Original controller did not exit")
                time.sleep(0.2)
            for path in plan["run_locks"]:
                handle = stack.enter_context(Path(path).open("a+"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = json.loads(Path(takeover["status_path"]).read_text())
            previous.update(status="superseded_by_early_step100", updated_at=queue.now(),
                            replacement_status=str(job / "status.json"), replacement_controller_pid=os.getpid())
            queue.write_json(takeover["status_path"], previous)
            status("waiting_for_checkpoint100", stage_pid=takeover["training"]["pid"],
                   stage_token=takeover["stage_token"], stage_log=takeover["train_log"])
            checkpoint_plan = {**plan, "target_step": 100}
            last_heartbeat = 0
            while not queue.training_complete(checkpoint_plan):
                if not same_live_process(takeover["training"]):
                    raise RuntimeError("Training exited before complete checkpoint100")
                marker = int((Path(plan["checkpoint_root"]) / "latest_checkpointed_iteration.txt").read_text())
                if marker > 100:
                    raise RuntimeError("Missed checkpoint100 boundary")
                if time.monotonic() - last_heartbeat > 30:
                    status("waiting_for_checkpoint100", latest_logged_step=queue.recent_step(takeover["train_log"]))
                    last_heartbeat = time.monotonic()
                time.sleep(5)
            report = validate_resume(plan, 100)
            queue.write_json(job / "step100_checkpoint_verification.json", report)
            status("stopping_after_verified_checkpoint100", checkpoint=report["checkpoint"])
            queue.cleanup_stage(takeover["stage_token"])
            queue.wait_idle(status)
            status("checkpoint100_ready_for_evaluation", stage_pid=None, latest_logged_step=100)
            completed = {100: evaluate_step(plan, 100, status)}
            write_results(plan, completed)
            status("evaluation_step100_complete", completed_evaluations=[50, 100], results=str(job / "results.md"))
            prune_step50_training_state(plan)
            env = training_environment(plan)
            env["RESUME_MODE"] = "auto"
            launch = ["bash", str(Path(plan["root"]) / "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh")]
            queue.verify_code(plan)
            with (job / "resume100_preflight.log").open("x") as handle:
                subprocess.run(launch, env={**env, "DRY_RUN": "1"}, cwd=plan["root"],
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            try:
                queue.run_stage(plan, "train", launch, env, lambda: queue.training_complete(plan), status)
            finally:
                restore_step50_adapter(plan)
            queue.write_json(job / "step150_checkpoint_verification.json", validate_resume(plan, 150))
            queue.wait_idle(status)
            completed[150] = evaluate_step(plan, 150, status)
            write_results(plan, completed)
            status("complete", completed_evaluations=[50, 100, 150], results=str(job / "results.md"), auto_resume=False)
        except BaseException as error:
            status("error", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
