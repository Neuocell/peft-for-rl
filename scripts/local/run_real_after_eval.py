#!/usr/bin/env python3
"""Wait for verified baseline evaluation, then run REAL observation/train/eval."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def prerequisite_ready(plan):
    """A stale summary field or a dead evaluation process is not completion."""
    state = json.loads(Path(plan["prerequisite_status"]).read_text())
    if state.get("experiment") != plan["baseline_experiment"]:
        raise RuntimeError("Prerequisite status belongs to a different experiment")
    if state.get("status") == "error":
        raise RuntimeError(f"Prerequisite evaluation failed: {state.get('error')}")
    complete = state.get("status") == "all_evaluations_complete"
    if complete:
        if set(state.get("completed_steps", [])) != {50, 100} or state.get("current_step") is not None:
            raise RuntimeError("Prerequisite completion status is inconsistent")
        for path in plan["prerequisite_verifications"]:
            report = json.loads(Path(path).read_text())
            if report.get("status") != "verified" or report.get("num_samples") != 7248:
                raise RuntimeError(f"Prerequisite benchmark verification failed: {path}")
        comparison = json.loads(Path(state["comparison"]).read_text())
        if comparison.get("status") != "verified_three_checkpoints":
            raise RuntimeError("Prerequisite checkpoint comparison is not verified")
    identity = plan["prerequisite_process"]
    process = process_info(identity["pid"])
    alive = process is not None and process["state"] != "Z" and process["start_ticks"] == identity["start_ticks"]
    if not complete and not alive:
        raise RuntimeError("Prerequisite controller exited without verified completion")
    return complete and not alive


def gpu_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return sorted({int(value) for value in result.stdout.split()})


def wait_idle(status):
    while True:
        pids = gpu_pids()
        if not pids:
            return
        status("waiting_for_gpu_release", gpu_pids=pids)
        time.sleep(30)


def verify_code(plan):
    for path, expected in plan["code_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Queued implementation changed: {path}; review and requeue the plan")


def verify_assets(plan):
    protocol = json.loads(Path(plan["protocol_file"]).read_text())
    for name, spec in protocol["assets"].items():
        path = Path(spec["path"])
        if not path.is_absolute():
            path = Path(plan["root"]) / path
        if sha256(path) != spec["sha256"]:
            raise RuntimeError(f"Pinned REAL experiment asset changed: {name}")


def owned_stage_processes(token):
    marker = f"REAL_PIPELINE_STAGE_TOKEN={token}".encode()
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if marker in (entry / "environ").read_bytes().split(b"\0"):
                info = process_info(int(entry.name))
                if info and info["state"] != "Z":
                    result.append(info)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def cleanup_stage(token):
    """Only stop children carrying this stage's unique inherited token."""
    for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        for process in owned_stage_processes(token):
            current = process_info(process["pid"])
            if current and current["start_ticks"] == process["start_ticks"]:
                try:
                    os.kill(process["pid"], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + grace
        while owned_stage_processes(token) and time.monotonic() < deadline:
            time.sleep(1)
        if not owned_stage_processes(token):
            return
    raise RuntimeError("Owned REAL stage processes did not exit")


def common_environment(plan):
    env = os.environ.copy()
    python_dir = str(Path(plan["python"]).parent)
    env.update(
        PATH=f"{python_dir}:{env.get('PATH', '')}", CONDA_PREFIX=str(Path(python_dir).parent),
        PYTHON_BIN=plan["python"], PYTHONPATH=plan["root"], PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", CUDA_VISIBLE_DEVICES="0,1,2,3",
        RUNTIME_ROOT=f"{plan['root']}/runs", DRY_RUN="0", RESUME_MODE="disable",
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
    )
    return env


def recent_step(log):
    if not Path(log).is_file():
        return None
    with Path(log).open("rb") as handle:
        handle.seek(max(0, os.fstat(handle.fileno()).st_size - 131072))
        matches = re.findall(rb"training/global_step:(\d+)", handle.read())
    return int(matches[-1]) if matches else None


def run_stage(plan, stage, command, environment, completed, status):
    verify_code(plan)
    wait_idle(status)
    if shutil.disk_usage(plan["root"]).free < 15 * 1024**3:
        raise RuntimeError("Less than 15 GiB free before REAL stage; preserving existing files")
    token = uuid.uuid4().hex
    environment = {**environment, "REAL_PIPELINE_STAGE_TOKEN": token}
    log = Path(plan["job_dir"]) / f"{stage}.log"
    started = time.monotonic()
    completed_at = None
    cleaned_after_completion = False
    with log.open("x") as handle:
        child = subprocess.Popen(
            command, cwd=plan["root"], env=environment, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        status(stage, stage_pid=child.pid, stage_token=token, stage_log=str(log))
        try:
            while child.poll() is None:
                if completed() and completed_at is None:
                    completed_at = time.monotonic()
                    status(f"{stage}_output_ready", stage_log=str(log))
                if completed_at is not None and time.monotonic() - completed_at > 90:
                    status(f"{stage}_cleaning_completed_workers")
                    cleaned_after_completion = True
                    cleanup_stage(token)
                    child.wait(timeout=15)
                    break
                status(stage, elapsed_seconds=round(time.monotonic() - started),
                       latest_logged_step=recent_step(log))
                time.sleep(30)
            if not completed():
                raise RuntimeError(f"REAL {stage} exited {child.returncode} without complete outputs: {log}")
            if child.returncode != 0 and not cleaned_after_completion:
                raise RuntimeError(f"REAL {stage} exited {child.returncode}; inspect {log}")
        finally:
            cleanup_stage(token)
    write_json(Path(plan["job_dir"]) / f"{stage}_execution.json", {
        "status": "completed", "finished_at": now(), "wall_seconds": time.monotonic() - started,
        "process_returncode": child.returncode, "command": command, "log": str(log),
        "cleaned_after_complete_output": cleaned_after_completion,
    })
    status(f"{stage}_completed", stage_pid=None, elapsed_seconds=round(time.monotonic() - started))


def probe_complete(plan):
    directory = Path(plan["artifact_dir"])
    if not all((directory / name).is_file() for name in ("rank_map.json", "summary.json", "subspaces.safetensors")):
        return False
    summary = json.loads((directory / "summary.json").read_text())
    return summary.get("trainer_step") == 64 and summary.get("prompt_groups", 0) > 0 and summary.get("optimizer_updates") == 0


def training_complete(plan):
    root = Path(plan["checkpoint_root"])
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or marker.read_text().strip() != str(plan["target_step"]):
        return False
    checkpoint = root / f"global_step_{plan['target_step']}"
    files = [checkpoint / "data.pt", checkpoint / "actor/peft_adapter/adapter_config.json",
             checkpoint / "actor/peft_adapter/adapter_model.safetensors"]
    files.extend(checkpoint / "actor" / f"{kind}_world_size_4_rank_{rank}.pt"
                 for kind in ("model", "optim", "extra_state") for rank in range(4))
    return all(path.is_file() and path.stat().st_size > 0 for path in files)


def execute(plan, status):
    root = Path(plan["root"])
    status("verifying_experiment_assets")
    verify_assets(plan)
    artifact = Path(plan["artifact_dir"])
    if artifact.exists() and any(artifact.iterdir()):
        raise RuntimeError(f"Refusing to overwrite an existing probe artifact: {artifact}")
    if list(Path(plan["checkpoint_root"]).glob("global_step_*")):
        raise RuntimeError("REAL training checkpoints already exist; this queue starts from the base policy")
    probe_env = common_environment(plan)
    probe_env.update(
        EXP_NAME=plan["probe_experiment"], GRADIENT_PROBE_OUTPUT_DIR=str(artifact),
        REAL_PROBE_RANK="32", REAL_PROBE_INPUT_WIDTH="128", REAL_PROBE_OUTPUT_WIDTH="128",
        REAL_PROBE_STEPS="64", REAL_PROBE_PROMPTS="0", REAL_PROBE_SEED="42",
        TRAIN_PROMPT_BSZ="64", TRAIN_PROMPT_MINI_BSZ="16", N_RESP_PER_PROMPT="8",
        LR="1e-5", LORA_INIT_SEED="42", DATA_SEED="42", PPO_DATA_LOADER_SEED="42",
        RAY_TEMP_DIR="/tmp/realp_1013", RAY_NUM_CPUS="20",
        GPU_MEMORY_UTILIZATION="0.70", ACTOR_PPO_MAX_TOKEN_LEN="12288",
        INFER_PPO_MAX_TOKEN_LEN="12288", ROLLOUT_MAX_NUM_SEQS="256",
    )
    run_stage(plan, "probe", ["bash", str(root / "scripts/local/start_real_gradient_probe_4gpu.sh")],
              probe_env, lambda: probe_complete(plan), status)

    train_env = common_environment(plan)
    train_env.update(
        METHOD="real", REAL_ARTIFACT_DIR=str(artifact), EXP_NAME=plan["experiment"],
        PROTOCOL_FILE=plan["protocol_file"],
        LR="1e-5", LORA_RANK="32", TARGET_STEP=str(plan["target_step"]),
        RAY_NUM_CPUS="20", RAY_TEMP_BASE="/tmp/real_1013",
        LOG_FILE=str(Path(plan["job_dir"]) / "train_metrics.log"),
    )
    launch = ["bash", str(root / "scripts/local/run_rlvr_protocol_1p5b_4gpu.sh")]
    status("training_preflight")
    with (Path(plan["job_dir"]) / "training_preflight.log").open("x") as handle:
        subprocess.run(launch, env={**train_env, "DRY_RUN": "1"}, cwd=root,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    run_stage(plan, "train", launch, train_env, lambda: training_complete(plan), status)
    wait_idle(status)
    verify_code(plan)
    from real_experiment_eval import evaluate_real
    result = evaluate_real(plan, status)
    status("complete", result=result, auto_resume=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="validate queue dependencies without launching a stage")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    verify_code(plan)
    if args.check:
        print(json.dumps({"checked_at": now(), "prerequisite_ready": prerequisite_ready(plan),
                          "gpu_pids": gpu_pids(), "plan": str(args.plan)}))
        return
    job_dir = Path(plan["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((job_dir / "controller.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (job_dir / "status.json").exists():
            previous = json.loads((job_dir / "status.json").read_text())
            if previous.get("status") == "complete":
                raise RuntimeError("REAL pipeline already completed")
        state = {"controller_pid": os.getpid(), "controller_start_ticks": process_info(os.getpid())["start_ticks"],
                 "experiment": plan["experiment"], "plan": str(args.plan), "started_at": now()}

        def status(stage, **details):
            state.update(status=stage, updated_at=now(), **details)
            write_json(job_dir / "status.json", state)
            print(f"[{now()}] {stage}: {json.dumps(details, ensure_ascii=False)}", flush=True)

        try:
            while not prerequisite_ready(plan):
                status("waiting_for_baseline_evaluations", prerequisite=plan["prerequisite_status"])
                time.sleep(30)
            for filename in plan["prerequisite_locks"]:
                handle = stack.enter_context(Path(filename).open("a+"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status("baseline_evaluations_verified")
            execute(plan, status)
        except BaseException as error:
            status("error", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
