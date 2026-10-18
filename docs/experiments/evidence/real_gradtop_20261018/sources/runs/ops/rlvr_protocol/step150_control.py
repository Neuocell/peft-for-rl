#!/usr/bin/env python3
"""Validate this run's full checkpoints and stop/evaluate at its atomic step150 marker."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
NAME = "rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42"
PLAN = ROOT / f"{NAME}.step150_plan.json"


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(plan, step):
    import torch
    from safetensors import safe_open

    checkpoint = Path(plan["checkpoint_root"]) / f"global_step_{step}"
    sizes = {}
    for rank in range(4):
        for kind in ("model", "optim", "extra_state"):
            path = checkpoint / "actor" / f"{kind}_world_size_4_rank_{rank}.pt"
            size = path.stat().st_size
            assert size > {"model": 100_000_000, "optim": 1_000_000, "extra_state": 1000}[kind], str(path)
            with zipfile.ZipFile(path) as archive:
                assert archive.namelist(), str(path)
            sizes[str(path.relative_to(checkpoint))] = size
        extra = torch.load(checkpoint / "actor" / f"extra_state_world_size_4_rank_{rank}.pt", map_location="cpu", weights_only=False)
        assert {"cpu", "numpy", "random", "cuda"}.issubset(extra["rng"]), f"Missing rank{rank} RNG"
        scheduler = extra["lr_scheduler"]
        assert scheduler["last_epoch"] == step, scheduler
        assert scheduler["_last_lr"] == [1e-5], scheduler

    data = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
    sampler = data["_sampler_iter_state"]
    assert data["_num_yielded"] == step, data["_num_yielded"]
    assert data["_sampler_iter_yielded"] == step, data["_sampler_iter_yielded"]
    assert sampler["samples_yielded"] == step * 64, sampler["samples_yielded"]
    assert sampler["sampler_iter_state"]["yielded"] == step * 64
    adapter = checkpoint / "actor/peft_adapter"
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert (config["peft_type"], config["r"], config["lora_alpha"], config["lora_dropout"]) == ("LORA", 32, 64, 0.0)
    assert Path(config["base_model_name_or_path"]).resolve() == Path(plan["base_model"]).resolve()
    weights = adapter / "adapter_model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        assert keys, "Empty adapter"
        for key in keys:
            shape = handle.get_slice(key).get_shape()
            if "lora_A" in key:
                assert shape[0] == 32, (key, shape)
            elif "lora_B" in key:
                assert shape[1] == 32, (key, shape)
            else:
                raise ValueError(f"Unexpected adapter tensor: {key}")
    return {"status": "verified", "validated_at": now(), "step": step,
            "checkpoint": str(checkpoint), "model_optimizer_extra_files": sizes,
            "dataloader_batches_yielded": step, "dataloader_samples_yielded": step * 64,
            "all_four_rank_rng_and_scheduler_verified": True,
            "adapter_tensor_count": len(keys), "adapter_sha256": sha256(weights)}


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": int(pid), "state": fields[0], "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_group(plan, require_leader=True):
    pid, pgid = plan["training_pid"], plan["training_pgid"]
    assert pgid != os.getpgrp(), "Controller shares the training process group"
    leader = process_info(pid)
    if leader and leader["state"] != "Z":
        assert leader["start_ticks"] == plan["training_start_ticks"], "Training PID was reused"
        assert leader["pgid"] == leader["sid"] == pgid, leader
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            if require_leader:
                raise RuntimeError("Training launcher exited before checkpoint150 was saved")
        else:
            assert f"EXP_NAME={plan['experiment']}".encode() in env, "Training experiment changed"
    elif require_leader:
        raise RuntimeError("Training launcher exited before checkpoint150 was saved")
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            item = process_info(int(path.name))
            if item and item["pgid"] == pgid and item["state"] != "Z":
                assert item["sid"] == pgid, f"Unexpected process in group: {item}"
                members.append(item)
    return members


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    return sorted(set(int(line.strip()) for line in result.stdout.splitlines() if line.strip()))


def recent_step(path):
    with Path(path).open("rb") as handle:
        handle.seek(max(0, os.fstat(handle.fileno()).st_size - 131072))
        matches = re.findall(rb"training/global_step:(\d+)", handle.read())
    return int(matches[-1]) if matches else None


def control(plan):
    lock = (ROOT / f"{NAME}.step150_controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT / f"{NAME}.step150_status.json"
    state = {"controller_pid": os.getpid(), "experiment": NAME, "target_step": 150,
             "plan": str(PLAN), "started_at": now()}

    def status(stage, **details):
        state.update(status=stage, updated_at=now(), **details)
        write_json(status_path, state)
        print(f"[{now()}] {stage}: {json.dumps(details, ensure_ascii=False)}", flush=True)

    try:
        assert plan["target_step"] == 150
        assert sha256(plan["protocol_file"]) == plan["protocol_sha256"], "Training protocol changed"
        assert sha256(plan["benchmark_snapshot"]) == plan["benchmark_snapshot_sha256"], "Benchmark snapshot changed"
        output = Path(plan["evaluation_output"])
        assert not output.exists(), f"Evaluation output already exists: {output}"
        marker = Path(plan["checkpoint_root"]) / "latest_checkpointed_iteration.txt"
        next_heartbeat = 0
        while True:
            step = int(marker.read_text().strip()) if marker.exists() else 0
            assert step <= 150, f"Checkpoint marker advanced beyond150: {step}; refusing to stop"
            if step == 150:
                break
            owned_group(plan)
            if time.monotonic() >= next_heartbeat:
                status("waiting_for_checkpoint150", latest_full_checkpoint=step,
                       latest_completed_trainer_step=recent_step(plan["training_log"]))
                next_heartbeat = time.monotonic() + 60
            time.sleep(2)

        verified = validate_checkpoint(plan, 150)
        write_json(ROOT / f"{NAME}.step150_checkpoint_verification.json", verified)
        status("checkpoint150_verified", checkpoint=verified["checkpoint"], adapter_sha256=verified["adapter_sha256"])
        members = owned_group(plan)
        assert members, "Training group disappeared before stop"
        status("stopping_training", training_group_members=len(members))
        os.killpg(plan["training_pgid"], signal.SIGTERM)
        deadline = time.monotonic() + 120
        while owned_group(plan, require_leader=False) and time.monotonic() < deadline:
            time.sleep(2)
        remaining = owned_group(plan, require_leader=False)
        if remaining:
            status("stopping_remaining_training_processes", remaining_pids=[item["pid"] for item in remaining])
            os.killpg(plan["training_pgid"], signal.SIGKILL)
        deadline = time.monotonic() + 180
        status("waiting_for_gpu_release")
        while True:
            remaining_gpu_pids = gpu_processes()
            if not remaining_gpu_pids and not owned_group(plan, require_leader=False):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Processes remain; refusing evaluation: {remaining_gpu_pids}")
            time.sleep(2)
        process_file = ROOT / f"{NAME}.process.json"
        process = json.loads(process_file.read_text())
        process.update(status="stopped_at_checkpoint150", stopped_at=now(), latest_completed_trainer_step=150,
                       latest_full_checkpoint=150, resume_checkpoint=verified["checkpoint"])
        write_json(process_file, process)
        eval_process = subprocess.Popen(["bash", plan["evaluation_launcher"]], start_new_session=True)
        status("evaluating_checkpoint150", evaluation_pid=eval_process.pid, evaluation_output=plan["evaluation_output"])
        code = eval_process.wait()
        assert code == 0, f"Evaluation/verification failed with exit{code}; see evaluation logs"
        report = json.loads(Path(plan["evaluation_verification"]).read_text())
        assert report["status"] == "verified" and report["num_samples"] == 7248, report
        assert sha256(Path(verified["checkpoint"]) / "actor/peft_adapter/adapter_model.safetensors") == verified["adapter_sha256"], "Adapter changed during evaluation"
        status("evaluation_complete", verification=plan["evaluation_verification"],
               summary=plan["evaluation_summary"], resume_checkpoint=verified["checkpoint"], auto_resume=False)
    except BaseException as error:
        status("error", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("watch", "validate", "require-idle"))
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if args.action == "validate":
        step = args.step
        if step is None:
            step = int((Path(plan["checkpoint_root"]) / "latest_checkpointed_iteration.txt").read_text())
            assert step >= 150, "Resume requires the completed step150 checkpoint"
        print(json.dumps(validate_checkpoint(plan, step), indent=2))
    elif args.action == "require-idle":
        assert not gpu_processes(), "GPU computation is still active"
        assert not owned_group(plan, require_leader=False), "Original training process group is still active"
        try:
            lock = (ROOT / f"{NAME}.step150_controller.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("step150 controller/evaluation is still active")
    else:
        control(plan)


if __name__ == "__main__":
    main()
