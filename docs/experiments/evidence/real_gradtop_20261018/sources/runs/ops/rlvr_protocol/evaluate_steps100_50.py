#!/usr/bin/env python3
"""Run this experiment's step100 then step50 six-benchmark evaluations."""
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from step150_control import now, process_info, sha256, write_json

OPS = Path(__file__).resolve().parent
NAME = "rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42"
PLAN_PATH = OPS / f"{NAME}.eval100_50_plan.json"
SIZES = {"aime24": 30, "aime25": 30, "amc23": 40, "hmmt_feb": 30, "math500": 500, "minerva": 272}
SAMPLES = {key: (32 if key in ("aime24", "aime25", "amc23", "hmmt_feb") else 4) for key in SIZES}
EXPECTED_KEYS = {(key, problem, sample) for key, size in SIZES.items() for problem in range(size) for sample in range(SAMPLES[key])}


def environment(plan, job, dry=False):
    env = os.environ.copy()
    env.update(CONDA_PREFIX=str(Path(plan["python"]).parent.parent),
               PATH=f"{Path(plan['python']).parent}:{env.get('PATH', '')}",
               PYTHON_BIN=plan["python"], PYTHONPATH=plan["root"], PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", EXP_NAME=NAME,
               PROJECT_ROOT=f"{plan['root']}/tina_run", RUNTIME_ROOT=f"{plan['root']}/runs",
               BASE_MODEL=plan["base_model"], LORA_ADAPTER=job["adapter"],
               BENCHMARK_SNAPSHOT_RECORDS=plan["snapshot"], CHECKPOINT_NAME=job["evaluation_name"],
               OUTPUT_DIR=job["preflight_output"] if dry else job["output"],
               CUDA_VISIBLE_DEVICES="0,1,2,3", BENCHMARKS=",".join(SIZES),
               SEED="42", TEMPERATURE="0.6", TOP_P="0.95", MAX_NEW_TOKENS="32768",
               MAX_MODEL_LEN="34816", SAMPLES_SMALL="32", SAMPLES_LARGE="4",
               GPU_MEMORY_UTILIZATION="0.85", LIMIT_PER_BENCHMARK="",
               DRY_RUN_NO_MODEL="1" if dry else "0")
    return env


def check_assets(plan, job):
    adapter = Path(job["adapter"])
    assert sha256(adapter / "adapter_model.safetensors") == job["adapter_sha256"], "Adapter changed"
    assert sha256(adapter / "adapter_config.json") == job["adapter_config_sha256"], "Adapter config changed"
    config = json.loads((adapter / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["lora_dropout"], config["peft_type"]) == (32, 64, 0.0, "LORA")


def check_manifests(plan, job, output):
    for index in range(4):
        manifest = json.loads((Path(output) / "manifests" / f"{job['evaluation_name']}.shard-{index:02d}-of-04.json").read_text())
        expected = {"checkpoint": job["evaluation_name"], "benchmarks": list(SIZES),
                    "total_requests": 7248, "shard_requests": 1812, "num_shards": 4,
                    "shard_index": index, "seed": 42, "temperature": 0.6, "top_p": 0.95,
                    "max_new_tokens": 32768, "max_model_len": 34816, "samples_small": 32,
                    "samples_large": 4, "limit_per_benchmark": None, "benchmark_sizes": SIZES,
                    "benchmark_snapshot_sha256": plan["snapshot_sha256"]}
        for field, value in expected.items():
            assert manifest.get(field) == value, (field, manifest.get(field), value)
        for field, value in (("base_model", plan["base_model"]), ("lora_adapter", job["adapter"]),
                             ("benchmark_snapshot_records", plan["snapshot"])):
            assert Path(manifest[field]).resolve() == Path(value).resolve(), field


def completed_markers(job):
    done = []
    for index in range(4):
        log = Path(job["output"]) / "logs" / f"{job['evaluation_name']}.shard-{index}-of-4.log"
        if not log.exists():
            continue
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 4096))
            if f"Wrote 1812 rows for shard {index}/4:".encode() in handle.read():
                done.append(index)
    return done


def validate_shards(plan, job):
    keys = set()
    counts = Counter()
    shards = []
    for index in range(4):
        path = Path(job["output"]) / "shards" / f"{job['evaluation_name']}.shard-{index:02d}-of-04.jsonl"
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["benchmark"], int(row["problem_index"]), int(row["sample_index"]))
                assert key not in keys, f"Duplicate record: {key}"
                keys.add(key)
                counts[row["benchmark"]] += 1
                for field, value in (("checkpoint", job["evaluation_name"]), ("shard_index", index),
                                     ("num_shards", 4), ("seed", 42), ("temperature", 0.6),
                                     ("top_p", 0.95), ("max_new_tokens", 32768), ("max_model_len", 34816)):
                    assert row[field] == value, (field, row[field], value)
                assert Path(row["lora_adapter"]).resolve() == Path(job["adapter"]).resolve()
                count += 1
        assert count == 1812, (index, count)
        shards.append({"index": index, "records": count, "path": str(path), "sha256": sha256(path)})
    assert keys == EXPECTED_KEYS, "Incomplete or unexpected benchmark sample keys"
    check_manifests(plan, job, job["output"])
    report = {"status": "all_shards_verified", "checked_at": now(), "num_records": 7248,
              "benchmark_counts": dict(counts), "shards": shards}
    write_json(Path(job["output"]) / "shards_completion_verification.json", report)
    return report


def gpu_pids():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    return [int(value) for value in result.stdout.split()]


def group_members(pid, start_ticks, expected_name):
    assert pid != os.getpgrp(), "Evaluation shares the controller group"
    leader = process_info(pid)
    if leader and leader["state"] != "Z":
        assert leader["start_ticks"] == start_ticks and leader["pgid"] == leader["sid"] == pid, "Evaluation PID reused"
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            pass
        else:
            assert f"CHECKPOINT_NAME={expected_name}".encode() in env, "Wrong evaluation group"
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            item = process_info(int(entry.name))
            if item and item["pgid"] == pid and item["state"] != "Z":
                assert item["sid"] == pid and item["start_ticks"] >= start_ticks, item
                members.append(item)
    return members


def signal_owned(pid, start_ticks, expected_name, engines_only, sig):
    signaled = []
    for member in group_members(pid, start_ticks, expected_name):
        target = member["pid"]
        try:
            cmd = Path(f"/proc/{target}/cmdline").read_bytes()
            if engines_only and not cmd.startswith(b"VLLM::EngineCore"):
                continue
            current = process_info(target)
            if current and current["start_ticks"] == member["start_ticks"]:
                os.kill(target, sig)
                signaled.append(target)
        except (FileNotFoundError, ProcessLookupError):
            pass
    return signaled


def verify(plan, job):
    path = Path(plan["root"]) / "scripts/analysis/verify_full_bench_run.py"
    spec = importlib.util.spec_from_file_location("fullbench_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = Path(job["output"])
    report = module.verify_full_bench_run(output / "summary" / f"{job['evaluation_name']}.json",
        output / "records" / f"{job['evaluation_name']}.jsonl", output, job["evaluation_name"],
        Path(job["adapter"]), Path(plan["base_model"]), Path(plan["snapshot"]), plan["snapshot_sha256"])
    check_assets(plan, job)
    write_json(output / "verification.json", report)
    return json.loads((output / "summary" / f"{job['evaluation_name']}.json").read_text())


def preflight(plan):
    assert sha256(plan["snapshot"]) == plan["snapshot_sha256"], "Benchmark snapshot changed"
    assert sha256(Path(plan["base_model"]) / "model.safetensors") == plan["base_model_weights_sha256"], "Base model changed"
    wrapper = Path(plan["root"]) / "tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
    for job in plan["jobs"]:
        check_assets(plan, job)
        assert not Path(job["output"]).exists(), f"Formal output already exists: {job['output']}"
        subprocess.run(["bash", str(wrapper)], env=environment(plan, job, dry=True), cwd=plan["root"], check=True)
        check_manifests(plan, job, job["preflight_output"])
    write_json(OPS / f"{NAME}.eval100_50_preflight.json", {"status": "verified_no_model_loaded", "checked_at": now(),
        "steps": [100, 50], "requests_per_checkpoint": 7248, "shards_per_checkpoint": 4,
        "model_sha256": plan["base_model_weights_sha256"], "snapshot_sha256": plan["snapshot_sha256"]})


def run_job(plan, job, status):
    assert not gpu_pids(), "GPUs are busy; refusing to overlap evaluations"
    assert not Path(job["output"]).exists(), f"Evaluation output exists: {job['output']}"
    check_assets(plan, job)
    wrapper = Path(plan["root"]) / "tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
    child = subprocess.Popen(["bash", str(wrapper)], env=environment(plan, job), cwd=plan["root"], start_new_session=True)
    birth = process_info(child.pid)["start_ticks"]
    status(f"evaluating_step{job['step']}", current_step=job["step"], evaluation_pid=child.pid,
           evaluation_output=job["output"], completed_shards=[])
    completion = None
    verified_shards = None
    engines_signaled = False
    heartbeat = time.monotonic() + 60
    try:
        while child.poll() is None:
            done = completed_markers(job)
            if len(done) == 4 and verified_shards is None:
                verified_shards = validate_shards(plan, job)
                completion = time.monotonic()
                status(f"finalizing_step{job['step']}", completed_shards=done, num_records=7248)
            if completion is not None and time.monotonic() - completion >= 60 and not engines_signaled:
                targets = signal_owned(child.pid, birth, job["evaluation_name"], True, signal.SIGTERM)
                write_json(Path(job["output"]) / "exit_cleanup.json", {"at": now(), "signal": "SIGTERM", "engine_pids": targets,
                    "reason": "All7248records validated; normal process exit exceeded60seconds"})
                engines_signaled = True
                status(f"cleaning_completed_step{job['step']}", cleanup_engine_pids=targets)
            if completion is not None and time.monotonic() - completion >= 120:
                signal_owned(child.pid, birth, job["evaluation_name"], False, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    signal_owned(child.pid, birth, job["evaluation_name"], False, signal.SIGKILL)
                    child.wait(timeout=15)
                break
            if time.monotonic() >= heartbeat:
                status(f"evaluating_step{job['step']}" if completion is None else f"finalizing_step{job['step']}", completed_shards=done)
                heartbeat = time.monotonic() + 60
            time.sleep(10)
        if verified_shards is None:
            verified_shards = validate_shards(plan, job)
        summary = Path(job["output"]) / "summary" / f"{job['evaluation_name']}.json"
        if child.returncode != 0 or not summary.exists():
            args = [plan["python"], "scripts/local/eval/eval_full_bench_vllm.py", "--base_model", plan["base_model"],
                    "--checkpoint_name", job["evaluation_name"], "--output_dir", job["output"],
                    "--benchmarks", ",".join(SIZES), "--num_shards", "4", "--aggregate_only"]
            with (Path(job["output"]) / "logs/recovery_aggregate.log").open("w") as log:
                subprocess.run(args, cwd=f"{plan['root']}/tina_run", env=environment(plan, job), stdout=log, stderr=subprocess.STDOUT, check=True)
        result = verify(plan, job)
        for shard in verified_shards["shards"]:
            assert sha256(shard["path"]) == shard["sha256"], "Generated records changed during cleanup"
        signal_owned(child.pid, birth, job["evaluation_name"], True, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while group_members(child.pid, birth, job["evaluation_name"]) and time.monotonic() < deadline:
            time.sleep(1)
        if group_members(child.pid, birth, job["evaluation_name"]):
            signal_owned(child.pid, birth, job["evaluation_name"], False, signal.SIGKILL)
        deadline = time.monotonic() + 30
        while gpu_pids() and time.monotonic() < deadline:
            time.sleep(1)
        assert not gpu_pids(), "GPU computation remains; refusing next evaluation"
        status(f"step{job['step']}_verified", summary=str(summary), macro_avg_at_k=result["macro_avg_at_k_over_benchmarks"])
        return result
    except BaseException:
        signal_owned(child.pid, birth, job["evaluation_name"], False, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            signal_owned(child.pid, birth, job["evaluation_name"], False, signal.SIGKILL)
            child.wait(timeout=15)
        if group_members(child.pid, birth, job["evaluation_name"]):
            signal_owned(child.pid, birth, job["evaluation_name"], False, signal.SIGKILL)
        raise


def comparison(plan, results):
    reference = Path(plan["reference_step150"])
    verification = json.loads((reference / "verification.json").read_text())
    assert verification["status"] == "verified" and verification["num_samples"] == 7248
    paths = list((reference / "summary").glob("*.json"))
    assert len(paths) == 1
    results[150] = json.loads(paths[0].read_text())
    rows = []
    for step in (50, 100, 150):
        summary = results[step]
        rows.append({"step": step, "macro_avg_at_k": summary["macro_avg_at_k_over_benchmarks"],
            "macro_pass_at_k": summary["macro_pass_at_k_over_benchmarks"], "parse_rate": summary["parse_rate_overall"],
            "hit_max_rate": summary["hit_max_rate_overall"], "mean_length": summary["length_mean_overall"],
            "benchmarks": {row["benchmark"]: row["macro_avg_at_k"] for row in summary["benchmarks"]}})
    output = OPS / f"{NAME}.step50_100_150_comparison.json"
    write_json(output, {"status": "verified_three_checkpoints", "created_at": now(), "rows": rows})
    table = ["标准LoRA r32、lr=1e-5：step50、100、150六榜结果对照", "", "全部使用相同题目快照、eval seed42、temperature0.6、top-p0.95、最大生成长度32768；每个点7248条记录。", "",
             "| Step | Macro Avg@k | Macro Pass@k | Parse | Hit max | Mean length |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        table.append(f"| {row['step']} | {row['macro_avg_at_k']*100:.4f}% | {row['macro_pass_at_k']*100:.4f}% | {row['parse_rate']*100:.4f}% | {row['hit_max_rate']*100:.4f}% | {row['mean_length']:.1f} |")
    table.extend(["", "| Step | AIME24 | AIME25 | AMC23 | HMMT Feb | MATH500 | Minerva |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        table.append(f"| {row['step']} | " + " | ".join(f"{row['benchmarks'][key]*100:.4f}%" for key in SIZES) + " |")
    table.extend(["", "Avg@k为每题采样的平均正确率；Pass@k为至少一次答对的题目比例。前四榜k=32，后两榜k=4。训练保持暂停在step150。", ""])
    output.with_suffix(".md").write_text("\n".join(table))
    return str(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    assert plan["steps"] == [100, 50]
    if args.preflight:
        preflight(plan)
        return
    locks = []
    for name in (f"{NAME}.eval100_50.lock", f"{NAME}.step150_controller.lock"):
        lock = (OPS / name).open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    report = json.loads((OPS / f"{NAME}.eval100_50_preflight.json").read_text())
    assert report["status"] == "verified_no_model_loaded"
    assert sha256(plan["snapshot"]) == plan["snapshot_sha256"]
    state = {"controller_pid": os.getpid(), "experiment": NAME, "started_at": now(), "steps": [100, 50], "completed_steps": [], "auto_resume": False}

    def status(stage, **details):
        state.update(status=stage, updated_at=now(), **details)
        write_json(OPS / f"{NAME}.eval100_50_status.json", state)
        print(f"[{now()}] {stage}: {json.dumps(details, ensure_ascii=False)}", flush=True)

    try:
        status("starting_evaluation_queue")
        results = {}
        for job in plan["jobs"]:
            results[job["step"]] = run_job(plan, job, status)
            state["completed_steps"].append(job["step"])
            status(f"step{job['step']}_complete")
        output = comparison(plan, results)
        status("all_evaluations_complete", comparison=output, completed_steps=[100, 50], current_step=None)
    except BaseException as error:
        status("error", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
