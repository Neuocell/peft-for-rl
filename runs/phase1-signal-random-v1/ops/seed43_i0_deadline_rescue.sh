#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/peft-for-rl"
jq_bin="/home/node/anaconda3/bin/jq"
controller_pgid=27424
eval_name="phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42"
eval_dir="${repo_root}/runs/phase1-signal-random-v1/outputs/full_bench_eval/${eval_name}"
adapter="${repo_root}/runs/phase1-signal-random-v1/ckpts/verl/DAPO-Math-17k/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3/global_step_50/actor/peft_adapter"
snapshot="/data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
snapshot_sha256="3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da"
summary="${eval_dir}/summary/${eval_name}.json"

date "+DEADLINE_RESCUE_CHECK %Y-%m-%d %H:%M:%S %z"
if [[ -s "${summary}" ]]; then
    echo "NO_ACTION summary already exists: ${summary}"
    exit 0
fi

for shard_index in 0 1 2 3; do
    shard_tag="$(printf '%02d' "${shard_index}")"
    shard="${eval_dir}/shards/${eval_name}.shard-${shard_tag}-of-04.jsonl"
    manifest="${eval_dir}/manifests/${eval_name}.shard-${shard_tag}-of-04.json"
    log="${eval_dir}/logs/${eval_name}.shard-${shard_index}-of-4.log"

    [[ -s "${shard}" ]] || { echo "NO_ACTION missing shard ${shard_index}"; exit 0; }
    [[ "$(wc -l < "${shard}")" -eq 1812 ]] || {
        echo "NO_ACTION shard ${shard_index} does not contain 1812 rows"
        exit 0
    }
    "${jq_bin}" -e . "${shard}" >/dev/null || {
        echo "NO_ACTION shard ${shard_index} contains invalid JSON"
        exit 0
    }

    [[ -s "${manifest}" ]] || { echo "NO_ACTION missing manifest ${shard_index}"; exit 0; }
    "${jq_bin}" -e \
        --arg checkpoint "${eval_name}" \
        --arg adapter "${adapter}" \
        --arg snapshot "${snapshot}" \
        --arg snapshot_sha256 "${snapshot_sha256}" \
        --argjson shard_index "${shard_index}" \
        '.checkpoint == $checkpoint and
         .lora_adapter == $adapter and
         .benchmark_snapshot_records == $snapshot and
         .benchmark_snapshot_sha256 == $snapshot_sha256 and
         .total_requests == 7248 and
         .shard_requests == 1812 and
         .num_shards == 4 and
         .shard_index == $shard_index and
         .seed == 42 and
         .temperature == 0.6 and
         .top_p == 0.95 and
         .max_new_tokens == 32768 and
         .max_model_len == 34816 and
         .samples_small == 32 and
         .samples_large == 4 and
         .limit_per_benchmark == null and
         .benchmarks == ["aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva"]' \
        "${manifest}" >/dev/null || {
            echo "NO_ACTION manifest ${shard_index} does not match the fixed protocol"
            exit 0
        }

    [[ -s "${log}" ]] || { echo "NO_ACTION missing log ${shard_index}"; exit 0; }
    grep -Fq "Wrote 1812 rows for shard ${shard_index}/4:" "${log}" || {
        echo "NO_ACTION shard ${shard_index} has no completion marker"
        exit 0
    }
done

if grep -Ein \
    'traceback|cuda out of memory|nccl.*(error|timeout)|no space left|segmentation fault|segfault' \
    "${eval_dir}"/logs/*.shard-*-of-4.log; then
    echo "NO_ACTION fatal error pattern found in shard logs"
    exit 0
fi

mapfile -t engine_pids < <(
    ps -eo pid=,pgid=,comm= | awk -v pgid="${controller_pgid}" \
        '$2 == pgid && $3 == "VLLM::EngineCor" {print $1}'
)
if (( ${#engine_pids[@]} == 0 )); then
    echo "NO_ACTION all outputs are complete but no residual EngineCore is present"
    exit 0
fi

echo "STRICT_PRECONDITIONS_PASSED residual EngineCore PIDs: ${engine_pids[*]}"
ps -o pid,ppid,pgid,sid,stat,etime,comm,cmd -p "$(IFS=,; echo "${engine_pids[*]}")"
kill -TERM "${engine_pids[@]}"
echo "SIGTERM_SENT only to residual EngineCore PIDs: ${engine_pids[*]}"
