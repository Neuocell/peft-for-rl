#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/peft-for-rl"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/peft-for-rl-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
EVAL_SERVICE="${EVAL_SERVICE:-spar-lora-tokenmask-step50-eval.service}"
EVAL_NAME="tokenmask_cov_uncentered_adaptive_eqr28_rmin16_b64m16n8_step50_fullbench_32768_seed42_v1"
EVAL_DIR="${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}"
SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"

echo "[$(date --iso-8601=seconds)] watching step-50 evaluation finalization"
while [[ ! -s "${SUMMARY}" ]]; do
    complete_shards=0
    for shard in 0 1 2 3; do
        shard_file="${EVAL_DIR}/shards/${EVAL_NAME}.shard-0${shard}-of-04.jsonl"
        if [[ -s "${shard_file}" ]] && [[ "$(wc -l <"${shard_file}")" == "1812" ]]; then
            ((complete_shards += 1))
        fi
    done

    if (( complete_shards == 4 )); then
        echo "[$(date --iso-8601=seconds)] all four shard files are complete; allowing normal aggregation to finish"
        for _ in $(seq 1 18); do
            [[ -s "${SUMMARY}" ]] && exit 0
            sleep 10
        done

        echo "[$(date --iso-8601=seconds)] shard cleanup is stalled; aggregating completed records directly"
        cd "${REPO_ROOT}/tina_run"
        "${PYTHON_BIN}" scripts/local/eval/eval_full_bench_vllm.py \
            --base_model "${RUNTIME_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base" \
            --checkpoint_name "${EVAL_NAME}" \
            --output_dir "${EVAL_DIR}" \
            --benchmarks aime24,aime25,amc23,hmmt_feb,math500,minerva \
            --num_shards 4 \
            --aggregate_only
        systemctl stop "${EVAL_SERVICE}" || true
        exit 0
    fi

    if ! systemctl is-active --quiet "${EVAL_SERVICE}"; then
        echo "Evaluation service exited before all four shards completed" >&2
        exit 3
    fi
    sleep 20
done
