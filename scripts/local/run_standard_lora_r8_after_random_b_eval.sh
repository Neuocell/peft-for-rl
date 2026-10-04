#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/standard-lora-rank8-v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
WAIT_EVAL_PID="${WAIT_EVAL_PID:-2775469}"
WAIT_SUMMARY="/data/peft-for-rl-runtime/outputs/full_bench_eval/random_b_energy_uniform_r8_b64m16n8_step50_fullbench_32768_seed42_v1/summary/random_b_energy_uniform_r8_b64m16n8_step50_fullbench_32768_seed42_v1.json"
TRAIN_NAME="standard_lora_r8_b64m16n8_50_seed42_v1"
ADAPTER_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}/global_step_50/actor/peft_adapter"
EVAL_NAME="standard_lora_r8_b64m16n8_step50_fullbench_32768_seed42_v1"
EVAL_DIR="/data/peft-for-rl-runtime/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
BENCHMARK_SNAPSHOT_RECORDS="/data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"

mkdir -p "${RUNTIME_ROOT}/logs/verl"
cd "${REPO_ROOT}"

echo "[$(date --iso-8601=seconds)] standard LoRA rank-8 pipeline queued"
while [[ ! -s "${WAIT_SUMMARY}" ]]; do
    if ! kill -0 "${WAIT_EVAL_PID}" 2>/dev/null; then
        echo "[$(date --iso-8601=seconds)] prerequisite random-B evaluation exited without summary" >&2
        exit 1
    fi
    sleep 60
done
echo "[$(date --iso-8601=seconds)] prerequisite random-B evaluation complete"

if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] starting matched 50-step standard LoRA rank-8 training"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    EXP_NAME="${TRAIN_NAME}" \
    bash "${SCRIPT_DIR}/start_standard_lora_r8_b64_50_4gpu.sh"
fi

if [[ ! -s "${EVAL_SUMMARY}" ]]; then
    echo "[$(date --iso-8601=seconds)] starting matched six-benchmark evaluation"
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PROJECT_ROOT="${REPO_ROOT}/tina_run" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BASE_MODEL="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base" \
    LORA_ADAPTER="${ADAPTER_DIR}" \
    DIRECT_OPD_EVAL_ROOT="/data/peft-for-rl-runtime/datasets/Direct-OPD/datasets/eval" \
    CHECKPOINT_NAME="${EVAL_NAME}" \
    OUTPUT_DIR="${EVAL_DIR}" \
    BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash "${REPO_ROOT}/tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
fi

echo "[$(date --iso-8601=seconds)] standard LoRA rank-8 pipeline complete: ${EVAL_SUMMARY}"
