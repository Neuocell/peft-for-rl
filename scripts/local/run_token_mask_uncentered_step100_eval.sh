#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/peft-for-rl-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"

TRAIN_NAME="tokenmask_cov_uncentered_adaptive_eqr28_rmin16_b64m16n8_270_seed42_v1"
ADAPTER_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}/global_step_100/actor/peft_adapter"
EVAL_NAME="tokenmask_cov_uncentered_adaptive_eqr28_rmin16_b64m16n8_step100_fullbench_32768_seed42_v1"
EVAL_DIR="${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
BENCHMARK_SNAPSHOT_RECORDS="${RUNTIME_ROOT}/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"

if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "Missing step-100 adapter: ${ADAPTER_DIR}/adapter_model.safetensors" >&2
    exit 2
fi
if [[ ! -s "${BENCHMARK_SNAPSHOT_RECORDS}" ]]; then
    echo "Missing benchmark snapshot: ${BENCHMARK_SNAPSHOT_RECORDS}" >&2
    exit 2
fi
if [[ -s "${EVAL_SUMMARY}" ]]; then
    echo "Evaluation already complete: ${EVAL_SUMMARY}"
    exit 0
fi

echo "[$(date --iso-8601=seconds)] starting masked uncentered-covariance step-100 evaluation"
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
PROJECT_ROOT="${REPO_ROOT}/tina_run" \
PYTHON_BIN="${PYTHON_BIN}" \
BASE_MODEL="${RUNTIME_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base" \
LORA_ADAPTER="${ADAPTER_DIR}" \
DIRECT_OPD_EVAL_ROOT="${RUNTIME_ROOT}/datasets/Direct-OPD/datasets/eval" \
CHECKPOINT_NAME="${EVAL_NAME}" \
OUTPUT_DIR="${EVAL_DIR}" \
BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS}" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash "${REPO_ROOT}/tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"

echo "[$(date --iso-8601=seconds)] evaluation complete: ${EVAL_SUMMARY}"
