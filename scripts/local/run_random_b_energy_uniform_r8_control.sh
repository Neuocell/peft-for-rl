#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/random-b-rank8-v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
PROBE_NAME="random_b_energy_probe_b16n8_w8_s6to12_seed42_v2"
PROBE_DIR="${RUNTIME_ROOT}/analysis/${PROBE_NAME}"
ALLOCATION_DIR="${PROBE_DIR}/uniform_r8"
TRAIN_NAME="random_b_energy_uniform_r8_b64m16n8_50_seed42_v1"
ADAPTER_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}/global_step_50/actor/peft_adapter"
EVAL_NAME="random_b_energy_uniform_r8_b64m16n8_step50_fullbench_32768_seed42_v1"
EVAL_DIR="/data/peft-for-rl-runtime/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"

mkdir -p "${RUNTIME_ROOT}/logs/verl"
cd "${REPO_ROOT}"

echo "[$(date --iso-8601=seconds)] random-B rank-8 control pipeline started"

if [[ ! -s "${PROBE_DIR}/summary.json" ]] || [[ ! -s "${PROBE_DIR}/subspaces.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] starting legacy random-B energy probe"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    EXP_NAME="${PROBE_NAME}" \
    GRADIENT_PROBE_OUTPUT_DIR="${PROBE_DIR}" \
    RAY_TEMP_DIR="/tmp/ray-random-b-r8-probe-v2" \
    RAY_LOCAL_FS_CAPACITY_THRESHOLD=0.99 \
    MODEL_PATH="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base" \
    TRAIN_FILE="/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet" \
    TEST_FILE="/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet" \
    bash "${SCRIPT_DIR}/start_gradient_probe_4gpu.sh"
fi

if [[ ! -s "${ALLOCATION_DIR}/rank_map.json" ]] || [[ ! -s "${ALLOCATION_DIR}/subspaces.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] building uniform rank-8 allocation"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/analysis/build_gradient_probe_uniform_allocation.py" \
        --artifact-dir "${PROBE_DIR}" \
        --output-dir "${ALLOCATION_DIR}" \
        --uniform-rank 8
fi

if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] starting matched 50-step rank-8 training"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    PROBE_DIR="${PROBE_DIR}" \
    ALLOCATION_DIR="${ALLOCATION_DIR}" \
    EXP_NAME="${TRAIN_NAME}" \
    bash "${SCRIPT_DIR}/start_random_b_energy_uniform_r8_b64_50_4gpu.sh"
fi

if [[ ! -s "${EVAL_SUMMARY}" ]]; then
    echo "[$(date --iso-8601=seconds)] starting six-benchmark evaluation"
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PROJECT_ROOT="${REPO_ROOT}/tina_run" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BASE_MODEL="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base" \
    LORA_ADAPTER="${ADAPTER_DIR}" \
    DIRECT_OPD_EVAL_ROOT="/data/peft-for-rl-runtime/datasets/Direct-OPD/datasets/eval" \
    CHECKPOINT_NAME="${EVAL_NAME}" \
    OUTPUT_DIR="${EVAL_DIR}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash "${REPO_ROOT}/tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
fi

echo "[$(date --iso-8601=seconds)] pipeline complete: ${EVAL_SUMMARY}"
