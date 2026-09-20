#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"

BASE_MODEL="${BASE_MODEL:-${RUNTIME_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
ADAPTER="${ADAPTER:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/rlpo_init_r32a64_b64m16n8_270_4gpu_boxed_v2/global_step_270/actor/peft_adapter}"
RECORDS="${RECORDS:-${RUNTIME_ROOT}/outputs/full_bench_eval/rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1/records/rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1.jsonl}"
OUT_DIR="${OUT_DIR:-${RUNTIME_ROOT}/analysis/d1_rlpo_r32_step270_activation_rank}"
LOG_FILE="${LOG_FILE:-${RUNTIME_ROOT}/logs/analysis/d1_rlpo_r32_step270_activation_rank.log}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_SEQUENCES="${NUM_SEQUENCES:-48}"
MAX_SEQ_TOKENS="${MAX_SEQ_TOKENS:-4096}"
TOKENS_PER_SEQUENCE="${TOKENS_PER_SEQUENCE:-256}"

mkdir -p "$(dirname -- "${LOG_FILE}")" "${OUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "D1 base model: ${BASE_MODEL}"
echo "D1 adapter:    ${ADAPTER}"
echo "D1 records:    ${RECORDS}"
echo "D1 output:     ${OUT_DIR}"
echo "D1 GPU:        ${CUDA_DEVICE}"

exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
  python "${REPO_ROOT}/scripts/analysis/diagnose_activation_weighted_rank.py" \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --records "${RECORDS}" \
    --out-dir "${OUT_DIR}" \
    --device cuda:0 \
    --num-sequences "${NUM_SEQUENCES}" \
    --max-seq-tokens "${MAX_SEQ_TOKENS}" \
    --tokens-per-sequence "${TOKENS_PER_SEQUENCE}"
