#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-full}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

BASE_MODEL="${REPO_ROOT}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
RUN_NAME="gradtop_probe12_r8to32_mean31p63_a2_b64m16n8_270_v1"

if [[ "$#" -eq 0 ]]; then
  STEPS=(100 150)
else
  STEPS=("$@")
fi

if [[ "${MODE}" == "debug" ]]; then
  NUM_SEQUENCES=12
  MAX_SEQ_TOKENS=1024
  TOKENS_PER_SEQUENCE=32
  TARGETS="0.90"
  SUFFIX="debug"
else
  NUM_SEQUENCES=48
  MAX_SEQ_TOKENS=8192
  TOKENS_PER_SEQUENCE=256
  TARGETS="0.90,0.95,0.98"
  SUFFIX="full"
fi

for STEP in "${STEPS[@]}"; do
  ADAPTER="${REPO_ROOT}/runs/ckpts/verl/DAPO-Math-17k/${RUN_NAME}/global_step_${STEP}/actor/peft_adapter"
  RECORDS="${REPO_ROOT}/runs/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step${STEP}_fullbench_32768_v1"
  OUT_DIR="${REPO_ROOT}/runs/analysis/gradtop_probe12_step${STEP}_rank_pruning_policy_${SUFFIX}"
  LOG_FILE="${REPO_ROOT}/runs/logs/analysis/gradtop_probe12_step${STEP}_rank_pruning_policy_${SUFFIX}.log"

  mkdir -p "$(dirname "${LOG_FILE}")"
  echo "[$(date -Is)] step=${STEP} mode=${MODE} device=${DEVICE}" | tee "${LOG_FILE}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/analysis/diagnose_rank_pruning_policy_distortion.py" \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --records "${RECORDS}" \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --num-sequences "${NUM_SEQUENCES}" \
    --max-seq-tokens "${MAX_SEQ_TOKENS}" \
    --tokens-per-sequence "${TOKENS_PER_SEQUENCE}" \
    --targets "${TARGETS}" \
    2>&1 | tee -a "${LOG_FILE}"
done
