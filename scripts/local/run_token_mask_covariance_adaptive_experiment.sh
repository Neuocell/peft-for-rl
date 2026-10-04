#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/peft-for-rl-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"

PROBE_NAME="full_gradient_tokenmask_top50_cov_centered_r32_d64c32a16_seed42_v1"
ARTIFACT_DIR="${RUNTIME_ROOT}/analysis/${PROBE_NAME}"
ALLOCATION_DIR="${ARTIFACT_DIR}/covariance_adaptive_eqr28_rmin16_stable_energy"
TRAIN_NAME="tokenmask_cov_centered_adaptive_eqr28_rmin16_b64m16n8_50_seed42_v1"
ADAPTER_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}/global_step_50/actor/peft_adapter"
EVAL_NAME="${TRAIN_NAME%_50_seed42_v1}_step50_fullbench_32768_seed42_v1"
EVAL_DIR="${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
BENCHMARK_SNAPSHOT_RECORDS="${RUNTIME_ROOT}/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"

mkdir -p "${RUNTIME_ROOT}/logs/verl"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

if [[ ! -s "${ARTIFACT_DIR}/probe_summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] starting token-selective full-gradient probe"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    EXP_NAME="${PROBE_NAME}" \
    GRADIENT_PROBE_OUTPUT_DIR="${ARTIFACT_DIR}" \
    FULL_GRADIENT_PROBE_RANK=32 \
    FULL_GRADIENT_PROBE_SKETCH_WIDTH=40 \
    FULL_GRADIENT_PROBE_DISCOVERY_PROMPTS=64 \
    FULL_GRADIENT_PROBE_CALIBRATION_PROMPTS=32 \
    FULL_GRADIENT_PROBE_AUDIT_PROMPTS=16 \
    FULL_GRADIENT_PROBE_TOKEN_MASK_MODE=top_surprisal \
    FULL_GRADIENT_PROBE_TOKEN_KEEP_RATIO=0.5 \
    FULL_GRADIENT_PROBE_TOKEN_MIN_KEEP=128 \
    FULL_GRADIENT_PROBE_TOKEN_KEEP_FINAL=128 \
    FULL_GRADIENT_PROBE_TOKEN_MASK_DISCOVERY_ONLY=True \
    TRAIN_PROMPT_BSZ=16 \
    TRAIN_PROMPT_MINI_BSZ=16 \
    N_RESP_PER_PROMPT=8 \
    TOTAL_TRAINING_STEPS=30 \
    MAX_RESPONSE_LENGTH=8192 \
    ACTOR_PPO_MAX_TOKEN_LEN=12288 \
    INFER_PPO_MAX_TOKEN_LEN=12288 \
    DATA_SEED=42 \
    PPO_DATA_LOADER_SEED=42 \
    bash "${SCRIPT_DIR}/start_full_gradient_rl_probe_4gpu.sh"
fi

if [[ ! -s "${ARTIFACT_DIR}/probe_summary.json" ]]; then
    echo "Probe exited without a complete artifact: ${ARTIFACT_DIR}" >&2
    exit 1
fi

if [[ ! -s "${ALLOCATION_DIR}/rank_map.json" ]] || [[ ! -s "${ALLOCATION_DIR}/subspaces.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] building covariance adaptive rank allocation"
    "${PYTHON_BIN}" scripts/analysis/build_full_gradient_uniform_allocation.py \
        --artifact-dir "${ARTIFACT_DIR}" \
        --output-dir "${ALLOCATION_DIR}" \
        --candidate-method covariance \
        --allocation-mode adaptive \
        --uniform-rank 28 \
        --r-min 16 \
        --adaptive-utility stable_energy
fi

if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] starting matched 50-step adaptive LoRA training"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    ALLOCATION_DIR="${ALLOCATION_DIR}" \
    CANDIDATE_METHOD=covariance \
    EXP_NAME="${TRAIN_NAME}" \
    TRAIN_PROMPT_BSZ=64 \
    TRAIN_PROMPT_MINI_BSZ=16 \
    N_RESP_PER_PROMPT=8 \
    TOTAL_TRAINING_STEPS=50 \
    SAVE_FREQ=50 \
    MAX_ACTOR_CKPT_TO_KEEP=2 \
    LORA_DROPOUT=0.05 \
    LR=1e-6 \
    DATA_SEED=42 \
    PPO_DATA_LOADER_SEED=42 \
    bash "${SCRIPT_DIR}/start_full_gradient_adaptive_eqr8_4gpu.sh"
fi

if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "Training exited without the step-50 adapter: ${ADAPTER_DIR}" >&2
    exit 1
fi
if [[ ! -s "${BENCHMARK_SNAPSHOT_RECORDS}" ]]; then
    echo "Missing benchmark snapshot: ${BENCHMARK_SNAPSHOT_RECORDS}" >&2
    exit 2
fi

if [[ ! -s "${EVAL_SUMMARY}" ]]; then
    echo "[$(date --iso-8601=seconds)] starting matched six-benchmark evaluation"
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
fi

echo "[$(date --iso-8601=seconds)] experiment complete: ${EVAL_SUMMARY}"
