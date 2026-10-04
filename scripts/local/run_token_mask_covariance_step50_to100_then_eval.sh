#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/peft-for-rl-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"

SOURCE_EXP="tokenmask_cov_centered_adaptive_eqr28_rmin16_b64m16n8_50_seed42_v1"
TARGET_EXP="tokenmask_cov_centered_adaptive_eqr28_rmin16_b64m16n8_step50_to100_seed42_v1"
CKPT_ROOT="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k"
SOURCE_CKPT="${CKPT_ROOT}/${SOURCE_EXP}/global_step_50"
TARGET_ROOT="${CKPT_ROOT}/${TARGET_EXP}"
TARGET_CKPT="${TARGET_ROOT}/global_step_50"
LATEST_CKPT_FILE="${TARGET_ROOT}/latest_checkpointed_iteration.txt"
ALLOCATION_DIR="${RUNTIME_ROOT}/analysis/full_gradient_tokenmask_top50_cov_centered_r32_d64c32a16_seed42_v1/covariance_adaptive_eqr28_rmin16_stable_energy"

EVAL_NAME="tokenmask_cov_centered_adaptive_eqr28_rmin16_b64m16n8_step100_fullbench_32768_seed42_v1"
EVAL_DIR="${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
BENCHMARK_SNAPSHOT_RECORDS="${RUNTIME_ROOT}/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"

required=(
    "actor/model_world_size_4_rank_0.pt"
    "actor/model_world_size_4_rank_1.pt"
    "actor/model_world_size_4_rank_2.pt"
    "actor/model_world_size_4_rank_3.pt"
    "actor/extra_state_world_size_4_rank_0.pt"
    "actor/extra_state_world_size_4_rank_1.pt"
    "actor/extra_state_world_size_4_rank_2.pt"
    "actor/extra_state_world_size_4_rank_3.pt"
    "actor/peft_adapter/adapter_model.safetensors"
    "data.pt"
)
for relative_path in "${required[@]}"; do
    if [[ ! -s "${SOURCE_CKPT}/${relative_path}" ]]; then
        echo "Incomplete source checkpoint: ${SOURCE_CKPT}/${relative_path}" >&2
        exit 2
    fi
done
if [[ ! -s "${ALLOCATION_DIR}/rank_map.json" ]] || [[ ! -s "${ALLOCATION_DIR}/subspaces.safetensors" ]]; then
    echo "Missing adaptive allocation: ${ALLOCATION_DIR}" >&2
    exit 2
fi

# The source checkpoint has model, extra and data state but no optimizer state.
# Hard-link it into a distinct run so the source remains intact while resume=auto
# restores weights, global step and dataloader position with a fresh optimizer.
if [[ ! -d "${TARGET_CKPT}" ]]; then
    mkdir -p "${TARGET_ROOT}"
    cp -al "${SOURCE_CKPT}" "${TARGET_CKPT}"
fi
for relative_path in "${required[@]}"; do
    if [[ ! -s "${TARGET_CKPT}/${relative_path}" ]]; then
        echo "Incomplete staged checkpoint: ${TARGET_CKPT}/${relative_path}" >&2
        exit 2
    fi
done
if [[ ! -s "${LATEST_CKPT_FILE}" ]]; then
    printf '50\n' >"${LATEST_CKPT_FILE}"
fi

mkdir -p "${RUNTIME_ROOT}/logs/verl"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

if [[ ! -s "${TARGET_ROOT}/global_step_100/actor/peft_adapter/adapter_model.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] continuing adaptive LoRA from step 50 to step 100"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    ALLOCATION_DIR="${ALLOCATION_DIR}" \
    CANDIDATE_METHOD=covariance \
    EXP_NAME="${TARGET_EXP}" \
    TRAIN_PROMPT_BSZ=64 \
    TRAIN_PROMPT_MINI_BSZ=16 \
    N_RESP_PER_PROMPT=8 \
    TOTAL_TRAINING_STEPS=100 \
    SAVE_FREQ=25 \
    MAX_ACTOR_CKPT_TO_KEEP=3 \
    SAVE_CONTENTS="['model','extra']" \
    RESUME_MODE=auto \
    LORA_DROPOUT=0.05 \
    LR=1e-6 \
    DATA_SEED=42 \
    PPO_DATA_LOADER_SEED=42 \
    RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_tokenmask_cov_50_100}" \
    bash "${SCRIPT_DIR}/start_full_gradient_adaptive_eqr8_4gpu.sh"
fi

ADAPTER_DIR="${TARGET_ROOT}/global_step_100/actor/peft_adapter"
if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "Continuation exited without the step-100 adapter: ${ADAPTER_DIR}" >&2
    exit 1
fi
if [[ ! -s "${BENCHMARK_SNAPSHOT_RECORDS}" ]]; then
    echo "Missing benchmark snapshot: ${BENCHMARK_SNAPSHOT_RECORDS}" >&2
    exit 2
fi

if [[ ! -s "${EVAL_SUMMARY}" ]]; then
    echo "[$(date --iso-8601=seconds)] starting matched step-100 evaluation"
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

echo "[$(date --iso-8601=seconds)] step-100 experiment complete: ${EVAL_SUMMARY}"
