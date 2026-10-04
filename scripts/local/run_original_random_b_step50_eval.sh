#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/random-b-original-repro-v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"

TRAIN_NAME="gradtop_probe12_r8to32_original_random_b_a2_b64m16n8_270_seed42_repro_v1"
CHECKPOINT_ROOT="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}"
ADAPTER_DIR="${CHECKPOINT_ROOT}/global_step_50/actor/peft_adapter"
EVAL_NAME="gradtop_original_random_b_r8to32_step50_fullbench_32768_seed42_repro_v1"
EVAL_DIR="${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS:-/data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl}"

required=(
    "global_step_50/actor/peft_adapter/adapter_config.json"
    "global_step_50/actor/peft_adapter/adapter_model.safetensors"
    "global_step_50/actor/model_world_size_4_rank_0.pt"
    "global_step_50/actor/model_world_size_4_rank_1.pt"
    "global_step_50/actor/model_world_size_4_rank_2.pt"
    "global_step_50/actor/model_world_size_4_rank_3.pt"
    "global_step_50/actor/optim_world_size_4_rank_0.pt"
    "global_step_50/actor/optim_world_size_4_rank_1.pt"
    "global_step_50/actor/optim_world_size_4_rank_2.pt"
    "global_step_50/actor/optim_world_size_4_rank_3.pt"
    "global_step_50/actor/extra_state_world_size_4_rank_0.pt"
    "global_step_50/actor/extra_state_world_size_4_rank_1.pt"
    "global_step_50/actor/extra_state_world_size_4_rank_2.pt"
    "global_step_50/actor/extra_state_world_size_4_rank_3.pt"
    "global_step_50/data.pt"
)
for relative_path in "${required[@]}"; do
    if [[ ! -s "${CHECKPOINT_ROOT}/${relative_path}" ]]; then
        echo "Incomplete resumable step-50 checkpoint: ${CHECKPOINT_ROOT}/${relative_path}" >&2
        exit 2
    fi
done
if [[ "$(tr -d '[:space:]' <"${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt")" != "50" ]]; then
    echo "The latest checkpoint marker is not step 50" >&2
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

echo "[$(date --iso-8601=seconds)] starting original random-B step-50 full-bench evaluation"
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

echo "[$(date --iso-8601=seconds)] evaluation complete: ${EVAL_SUMMARY}"
