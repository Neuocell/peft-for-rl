#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[$(date -Is)] FAILED at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TINA_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${TINA_ROOT}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs}"
VERL_ROOT="${VERL_ROOT:-${REPO_ROOT}}"
CONDA_SH="${CONDA_SH:-/home/wangls/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-verl-rl}"

PROJECT_NAME="${PROJECT_NAME:-DAPO-Math-17k}"
EXP_NAME="${EXP_NAME:-dapo_math_boxed_stable_oft_1p5b_4gpu_8k_v1_20260724}"
CKPTS_DIR="${CKPTS_DIR:-${RUN_ROOT}/ckpts/verl/${PROJECT_NAME}/${EXP_NAME}}"
BASE_MODEL="${BASE_MODEL:-${RUN_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_oft_1p5b_4gpu_8k.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${TINA_ROOT}/scripts/local/eval/run_full_bench_vllm_4gpu.sh}"
MERGE_OFT_SCRIPT="${MERGE_OFT_SCRIPT:-${TINA_ROOT}/scripts/local/eval/merge_oft_wrapped_hf.py}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUN_ROOT}/ray_oft_boxed_stable_8k}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/outputs/full_bench_eval}"
mkdir -p "${LOG_DIR}" "${EVAL_ROOT}"

TARGETS="${TARGETS:-100 200 300}"
EXTERNAL_STEP100_SESSION="${EXTERNAL_STEP100_SESSION:-train_oft_stable_v1_100}"
SAVE_FREQ="${SAVE_FREQ:-20}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-1024}"
EVAL_RETRY_SECONDS="${EVAL_RETRY_SECONDS:-300}"

BENCHMARKS="${BENCHMARKS:-aime24,aime25,amc23,hmmt_feb,math500,minerva}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
SAMPLES_SMALL="${SAMPLES_SMALL:-32}"
SAMPLES_LARGE="${SAMPLES_LARGE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >&2
}

actor_dir() {
  local step="$1"
  echo "${CKPTS_DIR}/global_step_${step}/actor"
}

actor_complete() {
  local dir="$1"
  [[ -f "${dir}/fsdp_config.json" ]] || return 1
  [[ -d "${dir}/huggingface" ]] || return 1
  find "${dir}" -maxdepth 1 -name 'model_world_size_*_rank_*.pt' -print -quit 2>/dev/null | grep -q .
}

summary_path() {
  local step="$1"
  echo "${EVAL_ROOT}/dapo_boxed_stable_oft_v1_step${step}_fullbench_32768/summary/dapo_boxed_stable_oft_v1_step${step}_fullbench_32768.json"
}

gpu_is_idle() {
  local gpu="$1"
  local used_memory
  local compute_pids
  used_memory="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  compute_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
  [[ -z "${compute_pids}" && "${used_memory}" -le "${GPU_IDLE_MEMORY_MB}" ]]
}

wait_for_eval_gpus_idle() {
  local all_idle
  while true; do
    all_idle=1
    for gpu in "${GPU_IDS[@]}"; do
      if ! gpu_is_idle "${gpu}"; then
        all_idle=0
        break
      fi
    done
    if [[ "${all_idle}" == "1" ]]; then
      log "GPUs idle: ${CUDA_DEVICES}"
      return 0
    fi
    log "Waiting for GPUs to become idle: ${CUDA_DEVICES}"
    sleep "${POLL_SECONDS}"
  done
}

cleanup_oft_ray() {
  log "Cleaning possible stale OFT Ray processes for ${RAY_TEMP_DIR}"
  pkill -TERM -f "${RAY_TEMP_DIR}" 2>/dev/null || true
  sleep 10
  pkill -KILL -f "${RAY_TEMP_DIR}" 2>/dev/null || true
}

wait_for_external_step100() {
  local step_dir
  step_dir="$(actor_dir 100)"
  if actor_complete "${step_dir}"; then
    log "global_step_100 already exists."
    return 0
  fi
  if tmux has-session -t "${EXTERNAL_STEP100_SESSION}" 2>/dev/null; then
    log "Waiting for existing session ${EXTERNAL_STEP100_SESSION} to finish."
    while tmux has-session -t "${EXTERNAL_STEP100_SESSION}" 2>/dev/null; do
      sleep "${POLL_SECONDS}"
    done
  fi
  while ! actor_complete "${step_dir}"; do
    log "Waiting for global_step_100 actor checkpoint: ${step_dir}"
    sleep "${POLL_SECONDS}"
  done
  cleanup_oft_ray
}

train_to_target() {
  local target="$1"
  local step_dir
  step_dir="$(actor_dir "${target}")"
  if actor_complete "${step_dir}"; then
    log "global_step_${target} already exists; skip training."
    return 0
  fi
  if [[ "${target}" == "100" ]]; then
    wait_for_external_step100
    return 0
  fi

  local train_log="${LOG_DIR}/dapo_math_boxed_stable_oft_1p5b_4gpu_8k_v1_resume_to${target}_$(date +%Y%m%d_%H%M%S).log"
  log "Training OFT to global_step_${target}; log=${train_log}"
  (
    cd "${VERL_ROOT}"
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
      EXP_NAME="${EXP_NAME}" \
      RUNTIME_ROOT="${RUN_ROOT}" \
      CKPTS_DIR="${CKPTS_DIR}" \
      TOTAL_TRAINING_STEPS="${target}" \
      SAVE_FREQ="${SAVE_FREQ}" \
      MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP}" \
      RESUME_MODE=auto \
      RAY_TEMP_DIR="${RAY_TEMP_DIR}" \
      bash "${TRAIN_SCRIPT}"
  ) > "${train_log}" 2>&1
  while ! actor_complete "${step_dir}"; do
    log "Training exited; waiting for global_step_${target} actor files to settle."
    sleep "${POLL_SECONDS}"
  done
  cleanup_oft_ray
}

hf_model_complete() {
  local model_dir="$1"
  [[ -f "${model_dir}/config.json" ]] || return 1
  [[ -f "${model_dir}/tokenizer_config.json" ]] || return 1
  [[ -f "${model_dir}/model.safetensors" || -f "${model_dir}/model.safetensors.index.json" || -f "${model_dir}/pytorch_model.bin" ]]
}

merge_oft_for_eval() {
  local step="$1"
  local ckpt_name="$2"
  local out_dir="$3"
  local actor
  local wrapped_model
  local merged_model
  actor="$(actor_dir "${step}")"
  wrapped_model="${CKPTS_DIR}/global_step_${step}/actor_hf_merged"
  merged_model="${CKPTS_DIR}/global_step_${step}/actor_hf_merged_vllm"

  mkdir -p "${out_dir}/logs"
  if ! hf_model_complete "${wrapped_model}"; then
    log "Merging FSDP actor to OFT-wrapped HF: ${wrapped_model}"
    (
      cd "${VERL_ROOT}"
      source "${CONDA_SH}"
      conda activate "${CONDA_ENV}"
      CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${actor}" \
        --target_dir "${wrapped_model}" \
        --use_cpu_initialization
    ) >> "${out_dir}/logs/${ckpt_name}.merge.log" 2>&1
  fi

  if ! hf_model_complete "${merged_model}"; then
    log "Merging OFT-wrapped HF to vLLM-ready HF: ${merged_model}"
    local oft_merge_extra=()
    if [[ "${OFT_COFT:-False}" == "True" ]]; then
      oft_merge_extra+=(--oft_coft)
    fi
    if [[ "${OFT_BLOCK_SHARE:-False}" == "True" ]]; then
      oft_merge_extra+=(--oft_block_share)
    fi
    if [[ "${OFT_USE_CAYLEY_NEUMANN:-True}" == "False" ]]; then
      oft_merge_extra+=(--no-oft_use_cayley_neumann)
    fi
    (
      cd "${TINA_ROOT}"
      source "${CONDA_SH}"
      conda activate "${CONDA_ENV}"
      CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" python "${MERGE_OFT_SCRIPT}" \
        --base_model "${BASE_MODEL}" \
        --wrapped_model "${wrapped_model}" \
        --output_dir "${merged_model}" \
        --oft_rank "${OFT_RANK:-0}" \
        --oft_block_size "${OFT_BLOCK_SIZE:-32}" \
        --oft_dropout "${OFT_DROPOUT:-0.0}" \
        --oft_eps "${OFT_EPS:-0.00006}" \
        --oft_num_cayley_neumann_terms "${OFT_NUM_CAYLEY_NEUMANN_TERMS:-5}" \
        "${oft_merge_extra[@]}" \
        --dtype bfloat16
    ) >> "${out_dir}/logs/${ckpt_name}.merge.log" 2>&1
  fi
  echo "${merged_model}"
}

eval_target() {
  local step="$1"
  local ckpt_name="dapo_boxed_stable_oft_v1_step${step}_fullbench_32768"
  local out_dir="${EVAL_ROOT}/${ckpt_name}"
  local summary
  local merged_model
  summary="$(summary_path "${step}")"
  if [[ -f "${summary}" ]]; then
    log "Eval summary for step ${step} already exists; skip eval."
    return 0
  fi

  while [[ ! -f "${summary}" ]]; do
    wait_for_eval_gpus_idle
    merged_model="$(merge_oft_for_eval "${step}" "${ckpt_name}" "${out_dir}")"
    wait_for_eval_gpus_idle
    log "Evaluating OFT global_step_${step}; output=${out_dir}"
    if (
      cd "${TINA_ROOT}"
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        PROJECT_ROOT="${TINA_ROOT}" \
        RUNTIME_ROOT="${RUN_ROOT}" \
        BASE_MODEL="${merged_model}" \
        CHECKPOINT_NAME="${ckpt_name}" \
        OUTPUT_DIR="${out_dir}" \
        BENCHMARKS="${BENCHMARKS}" \
        SEED="${SEED}" \
        TEMPERATURE="${TEMPERATURE}" \
        TOP_P="${TOP_P}" \
        MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        SAMPLES_SMALL="${SAMPLES_SMALL}" \
        SAMPLES_LARGE="${SAMPLES_LARGE}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
        bash "${EVAL_SCRIPT}"
    ); then
      break
    fi
    log "Eval failed or interrupted for step ${step}; will retry after ${EVAL_RETRY_SECONDS}s."
    sleep "${EVAL_RETRY_SECONDS}"
  done
  log "Eval done for step ${step}; removing temporary merged models."
  rm -rf "${CKPTS_DIR}/global_step_${step}/actor_hf_merged" "${CKPTS_DIR}/global_step_${step}/actor_hf_merged_vllm"
}

log "OFT stable v1 train/eval controller started. targets=${TARGETS}"
for target in ${TARGETS}; do
  train_to_target "${target}"
  eval_target "${target}"
done
log "OFT stable v1 train/eval controller finished."
