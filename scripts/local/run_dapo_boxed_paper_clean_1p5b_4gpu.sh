#!/usr/bin/env bash
set -euo pipefail

# Clean RLVR training entry for paper-facing PEFT experiments.
# Defaults are intentionally explicit and isolated from historical run scripts.

cd /home/wangls/CHERRL
export PATH="/home/wangls/miniconda3/envs/cherrl/bin:${PATH}"
export CONDA_PREFIX="/home/wangls/miniconda3/envs/cherrl"
export CONDA_DEFAULT_ENV="cherrl"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

METHOD="${METHOD:-lora}"
DATASET="${DATASET:-dapo_heldout637_seed42}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

PROJECT_NAME="${PROJECT_NAME:-DAPO-Math-17k-paper-clean}"
MODEL_PATH="${MODEL_PATH:-/home/wangls/Tina_orthres_run/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-/home/wangls/CHERRL/verl/utils/reward_score/boxed_math_accuracy.py}"

case "${DATASET}" in
  dapo_heldout637_seed42)
    TRAIN_FILE="${TRAIN_FILE:-/home/wangls/Tina_orthres_run/datasets/verl_dapo/data/dapo_math_boxed_heldout637_seed42_v1/train.parquet}"
    TEST_FILE="${TEST_FILE:-/home/wangls/Tina_orthres_run/datasets/verl_dapo/data/dapo_math_boxed_heldout637_seed42_v1/heldout.parquet}"
    DEFAULT_TOTAL_STEPS=270
    ;;
  dapo_heldout1021)
    TRAIN_FILE="${TRAIN_FILE:-/home/wangls/Tina_orthres_run/datasets/verl_dapo/data/dapo_math_boxed_heldout1021_v1/train.parquet}"
    TEST_FILE="${TEST_FILE:-/home/wangls/Tina_orthres_run/datasets/verl_dapo/data/dapo_math_boxed_heldout1021_v1/heldout.parquet}"
    DEFAULT_TOTAL_STEPS=264
    ;;
  dapo_heldout1019)
    TRAIN_FILE="${TRAIN_FILE:-/home/wangls/Tina_orthres_run/datasets/verl_dapo/data/dapo_math_boxed_heldout1019_v1/train.parquet}"
    TEST_FILE="${TEST_FILE:-/home/wangls/Tina_orthres_run/datasets/verl_dapo/data/dapo_math_boxed_heldout1019_v1/heldout.parquet}"
    DEFAULT_TOTAL_STEPS=264
    ;;
  dapo_full_boxed)
    TRAIN_FILE="${TRAIN_FILE:-/home/wangls/Tina_orthres_run/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
    TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
    DEFAULT_TOTAL_STEPS=279
    ;;
  open_rs3)
    TRAIN_FILE="${TRAIN_FILE:-/home/wangls/Tina_orthres_run/datasets/verl/open_rs3_justrl_answer.parquet}"
    TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
    DEFAULT_TOTAL_STEPS=109
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${DEFAULT_TOTAL_STEPS}}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"

LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"

TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
VAL_TOP_P="${VAL_TOP_P:-0.7}"

CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"

DATA_SEED="${DATA_SEED:-42}"
PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
ACTOR_SHUFFLE="${ACTOR_SHUFFLE:-False}"

SAVE_FREQ="${SAVE_FREQ:-20}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-20}"
SAVE_CONTENTS="${SAVE_CONTENTS:-['model','extra']}"
RESUME_MODE="${RESUME_MODE:-disable}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/home/wangls/ray_pc_${METHOD}_${RUN_TAG}}"

TARGET_MODULES="${TARGET_MODULES:-all-linear}"
OFFLOAD="${OFFLOAD:-True}"
FSDP_SIZE="${FSDP_SIZE:--1}"
GEN_TP="${GEN_TP:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"

ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))}"
INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-$((TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT))}"

PEFT_TYPE="${PEFT_TYPE:-${METHOD}}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
USE_ORIG_PARAMS="${USE_ORIG_PARAMS:-False}"

OFT_BLOCK_SIZE="${OFT_BLOCK_SIZE:-32}"
OFT_RANK="${OFT_RANK:-0}"
OFT_DROPOUT="${OFT_DROPOUT:-0.0}"
OFT_COFT="${OFT_COFT:-False}"
OFT_EPS="${OFT_EPS:-0.00006}"
OFT_BLOCK_SHARE="${OFT_BLOCK_SHARE:-False}"
OFT_USE_CAYLEY_NEUMANN="${OFT_USE_CAYLEY_NEUMANN:-True}"
OFT_NUM_CAYLEY_NEUMANN_TERMS="${OFT_NUM_CAYLEY_NEUMANN_TERMS:-5}"

SKEW_RANK="${SKEW_RANK:-32}"
SKEW_ALPHA="${SKEW_ALPHA:-64}"
SKEW_DROPOUT="${SKEW_DROPOUT:-0.0}"
SKEW_INIT_STD="${SKEW_INIT_STD:-0.01}"

BOET_RANK="${BOET_RANK:-32}"
BOET_ALPHA="${BOET_ALPHA:-64}"
BOET_DROPOUT="${BOET_DROPOUT:-0.0}"
BOET_INIT_STD="${BOET_INIT_STD:-0.01}"
BOET_USE_CAYLEY_NEUMANN="${BOET_USE_CAYLEY_NEUMANN:-False}"
BOET_NUM_CAYLEY_NEUMANN_TERMS="${BOET_NUM_CAYLEY_NEUMANN_TERMS:-5}"
BOET_CAYLEY_NEUMANN_EPS="${BOET_CAYLEY_NEUMANN_EPS:-0.9}"

BISO_BLOCK_SIZE="${BISO_BLOCK_SIZE:-32}"
BISO_ALPHA="${BISO_ALPHA:-16}"
BISO_INIT_STD="${BISO_INIT_STD:-0.0}"
BISO_PARAMETERIZATION="${BISO_PARAMETERIZATION:-primitive_cn}"
BISO_USE_CAYLEY_NEUMANN="${BISO_USE_CAYLEY_NEUMANN:-True}"
BISO_NUM_CAYLEY_NEUMANN_TERMS="${BISO_NUM_CAYLEY_NEUMANN_TERMS:-5}"
BISO_CAYLEY_NEUMANN_EPS="${BISO_CAYLEY_NEUMANN_EPS:-0.9}"
BISO_SELECTIVE_MODE="${BISO_SELECTIVE_MODE:-none}"
BISO_SELECTIVE_KEEP_RATIO="${BISO_SELECTIVE_KEEP_RATIO:-0.3}"
BISO_RAW_L2_COEF="${BISO_RAW_L2_COEF:-0.0}"
BISO_GATE_LR_SCALE="${BISO_GATE_LR_SCALE:-1.0}"

SPO_BLOCK_SIZE="${SPO_BLOCK_SIZE:-16}"
SPO_DEPTH="${SPO_DEPTH:-1}"
SPO_ALPHA="${SPO_ALPHA:-8}"
SPO_INIT_STD="${SPO_INIT_STD:-0.0}"
SPO_USE_CAYLEY_NEUMANN="${SPO_USE_CAYLEY_NEUMANN:-True}"
SPO_NUM_CAYLEY_NEUMANN_TERMS="${SPO_NUM_CAYLEY_NEUMANN_TERMS:-5}"
SPO_CAYLEY_NEUMANN_EPS="${SPO_CAYLEY_NEUMANN_EPS:-0.9}"
SPO_SEED="${SPO_SEED:-42}"

GEORA_SPARSITY_RATIO="${GEORA_SPARSITY_RATIO:-0.2}"
GEORA_OVERSAMPLE="${GEORA_OVERSAMPLE:-8}"
GEORA_NITER="${GEORA_NITER:-2}"
GEORA_SVD_DEVICE="${GEORA_SVD_DEVICE:-auto}"
GEORA_RESIDUAL_ANCHOR="${GEORA_RESIDUAL_ANCHOR:-True}"
GEORA_INIT_SCALE="${GEORA_INIT_SCALE:-1.0}"
GEORA_SEED="${GEORA_SEED:-42}"

case "${METHOD}" in
  lora)
    PEFT_TYPE="lora"
    ;;
  geora)
    PEFT_TYPE="geora"
    ;;
  oft)
    PEFT_TYPE="oft"
    LORA_RANK=0
    ;;
  skew)
    PEFT_TYPE="skew"
    LORA_RANK=0
    USE_ORIG_PARAMS="${USE_ORIG_PARAMS_OVERRIDE:-True}"
    ;;
  boet)
    PEFT_TYPE="boet"
    LORA_RANK=0
    USE_ORIG_PARAMS="${USE_ORIG_PARAMS_OVERRIDE:-True}"
    ;;
  biso)
    PEFT_TYPE="biso"
    LORA_RANK=0
    USE_ORIG_PARAMS="${USE_ORIG_PARAMS_OVERRIDE:-True}"
    ;;
  biso_mask)
    PEFT_TYPE="biso"
    LORA_RANK=0
    USE_ORIG_PARAMS="${USE_ORIG_PARAMS_OVERRIDE:-True}"
    BISO_SELECTIVE_MODE="${BISO_SELECTIVE_MODE_OVERRIDE:-geora_block_mask}"
    ;;
  spo)
    PEFT_TYPE="spo"
    LORA_RANK=0
    USE_ORIG_PARAMS="${USE_ORIG_PARAMS_OVERRIDE:-True}"
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

EXP_NAME="${EXP_NAME:-paper_clean_${DATASET}_${METHOD}_b${TRAIN_PROMPT_BSZ}m${TRAIN_PROMPT_MINI_BSZ}n${N_RESP_PER_PROMPT}_lr${LR}_wd0_${RUN_TAG}}"
CKPTS_DIR="${CKPTS_DIR:-/home/wangls/Tina_orthres_run/ckpts/verl/${PROJECT_NAME}/${EXP_NAME}}"
LOG_DIR="${LOG_DIR:-/home/wangls/Tina_orthres_run/logs/paper_clean}"
mkdir -p "${LOG_DIR}"

if [[ "${RESUME_MODE}" == "disable" && -d "${CKPTS_DIR}" ]] && find "${CKPTS_DIR}" -maxdepth 1 -type d -name 'global_step_*' | grep -q .; then
  echo "Refusing to start: CKPTS_DIR already has checkpoints and RESUME_MODE=disable: ${CKPTS_DIR}" >&2
  echo "Set a new EXP_NAME/RUN_TAG, or set ALLOW_EXISTING_CKPT_DIR=1 to bypass this guard." >&2
  if [[ "${ALLOW_EXISTING_CKPT_DIR:-0}" != "1" ]]; then
    exit 3
  fi
fi

cleanup_ray_temp() {
  if [[ -d "${RAY_TEMP_DIR}" ]]; then
    pkill -TERM -f "${RAY_TEMP_DIR}" 2>/dev/null || true
    sleep 5
    pkill -KILL -f "${RAY_TEMP_DIR}" 2>/dev/null || true
    rm -rf "${RAY_TEMP_DIR}"
  fi
  mkdir -p "${RAY_TEMP_DIR}"
}

TRAIN_ROWS="$(python3 - "${TRAIN_FILE}" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)"

printf '%s\n' \
  "===== paper clean config =====" \
  "METHOD=${METHOD}" \
  "PEFT_TYPE=${PEFT_TYPE}" \
  "DATASET=${DATASET}" \
  "TRAIN_FILE=${TRAIN_FILE}" \
  "TEST_FILE=${TEST_FILE}" \
  "TRAIN_ROWS=${TRAIN_ROWS}" \
  "TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}" \
  "TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ}" \
  "N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}" \
  "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}" \
  "PROMPT_EXPOSURE=$((TOTAL_TRAINING_STEPS * TRAIN_PROMPT_BSZ))" \
  "ROLLOUT_TRAJECTORIES=$((TOTAL_TRAINING_STEPS * TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT))" \
  "LR=${LR}" \
  "LR_WARMUP_STEPS=${LR_WARMUP_STEPS}" \
  "WEIGHT_DECAY=${WEIGHT_DECAY}" \
  "SPO_BLOCK_SIZE=${SPO_BLOCK_SIZE}" \
  "SPO_DEPTH=${SPO_DEPTH}" \
  "SPO_ALPHA=${SPO_ALPHA}" \
  "SPO_INIT_STD=${SPO_INIT_STD}" \
  "SPO_USE_CAYLEY_NEUMANN=${SPO_USE_CAYLEY_NEUMANN}" \
  "SPO_NUM_CAYLEY_NEUMANN_TERMS=${SPO_NUM_CAYLEY_NEUMANN_TERMS}" \
  "SPO_CAYLEY_NEUMANN_EPS=${SPO_CAYLEY_NEUMANN_EPS}" \
  "SPO_SEED=${SPO_SEED}" \
  "DATA_SEED=${DATA_SEED}" \
  "PPO_DATA_LOADER_SEED=${PPO_DATA_LOADER_SEED}" \
  "TRAIN_SHUFFLE=${TRAIN_SHUFFLE}" \
  "RESUME_MODE=${RESUME_MODE}" \
  "EXP_NAME=${EXP_NAME}" \
  "CKPTS_DIR=${CKPTS_DIR}" \
  "RAY_TEMP_DIR=${RAY_TEMP_DIR}" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" \
  "=============================="

CMD=(
  python3 -m verl.trainer.main_ppo
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${TEST_FILE}"
  "data.prompt_key=prompt"
  "data.reward_fn_key=data_source"
  "data.truncation=left"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.train_batch_size=${TRAIN_PROMPT_BSZ}"
  "data.val_batch_size=${TRAIN_PROMPT_BSZ}"
  "data.dataloader_num_workers=0"
  "data.shuffle=${TRAIN_SHUFFLE}"
  "data.seed=${DATA_SEED}"
  "data.validation_shuffle=False"
  "data.filter_overlong_prompts=False"
  "actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}"
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=True"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
  "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
  "actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.shuffle=${ACTOR_SHUFFLE}"
  "actor_rollout_ref.actor.data_loader_seed=${PPO_DATA_LOADER_SEED}"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "+actor_rollout_ref.model.peft_type=${PEFT_TYPE}"
  "actor_rollout_ref.model.lora_rank=${LORA_RANK}"
  "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}"
  "+actor_rollout_ref.model.lora_dropout=${LORA_DROPOUT}"
  "+actor_rollout_ref.model.oft_block_size=${OFT_BLOCK_SIZE}"
  "+actor_rollout_ref.model.oft_rank=${OFT_RANK}"
  "+actor_rollout_ref.model.oft_dropout=${OFT_DROPOUT}"
  "+actor_rollout_ref.model.oft_coft=${OFT_COFT}"
  "+actor_rollout_ref.model.oft_eps=${OFT_EPS}"
  "+actor_rollout_ref.model.oft_block_share=${OFT_BLOCK_SHARE}"
  "+actor_rollout_ref.model.oft_use_cayley_neumann=${OFT_USE_CAYLEY_NEUMANN}"
  "+actor_rollout_ref.model.oft_num_cayley_neumann_terms=${OFT_NUM_CAYLEY_NEUMANN_TERMS}"
  "+actor_rollout_ref.model.skew_rank=${SKEW_RANK}"
  "+actor_rollout_ref.model.skew_alpha=${SKEW_ALPHA}"
  "+actor_rollout_ref.model.skew_dropout=${SKEW_DROPOUT}"
  "+actor_rollout_ref.model.skew_init_std=${SKEW_INIT_STD}"
  "+actor_rollout_ref.model.boet_rank=${BOET_RANK}"
  "+actor_rollout_ref.model.boet_alpha=${BOET_ALPHA}"
  "+actor_rollout_ref.model.boet_dropout=${BOET_DROPOUT}"
  "+actor_rollout_ref.model.boet_init_std=${BOET_INIT_STD}"
  "+actor_rollout_ref.model.boet_use_cayley_neumann=${BOET_USE_CAYLEY_NEUMANN}"
  "+actor_rollout_ref.model.boet_num_cayley_neumann_terms=${BOET_NUM_CAYLEY_NEUMANN_TERMS}"
  "+actor_rollout_ref.model.boet_cayley_neumann_eps=${BOET_CAYLEY_NEUMANN_EPS}"
  "+actor_rollout_ref.model.biso_block_size=${BISO_BLOCK_SIZE}"
  "+actor_rollout_ref.model.biso_alpha=${BISO_ALPHA}"
  "+actor_rollout_ref.model.biso_init_std=${BISO_INIT_STD}"
  "+actor_rollout_ref.model.biso_parameterization=${BISO_PARAMETERIZATION}"
  "+actor_rollout_ref.model.biso_use_cayley_neumann=${BISO_USE_CAYLEY_NEUMANN}"
  "+actor_rollout_ref.model.biso_num_cayley_neumann_terms=${BISO_NUM_CAYLEY_NEUMANN_TERMS}"
  "+actor_rollout_ref.model.biso_cayley_neumann_eps=${BISO_CAYLEY_NEUMANN_EPS}"
  "+actor_rollout_ref.model.biso_selective_mode=${BISO_SELECTIVE_MODE}"
  "+actor_rollout_ref.model.biso_selective_keep_ratio=${BISO_SELECTIVE_KEEP_RATIO}"
  "+actor_rollout_ref.model.spo_block_size=${SPO_BLOCK_SIZE}"
  "+actor_rollout_ref.model.spo_depth=${SPO_DEPTH}"
  "+actor_rollout_ref.model.spo_alpha=${SPO_ALPHA}"
  "+actor_rollout_ref.model.spo_init_std=${SPO_INIT_STD}"
  "+actor_rollout_ref.model.spo_use_cayley_neumann=${SPO_USE_CAYLEY_NEUMANN}"
  "+actor_rollout_ref.model.spo_num_cayley_neumann_terms=${SPO_NUM_CAYLEY_NEUMANN_TERMS}"
  "+actor_rollout_ref.model.spo_cayley_neumann_eps=${SPO_CAYLEY_NEUMANN_EPS}"
  "+actor_rollout_ref.model.spo_seed=${SPO_SEED}"
  "+actor_rollout_ref.model.geora_sparsity_ratio=${GEORA_SPARSITY_RATIO}"
  "+actor_rollout_ref.model.geora_oversample=${GEORA_OVERSAMPLE}"
  "+actor_rollout_ref.model.geora_niter=${GEORA_NITER}"
  "+actor_rollout_ref.model.geora_svd_device=${GEORA_SVD_DEVICE}"
  "+actor_rollout_ref.model.geora_residual_anchor=${GEORA_RESIDUAL_ANCHOR}"
  "+actor_rollout_ref.model.geora_init_scale=${GEORA_INIT_SCALE}"
  "+actor_rollout_ref.model.geora_seed=${GEORA_SEED}"
  "actor_rollout_ref.model.target_modules=${TARGET_MODULES}"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
  "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=constant"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ}"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD}"
  "actor_rollout_ref.actor.fsdp_config.use_orig_params=${USE_ORIG_PARAMS}"
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "+actor_rollout_ref.actor.biso_raw_l2_coef=${BISO_RAW_L2_COEF}"
  "+actor_rollout_ref.actor.biso_gate_lr_scale=${BISO_GATE_LR_SCALE}"
  "actor_rollout_ref.actor.grad_clip=1.0"
  "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.load_format=dummy"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=True"
  "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
  "actor_rollout_ref.rollout.top_p=${TOP_P}"
  "actor_rollout_ref.rollout.top_k=${TOP_K}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
  "actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD}"
  "actor_rollout_ref.ref.ulysses_sequence_parallel_size=1"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_PPO_MAX_TOKEN_LEN}"
  "actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS}"
  "reward_model.reward_manager=dapo"
  "custom_reward_function.path=${CUSTOM_REWARD_PATH}"
  "custom_reward_function.name=compute_score"
  "+reward_model.reward_kwargs.overlong_buffer_cfg.enable=False"
  "+reward_model.reward_kwargs.overlong_buffer_cfg.len=4096"
  "+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0"
  "+reward_model.reward_kwargs.overlong_buffer_cfg.log=True"
  "+reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}"
  "trainer.logger=['console']"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXP_NAME}"
  "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.val_before_train=False"
  "trainer.test_freq=0"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
  "trainer.total_epochs=1"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.default_local_dir=${CKPTS_DIR}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.log_val_generations=0"
  "+ray_kwargs.ray_init.address=${RAY_ADDRESS}"
  "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
  "+ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}"
  "+ray_kwargs.ray_init.include_dashboard=False"
  "+ray_kwargs.ray_init._temp_dir=${RAY_TEMP_DIR}"
)

if [[ "${DRY_RUN:-1}" == "1" ]]; then
  printf 'DRY_RUN=1, command not executed. Set DRY_RUN=0 to launch.\n'
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cleanup_ray_temp
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"
printf 'Launching clean run; log=%s\n' "${LOG_FILE}"
TMPDIR="${RAY_TEMP_DIR}" \
TEMP="${RAY_TEMP_DIR}" \
TMP="${RAY_TEMP_DIR}" \
RAY_TMPDIR="${RAY_TEMP_DIR}" \
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
