#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"

# Keep the runtime bound to this checkout even when the launcher is invoked
# from a directory whose shell environment references another verl tree.
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
export RAY_ADDRESS=${RAY_ADDRESS:-local}

project_name=${PROJECT_NAME:-DAPO-Math-17k}
exp_name=${EXP_NAME:-dapo_math_boxed_stable_lora_1p5b_4gpu_8k}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-4096}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode=${LOSS_AGG_MODE:-token-mean}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-16}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}
ppo_epochs=${PPO_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-100}
if (( train_prompt_bsz % train_prompt_mini_bsz != 0 )); then
    echo "TRAIN_PROMPT_BSZ must be divisible by TRAIN_PROMPT_MINI_BSZ" >&2
    exit 2
fi
actor_updates_per_step=$((train_prompt_bsz / train_prompt_mini_bsz * ppo_epochs))
adalora_total_step=${ADALORA_TOTAL_STEP:-$((total_training_steps * actor_updates_per_step))}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

MODEL_PATH=${MODEL_PATH:-${RUNTIME_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}
TRAIN_FILE=${TRAIN_FILE:-${RUNTIME_ROOT}/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}
TEST_FILE=${TEST_FILE:-${TRAIN_FILE}}
CKPTS_DIR=${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/${project_name}/${exp_name}}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${REPO_ROOT}/verl/utils/reward_score/boxed_math_accuracy.py}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray}

imported_verl_root="$(
    python3 -c 'import pathlib, verl; print(pathlib.Path(verl.__file__).resolve().parent)'
)"
expected_verl_root="${REPO_ROOT}/verl"
if [[ "${imported_verl_root}" != "${expected_verl_root}" ]]; then
    echo "Expected verl from ${expected_verl_root}, got ${imported_verl_root}" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]] || ! compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
    echo "Missing model config or safetensors weights under ${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -s "${TRAIN_FILE}" ]]; then
    echo "Missing or empty training data: ${TRAIN_FILE}" >&2
    exit 2
fi
if [[ ! -s "${TEST_FILE}" ]]; then
    echo "Missing or empty validation data: ${TEST_FILE}" >&2
    exit 2
fi
mkdir -p "${CKPTS_DIR}" "${RAY_TEMP_DIR}"

temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_top_p=${VAL_TOP_P:-0.7}

sp_size=${SP_SIZE:-1}
use_dynamic_bsz=True
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 2))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 3))}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-$((train_prompt_bsz * n_resp_per_prompt))}
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-1}
fsdp_size=${FSDP_SIZE:--1}

save_contents=${SAVE_CONTENTS:-"['model', 'extra']"}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Configuration preflight passed; DRY_RUN=1, Ray and training were not started."
    exit 0
fi

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${train_prompt_bsz} \
    data.dataloader_num_workers=0 \
    data.shuffle=${TRAIN_SHUFFLE:-True} \
    data.seed=${DATA_SEED:-42} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.peft_type=${PEFT_TYPE:-lora} \
    actor_rollout_ref.model.lora_rank=${LORA_RANK:-32} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-64} \
    +actor_rollout_ref.model.lora_freeze_a=${LORA_FREEZE_A:-False} \
    actor_rollout_ref.model.lora_rank_pattern_path=${LORA_RANK_PATTERN_PATH:-null} \
    actor_rollout_ref.model.lora_adapter_path=${LORA_ADAPTER_PATH:-null} \
    +actor_rollout_ref.model.lora_dropout=${LORA_DROPOUT:-0.05} \
    +actor_rollout_ref.model.rlpo_svd_device=${RLPO_SVD_DEVICE:-auto} \
    +actor_rollout_ref.model.tinylora_rank=${TINY_LORA_RANK:-2} \
    +actor_rollout_ref.model.tinylora_projection_dim=${TINY_LORA_PROJECTION_DIM:-1} \
    +actor_rollout_ref.model.tinylora_tie_factor=${TINY_LORA_TIE_FACTOR:-16} \
    +actor_rollout_ref.model.tinylora_tie_strategy=${TINY_LORA_TIE_STRATEGY:-tiled} \
    +actor_rollout_ref.model.tinylora_seed=${TINY_LORA_SEED:-42} \
    +actor_rollout_ref.model.tinylora_svd_device=${TINY_LORA_SVD_DEVICE:-auto} \
    +actor_rollout_ref.model.tinylora_svd_method=${TINY_LORA_SVD_METHOD:-lowrank} \
    +actor_rollout_ref.model.tinylora_svd_oversample=${TINY_LORA_SVD_OVERSAMPLE:-4} \
    +actor_rollout_ref.model.tinylora_svd_niter=${TINY_LORA_SVD_NITER:-2} \
    +actor_rollout_ref.model.tinylora_projection_std=${TINY_LORA_PROJECTION_STD:-1.0} \
    +actor_rollout_ref.model.adalora_init_r=${ADALORA_INIT_R:-32} \
    +actor_rollout_ref.model.adalora_target_r=${ADALORA_TARGET_R:-8} \
    +actor_rollout_ref.model.adalora_tinit=${ADALORA_TINIT:-100} \
    +actor_rollout_ref.model.adalora_tfinal=${ADALORA_TFINAL:-200} \
    +actor_rollout_ref.model.adalora_delta_t=${ADALORA_DELTA_T:-20} \
    +actor_rollout_ref.model.adalora_beta1=${ADALORA_BETA1:-0.85} \
    +actor_rollout_ref.model.adalora_beta2=${ADALORA_BETA2:-0.85} \
    +actor_rollout_ref.model.adalora_orth_reg_weight=${ADALORA_ORTH_REG_WEIGHT:-1e-3} \
    +actor_rollout_ref.model.adalora_total_step=${adalora_total_step} \
    +actor_rollout_ref.model.gradient_probe_width=${GRADIENT_PROBE_WIDTH:-8} \
    +actor_rollout_ref.model.gradient_probe_capacity=${GRADIENT_PROBE_CAPACITY:-64} \
    +actor_rollout_ref.model.gradient_probe_target_energy=${GRADIENT_PROBE_TARGET_ENERGY:-0.95} \
    +actor_rollout_ref.model.gradient_probe_rank_bins=${GRADIENT_PROBE_RANK_BINS:-'[8,12,16,20,24,28,32]'} \
    +actor_rollout_ref.model.gradient_probe_seed=${GRADIENT_PROBE_SEED:-42} \
    +actor_rollout_ref.model.gradient_probe_min_steps=${GRADIENT_PROBE_MIN_STEPS:-6} \
    +actor_rollout_ref.model.gradient_probe_max_steps=${GRADIENT_PROBE_MAX_STEPS:-12} \
    +actor_rollout_ref.model.gradient_probe_stability_patience=${GRADIENT_PROBE_STABILITY_PATIENCE:-3} \
    +actor_rollout_ref.model.gradient_probe_overlap_threshold=${GRADIENT_PROBE_OVERLAP_THRESHOLD:-0.98} \
    +actor_rollout_ref.model.gradient_probe_rank_tolerance=${GRADIENT_PROBE_RANK_TOLERANCE:-1.0} \
    +actor_rollout_ref.model.gradient_probe_output_dir=${GRADIENT_PROBE_OUTPUT_DIR:-null} \
    +actor_rollout_ref.model.gradient_probe_method=${GRADIENT_PROBE_METHOD:-energy} \
    +actor_rollout_ref.model.gradient_probe_window_size=${GRADIENT_PROBE_WINDOW_SIZE:-3} \
    +actor_rollout_ref.model.gradient_probe_num_windows=${GRADIENT_PROBE_NUM_WINDOWS:-5} \
    +actor_rollout_ref.model.gradient_probe_calibration_windows=${GRADIENT_PROBE_CALIBRATION_WINDOWS:-0} \
    +actor_rollout_ref.model.gradient_probe_validation_windows=${GRADIENT_PROBE_VALIDATION_WINDOWS:-0} \
    +actor_rollout_ref.model.gradient_probe_target_mean_rank=${GRADIENT_PROBE_TARGET_MEAN_RANK:-16.0} \
    +actor_rollout_ref.model.gradient_probe_snr_ridge=${GRADIENT_PROBE_SNR_RIDGE:-0.05} \
    +actor_rollout_ref.model.gradient_probe_clip_factor=${GRADIENT_PROBE_CLIP_FACTOR:-2.5} \
    +actor_rollout_ref.model.gradient_probe_normalize_rank_utility=${GRADIENT_PROBE_NORMALIZE_RANK_UTILITY:-False} \
    +actor_rollout_ref.model.gradient_probe_balance_rank_by_module_type=${GRADIENT_PROBE_BALANCE_RANK_BY_MODULE_TYPE:-False} \
    +actor_rollout_ref.model.gradient_probe_local_atoms=${GRADIENT_PROBE_LOCAL_ATOMS:-2} \
    +actor_rollout_ref.model.gradient_probe_gap_cap=${GRADIENT_PROBE_GAP_CAP:-4.0} \
    +actor_rollout_ref.model.gradient_subspace_rank_map_path=${GRADIENT_SUBSPACE_RANK_MAP_PATH:-null} \
    +actor_rollout_ref.model.gradient_subspace_path=${GRADIENT_SUBSPACE_PATH:-null} \
    +actor_rollout_ref.model.gradient_subspace_scaling=${GRADIENT_SUBSPACE_SCALING:-2.0} \
    +actor_rollout_ref.model.oft_block_size=${OFT_BLOCK_SIZE:-32} \
    +actor_rollout_ref.model.oft_rank=${OFT_RANK:-0} \
    +actor_rollout_ref.model.oft_dropout=${OFT_DROPOUT:-0.0} \
    +actor_rollout_ref.model.oft_coft=${OFT_COFT:-False} \
    +actor_rollout_ref.model.oft_eps=${OFT_EPS:-0.00006} \
    +actor_rollout_ref.model.oft_block_share=${OFT_BLOCK_SHARE:-False} \
    +actor_rollout_ref.model.oft_use_cayley_neumann=${OFT_USE_CAYLEY_NEUMANN:-True} \
    +actor_rollout_ref.model.oft_num_cayley_neumann_terms=${OFT_NUM_CAYLEY_NEUMANN_TERMS:-5} \
    actor_rollout_ref.model.target_modules=${TARGET_MODULES:-all-linear} \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:-0} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE:-constant} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY:-0} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.shuffle=${ACTOR_SHUFFLE:-False} \
    actor_rollout_ref.actor.data_loader_seed=${PPO_DATA_LOADER_SEED:-42} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=${USE_ORIG_PARAMS:-False} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${FSDP_RESHARD_AFTER_FORWARD:-True} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}" \
    reward_model.reward_manager=dapo \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=${TEST_FREQ:-0} \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    +trainer.warm_start_global_step=${WARM_START_GLOBAL_STEP:-null} \
    +trainer.warm_start_data_path=${WARM_START_DATA_PATH:-null} \
    trainer.log_val_generations=0 \
    +ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-64} \
    +ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE} \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}"
