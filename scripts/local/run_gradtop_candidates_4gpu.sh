#!/usr/bin/env bash
# Frozen observation recipes followed by the common RLVR training protocol.
# DRY_RUN=1 is the default and does not start Ray or claim GPUs.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
cd "${repo_root}"
method="${METHOD:-gradtop_prompt_sign}"
phase="${PHASE:-all}"
dry_run="${DRY_RUN:-1}"
rank="${LORA_RANK:-32}"
lr="${LR:-1e-5}"
target_step="${TARGET_STEP:-50}"
python_bin="${PYTHON_BIN:-python3}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/envs/peft-for-rl/bin/python" && -z "${PYTHON_BIN:-}" ]]; then
    python_bin="${CONDA_PREFIX}/envs/peft-for-rl/bin/python"
fi
case "${method}" in
    gradtop) variant=token_mean; observations=16 ;;
    gradtop_short) variant=token_mean; observations=8 ;;
    gradtop_prompt_mean) variant=prompt_mean; observations=8 ;;
    gradtop_prompt_sign) variant=prompt_sign; observations=8 ;;
    gradtop_masked) variant=masked; observations=16 ;;
    *) echo "Unsupported GradTop METHOD=${method}" >&2; exit 2 ;;
esac
case "${phase}" in probe|train|all) ;; *) echo "PHASE must be probe, train, or all" >&2; exit 2 ;; esac
case "${dry_run}" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
case "${lr}" in 1e-6|5e-6|1e-5) ;; *) echo "LR must be 1e-6, 5e-6, or 1e-5" >&2; exit 2 ;; esac
if [[ ! "${rank}" =~ ^[1-9][0-9]*$ ]] || (( rank > 64 )); then
    echo "LORA_RANK must be an integer from 1 to the frozen sketch capacity 64" >&2
    exit 2
fi
if [[ ! "${target_step}" =~ ^[1-9][0-9]*$ ]] || (( target_step > 270 || target_step % 10 != 0 )); then
    echo "TARGET_STEP must be a positive multiple of 10 and <=270" >&2
    exit 2
fi
if [[ "${phase}" == train && -z "${GRADTOP_ARTIFACT_DIR:-}" ]]; then
    echo "PHASE=train requires GRADTOP_ARTIFACT_DIR" >&2
    exit 2
fi
if [[ "${GRADTOP_PREVIEW_MISSING_ARTIFACT:-0}" == 1 && "${dry_run}" != 1 ]]; then
    echo "Artifact validation cannot be deferred when DRY_RUN=0" >&2
    exit 2
fi
python_path="$(command -v "${python_bin}")"
export PATH="$(dirname -- "${python_path}"):${PATH}"
export PYTHON_BIN="${python_path}"
trainer_steps=$((observations / 4))
run_tag="${RUN_TAG:-$("${python_path}" -c 'import datetime, uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])')}"
run_name="${EXP_NAME:-rlvr_v1_${method}_r${rank}_a64_lr${lr//-/m}_${run_tag}}"
run_dir="$(realpath -m -- "${GRADTOP_RUN_DIR:-${repo_root}/runs/gradtop_candidates/${run_name}}")"
artifact_dir="$(realpath -m -- "${GRADTOP_ARTIFACT_DIR:-${run_dir}/probe_artifact}")"
protocol="${PROTOCOL_FILE:-${repo_root}/configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json}"

lock_value() {
    local name="$1" expected="$2"
    if [[ -v "${name}" && "${!name}" != "${expected}" ]]; then
        echo "Frozen GradTop recipe rejects ${name}=${!name}; required ${expected}" >&2
        exit 2
    fi
    printf -v "${name}" '%s' "${expected}"
    export "${name}"
}

printf 'GradTop METHOD=%s variant=%s observations=%s trainer_steps=%s rank=%s alpha=64 PHASE=%s DRY_RUN=%s\n' \
    "${method}" "${variant}" "${observations}" "${trainer_steps}" "${rank}" "${phase}" "${dry_run}"
printf 'Run directory: %s\nArtifact directory: %s\n' "${run_dir}" "${artifact_dir}"

if [[ "${phase}" == probe || "${phase}" == all ]]; then
    if [[ -e "${run_dir}/probe_launch.json" ]] || compgen -G "${run_dir}/probe_work/global_step_*" >/dev/null; then
        echo "Refusing to reuse a run directory with probe provenance or checkpoints: ${run_dir}" >&2
        exit 2
    fi
    if [[ -e "${artifact_dir}" ]] && [[ ! -d "${artifact_dir}" || -n "$(ls -A -- "${artifact_dir}")" ]]; then
        echo "Refusing to overwrite or silently reuse an existing probe artifact directory: ${artifact_dir}" >&2
        echo "Use PHASE=train to consume a completed artifact, or choose a fresh run directory." >&2
        exit 2
    fi
    (
        lock_value RUNTIME_ROOT "${repo_root}/runs"
        lock_value MODEL_PATH "${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
        lock_value TRAIN_FILE "${repo_root}/runs/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet"
        lock_value TEST_FILE "${TRAIN_FILE}"
        lock_value CUSTOM_REWARD_PATH "${repo_root}/verl/utils/reward_score/boxed_math_accuracy.py"
        lock_value PEFT_TYPE grad_probe
        lock_value TRAIN_PROMPT_BSZ 64
        lock_value TRAIN_PROMPT_MINI_BSZ 16
        lock_value N_RESP_PER_PROMPT 8
        lock_value PPO_EPOCHS 1
        lock_value MAX_PROMPT_LENGTH 1024
        lock_value MAX_RESPONSE_LENGTH 8192
        lock_value LOSS_AGG_MODE token-mean
        lock_value TRAIN_SHUFFLE True
        lock_value ACTOR_SHUFFLE False
        lock_value DATA_SEED 42
        lock_value PPO_DATA_LOADER_SEED 42
        lock_value ROLLOUT_SEED 42
        lock_value FSDP_SEED 42
        lock_value TARGET_MODULES all-linear
        lock_value LORA_ALPHA 64
        lock_value LORA_DROPOUT 0.0
        lock_value LORA_FREEZE_A False
        lock_value USE_ORIG_PARAMS True
        lock_value LORA_ADAPTER_PATH null
        lock_value LORA_RANK_PATTERN_PATH null
        lock_value WARM_START_GLOBAL_STEP null
        lock_value WARM_START_DATA_PATH null
        lock_value SAVE_STEPS null
        lock_value RAY_ADDRESS local
        lock_value GRADIENT_PROBE_METHOD energy
        lock_value GRADIENT_PROBE_VARIANT "${variant}"
        lock_value GRADIENT_PROBE_EXPECTED_OBSERVATIONS "${observations}"
        lock_value GRADIENT_PROBE_WIDTH 8
        lock_value GRADIENT_PROBE_CAPACITY 64
        lock_value GRADIENT_PROBE_RANK_BINS "[${rank}]"
        lock_value GRADIENT_PROBE_TARGET_ENERGY 0.95
        lock_value GRADIENT_PROBE_SEED 42
        lock_value GRADIENT_PROBE_SIGN_SEED 42
        lock_value GRADIENT_PROBE_MIN_STEPS "${trainer_steps}"
        lock_value GRADIENT_PROBE_MAX_STEPS "${trainer_steps}"
        lock_value GRADIENT_PROBE_STABILITY_PATIENCE 1000000
        lock_value GRADIENT_SUBSPACE_SCALING "$("${python_path}" -c 'import sys; print(64 / int(sys.argv[1]))' "${rank}")"
        lock_value GRADIENT_PROBE_OUTPUT_DIR "${artifact_dir}"
        lock_value TOTAL_TRAINING_STEPS "${trainer_steps}"
        lock_value TEMPERATURE 1.0
        lock_value TOP_P 1.0
        lock_value TOP_K -1
        lock_value CLIP_RATIO_LOW 0.2
        lock_value CLIP_RATIO_HIGH 0.28
        lock_value USE_KL_LOSS False
        lock_value KL_LOSS_COEF 0.0
        lock_value ENABLE_OVERLONG_BUFFER False
        lock_value WEIGHT_DECAY 0
        lock_value LR_WARMUP_STEPS 0
        lock_value LR_SCHEDULER_TYPE constant
        lock_value OFFLOAD True
        lock_value FSDP_SIZE -1
        lock_value GEN_TP 1
        lock_value SP_SIZE 1
        lock_value GPU_MEMORY_UTILIZATION 0.70
        lock_value ACTOR_PPO_MAX_TOKEN_LEN 12288
        lock_value INFER_PPO_MAX_TOKEN_LEN 12288
        lock_value ROLLOUT_MAX_NUM_SEQS 256
        lock_value NGPUS_PER_NODE 4
        lock_value NNODES 1
        lock_value SAVE_FREQ -1
        lock_value TEST_FREQ 0
        lock_value LORA_SNAPSHOT_EVERY 0
        lock_value RESUME_MODE disable
        export EXP_NAME="${run_name}_probe" LR="${lr}" LORA_RANK="${rank}" DRY_RUN="${dry_run}"
        export CKPTS_DIR="${run_dir}/probe_work" RAY_TEMP_DIR="${run_dir}/ray"
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
        IFS=',' read -r -a gpu_array <<<"${CUDA_VISIBLE_DEVICES}"
        if (( ${#gpu_array[@]} != 4 )); then
            echo "The frozen GradTop recipe requires exactly four CUDA devices" >&2
            exit 2
        fi
        "${python_path}" "${repo_root}/scripts/analysis/verify_rlvr_protocol.py" \
            --protocol "${protocol}" --learning-rate "${lr}" --method "${method}" --rank "${rank}" \
            --target-step "${target_step}" --gradtop-artifact-dir "${artifact_dir}" \
            --preview-missing-gradtop-artifact >/dev/null
        if [[ "${dry_run}" == 0 ]]; then
            mkdir -p "${run_dir}"
            if [[ -e "${run_dir}/probe_launch.json" ]]; then
                echo "Refusing to overwrite run provenance: ${run_dir}/probe_launch.json" >&2
                exit 2
            fi
        fi
        "${python_path}" - "${run_dir}" "${dry_run}" "${method}" <<'PY'
import datetime, hashlib, json, os, pathlib, subprocess, sys
root, dry_run, method = sys.argv[1:]
keys = [key for key in os.environ if key.startswith(('GRADIENT_PROBE_', 'GRADIENT_SUBSPACE_'))]
keys += ['MODEL_PATH','TRAIN_FILE','TEST_FILE','CUSTOM_REWARD_PATH','PEFT_TYPE','TRAIN_PROMPT_BSZ','TRAIN_PROMPT_MINI_BSZ',
         'N_RESP_PER_PROMPT','MAX_PROMPT_LENGTH','MAX_RESPONSE_LENGTH','DATA_SEED','ROLLOUT_SEED','FSDP_SEED',
         'LORA_RANK','LORA_ALPHA','LORA_DROPOUT','TOTAL_TRAINING_STEPS','CUDA_VISIBLE_DEVICES','PYTHON_BIN','EXP_NAME',
         'LOSS_AGG_MODE','TARGET_MODULES','ACTOR_PPO_MAX_TOKEN_LEN','INFER_PPO_MAX_TOKEN_LEN',
         'USE_ORIG_PARAMS','LORA_ADAPTER_PATH','LORA_RANK_PATTERN_PATH','WARM_START_GLOBAL_STEP','WARM_START_DATA_PATH']
sources = ['verl/utils/peft_gradient_subspace.py', 'verl/workers/actor/dp_actor.py', 'verl/trainer/ppo/ray_trainer.py',
           'verl/workers/fsdp_workers.py', 'verl/workers/config/model.py',
           'scripts/local/run_gradtop_candidates_4gpu.sh', 'scripts/analysis/verify_rlvr_protocol.py',
           'examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh']
manifest = {'source_sha256': {name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in sources},
            'method': method, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'git_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git','status','--porcelain'], text=True)),
            'environment': {key: os.environ[key] for key in sorted(set(keys))},
            'command': ['bash', 'examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh']}
print(json.dumps(manifest, indent=2, sort_keys=True))
if dry_run == '0':
    with (pathlib.Path(root) / 'probe_launch.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
PY
        bash "${repo_root}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
        if [[ "${dry_run}" == 0 ]]; then
            "${python_path}" "${repo_root}/scripts/analysis/verify_rlvr_protocol.py" \
                --protocol "${protocol}" --learning-rate "${lr}" --method "${method}" --rank "${rank}" \
                --target-step "${target_step}" --gradtop-artifact-dir "${artifact_dir}" \
                --output "${run_dir}/probe_verified.json" >/dev/null
        fi
    )
fi

if [[ "${phase}" == train || "${phase}" == all ]]; then
    preview=0
    if [[ "${dry_run}" == 1 && ! -e "${artifact_dir}/rank_map.json" && ! -e "${artifact_dir}/summary.json" && ! -e "${artifact_dir}/subspaces.safetensors" ]]; then
        preview=1
        echo "DRY_RUN preview: probe artifacts do not exist; formal training artifact validation is deferred."
    fi
    METHOD="${method}" LR="${lr}" LORA_RANK="${rank}" TARGET_STEP="${target_step}" \
    EXP_NAME="${run_name}" DRY_RUN="${dry_run}" GRADTOP_ARTIFACT_DIR="${artifact_dir}" \
    GRADTOP_PREVIEW_MISSING_ARTIFACT="${preview}" \
        bash "${repo_root}/scripts/local/run_rlvr_protocol_1p5b_4gpu.sh"
fi
