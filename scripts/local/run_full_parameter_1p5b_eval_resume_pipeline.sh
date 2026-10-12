#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
run_root="${FULL_PARAM_RUN_ROOT:-${repo_root}/runs/full-parameter-1p5b-unified-lr1e6-v2}"
exp_name="${EXP_NAME:-full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2}"
checkpoint_root="${CKPTS_DIR:-${run_root}/ckpts/verl/DAPO-Math-17k/${exp_name}}"
initial_service="${INITIAL_TRAIN_SERVICE:-peft-full-parameter-1p5b-lr1e6-v1.service}"
snapshot="${BENCHMARK_SNAPSHOT_RECORDS:-${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl}"
snapshot_sha256="3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da"
direct_opd_eval_root="${DIRECT_OPD_EVAL_ROOT:-/home/wangls/Direct-OPD/datasets/eval}"
gpu_ids="0,1,2,3"
controller_log="${run_root}/logs/full_parameter_step50_eval_resume_step100.controller.log"

mkdir -p "${run_root}/logs" "${run_root}/analysis/evaluation_verification"
exec 9>"${run_root}/.step50-eval-resume-step100.lock"
if ! flock -n 9; then
    echo "Another full-parameter evaluation/resume controller holds the lock" >&2
    exit 3
fi
exec > >(tee -a "${controller_log}") 2>&1

log() {
    printf '[%s] %s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$*"
}

checkpoint_complete() {
    local step="$1"
    local checkpoint="${checkpoint_root}/global_step_${step}"
    local rank relative
    local required=(
        "actor/fsdp_config.json"
        "actor/huggingface/config.json"
        "data.pt"
    )
    for rank in 0 1 2 3; do
        required+=(
            "actor/model_world_size_4_rank_${rank}.pt"
            "actor/optim_world_size_4_rank_${rank}.pt"
            "actor/extra_state_world_size_4_rank_${rank}.pt"
        )
    done
    for relative in "${required[@]}"; do
        [[ -s "${checkpoint}/${relative}" ]] || return 1
    done
}

hf_model_complete() {
    local model_dir="$1"
    [[ -s "${model_dir}/.merge_complete" ]] || return 1
    [[ -s "${model_dir}/config.json" ]] || return 1
    [[ -s "${model_dir}/tokenizer_config.json" ]] || return 1
    [[ -s "${model_dir}/model.safetensors" || -s "${model_dir}/model.safetensors.index.json" ]]
}

hf_model_files_complete() {
    local model_dir="$1"
    [[ -s "${model_dir}/config.json" ]] || return 1
    [[ -s "${model_dir}/tokenizer_config.json" ]] || return 1
    [[ -s "${model_dir}/model.safetensors" || -s "${model_dir}/model.safetensors.index.json" ]]
}

require_gpus_free() {
    local gpu pids
    for gpu in 0 1 2 3; do
        pids="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
        if [[ -n "${pids//[[:space:]]/}" ]]; then
            log "GPU ${gpu} is unexpectedly busy with PIDs ${pids//$'\n'/,}"
            return 1
        fi
    done
}

wait_for_initial_training() {
    local active_state result status
    while true; do
        active_state="$(systemctl --user show "${initial_service}" -p ActiveState --value)"
        case "${active_state}" in
            active|activating|reloading|deactivating)
                log "Waiting for ${initial_service}; state=${active_state}"
                sleep 60
                ;;
            inactive)
                result="$(systemctl --user show "${initial_service}" -p Result --value)"
                status="$(systemctl --user show "${initial_service}" -p ExecMainStatus --value)"
                if [[ "${result}" != "success" || "${status}" != "0" ]]; then
                    echo "Initial training stopped unsuccessfully: result=${result} status=${status}" >&2
                    return 1
                fi
                break
                ;;
            failed)
                systemctl --user status "${initial_service}" --no-pager -l || true
                echo "Initial training service failed" >&2
                return 1
                ;;
            *)
                echo "Unexpected state for ${initial_service}: ${active_state}" >&2
                return 1
                ;;
        esac
    done
    checkpoint_complete 50 || {
        echo "Initial training exited without a complete step-50 checkpoint" >&2
        return 1
    }
    log "Verified complete step-50 model, optimizer, extra, and dataloader state"
}

evaluate_step() {
    local step="$1"
    local checkpoint="${checkpoint_root}/global_step_${step}"
    local actor_dir="${checkpoint}/actor"
    local merged_model="${run_root}/merged/step${step}/actor_hf"
    local eval_name="${exp_name}_step${step}_fullbench_32768_seed42"
    local eval_dir="${run_root}/outputs/full_bench_eval/${eval_name}"
    local summary="${eval_dir}/summary/${eval_name}.json"
    local records="${eval_dir}/records/${eval_name}.jsonl"
    local postverify="${run_root}/analysis/evaluation_verification/${eval_name}.json"
    local verification
    local verify_args=(
        "scripts/analysis/verify_full_bench_run.py"
        --summary "${summary}"
        --records "${records}"
        --eval-dir "${eval_dir}"
        --eval-name "${eval_name}"
        --no-adapter
        --base-model "${merged_model}"
        --snapshot "${snapshot}"
        --snapshot-sha256 "${snapshot_sha256}"
    )

    checkpoint_complete "${step}" || {
        echo "Incomplete step-${step} checkpoint: ${checkpoint}" >&2
        return 1
    }
    if [[ -s "${postverify}" ]] \
        && jq -e '.status == "verified"' "${postverify}" >/dev/null \
        && hf_model_complete "${merged_model}" \
        && "${python_bin}" "${verify_args[@]}" >/dev/null 2>&1; then
        log "Step-${step} fullbench already verified: ${postverify}"
        return 0
    fi

    mkdir -p "${merged_model}" "${eval_dir}/logs"
    if ! hf_model_complete "${merged_model}"; then
        log "Merging step-${step} four-shard FSDP actor into Hugging Face format"
        CUDA_VISIBLE_DEVICES=0 "${python_bin}" -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${actor_dir}" \
            --target_dir "${merged_model}" \
            --use_cpu_initialization \
            >>"${eval_dir}/logs/${eval_name}.merge.log" 2>&1
        hf_model_files_complete "${merged_model}" || {
            echo "FSDP merge did not produce a complete HF model: ${merged_model}" >&2
            return 1
        }
        printf 'step=%s\ncompleted_at=%s\n' \
            "${step}" "$(TZ=Asia/Shanghai date --iso-8601=seconds)" \
            >"${merged_model}/.merge_complete"
    fi
    hf_model_complete "${merged_model}" || {
        echo "FSDP merge did not produce a complete HF model: ${merged_model}" >&2
        return 1
    }

    verification=""
    if [[ -s "${summary}" ]] \
        && verification="$("${python_bin}" "${verify_args[@]}" 2>>"${eval_dir}/logs/${eval_name}.verify.log")"; then
        log "Found an existing complete step-${step} fullbench; rebuilding its verification marker"
    else
        verification=""
        require_gpus_free
        log "Starting frozen six-task 7248-record fullbench for step ${step} on GPUs ${gpu_ids}"
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        PROJECT_ROOT="${repo_root}/tina_run" PYTHON_BIN="${python_bin}" \
        BASE_MODEL="${merged_model}" LORA_ADAPTER= \
        DIRECT_OPD_EVAL_ROOT="${direct_opd_eval_root}" \
        CHECKPOINT_NAME="${eval_name}" OUTPUT_DIR="${eval_dir}" \
        BENCHMARK_SNAPSHOT_RECORDS="${snapshot}" \
        BENCHMARKS="aime24,aime25,amc23,hmmt_feb,math500,minerva" \
        SEED=42 TEMPERATURE=0.6 TOP_P=0.95 \
        MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=34816 \
        SAMPLES_SMALL=32 SAMPLES_LARGE=4 LIMIT_PER_BENCHMARK= \
        GPU_MEMORY_UTILIZATION=0.85 CUDA_VISIBLE_DEVICES="${gpu_ids}" \
        bash "${repo_root}/tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
    fi

    if [[ -z "${verification}" ]]; then
        log "Verifying step-${step} summary, 7248 records, four manifests, and frozen protocol"
        verification="$("${python_bin}" "${verify_args[@]}")"
    fi
    jq -n \
        --arg created_at "$(TZ=Asia/Shanghai date --iso-8601=seconds)" \
        --arg summary "${summary}" \
        --arg records "${records}" \
        --arg merged_model "${merged_model}" \
        --argjson verification "${verification}" \
        '{schema_version:1,status:"verified",created_at:$created_at,summary:$summary,records:$records,merged_model:$merged_model,verification:$verification}' \
        >"${postverify}.tmp"
    mv "${postverify}.tmp" "${postverify}"
    log "STEP_${step}_FULLBENCH_VERIFIED ${summary}"
}

continue_to_step_100() {
    if checkpoint_complete 100; then
        log "Complete step-100 checkpoint already exists; continuation is not needed"
        return 0
    fi
    require_gpus_free
    log "Resuming the same run from step 50 to step 100 with optimizer and dataloader state"
    FULL_PARAM_RUN_ROOT="${run_root}" \
    EXP_NAME="${exp_name}" \
    CKPTS_DIR="${checkpoint_root}" \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    RESUME_MODE=auto \
    TOTAL_TRAINING_STEPS=100 \
    SAVE_FREQ=25 \
    MAX_ACTOR_CKPT_TO_KEEP=3 \
    SAVE_CONTENTS="['model','optimizer','extra']" \
    RAY_TEMP_DIR=/dev/shm/ray-fp15b-u2-resume \
    TMPDIR=/dev/shm/tmp-fp15b-u2-resume \
    LOG_FILE="${run_root}/logs/full_parameter_unified_training_step50_to100.log" \
    bash "${script_dir}/start_full_parameter_1p5b_lr1e6_4gpu.sh"
    checkpoint_complete 100 || {
        echo "Continuation exited without a complete step-100 checkpoint" >&2
        return 1
    }
    log "Verified complete step-100 model, optimizer, extra, and dataloader state"
}

for required in "${python_bin}" "${snapshot}" "${direct_opd_eval_root}"; do
    [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 2; }
done
actual_snapshot_sha256="$(sha256sum "${snapshot}" | awk '{print $1}')"
if [[ "${actual_snapshot_sha256}" != "${snapshot_sha256}" ]]; then
    echo "Benchmark snapshot SHA-256 changed: expected=${snapshot_sha256} actual=${actual_snapshot_sha256}" >&2
    exit 2
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
log "Controller started for step 50 evaluation, continuation to step 100, and step 100 evaluation"
wait_for_initial_training
evaluate_step 50
continue_to_step_100
evaluate_step 100
log "FULL_PARAMETER_STEP100_PIPELINE_COMPLETE"
