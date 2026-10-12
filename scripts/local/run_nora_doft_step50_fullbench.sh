#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
run_root="${RUN_ROOT:-${repo_root}/runs/nora-doft-v1}"
base_model="${BASE_MODEL:-${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
train_name="nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed"
eval_name="${train_name}_step50_fullbench_32768_seed42"
actor_dir="${repo_root}/runs/ckpts/verl/DAPO-Math-17k/${train_name}/global_step_50/actor"
merge_root="${run_root}/merged/step50"
wrapped_model="${merge_root}/actor_hf_wrapped"
merged_model="${merge_root}/actor_hf_vllm"
eval_dir="${run_root}/outputs/full_bench_eval/${eval_name}"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"
postverify="${run_root}/analysis/evaluation_verification/${eval_name}.json"
snapshot="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
snapshot_sha256="3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da"
direct_opd_eval_root="${repo_root}/../Direct-OPD/datasets/eval"
gpu_ids="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
controller_log="${run_root}/logs/${eval_name}.controller.log"

mkdir -p "${run_root}/logs" "${run_root}/analysis/evaluation_verification" "${merge_root}" "${eval_dir}/logs"
exec 9>"${run_root}/${eval_name}.lock"
if ! flock -n 9; then
    echo "Another ${eval_name} controller already holds the lock" >&2
    exit 3
fi
exec > >(tee -a "${controller_log}") 2>&1

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

hf_model_complete() {
    local model_dir="$1"
    [[ -s "${model_dir}/config.json" ]] || return 1
    [[ -s "${model_dir}/tokenizer_config.json" ]] || return 1
    [[ -s "${model_dir}/model.safetensors" || -s "${model_dir}/model.safetensors.index.json" ]]
}

for required in \
    "${python_bin}" \
    "${base_model}/model.safetensors" \
    "${actor_dir}/fsdp_config.json" \
    "${snapshot}" \
    "${direct_opd_eval_root}"; do
    [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 2; }
done
actual_snapshot_sha256="$(sha256sum "${snapshot}" | awk '{print $1}')"
if [[ "${actual_snapshot_sha256}" != "${snapshot_sha256}" ]]; then
    echo "Benchmark snapshot SHA-256 changed: expected=${snapshot_sha256} actual=${actual_snapshot_sha256}" >&2
    exit 2
fi
for rank in 0 1 2 3; do
    [[ -s "${actor_dir}/model_world_size_4_rank_${rank}.pt" ]] || {
        echo "Missing FSDP model shard for rank ${rank}" >&2
        exit 2
    }
done

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -s "${postverify}" ]]; then
    jq -e '.status == "verified"' "${postverify}" >/dev/null
    log "Evaluation already verified: ${postverify}"
    exit 0
fi

if ! hf_model_complete "${wrapped_model}"; then
    log "Merging four FSDP shards into wrapped HF model"
    CUDA_VISIBLE_DEVICES="${gpu_ids%%,*}" "${python_bin}" -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${wrapped_model}" \
        --use_cpu_initialization \
        >>"${eval_dir}/logs/${eval_name}.merge.log" 2>&1
fi

if ! hf_model_complete "${merged_model}"; then
    log "Materializing NoRA-DOFT rotations and magnitudes into plain HF weights"
    CUDA_VISIBLE_DEVICES="${gpu_ids%%,*}" "${python_bin}" \
        tina_run/scripts/local/eval/merge_nora_doft_wrapped_hf.py \
        --base_model "${base_model}" \
        --wrapped_model "${wrapped_model}" \
        --output_dir "${merged_model}" \
        --rank 8 \
        --block_size 32 \
        --alpha 8 \
        --normalization init \
        --use_magnitude \
        --eps 1e-6 \
        --seed 42 \
        --dtype bfloat16 \
        >>"${eval_dir}/logs/${eval_name}.merge.log" 2>&1
fi

if [[ ! -s "${summary}" ]]; then
    log "Launching fixed six-task full-bench on GPUs ${gpu_ids}"
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
    bash tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh
fi

log "Verifying 7248 records, manifests, snapshot, and protocol"
verification="$("${python_bin}" scripts/analysis/verify_full_bench_run.py \
    --summary "${summary}" \
    --records "${records}" \
    --eval-dir "${eval_dir}" \
    --eval-name "${eval_name}" \
    --no-adapter \
    --base-model "${merged_model}" \
    --snapshot "${snapshot}" \
    --snapshot-sha256 "${snapshot_sha256}")"
jq -n \
    --arg created_at "$(date --iso-8601=seconds)" \
    --arg summary "${summary}" \
    --arg records "${records}" \
    --arg merged_model "${merged_model}" \
    --argjson verification "${verification}" \
    '{schema_version:1,status:"verified",created_at:$created_at,summary:$summary,records:$records,merged_model:$merged_model,verification:$verification}' \
    >"${postverify}.tmp"
mv "${postverify}.tmp" "${postverify}"
log "NORA_DOFT_FULLBENCH_VERIFIED ${summary}"
