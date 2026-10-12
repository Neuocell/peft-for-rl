#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
run_root="${RUN_ROOT:-${repo_root}/runs/nora-r32-lr1e5-probe-v1}"
base_model="${BASE_MODEL:-${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
train_name="nora_init_r32a64_lr1e5_b64m16n8_10_seed42_v1"
eval_name="nora_init_r32a64_lr1e5_b64m16n8_step50_seed42_v1_fullbench_32768_seed42"
adapter_dir="${run_root}/ckpts/verl/DAPO-Math-17k/${train_name}/global_step_50/actor/peft_adapter"
adapter_file="${adapter_dir}/adapter_model.safetensors"
adapter_config="${adapter_dir}/adapter_config.json"
eval_dir="${run_root}/outputs/full_bench_eval/${eval_name}"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"
postverify="${run_root}/analysis/evaluation_verification/${eval_name}.json"
snapshot="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
snapshot_sha256="3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da"
direct_opd_eval_root="${repo_root}/../Direct-OPD/datasets/eval"
gpu_ids="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
controller_log="${run_root}/logs/${eval_name}.controller.log"

mkdir -p "${run_root}/logs" "${run_root}/analysis/evaluation_verification" "${eval_dir}/logs"
exec 9>"${run_root}/${eval_name}.lock"
if ! flock -n 9; then
    echo "Another ${eval_name} controller already holds the lock" >&2
    exit 3
fi
exec > >(tee -a "${controller_log}") 2>&1

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

for required in \
    "${python_bin}" \
    "${base_model}/model.safetensors" \
    "${adapter_file}" \
    "${adapter_config}" \
    "${snapshot}" \
    "${direct_opd_eval_root}"; do
    [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 2; }
done

actual_snapshot_sha256="$(sha256sum "${snapshot}" | awk '{print $1}')"
if [[ "${actual_snapshot_sha256}" != "${snapshot_sha256}" ]]; then
    echo "Benchmark snapshot SHA-256 changed: expected=${snapshot_sha256} actual=${actual_snapshot_sha256}" >&2
    exit 2
fi
jq -e '
    .peft_type == "LORA" and
    .r == 32 and
    .lora_alpha == 64 and
    .lora_dropout == 0
' "${adapter_config}" >/dev/null || {
    echo "Unexpected step-50 adapter configuration: ${adapter_config}" >&2
    exit 2
}

cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -s "${postverify}" ]]; then
    jq -e '.status == "verified"' "${postverify}" >/dev/null
    log "Evaluation already verified: ${postverify}"
    exit 0
fi

if [[ ! -s "${summary}" ]]; then
    log "Launching fixed six-task full-bench on GPUs ${gpu_ids}"
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PROJECT_ROOT="${repo_root}/tina_run" PYTHON_BIN="${python_bin}" \
    BASE_MODEL="${base_model}" LORA_ADAPTER="${adapter_dir}" \
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

log "Verifying 7248 records, manifests, snapshot, adapter, and protocol"
verification="$("${python_bin}" scripts/analysis/verify_full_bench_run.py \
    --summary "${summary}" \
    --records "${records}" \
    --eval-dir "${eval_dir}" \
    --eval-name "${eval_name}" \
    --adapter "${adapter_dir}" \
    --base-model "${base_model}" \
    --snapshot "${snapshot}" \
    --snapshot-sha256 "${snapshot_sha256}")"
adapter_sha256="$(sha256sum "${adapter_file}" | awk '{print $1}')"
jq -n \
    --arg created_at "$(date --iso-8601=seconds)" \
    --arg summary "${summary}" \
    --arg records "${records}" \
    --arg adapter "${adapter_file}" \
    --arg adapter_sha256 "${adapter_sha256}" \
    --argjson verification "${verification}" \
    '{
        schema_version: 1,
        status: "verified",
        created_at: $created_at,
        summary: $summary,
        records: $records,
        adapter: {path: $adapter, sha256: $adapter_sha256},
        verification: $verification
    }' >"${postverify}.tmp"
mv "${postverify}.tmp" "${postverify}"
log "NORA_R32_LR1E5_FULLBENCH_VERIFIED ${summary}"
