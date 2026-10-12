#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 <s0|o0> <25|50|100>" >&2
    exit 2
fi
method="${1,,}"
step="$2"
case "${method}" in s0|o0) ;; *) echo "Expected s0 or o0" >&2; exit 2 ;; esac
case "${step}" in 25|50|100) ;; *) echo "Expected step 25, 50, or 100" >&2; exit 2 ;; esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/delta-stability-v1}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-delta-stability-v1}"
base_model="${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
snapshot="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
direct_opd_eval_root="${repo_root}/../Direct-OPD/datasets/eval"
if [[ "${method}" == s0 ]]; then
    train_name="delta_stability_s0_standard_r32_seed42_v1"
else
    train_name="delta_stability_o0_random_orthogonal_r32_seed42_v1"
fi
eval_name="${train_name}_step${step}_fullbench_32768_seed42"
adapter_dir="${storage_root}/ckpts/verl/DAPO-Math-17k/${train_name}/global_step_${step}/actor/peft_adapter"
eval_dir="${storage_root}/outputs/full_bench_eval/${eval_name}"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"
postverify="${artifact_root}/analysis/evaluation_verification/${eval_name}.json"
snapshot_sha256="$(sha256sum "${snapshot}" | awk '{print $1}')"

if [[ -s "${postverify}" ]]; then
    jq -e '.status == "verified"' "${postverify}" >/dev/null
    echo "DELTA_FULLBENCH_ALREADY_VERIFIED ${eval_name}"
    exit 0
fi
for required in "${base_model}/model.safetensors" "${snapshot}" "${adapter_dir}/adapter_model.safetensors" "${adapter_dir}/adapter_config.json" "${direct_opd_eval_root}"; do
    [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 2; }
done
[[ ! -e "${eval_dir}" ]] || { echo "Refusing unverified existing evaluation: ${eval_dir}" >&2; exit 2; }

mkdir -p "${artifact_root}/logs/eval" "$(dirname -- "${postverify}")"
exec > >(tee -a "${artifact_root}/logs/eval/${eval_name}.log") 2>&1
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PROJECT_ROOT="${repo_root}/tina_run" PYTHON_BIN="${python_bin}" \
BASE_MODEL="${base_model}" LORA_ADAPTER="${adapter_dir}" \
DIRECT_OPD_EVAL_ROOT="${direct_opd_eval_root}" \
CHECKPOINT_NAME="${eval_name}" OUTPUT_DIR="${eval_dir}" \
BENCHMARK_SNAPSHOT_RECORDS="${snapshot}" \
BENCHMARKS="aime24,aime25,amc23,hmmt_feb,math500,minerva" \
SEED=42 TEMPERATURE=0.6 TOP_P=0.95 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=34816 \
SAMPLES_SMALL=32 SAMPLES_LARGE=4 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
bash tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh

verification="$("${python_bin}" scripts/analysis/verify_full_bench_run.py \
    --summary "${summary}" --records "${records}" --eval-dir "${eval_dir}" --eval-name "${eval_name}" \
    --adapter "${adapter_dir}" --base-model "${base_model}" --snapshot "${snapshot}" \
    --snapshot-sha256 "${snapshot_sha256}")"
jq -n \
    --arg created_at "$(date --iso-8601=seconds)" \
    --arg method "${method^^}" --argjson step "${step}" \
    --arg summary "${summary}" --arg records "${records}" \
    --argjson verification "${verification}" \
    '{schema_version:1,status:"verified",created_at:$created_at,method:$method,trainer_step:$step,summary:$summary,records:$records,verification:$verification}' \
    >"${postverify}.tmp"
mv "${postverify}.tmp" "${postverify}"
echo "DELTA_FULLBENCH_VERIFIED ${eval_name}"
