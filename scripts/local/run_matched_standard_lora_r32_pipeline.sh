#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ "$1" != "42" && "$1" != "43" ]]; then
    echo "Usage: $0 <42|43>" >&2
    exit 2
fi
train_seed="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
run_revision="${MATCHED_LORA_RUN_REVISION:-2}"
[[ "${run_revision}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MATCHED_LORA_RUN_REVISION: ${run_revision}" >&2; exit 2; }
runtime_root="${repo_root}/runs/matched-standard-lora-r32-v${run_revision}"
train_name="matched_standard_lora_r32a64_b64m16n8_step50_seed${train_seed}_v${run_revision}"
eval_name="${train_name}_fullbench_32768_seed42"
base_model="${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
train_file="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet"
snapshot="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
ckpt_dir="${runtime_root}/ckpts/verl/DAPO-Math-17k/${train_name}"
adapter_dir="${ckpt_dir}/global_step_50/actor/peft_adapter"
adapter_file="${adapter_dir}/adapter_model.safetensors"
audit="${runtime_root}/analysis/initialization/${train_name}.json"
contract="${runtime_root}/analysis/training_manifests/${train_name}.json"
eval_dir="${runtime_root}/outputs/full_bench_eval/${eval_name}"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"
postverify="${runtime_root}/analysis/verification/${train_name}_postverify.json"
pipeline_log="${runtime_root}/logs/${train_name}_pipeline.log"
direct_opd_eval_root="/home/wangls/Direct-OPD/datasets/eval"
if [[ "${train_seed}" == "42" ]]; then
    gpu_group="0,1,2,3"
else
    gpu_group="4,5,6,7"
fi

for path in "${contract}" "${ckpt_dir}" "${eval_dir}" "${postverify}" "${audit}"; do
    [[ ! -e "${path}" ]] || { echo "Refusing existing output: ${path}" >&2; exit 2; }
done
mkdir -p "${runtime_root}/logs" "${runtime_root}/analysis/training_manifests"
exec > >(tee -a "${pipeline_log}") 2>&1
cd "${repo_root}"
export PYTHONPATH="${repo_root}"

echo "[$(date --iso-8601=seconds)] creating matched standard-LoRA seed-${train_seed} contract"
"${python_bin}" scripts/analysis/standard_lora_training_contract.py create \
    --experiment-name "${train_name}" --training-seed "${train_seed}" \
    --model "${base_model}" --train-file "${train_file}" --snapshot "${snapshot}" \
    --audit "${audit}" --contract "${contract}" >/dev/null

echo "[$(date --iso-8601=seconds)] starting matched standard-LoRA seed-${train_seed} training on GPUs ${gpu_group}"
CUDA_VISIBLE_DEVICES="${gpu_group}" \
RUNTIME_ROOT="${runtime_root}" \
EXP_NAME="${train_name}" \
MODEL_PATH="${base_model}" \
TRAIN_FILE="${train_file}" \
LORA_INIT_AUDIT_PATH="${audit}" \
MATCHED_LORA_RUN_REVISION="${run_revision}" \
RAY_TEMP_DIR="/dev/shm/ray-matched-lora-s${train_seed}-v${run_revision}" \
TMPDIR="/dev/shm/tmp-matched-lora-s${train_seed}-v${run_revision}" \
bash scripts/local/start_matched_standard_lora_r32_4gpu.sh "${train_seed}"
[[ -s "${adapter_file}" ]] || { echo "Training ended without adapter: ${adapter_file}" >&2; exit 1; }

"${python_bin}" scripts/analysis/standard_lora_training_contract.py finalize \
    --contract "${contract}" --adapter "${adapter_file}" >/dev/null
"${python_bin}" scripts/analysis/standard_lora_training_contract.py verify \
    --contract "${contract}" --adapter "${adapter_file}" >/dev/null

echo "[$(date --iso-8601=seconds)] starting fixed six-benchmark evaluation for seed ${train_seed}"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PROJECT_ROOT="${repo_root}/tina_run" PYTHON_BIN="${python_bin}" \
BASE_MODEL="${base_model}" LORA_ADAPTER="${adapter_dir}" \
DIRECT_OPD_EVAL_ROOT="${direct_opd_eval_root}" \
CHECKPOINT_NAME="${eval_name}" OUTPUT_DIR="${eval_dir}" \
BENCHMARK_SNAPSHOT_RECORDS="${snapshot}" \
BENCHMARKS="aime24,aime25,amc23,hmmt_feb,math500,minerva" \
SEED=42 TEMPERATURE=0.6 TOP_P=0.95 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=34816 \
SAMPLES_SMALL=32 SAMPLES_LARGE=4 CUDA_VISIBLE_DEVICES="${gpu_group}" \
bash tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh

"${python_bin}" scripts/analysis/standard_lora_training_contract.py postverify \
    --contract "${contract}" --adapter "${adapter_file}" \
    --summary "${summary}" --records "${records}" --eval-dir "${eval_dir}" \
    --eval-name "${eval_name}" --output "${postverify}" >/dev/null
echo "[$(date --iso-8601=seconds)] matched standard-LoRA seed-${train_seed} pipeline complete"
