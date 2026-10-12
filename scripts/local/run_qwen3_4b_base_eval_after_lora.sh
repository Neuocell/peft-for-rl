#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/wangls/peft-for-rl"
python_bin="/home/wangls/miniconda3/envs/peft-for-rl/bin/python"
wait_unit="peft-qwen3-4b-lora-baseline-8gpu-safetensors-v3.service"
artifact_root="${repo_root}/runs/qwen3-4b-standard-lora-8gpu-safetensors-v3"
storage_root="/media/nfs_share/wangls/peft-for-rl-external-validity-v1/qwen3-4b-standard-lora-8gpu-safetensors-v3"
base_model="/home/wangls/models/Qwen3-4B"
snapshot="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"

lora_eval_name="qwen3_4b_standard_lora_r32a64_b64m16n8_8gpu_safetensors_step50_seed42_v3_fullbench_32768_seed42"
lora_records="${storage_root}/outputs/full_bench_eval/${lora_eval_name}/records/${lora_eval_name}.jsonl"
base_eval_name="qwen3_4b_base_fullbench_32768_seed42_v1"
base_eval_dir="${storage_root}/outputs/full_bench_eval/${base_eval_name}"
base_records="${base_eval_dir}/records/${base_eval_name}.jsonl"
comparison_prefix="${artifact_root}/analysis/comparisons/${lora_eval_name}_vs_${base_eval_name}"
log_file="${artifact_root}/logs/${base_eval_name}_pipeline.log"

mkdir -p "${artifact_root}/logs" "${artifact_root}/analysis/comparisons"
exec > >(tee -a "${log_file}") 2>&1

echo "[$(date --iso-8601=seconds)] waiting for ${wait_unit} to release the evaluation GPUs"
while systemctl --user is-active --quiet "${wait_unit}"; do
    sleep 60
done

# Avoid starting eight model replicas while any previous workload still owns GPU memory.
while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '{ if ($1 >= 2048) busy=1 } END { exit busy ? 1 : 0 }'; do
    echo "[$(date --iso-8601=seconds)] GPUs are still occupied; retrying in 60 seconds"
    sleep 60
done

[[ ! -e "${base_eval_dir}" ]] || {
    echo "Refusing existing base-model evaluation output: ${base_eval_dir}" >&2
    exit 2
}

cd "${repo_root}"
echo "[$(date --iso-8601=seconds)] starting matched Qwen3-4B base-model evaluation"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PROJECT_ROOT="${repo_root}/tina_run" PYTHON_BIN="${python_bin}" \
BASE_MODEL="${base_model}" LORA_ADAPTER="" \
DIRECT_OPD_EVAL_ROOT="/home/wangls/Direct-OPD/datasets/eval" \
CHECKPOINT_NAME="${base_eval_name}" OUTPUT_DIR="${base_eval_dir}" \
BENCHMARK_SNAPSHOT_RECORDS="${snapshot}" \
BENCHMARKS="aime24,aime25,amc23,hmmt_feb,math500,minerva" \
SEED=42 TEMPERATURE=0.6 TOP_P=0.95 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=34816 \
SAMPLES_SMALL=32 SAMPLES_LARGE=4 GPU_MEMORY_UTILIZATION=0.85 \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
bash tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh

if [[ -s "${lora_records}" ]]; then
    echo "[$(date --iso-8601=seconds)] generating paired base-versus-LoRA comparison"
    "${python_bin}" scripts/analysis/compare_paired_full_bench.py \
        --baseline-records "${base_records}" \
        --candidate-records "${lora_records}" \
        --output-prefix "${comparison_prefix}" \
        --bootstrap-samples 10000 \
        --seed 42
else
    echo "[$(date --iso-8601=seconds)] LoRA records are unavailable; base evaluation completed without paired comparison"
fi

echo "[$(date --iso-8601=seconds)] matched Qwen3-4B base-model evaluation complete"
