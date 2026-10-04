#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/peft-for-rl"
python_bin="/home/node/anaconda3/envs/peft-for-rl/bin/python"
jq_bin="/home/node/anaconda3/bin/jq"
method="${1:-I0}"
case "${method}" in
    I0|I8) ;;
    *) echo "Usage: $0 [I0|I8]" >&2; exit 2 ;;
esac
method_lower="${method,,}"
train_name="phase1_${method_lower}_uniform_r32_b64m16n8_step50_seed43_v3"
eval_name="${train_name}_fullbench_32768_seed42"
runtime_root="${repo_root}/runs/phase1-signal-random-v1"
eval_dir="${runtime_root}/outputs/full_bench_eval/${eval_name}"
adapter="${runtime_root}/ckpts/verl/DAPO-Math-17k/${train_name}/global_step_50/actor/peft_adapter"
contract="${runtime_root}/analysis/training_manifests/${train_name}.json"
preparation="${runtime_root}/analysis/training_allocations/phase1_training_preparation.json"
snapshot="/data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
snapshot_sha256="3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"
script_path="${runtime_root}/ops/seed43_i0_postverify.sh"
audit="${runtime_root}/analysis/verification/${train_name}_postverify.json"
training_contract_verifier="${repo_root}/scripts/analysis/phase1_training_contract.py"
full_benchmark_verifier="${repo_root}/scripts/analysis/verify_full_bench_run.py"

cd "${repo_root}"
date "+POSTVERIFY_START %Y-%m-%d %H:%M:%S %z"

contract_output="$(
    "${python_bin}" scripts/analysis/phase1_training_contract.py verify \
        --preparation-kind phase1 \
        --preparation "${preparation}" \
        --method "${method}" \
        --training-seed 43 \
        --experiment-name "${train_name}" \
        --contract "${contract}" \
        --adapter "${adapter}/adapter_model.safetensors"
)"
printf '%s\n' "${contract_output}"

verified=0
for attempt in $(seq 1 12); do
    echo "FULL_BENCH_VERIFY_ATTEMPT ${attempt}/12"
    if output="$(
        "${python_bin}" scripts/analysis/verify_full_bench_run.py \
            --summary "${summary}" \
            --records "${records}" \
            --eval-dir "${eval_dir}" \
            --eval-name "${eval_name}" \
            --adapter "${adapter}" \
            --base-model /data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base \
            --snapshot "${snapshot}" \
            --snapshot-sha256 "${snapshot_sha256}" 2>&1
    )"; then
        printf '%s\n' "${output}"
        verified=1
        break
    fi
    printf '%s\n' "${output}"
    if (( attempt < 12 )); then
        sleep 5
    fi
done
if (( verified != 1 )); then
    echo "POSTVERIFY_FAILED after 12 attempts" >&2
    exit 1
fi

echo "RECORD_LINES"
record_lines="$(wc -l < "${records}")"
[[ "${record_lines}" -eq 7248 ]]
wc -l "${records}"

echo "SHARD_LINES"
manifest_paths=()
for shard_index in 0 1 2 3; do
    shard="${eval_dir}/shards/${eval_name}.shard-$(printf '%02d' "${shard_index}")-of-04.jsonl"
    manifest_path="${eval_dir}/manifests/${eval_name}.shard-$(printf '%02d' "${shard_index}")-of-04.json"
    [[ "$(wc -l < "${shard}")" -eq 1812 ]]
    [[ -s "${manifest_path}" ]]
    wc -l "${shard}"
    manifest_paths+=("${manifest_path}")
done

echo "SHA256"
hashes="$(sha256sum \
    "${records}" \
    "${summary}" \
    "${adapter}/adapter_model.safetensors" \
    "${contract}" \
    "${script_path}" \
    "${manifest_paths[@]}" \
    "${training_contract_verifier}" \
    "${full_benchmark_verifier}")"
printf '%s\n' "${hashes}"

records_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${records}" '$2 == path {print $1}')"
summary_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${summary}" '$2 == path {print $1}')"
adapter_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${adapter}/adapter_model.safetensors" '$2 == path {print $1}')"
contract_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${contract}" '$2 == path {print $1}')"
script_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${script_path}" '$2 == path {print $1}')"
manifest0_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${manifest_paths[0]}" '$2 == path {print $1}')"
manifest1_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${manifest_paths[1]}" '$2 == path {print $1}')"
manifest2_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${manifest_paths[2]}" '$2 == path {print $1}')"
manifest3_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${manifest_paths[3]}" '$2 == path {print $1}')"
training_contract_verifier_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${training_contract_verifier}" '$2 == path {print $1}')"
full_benchmark_verifier_sha256="$(printf '%s\n' "${hashes}" | awk -v path="${full_benchmark_verifier}" '$2 == path {print $1}')"

mkdir -p "$(dirname "${audit}")"
audit_tmp="${audit}.tmp.$$"
trap 'rm -f "${audit_tmp}"' EXIT
"${jq_bin}" -n \
    --arg created_at "$(date --iso-8601=seconds)" \
    --arg candidate_method "${method}" \
    --arg train_name "${train_name}" \
    --arg eval_name "${eval_name}" \
    --arg records_path "${records}" \
    --arg summary_path "${summary}" \
    --arg adapter_path "${adapter}/adapter_model.safetensors" \
    --arg contract_path "${contract}" \
    --arg operational_script_path "${script_path}" \
    --arg records_sha256 "${records_sha256}" \
    --arg summary_sha256 "${summary_sha256}" \
    --arg adapter_sha256 "${adapter_sha256}" \
    --arg contract_sha256 "${contract_sha256}" \
    --arg operational_script_sha256 "${script_sha256}" \
    --arg manifest0_path "${manifest_paths[0]}" \
    --arg manifest0_sha256 "${manifest0_sha256}" \
    --arg manifest1_path "${manifest_paths[1]}" \
    --arg manifest1_sha256 "${manifest1_sha256}" \
    --arg manifest2_path "${manifest_paths[2]}" \
    --arg manifest2_sha256 "${manifest2_sha256}" \
    --arg manifest3_path "${manifest_paths[3]}" \
    --arg manifest3_sha256 "${manifest3_sha256}" \
    --arg training_contract_verifier_sha256 "${training_contract_verifier_sha256}" \
    --arg full_benchmark_verifier_sha256 "${full_benchmark_verifier_sha256}" \
    --argjson record_lines "${record_lines}" \
    --argjson contract_verification "${contract_output}" \
    --argjson full_bench_verification "${output}" \
    '{
        schema_version: 1,
        status: "verified",
        method: "phase1_seed43_postverification_v1",
        created_at: $created_at,
        candidate_method: $candidate_method,
        training_seed: 43,
        train_name: $train_name,
        evaluation_name: $eval_name,
        counts: {
            merged_records: $record_lines,
            shard_records: [1812, 1812, 1812, 1812]
        },
        verification: {
            training_contract: $contract_verification,
            full_benchmark: $full_bench_verification
        },
        artifacts: {
            records: {path: $records_path, sha256: $records_sha256},
            summary: {path: $summary_path, sha256: $summary_sha256},
            adapter: {path: $adapter_path, sha256: $adapter_sha256},
            training_contract: {path: $contract_path, sha256: $contract_sha256},
            manifests: [
                {path: $manifest0_path, sha256: $manifest0_sha256},
                {path: $manifest1_path, sha256: $manifest1_sha256},
                {path: $manifest2_path, sha256: $manifest2_sha256},
                {path: $manifest3_path, sha256: $manifest3_sha256}
            ],
            operational_script: {
                path: $operational_script_path,
                sha256: $operational_script_sha256
            }
        },
        verifier_code: {
            training_contract_sha256: $training_contract_verifier_sha256,
            full_benchmark_sha256: $full_benchmark_verifier_sha256
        }
    }' > "${audit_tmp}"
"${jq_bin}" -e '
    .status == "verified" and
    .counts.merged_records == 7248 and
    .verification.training_contract.status == "verified" and
    .verification.full_benchmark.status == "verified" and
    (.artifacts.manifests | length) == 4 and
    all(.artifacts.manifests[]; (.path | type == "string" and length > 0) and (.sha256 | type == "string" and length == 64)) and
    (.verifier_code.training_contract_sha256 | type == "string" and length == 64) and
    (.verifier_code.full_benchmark_sha256 | type == "string" and length == 64)
' "${audit_tmp}" >/dev/null
mv -f "${audit_tmp}" "${audit}"
trap - EXIT
echo "AUDIT_ARTIFACT ${audit}"
date "+POSTVERIFY_DONE %Y-%m-%d %H:%M:%S %z"
