#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/peft-for-rl"
runtime_root="${repo_root}/runs/phase1-signal-random-v1"
jq_bin="/home/node/anaconda3/bin/jq"
train_name="phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3"
eval_name="${train_name}_fullbench_32768_seed42"
step25_root="${runtime_root}/ckpts/verl/DAPO-Math-17k/${train_name}/global_step_25"
postverify="${runtime_root}/analysis/verification/${train_name}_postverify.json"
expected_records="${runtime_root}/outputs/full_bench_eval/${eval_name}/records/${eval_name}.jsonl"
expected_summary="${runtime_root}/outputs/full_bench_eval/${eval_name}/summary/${eval_name}.json"
expected_adapter="${runtime_root}/ckpts/verl/DAPO-Math-17k/${train_name}/global_step_50/actor/peft_adapter/adapter_model.safetensors"
expected_contract="${runtime_root}/analysis/training_manifests/${train_name}.json"
retention_dir="${runtime_root}/analysis/checkpoint_retention"
inventory="${retention_dir}/${train_name}_step25_files.jsonl"
manifest="${retention_dir}/${train_name}_step25_inventory.json"

expected_relative_paths=(
    "actor/extra_state_world_size_4_rank_0.pt"
    "actor/extra_state_world_size_4_rank_1.pt"
    "actor/extra_state_world_size_4_rank_2.pt"
    "actor/extra_state_world_size_4_rank_3.pt"
    "actor/fsdp_config.json"
    "actor/huggingface/chat_template.jinja"
    "actor/huggingface/config.json"
    "actor/huggingface/generation_config.json"
    "actor/huggingface/special_tokens_map.json"
    "actor/huggingface/tokenizer.json"
    "actor/huggingface/tokenizer_config.json"
    "actor/model_world_size_4_rank_0.pt"
    "actor/model_world_size_4_rank_1.pt"
    "actor/model_world_size_4_rank_2.pt"
    "actor/model_world_size_4_rank_3.pt"
    "actor/optim_world_size_4_rank_0.pt"
    "actor/optim_world_size_4_rank_1.pt"
    "actor/optim_world_size_4_rank_2.pt"
    "actor/optim_world_size_4_rank_3.pt"
    "actor/peft_adapter/adapter_config.json"
    "actor/peft_adapter/adapter_model.safetensors"
    "data.pt"
)

fail() {
    echo "STEP25_INVENTORY_REFUSED $*" >&2
    exit 2
}

hash_matches() {
    local path="$1"
    local expected="$2"
    local actual
    [[ -f "${path}" && -s "${path}" ]] || fail "missing bound artifact: ${path}"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || {
        fail "bound artifact hash changed: ${path} expected=${expected} actual=${actual}"
    }
}

[[ $# -eq 0 ]] || fail "this inventory-only tool accepts no arguments"
[[ -x "${jq_bin}" ]] || fail "missing jq: ${jq_bin}"
[[ -s "${postverify}" ]] || fail "postverify artifact is not ready: ${postverify}"
[[ -d "${step25_root}" ]] || fail "step-25 recovery directory is missing: ${step25_root}"
[[ ! -e "${inventory}" ]] || fail "refusing to overwrite inventory: ${inventory}"
[[ ! -e "${manifest}" ]] || fail "refusing to overwrite manifest: ${manifest}"

"${jq_bin}" -e \
    --arg train_name "${train_name}" \
    --arg eval_name "${eval_name}" \
    --arg expected_records "${expected_records}" \
    --arg expected_summary "${expected_summary}" \
    --arg expected_adapter "${expected_adapter}" \
    --arg expected_contract "${expected_contract}" \
    '.schema_version == 1 and
     .status == "verified" and
     .method == "phase1_seed43_postverification_v1" and
     .candidate_method == "I0" and
     .training_seed == 43 and
     .train_name == $train_name and
     .evaluation_name == $eval_name and
     .counts.merged_records == 7248 and
     .counts.shard_records == [1812, 1812, 1812, 1812] and
     .verification.training_contract.status == "verified" and
     .verification.training_contract.training_seed == 43 and
     .verification.training_contract.candidate_method == "I0" and
     .verification.full_benchmark.status == "verified" and
     .verification.full_benchmark.num_samples == 7248 and
     .verification.full_benchmark.num_shards == 4 and
     .artifacts.records.path == $expected_records and
     .artifacts.summary.path == $expected_summary and
     .artifacts.adapter.path == $expected_adapter and
     .artifacts.training_contract.path == $expected_contract and
     (.artifacts.records.sha256 | type == "string" and length == 64) and
     (.artifacts.summary.sha256 | type == "string" and length == 64) and
     (.artifacts.adapter.sha256 | type == "string" and length == 64) and
     (.artifacts.training_contract.sha256 | type == "string" and length == 64)' \
    "${postverify}" >/dev/null || fail "postverify schema or verification status changed"

while IFS=$'\t' read -r path expected_sha256; do
    hash_matches "${path}" "${expected_sha256}"
done < <(
    "${jq_bin}" -r \
        '.artifacts | [.records, .summary, .adapter, .training_contract][] | [.path, .sha256] | @tsv' \
        "${postverify}"
)

mapfile -t actual_relative_paths < <(
    find "${step25_root}" -type f -printf '%P\n' | LC_ALL=C sort
)
if (( ${#actual_relative_paths[@]} != ${#expected_relative_paths[@]} )); then
    fail "step-25 file count changed: actual=${#actual_relative_paths[@]} expected=${#expected_relative_paths[@]}"
fi
for index in "${!expected_relative_paths[@]}"; do
    [[ "${actual_relative_paths[$index]}" == "${expected_relative_paths[$index]}" ]] || {
        fail "step-25 file set changed at index ${index}: actual=${actual_relative_paths[$index]} expected=${expected_relative_paths[$index]}"
    }
done

for relative_path in "${expected_relative_paths[@]}"; do
    path="${step25_root}/${relative_path}"
    [[ -f "${path}" && ! -L "${path}" ]] || fail "not a regular non-symlink file: ${path}"
done

mkdir -p "${retention_dir}"
inventory_tmp="$(mktemp "${retention_dir}/.${train_name}.step25-files.XXXXXX")"
manifest_tmp="$(mktemp "${retention_dir}/.${train_name}.step25-inventory.XXXXXX")"
trap 'rm -f "${inventory_tmp}" "${manifest_tmp}"' EXIT

total_file_bytes=0
for relative_path in "${expected_relative_paths[@]}"; do
    path="${step25_root}/${relative_path}"
    bytes="$(stat -c '%s' "${path}")"
    sha256="$(sha256sum "${path}" | awk '{print $1}')"
    total_file_bytes=$((total_file_bytes + bytes))
    "${jq_bin}" -cn \
        --arg path "${path}" \
        --arg relative_path "${relative_path}" \
        --arg sha256 "${sha256}" \
        --argjson bytes "${bytes}" \
        '{path: $path, relative_path: $relative_path, bytes: $bytes, sha256: $sha256}' \
        >>"${inventory_tmp}"
done

[[ "$(wc -l < "${inventory_tmp}")" -eq 22 ]] || fail "generated inventory does not contain 22 rows"
inventory_sha256="$(sha256sum "${inventory_tmp}" | awk '{print $1}')"
disk_bytes="$(du -s --block-size=1 "${step25_root}" | awk '{print $1}')"
postverify_sha256="$(sha256sum "${postverify}" | awk '{print $1}')"

records_path="$("${jq_bin}" -r '.artifacts.records.path' "${postverify}")"
records_sha256="$("${jq_bin}" -r '.artifacts.records.sha256' "${postverify}")"
summary_path="$("${jq_bin}" -r '.artifacts.summary.path' "${postverify}")"
summary_sha256="$("${jq_bin}" -r '.artifacts.summary.sha256' "${postverify}")"
adapter_path="$("${jq_bin}" -r '.artifacts.adapter.path' "${postverify}")"
adapter_sha256="$("${jq_bin}" -r '.artifacts.adapter.sha256' "${postverify}")"
contract_path="$("${jq_bin}" -r '.artifacts.training_contract.path' "${postverify}")"
contract_sha256="$("${jq_bin}" -r '.artifacts.training_contract.sha256' "${postverify}")"

"${jq_bin}" -n \
    --arg created_at "$(date --iso-8601=seconds)" \
    --arg source_root "${step25_root}" \
    --arg inventory_path "${inventory}" \
    --arg inventory_sha256 "${inventory_sha256}" \
    --arg postverify_path "${postverify}" \
    --arg postverify_sha256 "${postverify_sha256}" \
    --arg records_path "${records_path}" \
    --arg records_sha256 "${records_sha256}" \
    --arg summary_path "${summary_path}" \
    --arg summary_sha256 "${summary_sha256}" \
    --arg adapter_path "${adapter_path}" \
    --arg adapter_sha256 "${adapter_sha256}" \
    --arg contract_path "${contract_path}" \
    --arg contract_sha256 "${contract_sha256}" \
    --argjson total_file_bytes "${total_file_bytes}" \
    --argjson disk_bytes "${disk_bytes}" \
    '{
        schema_version: 1,
        status: "inventory_ready_no_deletion_performed",
        method: "phase1_seed43_step25_retention_inventory_v1",
        created_at: $created_at,
        source_root: $source_root,
        file_count: 22,
        total_file_bytes: $total_file_bytes,
        disk_bytes: $disk_bytes,
        deletion_performed: false,
        irreversible_action_authorized: false,
        file_inventory: {
            path: $inventory_path,
            sha256: $inventory_sha256
        },
        prerequisite_postverify: {
            path: $postverify_path,
            sha256: $postverify_sha256,
            status: "verified"
        },
        preserved: {
            evaluation_records: {path: $records_path, sha256: $records_sha256},
            evaluation_summary: {path: $summary_path, sha256: $summary_sha256},
            step50_adapter: {path: $adapter_path, sha256: $adapter_sha256},
            training_contract: {path: $contract_path, sha256: $contract_sha256}
        },
        next_action: "Review and re-hash this inventory before any explicit deletion of the exact 22 listed files."
    }' >"${manifest_tmp}"

"${jq_bin}" -e \
    --arg inventory_sha256 "${inventory_sha256}" \
    '.status == "inventory_ready_no_deletion_performed" and
     .file_count == 22 and
     .deletion_performed == false and
     .irreversible_action_authorized == false and
     .file_inventory.sha256 == $inventory_sha256 and
     .prerequisite_postverify.status == "verified"' \
    "${manifest_tmp}" >/dev/null

mv "${inventory_tmp}" "${inventory}"
mv "${manifest_tmp}" "${manifest}"
trap - EXIT
echo "STEP25_INVENTORY_READY ${inventory} sha256=${inventory_sha256}"
echo "STEP25_INVENTORY_MANIFEST ${manifest}"
echo "NO_DELETION_PERFORMED"
