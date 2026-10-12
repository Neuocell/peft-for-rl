#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
artifact_root="${repo_root}/runs/qwen3-4b-standard-lora-smoke-v1"
storage_root="/media/nfs_share/wangls/peft-for-rl-external-validity-v1/qwen3-4b-standard-lora-smoke-v1"
exp_name="qwen3_4b_standard_lora_r32a64_b64m16n8_smoke_seed42_v1"
ckpt_dir="${storage_root}/ckpts/verl/DAPO-Math-17k/${exp_name}"
audit="${artifact_root}/analysis/initialization/${exp_name}.json"
log_file="${artifact_root}/logs/verl/${exp_name}.log"
result="${artifact_root}/analysis/${exp_name}_result.json"

if [[ -d "${ckpt_dir}" && -n "$(find "${ckpt_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing non-empty smoke checkpoint directory: ${ckpt_dir}" >&2
    exit 2
fi
for path in "${audit}" "${result}"; do
    [[ ! -e "${path}" ]] || { echo "Refusing existing smoke output: ${path}" >&2; exit 2; }
done
mkdir -p "$(dirname -- "${result}")"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"

"${python_bin}" - <<'PY'
from verl.utils.reward_score.boxed_math_accuracy import compute_score
valid = compute_score("math_dapo", "Therefore, the final answer is: $\\boxed{2}$.", "2")
invalid = compute_score("math_dapo", "Therefore, the final answer is: $\\boxed{3}$.", "2")
assert valid["score"] == 1.0 and valid["parse_success"] and valid["has_boxed"]
assert invalid["score"] == 0.0 and invalid["parse_success"] and invalid["has_boxed"]
PY

common_env=(
    "ARTIFACT_ROOT=${artifact_root}"
    "STORAGE_ROOT=${storage_root}"
    "EXP_NAME=${exp_name}"
    "CKPTS_DIR=${ckpt_dir}"
    "LORA_INIT_AUDIT_PATH=${audit}"
    "LOG_FILE=${log_file}"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    "MAX_ACTOR_CKPT_TO_KEEP=2"
    "RAY_TEMP_DIR=/dev/shm/ray-q3-lora-smoke-v1"
    "TMPDIR=/dev/shm/tmp-q3-lora-smoke-v1"
)

echo "[$(date --iso-8601=seconds)] Qwen3-4B smoke phase 1: train and save step 1"
env "${common_env[@]}" TOTAL_TRAINING_STEPS=1 SAVE_FREQ=1 RESUME_MODE=disable \
    bash scripts/local/start_qwen3_4b_standard_lora_4gpu.sh
[[ -s "${ckpt_dir}/global_step_1/actor/peft_adapter/adapter_model.safetensors" ]]
[[ -s "${ckpt_dir}/global_step_1/data.pt" ]]
compgen -G "${ckpt_dir}/global_step_1/actor/model_world_size_4_rank_*.pt" >/dev/null
compgen -G "${ckpt_dir}/global_step_1/actor/optim_world_size_4_rank_*.pt" >/dev/null

echo "[$(date --iso-8601=seconds)] Qwen3-4B smoke phase 2: resume step 1 and save step 2"
env "${common_env[@]}" TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1 RESUME_MODE=auto \
    bash scripts/local/start_qwen3_4b_standard_lora_4gpu.sh
[[ -s "${ckpt_dir}/global_step_2/actor/peft_adapter/adapter_model.safetensors" ]]
[[ "$(tr -d '[:space:]' < "${ckpt_dir}/latest_checkpointed_iteration.txt")" == "2" ]]
/usr/bin/grep -Eq "Resuming from .*global_step_1" "${log_file}"

"${python_bin}" - "${result}" "${audit}" "${ckpt_dir}" "${log_file}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

result, audit, ckpt, log = map(lambda value: Path(value).resolve(), sys.argv[1:])
adapter = ckpt / "global_step_2/actor/peft_adapter/adapter_model.safetensors"
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
value = {
    "schema_version": 1,
    "status": "pass",
    "method": "qwen3_4b_standard_lora_two_step_fsdp_checkpoint_resume_smoke_v1",
    "training_steps": 2,
    "checkpoint_resume": "pass",
    "boxed_reward_control": "pass",
    "checkpoint_root": str(ckpt),
    "step2_adapter_sha256": sha(adapter),
    "initialization_audit_path": str(audit),
    "initialization_audit_sha256": sha(audit),
    "log_path": str(log),
}
result.parent.mkdir(parents=True, exist_ok=True)
temporary = result.with_suffix(result.suffix + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, result)
print(json.dumps(value, indent=2, sort_keys=True))
PY
