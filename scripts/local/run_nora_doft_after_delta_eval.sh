#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
eval_unit="${EVAL_UNIT:-peft-delta-stability-eval-v1.service}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/delta-stability-v1}"
eval_log="${artifact_root}/logs/delta_stability_eval_lane.log"
run_root="${NORA_DOFT_RUN_ROOT:-${repo_root}/runs/nora-doft-v1}"
train_log="${run_root}/logs/nora_doft_after_delta_eval.log"
launcher="${repo_root}/examples/verl_train/run_dapo_math_boxed_nora_doft_1p5b_4gpu_8k.sh"
exp_name="${EXP_NAME:-nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed}"
checkpoint_dir="${CKPTS_DIR:-${repo_root}/runs/ckpts/verl/DAPO-Math-17k/${exp_name}}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

mkdir -p "$(dirname -- "${train_log}")"
exec > >(tee -a "${train_log}") 2>&1

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

log "waiting for ${eval_unit} to finish the six-task evaluation lane"
while true; do
    eval_state="$(systemctl --user show "${eval_unit}" --property=ActiveState --value 2>/dev/null || true)"
    case "${eval_state}" in
        active|activating|reloading|deactivating)
            sleep 60
            ;;
        inactive|failed)
            break
            ;;
        *)
            echo "Cannot determine a valid state for ${eval_unit}: ${eval_state:-missing}" >&2
            exit 1
            ;;
    esac
done

eval_result="$(systemctl --user show "${eval_unit}" --property=Result --value)"
if [[ "${eval_result}" != "success" ]]; then
    echo "${eval_unit} stopped with Result=${eval_result}; refusing to start training" >&2
    exit 1
fi
if [[ ! -s "${eval_log}" ]] || ! grep -Fq "DELTA_STABILITY_POSTTRAIN_COMPLETE" "${eval_log}"; then
    echo "Evaluation completion marker is missing from ${eval_log}; refusing to start training" >&2
    exit 1
fi

verification_files=()
for method in s0 o0; do
    if [[ "${method}" == "s0" ]]; then
        train_name="delta_stability_s0_standard_r32_seed42_v1"
    else
        train_name="delta_stability_o0_random_orthogonal_r32_seed42_v1"
    fi
    for step in 25 50 100; do
        eval_name="${train_name}_step${step}_fullbench_32768_seed42"
        verification_files+=("${artifact_root}/analysis/evaluation_verification/${eval_name}.json")
    done
done

"${python_bin}" - "${verification_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

invalid = []
for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        invalid.append(f"{path}: {exc}")
        continue
    if payload.get("status") != "verified" or payload.get("verification", {}).get("status") != "verified":
        invalid.append(f"{path}: status is not verified")

if invalid:
    print("Evaluation verification failed:", file=sys.stderr)
    print("\n".join(invalid), file=sys.stderr)
    raise SystemExit(1)
PY

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
while true; do
    busy=()
    for gpu in "${gpu_array[@]}"; do
        if ! pids="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
            echo "Failed to inspect GPU ${gpu}; refusing to start training" >&2
            exit 1
        fi
        if [[ -n "${pids//[[:space:]]/}" ]]; then
            busy+=("GPU ${gpu}: ${pids//$'\n'/,}")
        fi
    done
    if (( ${#busy[@]} == 0 )); then
        break
    fi
    log "GPUs ${gpu_ids} still have compute processes (${busy[*]}); retrying in 60 seconds"
    sleep 60
done

if compgen -G "${checkpoint_dir}/global_step_*" >/dev/null; then
    echo "Checkpoint steps already exist under ${checkpoint_dir}; refusing a non-resuming duplicate run" >&2
    exit 1
fi

cd "${repo_root}"
log "evaluation verified and GPUs ${gpu_ids} are free; running training preflight"
DRY_RUN=1 bash "${launcher}"
log "starting NoRA+DoRA+OFT training: EXP_NAME=${exp_name}, CUDA_VISIBLE_DEVICES=${gpu_ids}"
exec bash "${launcher}"
