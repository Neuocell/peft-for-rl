#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
verl_root="${VERL_ROOT:-}"
tina_root="${TINA_ORTHRES_ROOT:-}"

if [[ -z "${verl_root}" ]]; then
  echo "Set VERL_ROOT to a local verl checkout." >&2
  echo "Example: VERL_ROOT=/path/to/verl bash scripts/sync_from_verl.sh" >&2
  exit 2
fi

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${repo_root}/${dst}")"
  cp "${src}" "${repo_root}/${dst}"
}

if [[ -n "${tina_root}" ]]; then
  copy_file "${tina_root}/scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh" \
    "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh"
fi

copy_file "${verl_root}/verl/workers/engine/fsdp/transformer_impl.py" \
  "verl/workers/engine/fsdp/transformer_impl.py"
copy_file "${verl_root}/verl/workers/fsdp_workers.py" \
  "verl/workers/fsdp_workers.py"
copy_file "${verl_root}/verl/workers/config/model.py" \
  "verl/workers/config/model.py"
copy_file "${verl_root}/verl/workers/config/actor.py" \
  "verl/workers/config/actor.py"
copy_file "${verl_root}/verl/workers/actor/dp_actor.py" \
  "verl/workers/actor/dp_actor.py"

copy_file "${verl_root}/verl/utils/peft_geora.py" \
  "verl/utils/peft_geora.py"
copy_file "${verl_root}/verl/utils/peft_biso.py" \
  "verl/utils/peft_biso.py"
copy_file "${verl_root}/verl/utils/peft_boet.py" \
  "verl/utils/peft_boet.py"
copy_file "${verl_root}/verl/utils/peft_skew.py" \
  "verl/utils/peft_skew.py"
copy_file "${verl_root}/verl/utils/peft_spo.py" \
  "verl/utils/peft_spo.py"
copy_file "${verl_root}/verl/utils/peft_oft_compat.py" \
  "verl/utils/peft_oft_compat.py"

copy_file "${verl_root}/verl/utils/fsdp_utils.py" \
  "verl/utils/fsdp_utils.py"
copy_file "${verl_root}/verl/utils/checkpoint/fsdp_checkpoint_manager.py" \
  "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
copy_file "${verl_root}/verl/model_merger/fsdp_model_merger.py" \
  "verl/model_merger/fsdp_model_merger.py"

echo "Synced PEFT-for-RL patches from:"
echo "  VERL_ROOT=${verl_root}"
if [[ -n "${tina_root}" ]]; then
  echo "  TINA_ORTHRES_ROOT=${tina_root}"
fi
echo
git -C "${repo_root}" status --short
