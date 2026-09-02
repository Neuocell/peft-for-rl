#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cherrl_root="${CHERRL_ROOT:-/home/wangls/CHERRL}"
tina_root="${TINA_ORTHRES_ROOT:-/home/wangls/Tina_orthres_run}"

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "${repo_root}/${dst}")"
  cp "${src}" "${repo_root}/${dst}"
}

copy_file "${tina_root}/scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh" \
  "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh"

copy_file "${cherrl_root}/verl/workers/engine/fsdp/transformer_impl.py" \
  "verl/workers/engine/fsdp/transformer_impl.py"
copy_file "${cherrl_root}/verl/workers/fsdp_workers.py" \
  "verl/workers/fsdp_workers.py"
copy_file "${cherrl_root}/verl/workers/config/model.py" \
  "verl/workers/config/model.py"
copy_file "${cherrl_root}/verl/workers/config/actor.py" \
  "verl/workers/config/actor.py"
copy_file "${cherrl_root}/verl/workers/actor/dp_actor.py" \
  "verl/workers/actor/dp_actor.py"

copy_file "${cherrl_root}/verl/utils/peft_geora.py" \
  "verl/utils/peft_geora.py"
copy_file "${cherrl_root}/verl/utils/peft_biso.py" \
  "verl/utils/peft_biso.py"
copy_file "${cherrl_root}/verl/utils/peft_boet.py" \
  "verl/utils/peft_boet.py"
copy_file "${cherrl_root}/verl/utils/peft_skew.py" \
  "verl/utils/peft_skew.py"
copy_file "${cherrl_root}/verl/utils/peft_spo.py" \
  "verl/utils/peft_spo.py"
copy_file "${cherrl_root}/verl/utils/peft_oft_compat.py" \
  "verl/utils/peft_oft_compat.py"

copy_file "${cherrl_root}/verl/utils/fsdp_utils.py" \
  "verl/utils/fsdp_utils.py"
copy_file "${cherrl_root}/verl/utils/checkpoint/fsdp_checkpoint_manager.py" \
  "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
copy_file "${cherrl_root}/verl/model_merger/fsdp_model_merger.py" \
  "verl/model_merger/fsdp_model_merger.py"

echo "Synced PEFT-for-RL files from:"
echo "  CHERRL_ROOT=${cherrl_root}"
echo "  TINA_ORTHRES_ROOT=${tina_root}"
echo
git -C "${repo_root}" status --short
