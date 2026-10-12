#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
smoke_unit="peft-qwen3-4b-lora-smoke-v1.service"
smoke_result="${repo_root}/runs/qwen3-4b-standard-lora-smoke-v1/analysis/qwen3_4b_standard_lora_r32a64_b64m16n8_smoke_seed42_v1_result.json"

while true; do
    if [[ -s "${smoke_result}" ]] && "${python_bin}" - "${smoke_result}" <<'PY'
import json, sys
from pathlib import Path
raise SystemExit(0 if json.loads(Path(sys.argv[1]).read_text()).get("status") == "pass" else 1)
PY
    then
        break
    fi
    if ! systemctl --user is-active --quiet "${smoke_unit}"; then
        echo "Qwen3-4B smoke stopped without a pass result; formal baseline will not start" >&2
        exit 1
    fi
    sleep 30
done

cd "${repo_root}"
exec bash scripts/local/run_qwen3_4b_standard_lora_pipeline.sh
