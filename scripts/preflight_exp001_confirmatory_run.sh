#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
OUTPUT_DIR="${PSA_PREFLIGHT_OUTPUT_DIR:-${PROJECT_ROOT}/results/development/impl5b_confirmatory_preflight}"
OUTPUT_PATH="${OUTPUT_DIR}/preflight.json"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0

"${PYTHON_BIN}" -m psa confirmatory-preflight \
  --final-package "${PROJECT_ROOT}/preregistration/exp001/final_v1" \
  --core-set-package "${PROJECT_ROOT}/preregistration/exp001/core_set_v1" \
  --model-config "${PROJECT_ROOT}/configs/models/rwkv7_g1h_2.9b.candidate.json" \
  --asset-manifest "${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json" \
  --asset-root "${ASSET_ROOT}" \
  --output "${OUTPUT_PATH}" \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001 non-inference confirmatory preflight finished."
echo "No model was loaded and no Core Set trial was scored."
echo "Preflight: ${OUTPUT_PATH}"
