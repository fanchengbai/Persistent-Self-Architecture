#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/rwkv7_world_0.4b.impl1.json"
GATE_CONFIG="${PROJECT_ROOT}/configs/gates/impl2c_random_matched.dev.json"
ASSET_MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_world_0.4b.json"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
OUTPUT_DIR="${PSA_IMPL2C_OUTPUT:-${PROJECT_ROOT}/results/development/impl2c_random_matched}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0

"${PYTHON_BIN}" -m psa environment-report \
  --output "${PROJECT_ROOT}/results/development/environment_manifest.json" \
  --project-root "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${ASSET_MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa random-state-gate \
  --config "${MODEL_CONFIG}" \
  --gate-config "${GATE_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "Impl-2c matched random state gate finished."
echo "Summary: ${OUTPUT_DIR}/summary.json"
