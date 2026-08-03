#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/rwkv7_g1h_2.9b.candidate.json"
ASSET_MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json"
OUTPUT_DIR="${PSA_IMPL5B_DEV_OUTPUT:-${PROJECT_ROOT}/results/development/impl5b_confirmatory_runner_dev}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${ASSET_MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa confirmatory-runner-dev-gate \
  --model-config "${MODEL_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "Impl-5b non-Core runner development gate finished."
echo "No frozen Core Set group was accepted or scored."
echo "Summary: ${OUTPUT_DIR}/summary.json"
