#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
FINAL_PACKAGE="${PROJECT_ROOT}/preregistration/exp001/final_v1"
AUTHORIZATION="${PROJECT_ROOT}/preregistration/exp001/core_set_authorization.json"
FORMAL_CONFIG="${PROJECT_ROOT}/configs/preregistration/exp001_track_s.formal_v3_holdout.json"
OUTPUT_DIR="${PSA_CORE_SET_OUTPUT:-${PROJECT_ROOT}/preregistration/exp001/core_set_v1}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${ASSET_MANIFEST}" \
  --root "${ASSET_ROOT}" \
  --only rwkv-world-tokenizer-20230424

"${PYTHON_BIN}" -m psa preregistration-final-verify \
  --package-dir "${FINAL_PACKAGE}"

"${PYTHON_BIN}" -m psa core-set-generate \
  --final-package "${FINAL_PACKAGE}" \
  --authorization "${AUTHORIZATION}" \
  --config "${FORMAL_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m psa core-set-verify \
  --package-dir "${OUTPUT_DIR}"

echo "EXP-001 Core Set is generated and frozen."
echo "No model weights were loaded."
echo "No confirmatory experiment was run."
echo "Manifest: ${OUTPUT_DIR}/manifest.json"
