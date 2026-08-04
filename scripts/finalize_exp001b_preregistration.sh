#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CANDIDATE_DIR="${PSA_EXP001B_CANDIDATE_OUTPUT:-${PROJECT_ROOT}/results/development/exp001b_preregistration_candidate_v1}"
OUTPUT_DIR="${PSA_EXP001B_FINAL_PACKAGE:-${PROJECT_ROOT}/preregistration/exp001b/final_v1}"
CONFIRMATION_TEXT="${PSA_EXP001B_CONFIRMATION_TEXT:-}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

if [[ -z "${CONFIRMATION_TEXT}" ]]; then
  echo "error: set PSA_EXP001B_CONFIRMATION_TEXT to the exact confirmed sentence" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m psa exp001b-preregistration-finalize \
  --candidate-dir "${CANDIDATE_DIR}" \
  --confirmation-text "${CONFIRMATION_TEXT}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m psa exp001b-preregistration-final-verify \
  --package-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001B final preregistration package is frozen."
echo "No supplemental set was generated or authorized."
echo "No supplemental experiment was authorized or run."
echo "Manifest: ${OUTPUT_DIR}/manifest.json"
