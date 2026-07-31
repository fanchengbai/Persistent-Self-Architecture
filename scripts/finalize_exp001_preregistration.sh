#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CANDIDATE_DIR="${PROJECT_ROOT}/preregistration/exp001/impl3t_candidate"
CONFIRMATION="${PROJECT_ROOT}/preregistration/exp001/human_confirmation.json"
OUTPUT_DIR="${PROJECT_ROOT}/preregistration/exp001/final_v1"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m psa preregistration-finalize \
  --candidate "${CANDIDATE_DIR}/candidate.json" \
  --verification "${CANDIDATE_DIR}/verification.json" \
  --confirmation "${CONFIRMATION}" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m psa preregistration-final-verify \
  --package-dir "${OUTPUT_DIR}"

echo "EXP-001 final preregistration package is frozen."
echo "No Core Set was generated or unsealed."
echo "No confirmatory experiment was run."
echo "Manifest: ${OUTPUT_DIR}/manifest.json"
