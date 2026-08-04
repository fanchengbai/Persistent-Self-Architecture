#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DESIGN="${PROJECT_ROOT}/configs/preregistration/exp001b_supplemental_controls.draft.json"
BDEV1_DIR="${PSA_EXP001B_BDEV1_OUTPUT:-${PROJECT_ROOT}/results/development/exp001b_bdev1_non_core_calibration}"
BDEV2_V01_DIR="${PSA_EXP001B_BDEV2_V01_OUTPUT:-${PROJECT_ROOT}/results/development/exp001b_bdev2_non_core_runner}"
BDEV2_V02_DIR="${PSA_EXP001B_BDEV2_OUTPUT:-${PROJECT_ROOT}/results/development/exp001b_bdev2_non_core_runner_v02}"
OUTPUT_DIR="${PSA_EXP001B_CANDIDATE_OUTPUT:-${PROJECT_ROOT}/results/development/exp001b_preregistration_candidate_v1}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m psa exp001b-candidate-build \
  --design "${DESIGN}" \
  --bdev1-dir "${BDEV1_DIR}" \
  --bdev2-v01-dir "${BDEV2_V01_DIR}" \
  --bdev2-v02-dir "${BDEV2_V02_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001B unconfirmed preregistration candidate is ready for checksum review."
echo "No Core Set or supplemental formal set was read or generated."
echo "Summary: ${OUTPUT_DIR}/summary.json"
