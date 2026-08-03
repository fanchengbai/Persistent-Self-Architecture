#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
AUTHORIZATION="${PSA_CONFIRMATORY_AUTHORIZATION:-${PROJECT_ROOT}/results/authorizations/exp001_run_authorization.json}"
RUN_DIR="${PSA_CONFIRMATORY_OUTPUT:-${PROJECT_ROOT}/results/confirmatory/exp001_v1}"
REPORT="${PSA_CONFIRMATORY_VERIFICATION:-${PROJECT_ROOT}/results/confirmatory/exp001_v1.raw_verification.json}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m psa confirmatory-raw-verify \
  --output-dir "${RUN_DIR}" \
  --core-set-package "${PROJECT_ROOT}/preregistration/exp001/core_set_v1" \
  --preflight "${PROJECT_ROOT}/results/development/impl5b_confirmatory_preflight/preflight.json" \
  --authorization "${AUTHORIZATION}" \
  --output "${REPORT}"

echo "EXP-001 raw package integrity verification finished."
echo "No accuracy, confidence interval, or research decision was derived."
echo "Verification: ${REPORT}"
