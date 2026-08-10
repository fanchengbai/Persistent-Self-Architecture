#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m psa exp001b-analyze \
  --parent-raw-output-dir results/confirmatory/exp001_v1 \
  --parent-raw-verification results/confirmatory/exp001_v1.raw_verification.json \
  --supplemental-raw-output-dir results/confirmatory/exp001b_v1 \
  --supplemental-raw-verification results/confirmatory/exp001b_v1.raw_verification.json \
  --core-set-package preregistration/exp001/core_set_v1 \
  --supplemental-set-package preregistration/exp001b/supplemental_set_v1 \
  --analysis-config configs/analysis/exp001b_supplemental_v1.json \
  --analysis-output-dir results/confirmatory/exp001b_v1_analysis \
  --project-root "${PROJECT_ROOT}"
