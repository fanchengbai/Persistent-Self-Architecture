#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi
if [[ -z "${PSA_EXP001B_RUN_AUTHORIZATION:-}" ]]; then
  echo "error: PSA_EXP001B_RUN_AUTHORIZATION is not set" >&2
  exit 2
fi

python -m psa exp001b-raw-verify \
  --output-dir results/confirmatory/exp001b_v1 \
  --core-set-package preregistration/exp001/core_set_v1 \
  --supplemental-set-package preregistration/exp001b/supplemental_set_v1 \
  --preflight results/development/exp001b_run_preflight/preflight.json \
  --authorization "${PSA_EXP001B_RUN_AUTHORIZATION}" \
  --output results/confirmatory/exp001b_v1.raw_verification.json
