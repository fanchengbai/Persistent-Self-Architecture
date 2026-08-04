#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
cd "${PROJECT_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0
if [[ "${PSA_EXP001B_RUN:-}" != "AUTHORIZED_EXP001B_SUPPLEMENTAL_RUN" ]]; then
  echo "error: EXP-001B supplemental run execution lock is absent" >&2
  exit 2
fi
if [[ -z "${PSA_EXP001B_RUN_AUTHORIZATION:-}" ]]; then
  echo "error: PSA_EXP001B_RUN_AUTHORIZATION is not set" >&2
  exit 2
fi

RESUME_ARGS=()
if [[ "${PSA_EXP001B_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${PYTHON_BIN}" -m psa exp001b-run \
  --final-package preregistration/exp001b/final_v1 \
  --core-set-package preregistration/exp001/core_set_v1 \
  --supplemental-set-package preregistration/exp001b/supplemental_set_v1 \
  --model-config configs/models/rwkv7_g1h_2.9b.candidate.json \
  --asset-manifest configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json \
  --asset-root "${ASSET_ROOT}" \
  --runner-evidence results/development/exp001b_formal_runner_dev/summary.json \
  --preflight results/development/exp001b_run_preflight/preflight.json \
  --authorization "${PSA_EXP001B_RUN_AUTHORIZATION}" \
  --output-dir results/confirmatory/exp001b_v1 \
  --project-root "${PROJECT_ROOT}" \
  "${RESUME_ARGS[@]}"
