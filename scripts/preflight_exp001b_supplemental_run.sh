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

"${PYTHON_BIN}" -m psa exp001b-run-preflight \
  --final-package preregistration/exp001b/final_v1 \
  --core-set-package preregistration/exp001/core_set_v1 \
  --supplemental-set-package preregistration/exp001b/supplemental_set_v1 \
  --model-config configs/models/rwkv7_g1h_2.9b.candidate.json \
  --asset-manifest configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json \
  --asset-root "${ASSET_ROOT}" \
  --runner-evidence results/development/exp001b_formal_runner_dev/summary.json \
  --output results/development/exp001b_run_preflight/preflight.json \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001B formal run preflight finished."
echo "No model was loaded and no supplemental record was scored."
