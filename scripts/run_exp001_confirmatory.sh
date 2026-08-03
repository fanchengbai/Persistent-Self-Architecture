#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
AUTHORIZATION="${PSA_CONFIRMATORY_AUTHORIZATION:-}"
OUTPUT_DIR="${PSA_CONFIRMATORY_OUTPUT:-${PROJECT_ROOT}/results/confirmatory/exp001_v1}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi
if [[ "${PSA_CONFIRMATORY_EXECUTE:-}" != "AUTHORIZED_EXP001_CONFIRMATORY_RUN" ]]; then
  echo "error: explicit PSA_CONFIRMATORY_EXECUTE safety switch is missing" >&2
  exit 2
fi
if [[ -z "${AUTHORIZATION}" || ! -f "${AUTHORIZATION}" ]]; then
  echo "error: PSA_CONFIRMATORY_AUTHORIZATION must name an existing authorization file" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0

RESUME_ARGS=()
if [[ "${PSA_CONFIRMATORY_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

"${PYTHON_BIN}" -m psa confirmatory-run \
  --final-package "${PROJECT_ROOT}/preregistration/exp001/final_v1" \
  --core-set-package "${PROJECT_ROOT}/preregistration/exp001/core_set_v1" \
  --model-config "${PROJECT_ROOT}/configs/models/rwkv7_g1h_2.9b.candidate.json" \
  --asset-manifest "${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json" \
  --asset-root "${ASSET_ROOT}" \
  --runner-evidence "${PROJECT_ROOT}/results/development/impl5b_confirmatory_runner_dev/summary.json" \
  --preflight "${PROJECT_ROOT}/results/development/impl5b_confirmatory_preflight/preflight.json" \
  --authorization "${AUTHORIZATION}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}" \
  "${RESUME_ARGS[@]}"

echo "EXP-001 raw confirmatory execution finished."
echo "No derived accuracy or interim decision was emitted by the runner."
echo "Completion: ${OUTPUT_DIR}/completion.json"
