#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORMAL_CONFIG="${PROJECT_ROOT}/configs/preregistration/exp001_track_s.formal_v1.json"
ASSET_MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
OUTPUT_DIR="${PSA_IMPL3Q_OUTPUT:-${PROJECT_ROOT}/results/development/impl3q_exp001_formal_freeze_candidate}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0
export PSA_DETERMINISTIC=1
export PSA_DETERMINISTIC_SEED=4045556568
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${ASSET_MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa formal-freeze-candidate-gate \
  --config "${FORMAL_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m psa preregistration-verify \
  --candidate "${OUTPUT_DIR}/preregistration_candidate.json" \
  --project-root "${PROJECT_ROOT}" \
  --output "${OUTPUT_DIR}/preregistration_verification.manual.json"

echo "Impl-3q EXP-001 formal freeze candidate gate finished."
echo "No Core Set was generated or unsealed."
echo "Summary: ${OUTPUT_DIR}/summary.json"
