#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/rwkv7_g1h_2.9b.candidate.json"
GATE_CONFIG="${PROJECT_ROOT}/configs/gates/impl3p_g1h_2.9b_history_binding.dev.json"
ASSET_MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
OUTPUT_DIR="${PSA_IMPL3P_OUTPUT:-${PROJECT_ROOT}/results/development/impl3p_g1h_2.9b_history_binding}"

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
export PSA_DETERMINISTIC_SEED=20260731
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${ASSET_MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa history-binding-gate \
  --config "${MODEL_CONFIG}" \
  --gate-config "${GATE_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "Impl-3p G1h 2.9B history binding gate finished."
echo "Summary: ${OUTPUT_DIR}/summary.json"
