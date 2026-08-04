#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/rwkv7_g1h_2.9b.candidate.json"
ASSET_MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json"
DESIGN="${PROJECT_ROOT}/configs/preregistration/exp001b_supplemental_controls.draft.json"
BDEV1_DIR="${PSA_EXP001B_BDEV1_OUTPUT:-${PROJECT_ROOT}/results/development/exp001b_bdev1_non_core_calibration}"
OUTPUT_DIR="${PSA_EXP001B_BDEV2_OUTPUT:-${PROJECT_ROOT}/results/development/exp001b_bdev2_non_core_runner_v02}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${ASSET_MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa exp001b-bdev2-gate \
  --design "${DESIGN}" \
  --model-config "${MODEL_CONFIG}" \
  --bdev1-summary "${BDEV1_DIR}/summary.json" \
  --bdev1-thresholds "${BDEV1_DIR}/state_norm_thresholds.json" \
  --bdev1-matched-report "${BDEV1_DIR}/matched_context_token_report.json" \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001B B-Dev2 non-Core runner gate finished."
echo "No Core Set or supplemental formal set was read or generated."
echo "Summary: ${OUTPUT_DIR}/summary.json"
