#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_g1h_1.5b_candidate.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "PSA asset root: ${ASSET_ROOT}"
echo "Pinned G1h 1.5B candidate plan:"
"${PYTHON_BIN}" -m psa assets-plan \
  --manifest "${MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa assets-fetch \
  --manifest "${MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${MANIFEST}" \
  --root "${ASSET_ROOT}"

echo "RWKV-7 G1h 1.5B candidate assets are ready."
