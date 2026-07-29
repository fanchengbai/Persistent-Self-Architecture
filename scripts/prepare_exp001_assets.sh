#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_ROOT="${PSA_ASSET_ROOT:-${PROJECT_ROOT}/.psa-assets}"
MANIFEST="${PROJECT_ROOT}/configs/assets/exp001_rwkv7_world_0.4b.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "PSA asset root: ${ASSET_ROOT}"
echo "Pinned asset plan:"
"${PYTHON_BIN}" -m psa assets-plan \
  --manifest "${MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa assets-fetch \
  --manifest "${MANIFEST}" \
  --root "${ASSET_ROOT}"

"${PYTHON_BIN}" -m psa task-generate \
  --output "${ASSET_ROOT}/datasets/exp001/identity_goal.synthetic.dev.json" \
  --config "${PROJECT_ROOT}/configs/tasks/exp001_identity_goal.dev.json"

"${PYTHON_BIN}" -m psa assets-verify \
  --manifest "${MANIFEST}" \
  --root "${ASSET_ROOT}"

echo "EXP-001 development assets are ready."
