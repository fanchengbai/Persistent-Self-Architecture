#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${PSA_IMPL3R_OUTPUT:-${PROJECT_ROOT}/results/development/impl3r_exp001_formal_freeze_candidate_v2}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m psa formal-freeze-review \
  --output-dir "${OUTPUT_DIR}"

echo "Impl-3r read-only review finished."
echo "No model was loaded and no confirmatory result was read."
echo "Review: ${OUTPUT_DIR}/formal_freeze_review.json"
