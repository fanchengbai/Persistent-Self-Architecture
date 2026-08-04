#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0

"${PYTHON_BIN}" -m psa exp001b-runner-dev-gate \
  --model-config configs/models/rwkv7_g1h_2.9b.candidate.json \
  --bdev1-thresholds preregistration/exp001b/final_v1/evidence/bdev1/state_norm_thresholds.json \
  --output-dir results/development/exp001b_formal_runner_dev \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001B formal runner non-Core development gate finished."
echo "No frozen supplemental record was read and no formal authorization was used."
echo "Summary: ${PROJECT_ROOT}/results/development/exp001b_formal_runner_dev/summary.json"
