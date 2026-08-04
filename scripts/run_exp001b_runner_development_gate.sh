#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

python -m psa exp001b-runner-dev-gate \
  --model-config configs/models/rwkv7_g1h_2.9b.candidate.json \
  --bdev1-thresholds preregistration/exp001b/final_v1/evidence/bdev1/state_norm_thresholds.json \
  --output-dir results/development/exp001b_formal_runner_dev \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001B formal runner non-Core development gate finished."
echo "No frozen supplemental record was read and no formal authorization was used."
echo "Summary: ${PROJECT_ROOT}/results/development/exp001b_formal_runner_dev/summary.json"
