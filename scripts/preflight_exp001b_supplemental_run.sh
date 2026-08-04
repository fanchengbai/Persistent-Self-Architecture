#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

python -m psa exp001b-run-preflight \
  --final-package preregistration/exp001b/final_v1 \
  --core-set-package preregistration/exp001/core_set_v1 \
  --supplemental-set-package preregistration/exp001b/supplemental_set_v1 \
  --model-config configs/models/rwkv7_g1h_2.9b.candidate.json \
  --asset-manifest configs/assets/exp001_rwkv7_g1h_2.9b_candidate.json \
  --asset-root .psa-assets \
  --runner-evidence results/development/exp001b_formal_runner_dev/summary.json \
  --output results/development/exp001b_run_preflight/preflight.json \
  --project-root "${PROJECT_ROOT}"

echo "EXP-001B formal run preflight finished."
echo "No model was loaded and no supplemental record was scored."
