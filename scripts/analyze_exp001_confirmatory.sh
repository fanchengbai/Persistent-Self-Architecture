#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python -m psa confirmatory-analyze \
  --raw-output-dir results/confirmatory/exp001_v1 \
  --raw-verification results/confirmatory/exp001_v1.raw_verification.json \
  --core-set-package preregistration/exp001/core_set_v1 \
  --final-package preregistration/exp001/final_v1 \
  --analysis-config configs/analysis/exp001_confirmatory_v1.json \
  --analysis-output-dir results/confirmatory/exp001_v1_analysis \
  --project-root "$project_root"

echo "EXP-001 frozen read-only analysis finished."
echo "Summary: $project_root/results/confirmatory/exp001_v1_analysis/summary.json"
