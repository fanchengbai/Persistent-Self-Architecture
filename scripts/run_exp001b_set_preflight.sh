#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

python -m psa exp001b-set-preflight \
  --final-package preregistration/exp001b/final_v1 \
  --core-set-package preregistration/exp001/core_set_v1 \
  --output results/development/exp001b_set_preflight/preflight.json \
  --project-root "${ROOT_DIR}"

echo "EXP-001B supplemental-set preflight finished."
echo "No model was loaded, no supplemental set was generated, and no trial was scored."
