#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi
if [[ "${PSA_EXP001B_SET_GENERATE:-}" != "AUTHORIZED_EXP001B_SET_GENERATION" ]]; then
  echo "error: EXP-001B supplemental-set execution lock is absent" >&2
  exit 2
fi
if [[ -z "${PSA_EXP001B_SET_AUTHORIZATION:-}" ]]; then
  echo "error: PSA_EXP001B_SET_AUTHORIZATION must name the owner authorization file" >&2
  exit 2
fi

python -m psa exp001b-set-generate \
  --final-package preregistration/exp001b/final_v1 \
  --core-set-package preregistration/exp001/core_set_v1 \
  --authorization "${PSA_EXP001B_SET_AUTHORIZATION}" \
  --formal-config configs/preregistration/exp001_track_s.formal_v3_holdout.json \
  --model-config configs/models/rwkv7_g1h_2.9b.candidate.json \
  --output-dir preregistration/exp001b/supplemental_set_v1 \
  --project-root "${ROOT_DIR}"

echo "EXP-001B supplemental set is frozen and independently verified."
echo "The formal supplemental experiment remains unauthorized and unrun."
