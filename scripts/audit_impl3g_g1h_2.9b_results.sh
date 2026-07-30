#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUT_DIR="${PROJECT_ROOT}/results/development/impl3g_g1h_2.9b_fake_think"

python -m psa g1-capability-audit --output-dir "${OUTPUT_DIR}"

echo "Audit: ${OUTPUT_DIR}/audit_report.json"
