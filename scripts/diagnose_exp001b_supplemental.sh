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

"${PYTHON_BIN}" -m psa exp001b-diagnose \
  --supplemental-raw-output-dir results/confirmatory/exp001b_v1 \
  --supplemental-raw-verification results/confirmatory/exp001b_v1.raw_verification.json \
  --supplemental-set-package preregistration/exp001b/supplemental_set_v1 \
  --analysis-output-dir results/confirmatory/exp001b_v1_analysis_v02 \
  --diagnostic-output-dir results/confirmatory/exp001b_v1_diagnostics_v04 \
  --tokenizer .psa-assets/tokenizers/rwkv_vocab_v20230424.txt
