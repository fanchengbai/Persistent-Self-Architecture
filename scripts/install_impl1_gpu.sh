#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPORT_PATH="${PSA_ENV_REPORT:-${PROJECT_ROOT}/results/development/environment_manifest.json}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate the project virtual environment before running this script" >&2
  exit 2
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
  torch==2.12.0 \
  --index-url https://download.pytorch.org/whl/cu132
"${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}[rwkv7]"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0
"${PYTHON_BIN}" -m psa environment-report --output "${REPORT_PATH}"

echo "Impl-1 GPU environment is ready."
echo "Environment report: ${REPORT_PATH}"
