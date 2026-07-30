#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export RWKV_V7_ON=1
export RWKV_JIT_ON=0
export RWKV_CUDA_ON=0
export PSA_DETERMINISTIC=1
export PSA_DETERMINISTIC_SEED=20260730
export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUTPUT_DIR="${PROJECT_ROOT}/results/development/impl3_development_v02"
mkdir -p "${OUTPUT_DIR}"

python -m psa impl3-development-gate \
  --config configs/models/rwkv7_world_0.4b.impl1.json \
  --gate-config configs/gates/impl3_development_v0.2.dev.json \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "Summary: ${OUTPUT_DIR}/summary.json"
