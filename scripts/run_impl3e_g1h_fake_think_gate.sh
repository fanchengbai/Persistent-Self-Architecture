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

OUTPUT_DIR="${PROJECT_ROOT}/results/development/impl3e_g1h_1.5b_fake_think"
mkdir -p "${OUTPUT_DIR}"

python -m psa g1-capability-ladder-gate \
  --config configs/models/rwkv7_g1h_1.5b.candidate.json \
  --gate-config configs/gates/impl3e_g1h_1.5b_fake_think.dev.json \
  --output-dir "${OUTPUT_DIR}" \
  --project-root "${PROJECT_ROOT}"

echo "Summary: ${OUTPUT_DIR}/summary.json"
