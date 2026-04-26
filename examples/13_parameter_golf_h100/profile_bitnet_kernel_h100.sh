#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEVICE="${DEVICE:-cuda:0}"
ACTIVATION_QUANT="${ACTIVATION_QUANT:-dynamic_int8}"
SHAPE="${SHAPE:-mlp_up:1024:3072}"
ROWS="${ROWS:-65536}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NCU_BIN="${NCU_BIN:-ncu}"
OUT="${OUT:-/tmp/model_stack_pg_bitnet_kernel_profile}"

cd "${ROOT_DIR}"

"${NCU_BIN}" \
  --force-overwrite \
  --set full \
  --target-processes all \
  --export "${OUT}" \
  "${PYTHON_BIN}" examples/13_parameter_golf_h100/bench_pg_bitnet_kernels.py \
    --device "${DEVICE}" \
    --no-preset-shapes \
    --shape "${SHAPE}" \
    --rows "${ROWS}" \
    --activation-quant "${ACTIVATION_QUANT}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeats 1 \
    --sync-wall \
    --consume-output \
    --jsonl

echo "Nsight Compute report: ${OUT}.ncu-rep"
