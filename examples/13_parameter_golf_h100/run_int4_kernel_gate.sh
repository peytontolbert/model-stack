#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROWS="${ROWS:-65536}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"
MIN_DEFAULT_SPEEDUP="${MIN_DEFAULT_SPEEDUP:-1.05}"
MIN_NATIVE_SPEEDUP="${MIN_NATIVE_SPEEDUP:-1.05}"
INCLUDE_ATTENTION="${INCLUDE_ATTENTION:-0}"
LOG_FILE="${LOG_FILE:-$(mktemp /tmp/model_stack_pg_int4_kernel_gate.XXXXXX.jsonl)}"

cd "${ROOT_DIR}"

run_shape() {
  local name="$1"
  local k="$2"
  local n="$3"
  echo "### ${name} M=${ROWS} K=${k} N=${n}" | tee -a "${LOG_FILE}"
  "${PYTHON_BIN}" examples/13_parameter_golf_h100/bench_pg_int4_training_matmul.py \
    --device "${DEVICE}" \
    --dtype bf16 \
    --rows "${ROWS}" \
    --in-features "${k}" \
    --out-features "${n}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --jsonl | tee -a "${LOG_FILE}"
}

run_shape mlp_up_relu2 1024 2048
run_shape mlp_down_relu2 2048 1024
if [[ "${INCLUDE_ATTENTION}" != "0" ]]; then
  run_shape attn_q 1024 1024
  run_shape attn_kv 1024 256
fi

echo "### int4 kernel gate log: ${LOG_FILE}"

"${PYTHON_BIN}" - "${LOG_FILE}" "${MIN_DEFAULT_SPEEDUP}" "${MIN_NATIVE_SPEEDUP}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
min_default = float(sys.argv[2])
min_native = float(sys.argv[3])

current = "unknown"
failures: list[str] = []
with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line.startswith("### "):
            current = line[4:]
            continue
        if not line.startswith("{"):
            continue
        item = json.loads(line)
        variant = str(item.get("variant", ""))
        speedup = float(item.get("speedup_vs_dense", math.nan))
        if variant == "int4_default_train" and (not math.isfinite(speedup) or speedup < min_default):
            failures.append(f"{current}: int4_default_train speedup {speedup:.3f} < {min_default:.3f}")
        if variant == "int4_native_forward_dense_backward_train" and (
            not math.isfinite(speedup) or speedup < min_native
        ):
            failures.append(
                f"{current}: native packed int4 speedup {speedup:.3f} < {min_native:.3f}"
            )

if failures:
    print("INT4 kernel gate failed:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("INT4 kernel gate passed.")
PY
