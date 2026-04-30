#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEVICE="${DEVICE:-cuda:0}"
PRESET="${PG_PRESET:-runtime_row_1024x7_relu2_mlp2}"
ROWS="${ROWS:-1,16,4096,65536}"
WARMUP="${WARMUP:-30}"
ITERS="${ITERS:-100}"
REPEATS="${REPEATS:-3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INCLUDE_INT8_CUTLASS_FUSED_VARIANT="${INCLUDE_INT8_CUTLASS_FUSED_VARIANT:-0}"
INCLUDE_INT8_BACKEND_VARIANTS="${INCLUDE_INT8_BACKEND_VARIANTS:-0}"
RUN_EXACT="${RUN_EXACT:-0}"
RUN_DYNAMIC="${RUN_DYNAMIC:-1}"
ENFORCE_GATE="${ENFORCE_GATE:-1}"
MIN_FORCED_SPEEDUP="${MIN_FORCED_SPEEDUP:-1.0}"
MIN_DIRECT_INT8_SPEEDUP="${MIN_DIRECT_INT8_SPEEDUP:-0.0}"
MIN_PREQUANT_INT8_SPEEDUP="${MIN_PREQUANT_INT8_SPEEDUP:-0.0}"
MIN_ACCUM_INT8_SPEEDUP="${MIN_ACCUM_INT8_SPEEDUP:-1.0}"
MIN_CUTLASS_PREQUANT_SPEEDUP_VS_CURRENT="${MIN_CUTLASS_PREQUANT_SPEEDUP_VS_CURRENT:-0.0}"
MIN_CUTLASS_PREQUANT_SPEEDUP_VS_DEQUANT="${MIN_CUTLASS_PREQUANT_SPEEDUP_VS_DEQUANT:-0.0}"
MIN_CUTLASS_DIRECT_MLP_SPEEDUP_VS_DEQUANT="${MIN_CUTLASS_DIRECT_MLP_SPEEDUP_VS_DEQUANT:-0.0}"
MIN_GATE_ROWS="${MIN_GATE_ROWS:-65536}"
MAX_EXACT_ERR="${MAX_EXACT_ERR:-0.001}"
MAX_DYNAMIC_ERR="${MAX_DYNAMIC_ERR:-1.0}"
LOG_FILE="${LOG_FILE:-$(mktemp /tmp/model_stack_pg_bitnet_kernel_gate.XXXXXX.jsonl)}"

cd "${ROOT_DIR}"

run_bench() {
  local mode="$1"
  local extra_args=()
  if [[ "${INCLUDE_INT8_CUTLASS_FUSED_VARIANT}" != "0" && "${mode}" == "dynamic_int8" ]]; then
    extra_args+=(--include-int8-cutlass-fused-variant)
  fi
  if [[ "${INCLUDE_INT8_BACKEND_VARIANTS}" != "0" && "${mode}" == "dynamic_int8" ]]; then
    extra_args+=(--include-int8-backend-variants)
  fi
  "${PYTHON_BIN}" examples/13_parameter_golf_h100/bench_pg_bitnet_kernels.py \
    --device "${DEVICE}" \
    --preset "${PRESET}" \
    --rows "${ROWS}" \
    --activation-quant "${mode}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --repeats "${REPEATS}" \
    --sync-wall \
    --consume-output \
    --jsonl \
    "${extra_args[@]}"
}

echo "### exact packed runtime-row BitNet, activation_quant=none"
if [[ "${RUN_EXACT}" != "0" ]]; then
  run_bench none | tee -a "${LOG_FILE}"
fi

echo "### dynamic-int8 BitNet candidate, activation_quant=dynamic_int8"
if [[ "${RUN_DYNAMIC}" != "0" ]]; then
  run_bench dynamic_int8 | tee -a "${LOG_FILE}"
fi

echo "### kernel gate log: ${LOG_FILE}"

if [[ "${ENFORCE_GATE}" != "0" ]]; then
  "${PYTHON_BIN}" - "${LOG_FILE}" "${MIN_FORCED_SPEEDUP}" "${MIN_DIRECT_INT8_SPEEDUP}" \
    "${MIN_PREQUANT_INT8_SPEEDUP}" "${MIN_ACCUM_INT8_SPEEDUP}" "${MIN_GATE_ROWS}" \
    "${MAX_EXACT_ERR}" "${MAX_DYNAMIC_ERR}" "${MIN_CUTLASS_PREQUANT_SPEEDUP_VS_CURRENT}" \
    "${MIN_CUTLASS_PREQUANT_SPEEDUP_VS_DEQUANT}" "${MIN_CUTLASS_DIRECT_MLP_SPEEDUP_VS_DEQUANT}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
min_forced = float(sys.argv[2])
min_direct = float(sys.argv[3])
min_prequant = float(sys.argv[4])
min_accum = float(sys.argv[5])
min_gate_rows = int(sys.argv[6])
max_exact_err = float(sys.argv[7])
max_dynamic_err = float(sys.argv[8])
min_cutlass_vs_current = float(sys.argv[9])
min_cutlass_vs_dequant = float(sys.argv[10])
min_cutlass_direct_mlp_vs_dequant = float(sys.argv[11])

failures: list[str] = []
results = []
with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if "header" in obj:
            continue
        results.append(obj)

for item in results:
    shape = item.get("shape")
    rows = int(item.get("rows"))
    mode = item.get("activation_quant")
    variant = item.get("env_variant")
    forced = float(item.get("forced_speedup_vs_dequant", math.nan))
    forced_err = float(item.get("forced_max_abs_err_vs_dequant", math.inf))
    if mode == "none":
        if not math.isfinite(forced) or forced < min_forced:
            failures.append(
                f"{variant} {mode} {shape} rows={rows}: forced_speedup_vs_dequant={forced:.3f} < {min_forced:.3f}"
            )
        if forced_err > max_exact_err:
            failures.append(
                f"{variant} {mode} {shape} rows={rows}: forced_max_abs_err_vs_dequant={forced_err:.6g} > {max_exact_err:.6g}"
            )
    elif mode in {"dynamic_int8", "static_int8"}:
        direct = float(item.get("direct_bitnet_int8_from_float_speedup_vs_dequant", math.nan))
        prequant = float(item.get("prequant_int8_matmul_speedup_vs_dequant", math.nan))
        accum = float(item.get("prequant_int8_accum_only_speedup_vs_dequant", math.nan))
        direct_err = float(item.get("direct_bitnet_int8_from_float_max_abs_err_vs_dequant", math.inf))
        if rows >= min_gate_rows and min_direct > 0.0 and (not math.isfinite(direct) or direct < min_direct):
            failures.append(
                f"{variant} {mode} {shape} rows={rows}: direct_bitnet_speedup_vs_dequant={direct:.3f} < {min_direct:.3f}"
            )
        if rows >= min_gate_rows and min_prequant > 0.0 and (not math.isfinite(prequant) or prequant < min_prequant):
            failures.append(
                f"{variant} {mode} {shape} rows={rows}: prequant_int8_matmul_speedup_vs_dequant={prequant:.3f} < {min_prequant:.3f}"
            )
        if rows >= min_gate_rows and min_accum > 0.0 and (not math.isfinite(accum) or accum < min_accum):
            failures.append(
                f"{variant} {mode} {shape} rows={rows}: prequant_int8_accum_only_speedup_vs_dequant={accum:.3f} < {min_accum:.3f}"
            )
        if direct_err > max_dynamic_err:
            failures.append(
                f"{variant} {mode} {shape} rows={rows}: direct_bitnet_max_abs_err_vs_dequant={direct_err:.6g} > {max_dynamic_err:.6g}"
            )

if min_cutlass_vs_current > 0.0 or min_cutlass_vs_dequant > 0.0 or min_cutlass_direct_mlp_vs_dequant > 0.0:
    current_by_key = {}
    cutlass_by_key = {}
    for item in results:
        if item.get("activation_quant") not in {"dynamic_int8", "static_int8"}:
            continue
        rows = int(item.get("rows"))
        if rows < min_gate_rows:
            continue
        key = (item.get("activation_quant"), item.get("shape"), rows)
        variant = item.get("env_variant")
        if variant == "current":
            current_by_key[key] = item
        elif variant == "int8_cutlass_fused":
            cutlass_by_key[key] = item
    for key, current in current_by_key.items():
        cutlass = cutlass_by_key.get(key)
        mode, shape, rows = key
        if cutlass is None:
            failures.append(
                f"missing int8_cutlass_fused result for {mode} {shape} rows={rows}; "
                "set INCLUDE_INT8_BACKEND_VARIANTS=1"
            )
            continue
        current_ms = float(current.get("prequant_int8_matmul_ms", math.nan))
        cutlass_ms = float(cutlass.get("prequant_int8_matmul_ms", math.nan))
        ratio = current_ms / cutlass_ms if cutlass_ms > 0.0 else math.inf
        if not math.isfinite(ratio) or ratio < min_cutlass_vs_current:
            failures.append(
                f"int8_cutlass_fused {mode} {shape} rows={rows}: "
                f"prequant speedup vs current={ratio:.3f} < {min_cutlass_vs_current:.3f} "
                f"(current_ms={current_ms:.6g}, cutlass_ms={cutlass_ms:.6g})"
            )
        cutlass_vs_dequant = float(cutlass.get("prequant_int8_matmul_speedup_vs_dequant", math.nan))
        if min_cutlass_vs_dequant > 0.0 and (
            not math.isfinite(cutlass_vs_dequant) or cutlass_vs_dequant < min_cutlass_vs_dequant
        ):
            failures.append(
                f"int8_cutlass_fused {mode} {shape} rows={rows}: "
                f"prequant speedup vs dequant={cutlass_vs_dequant:.3f} < {min_cutlass_vs_dequant:.3f}"
            )
        if min_cutlass_direct_mlp_vs_dequant > 0.0 and str(shape).startswith("mlp_"):
            direct_vs_dequant = float(cutlass.get("direct_bitnet_int8_from_float_speedup_vs_dequant", math.nan))
            auto_vs_dequant = float(cutlass.get("auto_speedup_vs_dequant", math.nan))
            native_module_vs_dequant = cutlass.get("native_module_auto_speedup_vs_dequant")
            if not math.isfinite(direct_vs_dequant) or direct_vs_dequant < min_cutlass_direct_mlp_vs_dequant:
                failures.append(
                    f"int8_cutlass_fused {mode} {shape} rows={rows}: "
                    f"direct speedup vs dequant={direct_vs_dequant:.3f} < {min_cutlass_direct_mlp_vs_dequant:.3f}"
                )
            if not math.isfinite(auto_vs_dequant) or auto_vs_dequant < min_cutlass_direct_mlp_vs_dequant:
                failures.append(
                    f"int8_cutlass_fused {mode} {shape} rows={rows}: "
                    f"auto speedup vs dequant={auto_vs_dequant:.3f} < {min_cutlass_direct_mlp_vs_dequant:.3f}"
                )
            if native_module_vs_dequant is not None:
                native_module_vs_dequant = float(native_module_vs_dequant)
                if (
                    not math.isfinite(native_module_vs_dequant) or
                    native_module_vs_dequant < min_cutlass_direct_mlp_vs_dequant
                ):
                    failures.append(
                        f"int8_cutlass_fused {mode} {shape} rows={rows}: "
                        f"native module auto speedup vs dequant={native_module_vs_dequant:.3f} "
                        f"< {min_cutlass_direct_mlp_vs_dequant:.3f}"
                    )

if failures:
    print("Kernel gate failed:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)

print("Kernel gate passed.")
PY
fi
