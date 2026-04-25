#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PG_DIR="${PG_DIR:-${ROOT_DIR}/other_repos/parameter-golf}"
ARTIFACT="${ARTIFACT:-${PG_DIR}/final_model.ternary.ptz}"
OUT="${OUT:-${ROOT_DIR}/artifacts/parameter_golf/final_model.runtime_bitnet.pt}"

cd "${ROOT_DIR}"
mkdir -p "$(dirname "${OUT}")"

python tests/bench_parameter_golf_bitnet_export.py \
  --pg-script "${PG_DIR}/train_gpt.py" \
  --artifact "${ARTIFACT}" \
  --export-packed "${OUT}" \
  --verify-export \
  --summary-json

