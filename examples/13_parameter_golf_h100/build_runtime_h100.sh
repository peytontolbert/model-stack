#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

MODEL_STACK_BUILD_NATIVE="${MODEL_STACK_BUILD_NATIVE:-1}" \
MODEL_STACK_BUILD_CUDA="${MODEL_STACK_BUILD_CUDA:-1}" \
MODEL_STACK_CUDA_ARCH_LIST="${MODEL_STACK_CUDA_ARCH_LIST:-9.0}" \
MODEL_STACK_MAX_JOBS="${MODEL_STACK_MAX_JOBS:-4}" \
python setup.py build_ext --inplace

