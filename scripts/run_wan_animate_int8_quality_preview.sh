#!/usr/bin/env bash
set -euo pipefail

WAN_ROOT=${WAN_ROOT:-/home/peyton/src/Wan2.2}
CKPT_DIR=${CKPT_DIR:-/data/models/Wan-AI--Wan2.2-Animate-14B}
SRC_ROOT=${SRC_ROOT:-/data/clone/wan_animate_seated_trial/input33}
INT8_ARTIFACT_DIR=${INT8_ARTIFACT_DIR:-/data/transformer_10/checkpoints/wan_animate_int8_weightonly_v2}
OUT_DIR=${OUT_DIR:-/data/clone/wan_animate_seated_trial/output_int8}
SEED=${SEED:-2988027455084004927}
STEPS=${STEPS:-4}
# Wan Animate uses frame_num as clip length, not as a total-frame truncation.
# Keep this at 33 for the input33 source to avoid a second overlapping clip.
FRAME_NUM=${FRAME_NUM:-33}
GUIDE=${GUIDE:-1.0}
WAN_PROFILE=${WAN_PROFILE:-0}
DISABLE_BLOCK_PREFETCH=${DISABLE_BLOCK_PREFETCH:-0}

EXTRA_ARGS=()
if [[ "${DISABLE_BLOCK_PREFETCH}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_block_prefetch)
fi

mkdir -p "${OUT_DIR}"
cd "${WAN_ROOT}"
PYTHONPATH=/data/transformer_10:${WAN_ROOT}:${PYTHONPATH:-} \
PYTHONNOUSERSITE=1 \
WAN_PROFILE="${WAN_PROFILE}" \
python generate.py \
  --task animate-14B \
  --size 1280*720 \
  --frame_num "${FRAME_NUM}" \
  --ckpt_dir "${CKPT_DIR}" \
  --src_root_path "${SRC_ROOT}" \
  --refert_num 1 \
  --t5_cpu \
  --offload_model true \
  --dit_int8_offload \
  --int8_artifact_dir "${INT8_ARTIFACT_DIR}" \
  --int8_block_group 1 \
  --sample_solver unipc \
  --sample_steps "${STEPS}" \
  --sample_shift 5.0 \
  --sample_guide_scale "${GUIDE}" \
  --base_seed "${SEED}" \
  --save_file "${OUT_DIR}/peyton_wan_animate_int8_weightonly_${FRAME_NUM}f_${STEPS}step.mp4" \
  "${EXTRA_ARGS[@]}"
