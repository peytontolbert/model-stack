#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
PYTHON="${PYTHON:-/home/peyton/miniconda3/envs/ai/bin/python}"
ROOT="${ROOT:-/data/transformer_10}"
TEACHER="${TEACHER:-/data/resumebot/checkpoints/final_finetuned_model.pt}"
# Keep the dense/Q4 shadow weights for BitNet QAT. Starting from a
# pre-materialized BitNet projection discards the recoverable float shadow and
# substantially worsens F5 audio quality.
STUDENT_INIT="${STUDENT_INIT:-/data/transformer_10/checkpoints/f5tts_q4_8step_cfg2_libritts_fullq4_surface_v2/model_q4_12to4_best.pt}"
PROJECTED_BITNET_INIT="${PROJECTED_BITNET_INIT:-/data/transformer_10/checkpoints/f5tts_bitnet_projection_from_current_best/model_bitnet_projected.pt}"
VOCAB="${VOCAB:-/data/resumebot/checkpoints/F5TTS_Base_vocab.txt}"
LOG_DIR="${LOG_DIR:-/data/transformer_10/logs}"
MAX_STEPS="${MAX_STEPS:-800}"
SAVE_EVERY="${SAVE_EVERY:-100}"
DATASET="${DATASET:-blabble-io/libritts_r}"
CONFIG="${CONFIG:-clean}"
SPLIT="${SPLIT:-train.clean.360}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
TEXT_COLUMN="${TEXT_COLUMN:-text_normalized}"
WAIT_FREE_MB="${WAIT_FREE_MB:-0}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
MODE="${1:-both}"
if [[ $# -gt 0 ]]; then
  shift
fi
TRAIN_INCLUDE="${TRAIN_INCLUDE:-transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out}"
BITNET_INCLUDE="${BITNET_INCLUDE:-${TRAIN_INCLUDE}}"

mkdir -p "${LOG_DIR}" "${ROOT}/checkpoints"

gpu_free_mb() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v gpu="${GPU}" '$1 == gpu {gsub(/ /, "", $2); print $2; exit}'
}

wait_for_gpu_memory() {
  if [[ "${WAIT_FREE_MB}" -le 0 ]]; then
    return
  fi

  while true; do
    local free_mb
    free_mb="$(gpu_free_mb)"
    if [[ -z "${free_mb}" ]]; then
      echo "GPU ${GPU} was not found by nvidia-smi." >&2
      exit 1
    fi
    if [[ "${free_mb}" -ge "${WAIT_FREE_MB}" ]]; then
      echo "GPU ${GPU} has ${free_mb} MiB free; starting BitNet candidate."
      return
    fi
    echo "GPU ${GPU} has ${free_mb} MiB free; waiting for ${WAIT_FREE_MB} MiB..."
    sleep "${WAIT_POLL_SECONDS}"
  done
}

common_args=(
  "${ROOT}/scripts/distill_f5tts_12_to_4_q4.py"
  --checkpoint "${TEACHER}"
  --student-checkpoint "${STUDENT_INIT}"
  --vocab "${VOCAB}"
  --dataset "${DATASET}"
  --config "${CONFIG}"
  --split "${SPLIT}"
  --audio-column "${AUDIO_COLUMN}"
  --text-column "${TEXT_COLUMN}"
  --sample-rate 24000
  --local-sample-prob 0.0
  --min-duration 1.0
  --max-duration 5.0
  --max-frames "${MAX_FRAMES:-256}"
  --cond-frames "${COND_FRAMES:-192}"
  --min-gen-frames "${MIN_GEN_FRAMES:-32}"
  --batch-size 1
  --max-steps "${MAX_STEPS}"
  --teacher-steps 24
  --teacher-cfg-strength 2.0
  --sway-sampling-coef -1.0
  --student-quant-scheme bitnet_qat
  ${BITNET_QAT_LEARNED_SCALE:+--bitnet-qat-learned-scale}
  --bitnet-scale-lr-multiplier "${BITNET_SCALE_LR_MULTIPLIER:-1.0}"
  --q4-include "${BITNET_INCLUDE}"
  --q4-exclude text_embed.text_embed,mel_spec
  --train-include "${TRAIN_INCLUDE}"
  --detach-null-grad
  --train-in-eval-mode
  --convert-text-to-pinyin
  --lr 1e-7
  --weight-decay 0.0
  --max-grad-norm 0.5
  --save-every "${SAVE_EVERY}"
  --prune-step-checkpoints
  --log-every 1
  --rollout-loss-weight 1.0
  --teacher-flow-loss-weight 0.12
  --real-mel-loss-weight 0.14
  --anchor-weight-loss-weight 0.08
  --temporal-delta-loss-weight 0.35
  --mel-energy-loss-weight 0.08
  --high-mel-excess-loss-weight 0.03
  --high-mel-match-loss-weight 0.10
  --high-mel-start-bin 80
  --loss-schedule-steps 200
  --rollout-loss-weight-final 1.0
  --teacher-flow-loss-weight-final 0.04
  --real-mel-loss-weight-final 0.20
  --anchor-weight-loss-weight-final 0.10
  --temporal-delta-loss-weight-final 0.40
  --mel-energy-loss-weight-final 0.10
  --high-mel-excess-loss-weight-final 0.04
  --high-mel-match-loss-weight-final 0.12
  --device cuda
  --cuda-visible-devices "${GPU}"
  --seed 20260521
)

run_candidate() {
  local name="$1"
  shift
  local output_name="${name}${OUTPUT_SUFFIX}"
  local out_dir="${ROOT}/checkpoints/${output_name}"
  local log="${LOG_DIR}/${output_name}_active.log"
  mkdir -p "${out_dir}"
  wait_for_gpu_memory
  echo "Launching ${output_name} on CUDA_VISIBLE_DEVICES=${GPU}; log=${log}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${common_args[@]}" --output-dir "${out_dir}" "$@" 2>&1 | tee "${log}"
}

case "${MODE}" in
  2step)
    run_candidate f5tts_bitnet_current_best_to_2step_cfgfree_v0 --student-steps 2 --student-cfg-strength 0.0 --cfg-strength 2.0 "$@"
    ;;
  1step)
    run_candidate f5tts_bitnet_current_best_to_1step_cfg2_v0 --student-steps 1 --student-cfg-strength 2.0 --cfg-strength 2.0 "$@"
    ;;
  both)
    run_candidate f5tts_bitnet_current_best_to_2step_cfgfree_v0 --student-steps 2 --student-cfg-strength 0.0 --cfg-strength 2.0 "$@"
    run_candidate f5tts_bitnet_current_best_to_1step_cfg2_v0 --student-steps 1 --student-cfg-strength 2.0 --cfg-strength 2.0 "$@"
    ;;
  *)
    echo "Usage: GPU=1 MAX_STEPS=800 $0 [2step|1step|both]" >&2
    exit 2
    ;;
esac
