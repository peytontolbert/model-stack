#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
PYTHON="${PYTHON:-/home/peyton/miniconda3/envs/ai/bin/python}"
ROOT="${ROOT:-/data/transformer_10}"
TEACHER="${TEACHER:-/data/resumebot/checkpoints/final_finetuned_model.pt}"
VOCAB="${VOCAB:-/data/resumebot/checkpoints/F5TTS_Base_vocab.txt}"
DATASET="${DATASET:-blabble-io/libritts_r}"
CONFIG="${CONFIG:-clean}"
SPLIT="${SPLIT:-train.clean.360}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
TEXT_COLUMN="${TEXT_COLUMN:-text_normalized}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
MAX_STEPS="${MAX_STEPS:-800}"
SAVE_EVERY="${SAVE_EVERY:-100}"
SEED="${SEED:-20260522}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
MODE="${1:-24to4}"
if [[ $# -gt 0 ]]; then
  shift
fi

# The good 8-step models used a broad Q4 surface while training only the late
# denoiser blocks. Keep that separation: quantization coverage should match the
# shipping path; training remains scoped.
TRAIN_INCLUDE="${TRAIN_INCLUDE:-transformer.transformer_blocks.16,transformer.transformer_blocks.17,transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out}"
Q4_INCLUDE="${Q4_INCLUDE:-}"
Q4_STE_INCLUDE="${Q4_STE_INCLUDE:-${TRAIN_INCLUDE}}"
Q4_EXCLUDE="${Q4_EXCLUDE:-text_embed.text_embed,mel_spec}"

STUDENT_8STEP="${STUDENT_8STEP:-${ROOT}/checkpoints/f5tts_q4_8step_cfg2_libritts_fullq4_surface_v2/model_q4_12to4_best.pt}"
STUDENT_4STEP="${STUDENT_4STEP:-${ROOT}/checkpoints/f5tts_q4_4step_cfg2_clean360_progressive_v0/model_q4_12to4_best.pt}"
STUDENT_2STEP="${STUDENT_2STEP:-${ROOT}/checkpoints/f5tts_q4_2step_cfg2_clean360_progressive_v0/model_q4_12to4_best.pt}"
TEACHER_8TO4="${TEACHER_8TO4:-${STUDENT_8STEP}}"
TEACHER_4TO2="${TEACHER_4TO2:-${STUDENT_4STEP}}"
TEACHER_BITNET2="${TEACHER_BITNET2:-${STUDENT_2STEP}}"

mkdir -p "${LOG_DIR}" "${ROOT}/checkpoints"

common_args=(
  "${ROOT}/scripts/distill_f5tts_12_to_4_q4.py"
  --checkpoint "${TEACHER}"
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
  --batch-size "${BATCH_SIZE:-1}"
  --max-steps "${MAX_STEPS}"
  --teacher-steps 24
  --teacher-cfg-strength 2.0
  --sway-sampling-coef -1.0
  --q4-include "${Q4_INCLUDE}"
  --q4-exclude "${Q4_EXCLUDE}"
  --q4-ste-include "${Q4_STE_INCLUDE}"
  --train-include "${TRAIN_INCLUDE}"
  --detach-null-grad
  --train-in-eval-mode
  --convert-text-to-pinyin
  --weight-decay 0.0
  --save-every "${SAVE_EVERY}"
  --prune-step-checkpoints
  --log-every "${LOG_EVERY:-1}"
  --high-mel-start-bin 80
  --device cuda
  --cuda-visible-devices "${GPU}"
  --seed "${SEED}"
)

run_stage() {
  local name="$1"
  shift
  local output_name="${name}${OUTPUT_SUFFIX}"
  local out_dir="${ROOT}/checkpoints/${output_name}"
  local log="${LOG_DIR}/${output_name}_active.log"
  mkdir -p "${out_dir}"
  echo "Launching ${output_name} on CUDA_VISIBLE_DEVICES=${GPU}; log=${log}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${common_args[@]}" --output-dir "${out_dir}" "$@" 2>&1 | tee "${log}"
}

case "${MODE}" in
  24to4)
    run_stage f5tts_q4_4step_cfg2_original_teacher_progressive_v0 \
      --checkpoint "${TEACHER}" \
      --teacher-steps 24 \
      --student-checkpoint "${STUDENT_8STEP}" \
      --student-quant-scheme q4 \
      --student-steps 4 \
      --student-cfg-strength "${STUDENT_CFG_STRENGTH:-1.0}" \
      --cfg-strength 2.0 \
      --lr "${LR:-5e-8}" \
      --max-grad-norm "${MAX_GRAD_NORM:-0.35}" \
      --rollout-loss-weight 1.0 \
      --teacher-flow-loss-weight 0.0 \
      --trajectory-loss-weight 0.0 \
      --real-mel-loss-weight 0.1425 \
      --anchor-weight-loss-weight 0.10 \
      --temporal-delta-loss-weight 0.225 \
      --mel-energy-loss-weight 0.0 \
      --high-mel-excess-loss-weight 0.0 \
      --high-mel-match-loss-weight 0.025 \
      --loss-schedule-steps 200 \
      --rollout-loss-weight-final 1.0 \
      --teacher-flow-loss-weight-final 0.0 \
      --real-mel-loss-weight-final 0.18 \
      --anchor-weight-loss-weight-final 0.18 \
      --temporal-delta-loss-weight-final 0.30 \
      --mel-energy-loss-weight-final 0.0 \
      --high-mel-excess-loss-weight-final 0.0 \
      --high-mel-match-loss-weight-final 0.04 \
      "$@"
    ;;
  8to4)
    run_stage f5tts_q4_4step_cfg2_clean360_progressive_v0 \
      --checkpoint "${TEACHER_8TO4}" \
      --teacher-steps 8 \
      --student-checkpoint "${STUDENT_8STEP}" \
      --student-quant-scheme q4 \
      --student-steps 4 \
      --student-cfg-strength 2.0 \
      --cfg-strength 2.0 \
      --lr "${LR:-1e-7}" \
      --max-grad-norm "${MAX_GRAD_NORM:-0.5}" \
      --rollout-loss-weight 1.0 \
      --teacher-flow-loss-weight 0.0 \
      --trajectory-loss-weight 0.0 \
      --real-mel-loss-weight 0.1425 \
      --anchor-weight-loss-weight 0.08 \
      --temporal-delta-loss-weight 0.225 \
      --mel-energy-loss-weight 0.0 \
      --high-mel-excess-loss-weight 0.0 \
      --high-mel-match-loss-weight 0.025 \
      --loss-schedule-steps 200 \
      --rollout-loss-weight-final 1.0 \
      --teacher-flow-loss-weight-final 0.0 \
      --real-mel-loss-weight-final 0.18 \
      --anchor-weight-loss-weight-final 0.10 \
      --temporal-delta-loss-weight-final 0.30 \
      --mel-energy-loss-weight-final 0.0 \
      --high-mel-excess-loss-weight-final 0.0 \
      --high-mel-match-loss-weight-final 0.04 \
      "$@"
    ;;
  4to2)
    run_stage f5tts_q4_2step_cfg2_clean360_progressive_v0 \
      --checkpoint "${TEACHER_4TO2}" \
      --teacher-steps 4 \
      --student-checkpoint "${STUDENT_4STEP}" \
      --student-quant-scheme q4 \
      --student-steps 2 \
      --student-cfg-strength 2.0 \
      --cfg-strength 2.0 \
      --lr "${LR:-5e-8}" \
      --max-grad-norm "${MAX_GRAD_NORM:-0.35}" \
      --rollout-loss-weight 1.0 \
      --teacher-flow-loss-weight 0.05 \
      --trajectory-loss-weight 0.45 \
      --real-mel-loss-weight 0.18 \
      --anchor-weight-loss-weight 0.12 \
      --temporal-delta-loss-weight 0.38 \
      --mel-energy-loss-weight 0.06 \
      --high-mel-excess-loss-weight 0.02 \
      --high-mel-match-loss-weight 0.10 \
      --loss-schedule-steps 200 \
      --rollout-loss-weight-final 1.0 \
      --teacher-flow-loss-weight-final 0.03 \
      --real-mel-loss-weight-final 0.22 \
      --anchor-weight-loss-weight-final 0.16 \
      --temporal-delta-loss-weight-final 0.42 \
      --mel-energy-loss-weight-final 0.08 \
      --high-mel-excess-loss-weight-final 0.03 \
      --high-mel-match-loss-weight-final 0.12 \
      "$@"
    ;;
  bitnet2)
    BITNET_INCLUDE="${BITNET_INCLUDE:-${TRAIN_INCLUDE}}"
    run_stage f5tts_bitnet_2step_cfg2_clean360_progressive_v0 \
      --checkpoint "${TEACHER_BITNET2}" \
      --teacher-steps 2 \
      --student-checkpoint "${STUDENT_2STEP}" \
      --student-quant-scheme bitnet_qat \
      --bitnet-qat-learned-scale \
      --bitnet-scale-lr-multiplier "${BITNET_SCALE_LR_MULTIPLIER:-0.1}" \
      --q4-include "${BITNET_INCLUDE}" \
      --student-steps 2 \
      --student-cfg-strength 2.0 \
      --cfg-strength 2.0 \
      --lr "${LR:-2e-8}" \
      --max-grad-norm "${MAX_GRAD_NORM:-0.25}" \
      --rollout-loss-weight 1.0 \
      --teacher-flow-loss-weight 0.04 \
      --trajectory-loss-weight 0.40 \
      --real-mel-loss-weight 0.18 \
      --anchor-weight-loss-weight 0.18 \
      --temporal-delta-loss-weight 0.40 \
      --mel-energy-loss-weight 0.08 \
      --high-mel-excess-loss-weight 0.03 \
      --high-mel-match-loss-weight 0.12 \
      --loss-schedule-steps 200 \
      --rollout-loss-weight-final 1.0 \
      --teacher-flow-loss-weight-final 0.03 \
      --real-mel-loss-weight-final 0.22 \
      --anchor-weight-loss-weight-final 0.22 \
      --temporal-delta-loss-weight-final 0.44 \
      --mel-energy-loss-weight-final 0.10 \
      --high-mel-excess-loss-weight-final 0.04 \
      --high-mel-match-loss-weight-final 0.14 \
      "$@"
    ;;
  *)
    echo "Usage: GPU=2 MAX_STEPS=800 $0 [24to4|8to4|4to2|bitnet2] [extra distill args...]" >&2
    exit 2
    ;;
esac
