#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
PYTHON="${PYTHON:-/home/peyton/miniconda3/envs/ai/bin/python}"
ROOT="${ROOT:-/data/transformer_10}"
TEACHER="${TEACHER:-/data/resumebot/checkpoints/final_finetuned_model.pt}"
STUDENT_INIT="${STUDENT_INIT:-${ROOT}/checkpoints/f5tts_q4_current_best_to_2step_cfgfree_v0/model_q4_12to4_best.pt}"
VOCAB="${VOCAB:-/data/resumebot/checkpoints/F5TTS_Base_vocab.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/f5tts_bitnet_late_attention_2step_cfgfree_libritts_v0}"
LOG="${LOG:-${ROOT}/logs/f5tts_bitnet_late_attention_2step_cfgfree_libritts_v0_active.log}"
DATASET="${DATASET:-blabble-io/libritts_r}"
CONFIG="${CONFIG:-clean}"
SPLIT="${SPLIT:-train.clean.360}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
TEXT_COLUMN="${TEXT_COLUMN:-text_normalized}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_EVERY="${SAVE_EVERY:-100}"
WAIT_FREE_MB="${WAIT_FREE_MB:-9000}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

LATE_ATTENTION_QKVO="$(
  for block in 16 17 18 19 20 21; do
    printf 'transformer_blocks.%s.attn.to_q,transformer_blocks.%s.attn.to_k,transformer_blocks.%s.attn.to_v,transformer_blocks.%s.attn.to_out.0,' "$block" "$block" "$block" "$block"
  done | sed 's/,$//'
)"

gpu_free_mb() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v gpu="${GPU}" '$1 == gpu {gsub(/ /, "", $2); print $2; exit}'
}

if [[ "${WAIT_FREE_MB}" -gt 0 ]]; then
  while true; do
    free_mb="$(gpu_free_mb)"
    if [[ -z "${free_mb}" ]]; then
      echo "GPU ${GPU} was not found by nvidia-smi." >&2
      exit 1
    fi
    if [[ "${free_mb}" -ge "${WAIT_FREE_MB}" ]]; then
      break
    fi
    echo "GPU ${GPU} has ${free_mb} MiB free; waiting for ${WAIT_FREE_MB} MiB..."
    sleep "${WAIT_POLL_SECONDS}"
  done
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG}")"
cd "${ROOT}"

PYTHONPATH="${ROOT}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"${PYTHON}" scripts/distill_f5tts_12_to_4_q4.py \
  --checkpoint "${TEACHER}" \
  --student-checkpoint "${STUDENT_INIT}" \
  --vocab "${VOCAB}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --audio-column "${AUDIO_COLUMN}" \
  --text-column "${TEXT_COLUMN}" \
  --local-sample-prob 0.0 \
  --max-steps "${MAX_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --prune-step-checkpoints \
  --teacher-steps 24 \
  --student-steps 2 \
  --teacher-cfg-strength 2.0 \
  --student-cfg-strength 0.0 \
  --cfg-strength 2.0 \
  --detach-null-grad \
  --max-frames "${MAX_FRAMES:-256}" \
  --cond-frames "${COND_FRAMES:-192}" \
  --min-gen-frames "${MIN_GEN_FRAMES:-32}" \
  --lr "${LR:-1e-7}" \
  --weight-decay 0.0 \
  --max-grad-norm "${MAX_GRAD_NORM:-0.1}" \
  --student-quant-scheme bitnet_qat \
  --q4-include "${LATE_ATTENTION_QKVO}" \
  --train-include "${LATE_ATTENTION_QKVO}" \
  --rollout-loss-weight 1.0 \
  --teacher-flow-loss-weight 0.04 \
  --real-mel-loss-weight 0.16 \
  --anchor-weight-loss-weight 0.08 \
  --temporal-delta-loss-weight 0.32 \
  --mel-energy-loss-weight 0.02 \
  --high-mel-excess-loss-weight 0.0 \
  --high-mel-match-loss-weight 0.06 \
  --loss-schedule-steps 120 \
  --teacher-flow-loss-weight-final 0.02 \
  --real-mel-loss-weight-final 0.20 \
  --anchor-weight-loss-weight-final 0.10 \
  --temporal-delta-loss-weight-final 0.36 \
  --mel-energy-loss-weight-final 0.03 \
  --high-mel-excess-loss-weight-final 0.0 \
  --high-mel-match-loss-weight-final 0.08 \
  --device cuda 2>&1 | tee "${LOG}"
