#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PG_DIR="${PG_DIR:-${ROOT_DIR}/other_repos/parameter-golf}"
PRESET="${PG_PRESET:-runtime_row_1024}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "${PG_DIR}"

common_env=(
  DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
  VOCAB_SIZE="${VOCAB_SIZE:-1024}"
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-4096}"
  YARN_MAX_LEN="${YARN_MAX_LEN:-4096}"
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-599}"
  VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
  TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
  WARMUP_STEPS="${WARMUP_STEPS:-1}"
  BITNET_GROUP_SIZE="${BITNET_GROUP_SIZE:-64}"
  BITNET_SCALE_LAYOUT="${BITNET_SCALE_LAYOUT:-runtime_row}"
  FP_STORAGE="${FP_STORAGE:-0}"
  SEQ_SCHEDULE_FRACTION="${SEQ_SCHEDULE_FRACTION:-0.33}"
  BATCH_SCHEDULE_FRACTION="${BATCH_SCHEDULE_FRACTION:-0.33}"
  MATRIX_LR="${MATRIX_LR:-0.04}"
  SCALAR_LR="${SCALAR_LR:-0.04}"
  TIED_EMBED_LR="${TIED_EMBED_LR:-0.05}"
  HEAD_LR="${HEAD_LR:-0.008}"
  MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-5}"
  LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30}"
  QK_GAIN_INIT="${QK_GAIN_INIT:-1.5}"
)

case "${PRESET}" in
  runtime_row_512)
    preset_env=(
      RUN_ID="${RUN_ID:-ternary_runtime_row_sp1024_ctx4096_8xh100_512}"
      MODEL_DIM="${MODEL_DIM:-512}"
      NUM_LAYERS="${NUM_LAYERS:-9}"
      NUM_HEADS="${NUM_HEADS:-8}"
      NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
      MLP_MULT="${MLP_MULT:-2}"
    )
    ;;
  runtime_row_1024)
    preset_env=(
      RUN_ID="${RUN_ID:-ternary_runtime_row_sp1024_ctx4096_8xh100_1024}"
      MODEL_DIM="${MODEL_DIM:-1024}"
      NUM_LAYERS="${NUM_LAYERS:-9}"
      NUM_HEADS="${NUM_HEADS:-16}"
      NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
      MLP_MULT="${MLP_MULT:-2}"
      TRAINING_DEPTH_RECURRENCE="${TRAINING_DEPTH_RECURRENCE:-1}"
      EVAL_DEPTH_RECURRENCE="${EVAL_DEPTH_RECURRENCE:-1}"
      ROPE_TYPE="${ROPE_TYPE:-yarn}"
    )
    ;;
  *)
    echo "Unknown PG_PRESET=${PRESET}; expected runtime_row_512 or runtime_row_1024" >&2
    exit 2
    ;;
esac

env "${common_env[@]}" "${preset_env[@]}" \
  torchrun --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py

