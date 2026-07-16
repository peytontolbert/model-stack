#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/transformer_10}"
PYTHON="${PYTHON:-/home/peyton/miniconda3/envs/ai/bin/python}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the F5TTS checkpoint to render}"
LABEL="${LABEL:-candidate_nfe4_cfg085_speed1}"
OUT_DIR="${OUT_DIR:-${ROOT}/evals/f5tts_safe_gate_$(date -u +%Y%m%d_%H%M%S)}"
VOCAB="${VOCAB:-/data/resumebot/checkpoints/F5TTS_Base_vocab.txt}"
VOICE_PROFILE_DIR="${VOICE_PROFILE_DIR:-/data/resumebot/voice_profiles/Peyton}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-${ROOT}/evals/f5tts_voice_profile_suite_2step_textwin_20260523/teacher_fp32_12step/manifest.json}"

GEN_GPU="${GEN_GPU:-0}"
ASR_GPU="${ASR_GPU:-1}"
MIN_CUDA_TOTAL_GB="${MIN_CUDA_TOTAL_GB:-16}"
NFE_STEP="${NFE_STEP:-4}"
CFG_STRENGTH="${CFG_STRENGTH:-0.85}"
SPEED="${SPEED:-1.0}"
SEED="${SEED:-1337}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-large-v3-turbo}"
NORMALIZE_F5TTS_TEXT="${NORMALIZE_F5TTS_TEXT:-0}"
ASR_REFERENCE_FIELD="${ASR_REFERENCE_FIELD:-text}"

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GEN_GPU}" PYTHONPATH="${ROOT}:/data/resumebot" "${PYTHON}" "${ROOT}/scripts/eval_f5tts_voice_profile_suite.py" \
  --checkpoint "${CHECKPOINT}" \
  --vocab "${VOCAB}" \
  --voice-profile-dir "${VOICE_PROFILE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --label "${LABEL}" \
  --nfe-step "${NFE_STEP}" \
  --cfg-strength "${CFG_STRENGTH}" \
  --speed "${SPEED}" \
  --seed "${SEED}" \
  --device cuda \
  --min-cuda-total-gb "${MIN_CUDA_TOTAL_GB}" \
  $([[ "${NORMALIZE_F5TTS_TEXT}" == "1" ]] && printf '%s' "--normalize-f5tts-text")

CUDA_VISIBLE_DEVICES="${ASR_GPU}" PYTHONPATH="${ROOT}:/data/resumebot" "${PYTHON}" "${ROOT}/scripts/eval_f5tts_suite_gate.py" \
  --teacher-manifest "${TEACHER_MANIFEST}" \
  --candidate-manifest "${OUT_DIR}/${LABEL}/manifest.json" \
  --asr-model "${ASR_MODEL}" \
  --asr-device cuda \
  --asr-local-files-only \
  --require-candidate-min-cuda-gb "${MIN_CUDA_TOTAL_GB}" \
  --asr-reference-field "${ASR_REFERENCE_FIELD}" \
  > "${OUT_DIR}/${LABEL}/gate_vs_teacher.json"

jq '.candidate.summary' "${OUT_DIR}/${LABEL}/gate_vs_teacher.json"
