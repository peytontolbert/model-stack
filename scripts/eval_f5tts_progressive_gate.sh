#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
PYTHON="${PYTHON:-/home/peyton/miniconda3/envs/ai/bin/python}"
ROOT="${ROOT:-/data/transformer_10}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the checkpoint to sample}"
STAGE="${STAGE:-4step}"
NFE_STEP="${NFE_STEP:-4}"
CFG_STRENGTH="${CFG_STRENGTH:-2.0}"
SWAY_SAMPLING_COEF="${SWAY_SAMPLING_COEF:--1.0}"
SEED="${SEED:-1234}"
VOCAB="${VOCAB:-/data/resumebot/checkpoints/F5TTS_Base_vocab.txt}"
TEACHER_WAV="${TEACHER_WAV:-${ROOT}/evals/f5tts_voice_profiles/peyton_teacher_24step_cfg2_seed1234_20260522.wav}"
BASELINE_WAV="${BASELINE_WAV:-${ROOT}/evals/f5tts_voice_profiles/peyton_q4_8step_clean360_v4_energy_best_same_text_20260522.wav}"
OUT_DIR="${OUT_DIR:-${ROOT}/evals/f5tts_voice_profiles}"
TEXT="${TEXT:-This is Peyton speaking from Agent Kernel Lite. This is the current ${STAGE} F5 TTS student model running with the Peyton voice profile.}"
SLUG="${SLUG:-peyton_${STAGE}_progressive_seed${SEED}_$(date -u +%Y%m%d_%H%M%S)}"
WAV="${OUT_DIR}/${SLUG}.wav"

mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT}:/data/resumebot" "${PYTHON}" "${ROOT}/scripts/eval_f5tts_voice_profile.py" \
  --checkpoint "${CHECKPOINT}" \
  --vocab "${VOCAB}" \
  --output-path "${WAV}" \
  --text "${TEXT}" \
  --nfe-step "${NFE_STEP}" \
  --cfg-strength "${CFG_STRENGTH}" \
  --sway-sampling-coef "${SWAY_SAMPLING_COEF}" \
  --seed "${SEED}" \
  --device cuda

HF_HOME="${HF_HOME:-/data/.cache/huggingface}" \
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/.cache/huggingface/hub}" \
PYTHONPATH="${ROOT}" \
"${PYTHON}" "${ROOT}/scripts/eval_f5tts_audio_gate.py" \
  --teacher "${TEACHER_WAV}" \
  --candidate "${WAV}" \
  --baseline "${BASELINE_WAV}" \
  --text "${TEXT}" \
  --asr-model "${ASR_MODEL:-openai/whisper-tiny.en}" \
  --asr-device cuda \
  --asr-local-files-only

echo "sample=${WAV}"
