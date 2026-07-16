#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
PYTHON="${PYTHON:-/home/peyton/miniconda3/envs/ai/bin/python}"
ROOT="${ROOT:-/data/transformer_10}"
TEACHER="${TEACHER:-/data/resumebot/checkpoints/final_finetuned_model.pt}"
START_CHECKPOINT="${START_CHECKPOINT:-${ROOT}/checkpoints/f5tts_q4_current_best_to_2step_cfgfree_v0_realmel_20260523/model_q4_12to4_best.pt}"
BEST_GATE="${BEST_GATE:-${ROOT}/evals/f5tts_q4_8step_release_regate_20260524/fullq4_surface_v2_nfe8_cfg2_speed115/gate_vs_teacher.json}"
RUN_NAME="${RUN_NAME:-f5tts_bitnet_2step_scale_gated_clean360_from_q4_2step_realmel_v0}"
PHASES="${PHASES:-8}"
STEPS_PER_PHASE="${STEPS_PER_PHASE:-100}"
MAX_FRAMES="${MAX_FRAMES:-256}"
COND_FRAMES="${COND_FRAMES:-192}"
LR="${LR:-2e-8}"

exec env CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON}" "${ROOT}/scripts/run_f5tts_scale_tracked_distill.py" \
    --run-name "${RUN_NAME}" \
    --teacher-checkpoint "${TEACHER}" \
    --start-checkpoint "${START_CHECKPOINT}" \
    --best-gate-json "${BEST_GATE}" \
    --dataset "${DATASET:-blabble-io/libritts_r}" \
    --config "${CONFIG:-clean}" \
    --split "${SPLIT:-train.clean.360}" \
    --audio-column "${AUDIO_COLUMN:-audio}" \
    --text-column "${TEXT_COLUMN:-text_normalized}" \
    --student-quant-scheme bitnet_qat \
    --bitnet-qat-learned-scale \
    --bitnet-scale-lr-multiplier "${BITNET_SCALE_LR_MULTIPLIER:-0.1}" \
    --student-steps 2 \
    --eval-nfe-step 2 \
    --eval-cfg-strength "${EVAL_CFG_STRENGTH:-0.5}" \
    --eval-speed "${EVAL_SPEED:-1.15}" \
    --teacher-steps 24 \
    --teacher-cfg-strength 2.0 \
    --student-cfg-strength "${STUDENT_CFG_STRENGTH:-0.0}" \
    --student-cfg-strengths "${STUDENT_CFG_STRENGTHS:-0.0,0.25,0.5}" \
    --phases "${PHASES}" \
    --steps-per-phase "${STEPS_PER_PHASE}" \
    --max-frames "${MAX_FRAMES}" \
    --cond-frames "${COND_FRAMES}" \
    --min-gen-frames "${MIN_GEN_FRAMES:-32}" \
    --shuffle-buffer "${SHUFFLE_BUFFER:-4096}" \
    --lr "${LR}" \
    --max-grad-norm "${MAX_GRAD_NORM:-0.25}" \
    --rollout-loss-weight "${ROLLOUT_LOSS_WEIGHT:-1.0}" \
    --teacher-flow-loss-weight "${TEACHER_FLOW_LOSS_WEIGHT:-0.08}" \
    --real-mel-loss-weight "${REAL_MEL_LOSS_WEIGHT:-0.20}" \
    --anchor-weight-loss-weight "${ANCHOR_WEIGHT_LOSS_WEIGHT:-0.20}" \
    --temporal-delta-loss-weight "${TEMPORAL_DELTA_LOSS_WEIGHT:-0.42}" \
    --mel-energy-loss-weight "${MEL_ENERGY_LOSS_WEIGHT:-0.08}" \
    --high-mel-excess-loss-weight "${HIGH_MEL_EXCESS_LOSS_WEIGHT:-0.03}" \
    --high-mel-match-loss-weight "${HIGH_MEL_MATCH_LOSS_WEIGHT:-0.12}" \
    --q4-include "${Q4_INCLUDE:-transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out}" \
    --q4-exclude "${Q4_EXCLUDE:-text_embed.text_embed,mel_spec}" \
    --q4-ste-include "${Q4_STE_INCLUDE:-transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out}" \
    --train-include "${TRAIN_INCLUDE:-transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out}" \
    --local-sample-prob 0.0 \
    --save-every "${SAVE_EVERY:-1000000000}" \
    --no-save-best \
    --max-promote-clipped-samples "${MAX_PROMOTE_CLIPPED_SAMPLES:-64}" \
    --max-promote-raw-clip-outputs "${MAX_PROMOTE_RAW_CLIP_OUTPUTS:-0}" \
    --cuda-visible-devices "${GPU}" \
    --device cuda \
    "$@"
