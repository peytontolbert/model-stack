#!/usr/bin/env bash
set -eo pipefail

source /home/peyton/miniconda3/etc/profile.d/conda.sh
conda activate ai

export CUDA_VISIBLE_DEVICES=0

python /data/transformer_10/scripts/train_f5tts_2step_from_teacher_cache.py \
  --cache-dir /data/transformer_10/cache/f5tts_voiceclone_libritts_full_hardphrase_teacher24_normgen_v0 \
  --student-checkpoint /data/transformer_10/checkpoints/f5tts_q4_6step_hardterm_targeted_v10/model_q4_12to6_best_val_rollout.pt \
  --vocab /data/resumebot/checkpoints/F5TTS_Base_vocab.txt \
  --output-dir /data/transformer_10/checkpoints/f5tts_q4_6step_teacher24_normgen_text_contrast_v34 \
  --teacher-steps 24 \
  --student-steps 6 \
  --student-quant-scheme q4 \
  --q4-exclude text_embed.text_embed,mel_spec \
  --q4-ste-include transformer.transformer_blocks.20,transformer.transformer_blocks.21 \
  --train-include transformer.text_embed,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out \
  --train-exclude "" \
  --checkpoint-activations \
  --train-in-eval-mode \
  --student-cfg-strength 1.25 \
  --student-cfg-strengths 1.05,1.15,1.25 \
  --sway-sampling-coef -1.0 \
  --path-target teacher_trajectory \
  --steps 260 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --lr 8e-8 \
  --weight-decay 0.0 \
  --max-grad-norm 0.06 \
  --bridge-flow-loss-weight 0.9 \
  --teacher-instant-flow-loss-weight 0.02 \
  --bridge-direction-loss-weight 0.22 \
  --bridge-norm-loss-weight 0.05 \
  --next-state-loss-weight 4.0 \
  --text-flow-contrastive-loss-weight 0.10 \
  --text-flow-contrastive-margin 0.08 \
  --hard-negative-text-prob 1.0 \
  --anchor-weight-loss-weight 0.36 \
  --final-rollout-loss-weight 0.0 \
  --final-rollout-loss-weight-final 0.0 \
  --temporal-delta-loss-weight 0.0 \
  --frame-energy-loss-weight 0.0 \
  --high-band-match-loss-weight 0.0 \
  --loss-schedule-steps 160 \
  --log-every 10 \
  --save-every 80 \
  --val-every 65 \
  --val-fraction 0.05 \
  --val-max-rows 24 \
  --device cuda \
  --seed 2026052605
