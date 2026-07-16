#!/usr/bin/env bash
set -eo pipefail

source /home/peyton/miniconda3/etc/profile.d/conda.sh
conda activate ai

export CUDA_VISIBLE_DEVICES=0

python /data/transformer_10/scripts/run_f5tts_scale_tracked_distill.py \
  --run-name f5tts_q4_6step_from_v10_scale_refine_v31_3090 \
  --start-checkpoint /data/transformer_10/checkpoints/f5tts_q4_6step_hardterm_targeted_v10/model_q4_12to6_best_val_rollout.pt \
  --teacher-checkpoint /data/resumebot/checkpoints/final_finetuned_model.pt \
  --teacher-manifest /data/agent_kernel_lite/tts_quality_samples/20260526_6step_teacher24_v14_eval/teacher_fp32_24step_cfg2/manifest.json \
  --best-gate-json /data/transformer_10/evals/f5tts_6step_broad_eval_20260526/hardterm_targeted_v10_bestval_nfe6_cfg125_broad/gate_vs_teacher.json \
  --dataset blabble-io/libritts_r \
  --config all \
  --split train.clean.100,train.clean.360,train.other.500 \
  --split-probabilities 0.25,0.45,0.30 \
  --text-column text_normalized \
  --audio-column audio \
  --student-quant-scheme q4 \
  --teacher-steps 24 \
  --student-steps 6 \
  --teacher-cfg-strength 2.0 \
  --student-cfg-strength 1.15 \
  --student-cfg-strengths 1.05,1.15,1.25 \
  --phases 3 \
  --steps-per-phase 180 \
  --batch-size 1 \
  --lr 8e-8 \
  --max-grad-norm 0.07 \
  --weight-decay 0.0 \
  --rollout-loss-weight 0.96 \
  --teacher-flow-loss-weight 0.035 \
  --real-mel-loss-weight 0.10 \
  --anchor-weight-loss-weight 0.58 \
  --temporal-delta-loss-weight 0.42 \
  --mel-energy-loss-weight 0.01 \
  --energy-envelope-loss-weight 0.03 \
  --silence-envelope-loss-weight 0.006 \
  --high-mel-match-loss-weight 0.05 \
  --high-mel-excess-loss-weight 0.008 \
  --high-mel-ratio-loss-weight 0.08 \
  --high-mel-temporal-loss-weight 0.22 \
  --high-mel-start-bin 56 \
  --low-mid-mel-body-loss-weight 0.045 \
  --low-mid-mel-end-bin 72 \
  --text-delta-loss-weight 0.012 \
  --eval-nfe-step 6 \
  --eval-cfg-strength 1.25 \
  --eval-speed 1.0 \
  --eval-seed 1337 \
  --eval-text-file /data/transformer_10/data/f5tts_peyton_heldout_broad_eval_v0.txt \
  --normalize-f5tts-text \
  --cuda-visible-devices 0 \
  --seed 20260526 \
  --chain-phases \
  --must-beat-teacher \
  --max-promote-mean-wer 0.18 \
  --max-promote-mean-phonetic-wer 0.16 \
  --no-save-best
