# Bddy Conversational ASR Plan

## Goal

Bddy needs the best possible faithful conversational speech transcription for
multi-participant meetings. ASR must produce the transcript only. Any coaching,
command detection, or assistant behavior stays downstream of transcription.

## What We Reused From AgentKernel Lite

`/data/agent_kernel_lite` has a useful multimodal training pattern:

- keep model-stack runtime kernels separate from product behavior;
- use teacher models to generate stronger targets;
- save compact artifacts that can be reused by lightweight students;
- reject checkpoints with behavior gates, not just lower training loss;
- keep WASM/Core ML/ONNX runtime paths separate from PyTorch training.

For ASR, that maps to:

1. Build a teacher-labeled Parquet dataset from strong Whisper models.
2. Train `distil-whisper/distil-small.en` or another lightweight student on the
   accepted teacher rows.
3. Reject rows where the teacher hallucinated, repeated itself, dropped critical
   conversational terms, or emitted empty text.
4. Promote only when the student improves meeting WER, deletion/substitution
   rates, conversational-term recall, and repetition rate on hard meeting/audio
   evals.
5. Convert and package only promoted checkpoints.

## Dataset Shape

The teacher dataset is Parquet, not JSONL:

```text
audio: Audio
teacher_text: string
reference_text: string
duration_seconds: float
```

Use `teacher_text` as the training target when the teacher is stronger than the
provided transcript, especially for noisy conversational sources.

## Commands

Build teacher labels:

```bash
python scripts/build_whisper_teacher_dataset.py \
  --teacher-model openai/whisper-large-v3-turbo \
  --dataset 'edinburghcstr/ami:ihm:train[:1000]:text' \
  --dataset 'edinburghcstr/ami:sdm:train[:1000]:text' \
  --output /data/model/bddy-whisper-teacher/ami_teacher_v1.parquet
```

Train a lightweight student from the teacher Parquet:

```bash
python scripts/train_whisper_asr_lora.py \
  --model distil-whisper/distil-small.en \
  --dataset 'parquet:/data/model/bddy-whisper-teacher/ami_teacher_v1.parquet:train:teacher_text' \
  --eval-dataset 'edinburghcstr/ami:sdm:validation[:1500]:text' \
  --limit-per-dataset 1000 \
  --eval-limit-per-dataset 100 \
  --eval-samples 80 \
  --min-words 4 \
  --max-duration-seconds 24 \
  --max-steps 300 \
  --learning-rate 3e-6 \
  --output-dir /data/model/bddy-distil-small-en-teacher-lora-v1
```

Train with conversational robustness augmentation:

```bash
python scripts/train_whisper_asr_lora.py \
  --model distil-whisper/distil-small.en \
  --dataset 'parquet:/data/model/bddy-whisper-teacher/ami_teacher_v1.parquet:train:teacher_text' \
  --eval-dataset 'edinburghcstr/ami:sdm:validation[:1500]:text' \
  --conversation-window-seconds 24 \
  --conversation-window-gap-seconds 1.0 \
  --conversation-window-group-columns meeting_id,speaker_id \
  --conversation-window-min-words 8 \
  --train-gain-db-min -6 \
  --train-gain-db-max 6 \
  --train-noise-prob 0.35 \
  --train-noise-snr-db-min 12 \
  --train-noise-snr-db-max 28 \
  --train-overlap-prob 0.12 \
  --train-overlap-gain-db-min -20 \
  --train-overlap-gain-db-max -10 \
  --max-steps 300 \
  --learning-rate 3e-6 \
  --output-dir /data/model/bddy-distil-small-en-teacher-lora-robust-v1
```

Use `--eval-only` to measure a base model or candidate adapter against the
prepared eval split without running a training loop.

Run a meeting-focused ASR eval:

```bash
python scripts/eval_asr_quality.py \
  --model distil-whisper/distil-small.en \
  --dataset edinburghcstr/ami \
  --config sdm \
  --split 'validation[:1500]' \
  --text-column text \
  --limit 80 \
  --min-words 4 \
  --min-duration-seconds 1.0 \
  --max-duration-seconds 24
```

The meeting eval reports WER plus substitution, deletion, insertion,
repetition, question recall, and conversational critical-term recall. These
metrics matter because they measure whether the transcript preserves the actual
conversation, including uncertainty, objections, clarifications, and questions.

## Promotion Gate

Do not promote a checkpoint unless it improves all of:

- mean WER on AMI SDM room-mic validation;
- deletion and substitution rates on room-mic meeting turns;
- question recall on multi-speaker meeting turns;
- conversational critical-term recall for uncertainty, objections, risks, budget,
  timeline, clarification, and confusion;
- worst-case repetition/hallucination examples;
- speaker/stream-aware multi-turn meeting transcript smoke set.

The latest plain LoRA and augmented LoRA runs did not pass this gate, so those
checkpoints are rejected. The current baseline `distil-whisper/distil-small.en`
on a filtered AMI SDM meeting slice is still around `0.43` WER, so meeting ASR
quality remains the active blocker.

## Runtime Boundary

The teacher dataset and LoRA trainer are not runtime dependencies. Promoted
checkpoints should enter the runtime through a packaged backend that implements
`transcribe_features(log_mel, options)`. Product behavior such as command
detection, coaching decisions, and assistant actions must stay downstream of the
ASR transcript.
