# Bddy ASR Teacher Distillation Plan

## Goal

Bddy needs faithful conversational transcription for JARVIS and coaching. ASR must
produce the transcript only. Command detection, coaching decisions, and JARVIS
actions stay downstream.

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
3. Reject rows where the teacher hallucinated, repeated itself, or emitted empty
   text.
4. Promote only when the student improves WER and repetition rate on hard
   meeting/audio evals.
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

## Promotion Gate

Do not promote a checkpoint unless it improves all of:

- mean WER on AMI SDM room-mic validation;
- worst-case repetition/hallucination examples;
- JARVIS-style short command transcription smoke set;
- coaching-style multi-turn meeting transcript smoke set.

The latest plain LoRA and augmented LoRA runs did not pass this gate, so those
checkpoints are rejected.
