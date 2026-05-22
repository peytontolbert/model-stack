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

See `docs/conversational_asr_dataset_plan.md` for the broader dataset map,
including AMI, ICSI, VoxConverse, Earnings22, gated large-corpus options, and
the F5TTS synthetic meeting augmentation path.

## Commands

Build teacher labels:

```bash
python scripts/build_whisper_teacher_dataset.py \
  --teacher-model openai/whisper-large-v3-turbo \
  --dataset 'edinburghcstr/ami:ihm:train[:1000]:text' \
  --dataset 'edinburghcstr/ami:sdm:train[:1000]:text' \
  --output /data/model/bddy-whisper-teacher/ami_teacher_v1.parquet
```

Build teacher labels for meeting windows. This is the preferred path for bddy
coaching/JARVIS ASR because the app transcribes rolling conversation context,
not only isolated one-second fragments:

```bash
python scripts/build_whisper_teacher_dataset.py \
  --teacher-model openai/whisper-large-v3-turbo \
  --dataset 'edinburghcstr/ami:sdm:train[:3000]:text' \
  --output /data/model/bddy-whisper-teacher/ami_sdm_window_large_turbo_teacher_v1.parquet \
  --limit-per-dataset 120 \
  --conversation-window-seconds 18 \
  --conversation-window-gap-seconds 1.0 \
  --conversation-window-min-words 20 \
  --min-words 8 \
  --min-duration-seconds 5 \
  --max-duration-seconds 24 \
  --max-new-tokens 192
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

Run the same eval against synthetic meeting Parquet:

```bash
python scripts/eval_asr_quality.py \
  --model distil-whisper/distil-small.en \
  --dataset parquet \
  --config /data/model/bddy-synthetic-asr/synthetic_meetings.parquet \
  --split train \
  --text-column text \
  --limit 80 \
  --min-words 4 \
  --max-duration-seconds 24
```

The meeting eval reports WER plus substitution, deletion, insertion,
repetition, question recall, and conversational critical-term recall. These
metrics matter because they measure whether the transcript preserves the actual
conversation, including uncertainty, objections, clarifications, and questions.

Run the full bddy teacher-window experiment loop. This is the preferred
repeatable environment for improving transcription accuracy because it creates
teacher-cleaned meeting windows, trains the LoRA student, merges it, and writes
short/window promotion reports under one run directory:

```bash
python scripts/run_bddy_asr_teacher_window_experiment.py \
  --run-name medium_teacher_windows_v1 \
  --source-dataset 'edinburghcstr/ami:sdm:train[:8000]:text' \
  --limit-windows 500 \
  --train-steps 180
```

Use multiple sources for a more diverse conversational run. Timestamped meeting
datasets such as AMI SDM/IHM can become windows; long-form chunked corpora such
as Earnings22 are included as row-level teacher examples when they do not expose
AMI-style timestamps:

```bash
python scripts/run_bddy_asr_teacher_window_experiment.py \
  --run-name medium_diverse_teacher_windows_v1 \
  --source-dataset 'edinburghcstr/ami:sdm:train[:8000]:text' \
  --source-dataset 'edinburghcstr/ami:ihm:train[:8000]:text' \
  --source-dataset 'distil-whisper/earnings22:chunked:test[:5000]:transcription' \
  --limit-windows 500 \
  --train-steps 180
```

For very large row-level corpora, stream from Hugging Face instead of
materializing a full split. Streaming is intentionally row-level only; timestamp
window building still materializes rows so they can be sorted by meeting/time:

```bash
python scripts/build_whisper_teacher_dataset.py \
  --streaming \
  --teacher-model openai/whisper-large-v3-turbo \
  --dataset 'distil-whisper/earnings22:chunked:test:transcription' \
  --output /data/model/bddy-whisper-teacher/earnings22_stream_teacher.parquet \
  --limit-per-dataset 1000 \
  --conversation-window-seconds 0 \
  --min-words 8 \
  --min-duration-seconds 5 \
  --max-duration-seconds 24
```

Build F5TTS render jobs without JSONL:

```bash
python scripts/collect_conversational_asr_sources.py \
  --dataset 'edinburghcstr/ami:ihm:train[:5000]:text:audio:speaker_id' \
  --dataset 'distil-whisper/earnings22:chunked:test[:5000]:transcription:audio:file_id' \
  --utterances-output /data/model/bddy-synthetic-asr/source_utterances.parquet \
  --speaker-refs-output /data/model/bddy-synthetic-asr/speaker_refs.parquet \
  --speaker-ref-dir /data/model/bddy-synthetic-asr/speaker_refs \
  --limit-per-dataset 5000 \
  --min-duration-seconds 7 \
  --max-duration-seconds 12

python scripts/build_synthetic_meeting_asr_dataset.py plan-f5 \
  --texts-parquet /data/model/bddy-synthetic-asr/source_utterances.parquet \
  --speaker-reference-manifest /data/model/bddy-synthetic-asr/speaker_refs.parquet \
  --output /data/model/bddy-synthetic-asr/f5_render_jobs.parquet \
  --render-output-dir /data/model/bddy-synthetic-asr/rendered_wavs \
  --count 10000 \
  --shuffle-texts \
  --shuffle-speakers
```

After rendering those jobs with the local AgentKernel Lite F5TTS Q4 runtime,
pack the rendered utterance paths into Parquet and mix hard meeting examples:

```bash
python scripts/render_f5tts_jobs_from_parquet.py \
  --jobs /data/model/bddy-synthetic-asr/f5_render_jobs.parquet \
  --output /data/model/bddy-synthetic-asr/rendered_utterances.parquet \
  --limit 10000 \
  --steps 24

python scripts/build_synthetic_meeting_asr_dataset.py mix-rendered \
  --rendered-utterances /data/model/bddy-synthetic-asr/rendered_utterances.parquet \
  --output /data/model/bddy-synthetic-asr/synthetic_meetings.parquet \
  --audio-output-dir /data/model/bddy-synthetic-asr/meeting_wavs \
  --meetings 5000 \
  --overlap-probability 0.25 \
  --noise-probability 0.35

python scripts/filter_synthetic_asr_with_teacher.py \
  --input /data/model/bddy-synthetic-asr/synthetic_meetings.parquet \
  --output /data/model/bddy-synthetic-asr/synthetic_meetings_verified.parquet \
  --rejected-output /data/model/bddy-synthetic-asr/synthetic_meetings_rejected.parquet \
  --teacher-model openai/whisper-large-v3-turbo \
  --max-wer 0.25
```

Only train on `synthetic_meetings_verified.parquet`. F5TTS renders are
augmentation candidates, not trusted labels. If the teacher transcript does not
match the intended target closely, the row is rejected so it cannot poison ASR
training.

For F5TTS specifically, use original-style `24` step generation when building
quality ASR augmentation and use 7-12 second speaker reference clips. Shorter
speaker clips are useful for smoke tests only; they are not a fair TTS quality
check.

For a stronger no-TTS baseline, build difficult conversation mixtures directly
from real LibriTTS utterances. This keeps exact labels and avoids synthetic
voice mismatch:

```bash
python scripts/collect_conversational_asr_sources.py \
  --dataset 'mythicinfinity/libritts:clean:train.clean.100:text_normalized:audio:speaker_id' \
  --utterances-output /data/model/bddy-real-mix-asr/libritts_7_12s_utterances.parquet \
  --speaker-refs-output /data/model/bddy-real-mix-asr/libritts_7_12s_speaker_refs.parquet \
  --speaker-ref-dir /data/model/bddy-real-mix-asr/speaker_refs \
  --utterance-audio-dir /data/model/bddy-real-mix-asr/utterance_wavs \
  --limit-per-dataset 20000 \
  --max-utterances-per-speaker 20 \
  --min-duration-seconds 7 \
  --max-duration-seconds 12

python scripts/build_synthetic_meeting_asr_dataset.py mix-rendered \
  --rendered-utterances /data/model/bddy-real-mix-asr/libritts_7_12s_utterances.parquet \
  --output /data/model/bddy-real-mix-asr/libritts_conversation_mix.parquet \
  --audio-output-dir /data/model/bddy-real-mix-asr/mixed_wavs \
  --meetings 10000 \
  --min-turns 3 \
  --max-turns 8 \
  --prefer-distinct-speakers \
  --source-label libritts_real_conversation_mix \
  --overlap-probability 0.3 \
  --noise-probability 0.35
```

This real-audio mixture is a better first ASR training source than F5TTS when
the goal is robust transcription under overlap, speaker changes, and noise.

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

See `docs/asr_training_runs.md` for the current run ledger. The latest
LibriTTS real-conversation LoRA attempts, `v5` and `v6`, are rejected because
they failed to improve AMI SDM room-mic validation.

## Runtime Boundary

The teacher dataset and LoRA trainer are not runtime dependencies. Promoted
checkpoints should enter the runtime through a packaged backend that implements
`transcribe_features(log_mel, options)`. Product behavior such as command
detection, coaching decisions, and assistant actions must stay downstream of the
ASR transcript.
