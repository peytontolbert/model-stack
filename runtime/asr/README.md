# ASR Runtime

`runtime.asr` is the model-stack speech recognition surface. It keeps audio
preprocessing, stream chunking, decode policy, and backend execution separate so
Whisper, Distil-Whisper, Parakeet-style encoders, ONNX Runtime, and native
custom kernels can share the same application-facing API.

Current surface:

- `AsrStreamState`: turns live mono PCM into fixed windows with overlap.
- `pcm_to_log_mel`: dependency-light PyTorch log-mel frontend for Whisper-like
  models.
- `AsrDecodeOptions` / `apply_asr_logit_policy`: token suppression and decode
  policy primitives.
- `AsrRuntime`: façade that feeds features into a pluggable backend.

Backend implementations should expose `transcribe_features(log_mel, options)`.
That keeps model-specific encoder/decoder kernels out of the application layer
and lets bddy switch between CPU, ONNX, CUDA, WebGPU, or custom kernels without
rewiring capture code.

## NeMo ASR Bridge

NeMo ASR catalog entries are local under `/arxiv/models` and include `.nemo`
archives. They should be validated in a dedicated `nemo_speech` conda env rather
than the general `ai` Diffusers env. Use
`scripts/smoke_nemo_asr_bridge.py <catalog-id>` for status-only checks and add
`--restore` only after `nemo_toolkit` imports cleanly.

## Training and Distillation

ASR training lives in scripts instead of the runtime package. The runtime package
defines the inference contract; the scripts build datasets, train adapters, and
write candidate checkpoints.

Current ASR scripts:

- `scripts/collect_conversational_asr_sources.py`: samples real HF ASR datasets
  into utterance and speaker-reference Parquet tables for F5TTS synthesis.
- `scripts/build_whisper_teacher_dataset.py`: runs a stronger Whisper teacher,
  rejects empty/repetitive/too-short/too-long transcripts, and writes a Parquet
  pseudo-label dataset with `audio`, `teacher_text`, `reference_text`, and
  `duration_seconds`.
- `scripts/train_whisper_asr_lora.py`: trains a LoRA student on Hugging Face or
  Parquet datasets, supports conversation windows, transcript quality filters,
  eval-only mode, WER reporting, and optional merged adapter output.
- `scripts/build_synthetic_meeting_asr_dataset.py`: writes Parquet F5TTS render
  jobs and mixes rendered utterances into overlapped/noisy synthetic meeting
  examples for ASR stress training.
- `scripts/render_f5tts_jobs_from_parquet.py`: consumes those F5 render jobs
  through the local AgentKernel Lite Q4 JS/WASM runtime and writes rendered
  utterance metadata back to Parquet.
- `scripts/filter_synthetic_asr_with_teacher.py`: transcribes synthetic rows
  with a stronger ASR teacher and rejects rows where rendered audio does not
  match the intended target text.

The LoRA trainer includes robustness augmentation for conversational audio:

- random gain
- additive Gaussian noise by SNR
- overlap mixing from another utterance
- conversation-window stitching using timing/group columns

Use these as training-time perturbations only. The runtime must continue to
receive clean PCM/log-mel inputs and must not depend on training augmentation
state.

Synthetic F5TTS meetings are also training-time data only. They are useful for
targeted overlap, interruption, speaker-change, and noise cases, but promotion
must still be decided on real conversational audio. See
`docs/conversational_asr_dataset_plan.md`.

Promotion remains metric-gated. A checkpoint is not a runtime default just
because training loss improves; it must improve hard conversational WER,
repetition/hallucination behavior, short-command smoke cases, and multi-turn
meeting transcript smoke cases. See
`docs/asr_teacher_distillation_plan.md` for the current Bddy ASR plan.
