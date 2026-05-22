# Conversational ASR Dataset Plan

## Objective

Build the strongest lightweight English conversational ASR stack for Bddy. The
model must faithfully transcribe meetings and conversations with many
participants. It must not detect commands, coach, summarize, or perform product
logic. Those behaviors live downstream of the transcript.

## Dataset Tiers

### Tier 1: Real Meeting Validation

These datasets decide whether a model is better. Synthetic data is not allowed
to replace this tier.

| Dataset | Role | Notes |
| --- | --- | --- |
| `edinburghcstr/ami`, `sdm` | Primary hard validation | Room-mic meeting audio. This catches the failures users care about: missed uncertainty, dropped questions, substitutions, and far-field noise. |
| `edinburghcstr/ami`, `ihm` | Clean meeting validation/train | Individual-headset meeting audio. Useful to separate language/model quality from room acoustics. |
| `argmaxinc/icsi-meetings` | Meeting diarization stress | Public HF mirror exposes audio plus speaker/timestamp tracks. Use for segmentation/diarization stress; it does not expose plain ASR text in the inspected schema. |
| `diarizers-community/voxconverse` | Overlap/diarization stress | Public HF dataset exposes multi-speaker audio plus speaker/timestamp tracks. Use to stress overlap and speaker changes, not as a direct ASR text benchmark unless aligned transcripts are added. |

### Tier 2: Real Conversational Training

These expand acoustic and language coverage. They should improve robustness, but
promotion still depends on Tier 1.

| Dataset | Role | Notes |
| --- | --- | --- |
| `distil-whisper/earnings22`, `chunked` | Long-form business speech | Public chunked audio/transcription schema. Useful for meetings, sales, finance, and formal discussion vocabulary. |
| `speechcolab/gigaspeech` | Large broad ASR corpus | Gated. Good if access is configured; use for acoustic diversity, not final meeting proof. |
| `kensho/spgispeech` | Financial/business speech | Gated. Similar role to Earnings22, with commercial/business vocabulary. |
| `sanchit-gandhi/tedlium-data` | Clean public speaking | Useful for speaker/accent diversity and clean references. Less meeting-like. |

Use streaming for large row-level corpora so the training pipeline can sample
from many sources without downloading entire splits. Timestamped meeting-window
sources still need materialized rows so the builder can sort by meeting/time
before concatenating utterances.

### Tier 3: F5TTS Synthetic Stress Augmentation

Use `/data/agent_kernel_lite/artifacts/hf_releases/f5tts-4bit-distill` to render
controlled speech from many reference speakers, then mix it into hard meeting
scenes:

- overlapping speakers;
- interruptions and backchannels;
- speaker changes with short gaps;
- far-field/static/noise perturbations;
- high-value meeting vocabulary such as budget, timeline, risk, deadline,
  concern, objection, clarification, and uncertainty.

Synthetic data is useful because it creates targeted failure cases at scale, but
it is not a source of truth by itself. A checkpoint promoted from synthetic
training must still improve real AMI SDM and other real conversation gates.

## Current Local F5TTS Bundle

Local path:

```text
/data/agent_kernel_lite/artifacts/hf_releases/f5tts-4bit-distill
```

Expected files:

```text
manifest.json
tensor_q4_index.json
tensors.q4.bin
tensor_fp16_index.json
tensors.fp16.bin
F5TTS_Base_vocab.txt
peyton_voice_q4.tar
samples/
```

The F5 bundle is used to render utterance-level WAV files from text plus a
reference speaker. Those rendered utterances are then packed into Parquet and
mixed into synthetic meeting examples.

## Parquet-First Data Flow

No large JSONL datasets. Use Parquet for every train/eval artifact.

```text
real HF datasets
  -> teacher labels with scripts/build_whisper_teacher_dataset.py
  -> Parquet: audio, teacher_text, reference_text, duration_seconds

speaker reference clips + meeting text prompts
  -> F5 render jobs Parquet
  -> rendered utterance Parquet
  -> synthetic meeting Parquet

real + synthetic Parquet
  -> scripts/train_whisper_asr_lora.py
  -> candidate checkpoint
  -> scripts/eval_asr_quality.py promotion gates
```

## Recommended First Training Mix

Start conservative:

- 70% real meeting/conversational audio:
  - AMI IHM train;
  - AMI SDM train;
  - Earnings22 chunks;
  - optional GigaSpeech/SPGISpeech if gated access is available.
- 30% synthetic stress:
  - F5TTS rendered multi-speaker meetings;
  - overlap probability between `0.15` and `0.35`;
  - noise SNR between `8` and `24` dB;
  - turn gaps from `-0.8` to `1.5` seconds, where negative means overlap.

Do not over-weight synthetic data at first. If real AMI SDM WER worsens, reduce
synthetic weight or restrict synthetic to late training.

## Promotion Gates

Promote only if the candidate improves:

- AMI SDM mean and p90 WER;
- substitution and deletion rates;
- question recall;
- critical conversational term recall;
- repetition/hallucination rate;
- robustness under noise and overlap perturbation;
- worst-case manual review examples.

Rejected checkpoints are still useful as training evidence, but they must not
become the default runtime model.

## Fixed Eval Suite

The current blocker is not only training volume; it is also evaluation
stability. New checkpoints must be compared against the same held-out meeting
cases so we can tell whether a change truly improved transcription instead of
moving on a small ad hoc sample.

Build the v14 Parquet eval suite:

```bash
python scripts/build_bddy_asr_eval_suite.py \
  --output-dir /data/model/bddy-asr-eval/v14 \
  --short-limit-per-source 120 \
  --window-limit-per-source 60 \
  --window-seconds 18 \
  --window-min-words 16
```

Artifacts:

```text
/data/model/bddy-asr-eval/v14/ami_sdm_validation_short.parquet
/data/model/bddy-asr-eval/v14/ami_sdm_validation_windows.parquet
/data/model/bddy-asr-eval/v14/ami_ihm_validation_short.parquet
/data/model/bddy-asr-eval/v14/ami_ihm_validation_windows.parquet
/data/model/bddy-asr-eval/v14/manifest.json
```

The window files are timestamp and meeting-group aware. They should be used as
the primary promotion gate for coaching/JARVIS conversational transcription
because Bddy transcribes rolling conversation context, not isolated snippets.

Run a fixed-suite eval:

```bash
python scripts/eval_asr_quality.py \
  --model distil-whisper/distil-small.en \
  --dataset parquet \
  --config /data/model/bddy-asr-eval/v14/ami_sdm_validation_windows.parquet \
  --split train \
  --text-column text \
  --limit 60 \
  --max-new-tokens 192
```

Use `ami_sdm_validation_windows.parquet` as the hardest room-mic gate and
`ami_ihm_validation_windows.parquet` to separate language quality from room
acoustics. A model should not be promoted if it improves clean IHM but regresses
SDM room-mic meeting windows.
