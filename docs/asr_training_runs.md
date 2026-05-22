# ASR Training Runs

## Current Rule

Do not promote an ASR checkpoint unless it improves real conversational audio,
especially AMI SDM room-mic validation. Synthetic or mixed data can be useful
for training pressure, but it is not enough by itself.

## Real LibriTTS Conversation Mix V1

Source:

```text
/data/model/bddy-real-mix-asr/libritts_7_12s_utterances_v1.parquet
/data/model/bddy-real-mix-asr/libritts_conversation_mix_v2.parquet
```

Build properties:

- 600 real LibriTTS utterances;
- 57 speakers;
- utterance duration: 7-12 seconds;
- 500 mixed conversation rows;
- 2-3 turns per row;
- distinct speakers preferred;
- overlap probability: 0.35;
- noise probability: 0.35.

This data path is valid because it uses real speech and exact text labels. It
is a better first ASR augmentation source than unverified F5TTS output.

## Rejected: `bddy-distil-small-conversation-lora-v5`

Output:

```text
/data/model/bddy-distil-small-conversation-lora-v5
/data/model/bddy-distil-small-conversation-lora-v5-merged
```

Training summary:

- base model: `distil-whisper/distil-small.en`;
- train rows after filtering: 1,276;
- LoRA target modules: `q_proj,v_proj`;
- LoRA rank: 8;
- trainable params: 491,520;
- max steps: 180;
- learning rate: `5e-6`.

Internal held-out mix eval:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.0394 | 0.0398 |

External AMI SDM eval, 30 filtered rows:

| Metric | Base | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4412 | 0.4412 |
| median WER | 0.4226 | 0.4226 |
| p90 WER | 0.8333 | 0.8333 |
| substitution rate | 0.1789 | 0.1789 |
| deletion rate | 0.1320 | 0.1320 |
| insertion rate | 0.0762 | 0.0762 |

Decision: reject. The adapter did not improve the promotion gate.

## Rejected: `bddy-distil-small-conversation-lora-v6`

Output:

```text
/data/model/bddy-distil-small-conversation-lora-v6
/data/model/bddy-distil-small-conversation-lora-v6-merged
```

Training summary:

- base model: `distil-whisper/distil-small.en`;
- train rows after filtering: 1,416;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj`;
- LoRA rank: 16;
- trainable params: 1,966,080;
- max steps: 220;
- learning rate: `2e-5`.

Internal held-out mix eval:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.0399 | 0.0401 |

External AMI SDM eval, 30 filtered rows:

| Metric | Base | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4412 | 0.4467 |
| median WER | 0.4226 | 0.4286 |
| p90 WER | 0.8333 | 0.8333 |
| substitution rate | 0.1789 | 0.1848 |
| deletion rate | 0.1320 | 0.1144 |
| insertion rate | 0.0762 | 0.0850 |

Decision: reject. The adapter moved the model, but it worsened AMI SDM mean WER
and substitution rate.

## Next Training Direction

The two LoRA runs show that generic LibriTTS conversation mixing alone does not
fix AMI room-mic conversational errors. The next useful run should focus on
real room-mic meeting data:

1. Build a larger teacher-labeled AMI SDM Parquet using
   `openai/whisper-large-v3-turbo`.
2. Keep LibriTTS conversation mixes at a low training weight.
3. Train against teacher text for AMI SDM, not just the original short
   normalized transcripts.
4. Evaluate on held-out AMI SDM meetings that are not used in teacher labeling.
5. Promote only if AMI SDM improves without hurting repetition rate or question
   recall.

## Rejected: `bddy-distil-small-ami-teacher-lora-v7`

Teacher data:

```text
/data/model/bddy-whisper-teacher/ami_sdm_large_turbo_teacher_v1.parquet
```

Teacher-labeling summary:

- teacher: `openai/whisper-large-v3-turbo`;
- source: AMI SDM train slice;
- requested rows: 450;
- accepted rows: 304;
- rejected as too short: 146.

Training summary:

- base model: `distil-whisper/distil-small.en`;
- train rows after filtering: 439;
- target text: `teacher_text` for AMI SDM pseudo-labels;
- LibriTTS real conversation mix included as a smaller robustness component;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj`;
- LoRA rank: 16;
- trainable params: 1,966,080;
- max steps: 180;
- learning rate: `1e-5`.

Internal held-out eval:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.2256 | 0.2231 |

External AMI SDM eval, 40 filtered rows:

| Metric | Base | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4581 | 0.4581 |
| median WER | 0.4524 | 0.4524 |
| p90 WER | 0.8000 | 0.8000 |
| substitution rate | 0.1812 | 0.1791 |
| deletion rate | 0.1365 | 0.1386 |
| insertion rate | 0.0682 | 0.0682 |
| question recall | 0.7778 | 0.7778 |

Decision: reject. The adapter slightly improved the trainer's mixed internal
eval, but it did not improve the external AMI SDM promotion gate.

## Updated Next Direction

The teacher-label run did not improve the external room-mic gate. The next
accuracy work should stop short-utterance LoRA on `distil-small.en` and focus on
one of these:

1. Evaluate a stronger runtime candidate, such as `distil-whisper/distil-medium.en`
   or `openai/whisper-small.en`, on the same AMI SDM gate before more training.
2. Build meeting windows from AMI SDM instead of individual short utterances, so
   training and eval match 10-25 second conversational context.
3. Train only after teacher labels are filtered by agreement with original AMI
   text or manually reviewed meeting windows, because large-v3-turbo still
   mistranscribes some short room-mic fragments.

## Baseline Model Comparison

External AMI SDM eval, 40 filtered rows:

| Model | Params | Mean WER | Median WER | P90 WER | Median Latency | Question Recall | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `distil-whisper/distil-small.en` | 166.1M | 0.4581 | 0.4524 | 0.8000 | 2.67s | 0.7778 | Current lightweight baseline. |
| `distil-whisper/distil-medium.en` | 394.4M | 0.4449 | 0.4226 | 0.7500 | 6.12s | 0.7778 | Best accuracy/latency tradeoff tested so far. |
| `openai/whisper-small.en` | 241.7M | 0.4582 | 0.4584 | 0.8000 | 3.32s | 0.7778 | No meaningful improvement over distil-small. |
| `openai/whisper-medium.en` | 763.9M | 0.4548 | 0.4226 | 0.9000 | 9.38s | 0.7222 | Slower and not better on this gate. |

Decision: `distil-whisper/distil-medium.en` is the strongest runtime candidate
tested so far for desktop-quality ASR. It improves mean WER and p90 WER versus
`distil-small.en`, but roughly doubles latency. It should be treated as an
optional higher-accuracy desktop mode, not a mobile/default runtime.

## Candidate: `bddy-distil-medium-ami-teacher-lora-v8`

Training data:

```text
/data/model/bddy-whisper-teacher/ami_sdm_large_turbo_teacher_v1.parquet
/data/model/bddy-real-mix-asr/libritts_conversation_mix_v2.parquet
```

Training summary:

- base model: `distil-whisper/distil-medium.en`;
- target text: `teacher_text` for AMI SDM pseudo-labels, `text` for LibriTTS
  real conversation mixes;
- train rows after filtering: 439;
- eval rows after filtering: 97;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj`;
- LoRA rank: 8;
- trainable params: 1,835,008;
- max steps: 140;
- learning rate: `8e-6`;
- finished epoch: 5.0;
- merged model: `/data/model/bddy-distil-medium-ami-teacher-lora-v8-merged`.

External AMI SDM eval, 40 filtered rows:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4449 | 0.4401 |
| median WER | 0.4226 | 0.4226 |
| p90 WER | 0.7500 | 0.7500 |
| median latency | 5.65s | 5.42s |
| p90 latency | 6.05s | 6.13s |
| substitution rate | 0.1450 | 0.1450 |
| deletion rate | 0.1727 | 0.1706 |
| insertion rate | 0.0384 | 0.0405 |
| question recall | 0.7778 | 0.7778 |

Decision: keep as a candidate. This is the first tuning run that improved the
external AMI SDM promotion gate, but the gain is small. It is acceptable for
continued desktop-quality testing, while the production default should still be
selected by runtime budget and real app measurements.

AMI SDM conversation-window smoke eval, 6 windows from `validation[:600]`:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4951 | 0.4951 |
| median WER | 0.4900 | 0.4900 |
| p90 WER | 0.6977 | 0.6977 |
| median latency | 6.42s | 6.60s |
| question recall | 0.6250 | 0.6250 |

The candidate does not improve the longer-window smoke test. This reinforces
that the next training run should use meeting windows directly instead of only
isolated short utterances.

Next accuracy work:

1. Add AMI conversation-window training/eval instead of relying on isolated
   short utterances, because bddy streams transcript windows during coaching and
   JARVIS.
2. Evaluate on a separate long-form business/conversation set such as
   `distil-whisper/earnings22` to ensure the adapter is not overfit to AMI.
3. Keep F5TTS-generated rows out of ASR training unless they pass teacher
   verification against the intended transcript.

## Rejected: `bddy-distil-medium-ami-window-lora-v9`

Training data:

```text
edinburghcstr/ami:sdm:train[:2500]:text
```

Training summary:

- base model: `distil-whisper/distil-medium.en`;
- timestamp-aware AMI windows up to 18 seconds;
- training rows after filtering: 177;
- eval rows after filtering: 39;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj`;
- LoRA rank: 8;
- trainable params: 1,835,008;
- max steps: 90;
- learning rate: `5e-6`;
- mild gain/noise augmentation plus 12% low-volume overlap augmentation;
- merged model: `/data/model/bddy-distil-medium-ami-window-lora-v9-merged`.

Internal trainer window eval, 12 samples:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.3934 | 0.3934 |
| median WER | 0.3611 | 0.3611 |

AMI SDM conversation-window smoke eval, 6 windows from `validation[:600]`:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4951 | 0.4951 |
| median WER | 0.4900 | 0.4900 |
| p90 WER | 0.6977 | 0.6977 |
| median latency | 6.71s | 6.65s |
| question recall | 0.6250 | 0.6250 |

External AMI SDM short-utterance eval, 40 filtered rows:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4449 | 0.4481 |
| median WER | 0.4226 | 0.4226 |
| p90 WER | 0.7500 | 0.7500 |
| median latency | 5.50s | 5.56s |
| question recall | 0.7778 | 0.7778 |

Decision: reject. The window-trained adapter did not improve the window eval and
regressed the short-utterance promotion gate. The likely issue is that the
training target uses noisy AMI room-mic labels directly; future window training
should use higher-quality teacher labels or a cleaner meeting corpus.

## Not Promoted: `bddy-distil-medium-ami-window-teacher-lora-v10`

Teacher data:

```text
/data/model/bddy-whisper-teacher/ami_sdm_window_large_turbo_teacher_v1.parquet
```

Teacher-labeling summary:

- teacher: `openai/whisper-large-v3-turbo`;
- source: AMI SDM train windows;
- window length: up to 18 seconds;
- requested windows: 120;
- accepted windows: 115;
- rejected too short: 2;
- rejected too short audio: 1;
- rejected repetitive: 2.

Training summary:

- base model: `distil-whisper/distil-medium.en`;
- target text: `teacher_text` for AMI window pseudo-labels;
- LibriTTS real conversation mix included as robustness data;
- train rows after filtering: 167;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj`;
- LoRA rank: 8;
- trainable params: 1,835,008;
- max steps: 80;
- learning rate: `4e-6`;
- merged model: `/data/model/bddy-distil-medium-ami-window-teacher-lora-v10-merged`.

Internal trainer eval, 6 samples:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.3147 | 0.3100 |

AMI SDM conversation-window smoke eval, 6 windows from `validation[:600]`:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4951 | 0.4951 |
| median WER | 0.4900 | 0.4900 |
| p90 WER | 0.6977 | 0.6977 |
| median latency | 7.16s | 7.55s |
| question recall | 0.6250 | 0.6250 |

External AMI SDM short-utterance eval, 40 filtered rows:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4449 | 0.4449 |
| median WER | 0.4226 | 0.4226 |
| p90 WER | 0.7500 | 0.7500 |
| median latency | 5.98s | 6.35s |
| question recall | 0.7778 | 0.7778 |

Decision: do not promote. The large-teacher meeting-window dataset path works,
and the internal eval moved slightly, but 115 teacher windows is too small to
move the external gates. The next meaningful improvement is to scale teacher
window generation into the low thousands, then train with a held-out meeting
window promotion gate.

## Not Promoted: `medium_teacher_windows_v11`

Run directory:

```text
/data/model/bddy-asr-runs/medium_teacher_windows_v11
```

Teacher-labeling summary:

- teacher: `openai/whisper-large-v3-turbo`;
- source: `edinburghcstr/ami:sdm:train[:6000]:text`;
- window length: up to 18 seconds;
- requested windows: 300;
- accepted windows: 285;
- rejected too short: 9;
- rejected too short audio: 1;
- rejected repetitive: 5.

Training summary:

- base model: `distil-whisper/distil-medium.en`;
- target text: `teacher_text` for AMI window pseudo-labels;
- LibriTTS real conversation mix included as robustness data;
- train rows after filtering: 437;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj`;
- LoRA rank: 8;
- trainable params: 1,835,008;
- max steps: 120;
- learning rate: `4e-6`;
- merged model: `/data/model/bddy-asr-runs/medium_teacher_windows_v11/merged`.

Internal trainer eval, 11 samples:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.1340 | 0.1302 |
| median WER | 0.1071 | 0.1071 |

External AMI SDM short-utterance eval, 40 filtered rows:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4449 | 0.4449 |
| median WER | 0.4226 | 0.4226 |
| p90 WER | 0.7500 | 0.7500 |
| question recall | 0.7778 | 0.7778 |

AMI SDM conversation-window eval, 8 windows from `validation[:1200]`:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4562 | 0.4562 |
| median WER | 0.4462 | 0.4462 |
| p90 WER | 0.6977 | 0.6977 |
| question recall | 0.7000 | 0.7000 |

Decision: do not promote. Scaling from 115 to 285 teacher windows produced a
cleaner internal eval but still did not move the external gates. This confirms
the repeatable experiment environment works, but the next accuracy gain likely
requires more diverse meeting data, better segmentation/channel handling, or
training more decoder/encoder capacity than the current small LoRA target.

## Rejected: `medium_diverse_teacher_v12`

Run directory:

```text
/data/model/bddy-asr-runs/medium_diverse_teacher_v12
```

Teacher-labeling summary:

- teacher: `openai/whisper-large-v3-turbo`;
- window source: `edinburghcstr/ami:sdm:train[:3000]:text`;
- streamed row-level source: `distil-whisper/earnings22:chunked:test:transcription`;
- accepted AMI windows: 115;
- accepted Earnings22 streamed rows: 30;
- LibriTTS real conversation mix included as robustness data;
- final train rows after filtering: 197.

Training summary:

- base model: `distil-whisper/distil-medium.en`;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj`;
- LoRA rank: 8;
- max steps: 80;
- learning rate: `4e-6`;
- merged model: `/data/model/bddy-asr-runs/medium_diverse_teacher_v12/merged`.

Internal trainer eval, 11 samples:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.1340 | 0.1302 |
| median WER | 0.1071 | 0.1071 |

External AMI SDM short-utterance eval, 40 filtered rows:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4449 | 0.4464 |
| median WER | 0.4226 | 0.4226 |
| p90 WER | 0.7500 | 0.7500 |
| question recall | 0.7778 | 0.7778 |

AMI SDM conversation-window eval, 8 windows from `validation[:1200]`:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4562 | 0.4562 |
| median WER | 0.4462 | 0.4462 |
| p90 WER | 0.6977 | 0.6977 |
| question recall | 0.7000 | 0.7000 |

Decision: reject. The diverse streaming source path works, but adding a small
amount of Earnings22 row-level data did not improve AMI meeting transcription
and slightly regressed the short gate. Next runs should either scale the
diverse row data substantially with per-source balancing, or change the training
capacity/objective instead of adding small unbalanced data slices.

## Not Promoted: `medium_multisource_capacity_v13`

Run directory:

```text
/data/model/bddy-asr-runs/medium_multisource_capacity_v13
```

Teacher-labeling summary:

- teacher: `openai/whisper-large-v3-turbo`;
- window sources:
  - `edinburghcstr/ami:sdm:train[:8000]:text`;
  - `edinburghcstr/ami:ihm:train[:8000]:text`;
- streamed row-level source: `distil-whisper/earnings22:chunked:test:transcription`;
- accepted AMI windows: 676;
- accepted Earnings22 streamed rows: 62;
- rejected AMI windows: 24 total, mostly short or short-audio windows;
- rejected Earnings22 rows: 58 total, mostly short-audio rows;
- LibriTTS real conversation mix included as robustness data.

Training summary:

- base model: `distil-whisper/distil-medium.en`;
- LoRA target modules: `q_proj,k_proj,v_proj,out_proj,fc1,fc2`;
- LoRA rank: 8;
- trainable params: 3,964,928;
- filtered training rows: 586;
- max steps: 180;
- learning rate: `3e-6`;
- mild gain/noise augmentation plus 8% low-volume overlap augmentation;
- merged model: `/data/model/bddy-asr-runs/medium_multisource_capacity_v13/merged`.

Internal trainer eval, 11 samples:

| Metric | Before | After |
| --- | ---: | ---: |
| mean WER | 0.1340 | 0.1302 |
| median WER | 0.1071 | 0.1071 |

External AMI SDM short-utterance eval, 40 filtered rows:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4449 | 0.4464 |
| median WER | 0.4226 | 0.4226 |
| p90 WER | 0.7500 | 0.7500 |
| median latency | 5.99s | 5.86s |
| substitution rate | 0.1450 | 0.1450 |
| deletion rate | 0.1727 | 0.1727 |
| insertion rate | 0.0384 | 0.0405 |
| question recall | 0.7778 | 0.7778 |

AMI SDM conversation-window eval, 12 windows from `validation[:1200]`:

| Metric | Base `distil-medium.en` | Candidate |
| --- | ---: | ---: |
| mean WER | 0.4739 | 0.4679 |
| median WER | 0.4719 | 0.4719 |
| p90 WER | 0.6977 | 0.6977 |
| median latency | 7.28s | 7.54s |
| substitution rate | 0.1951 | 0.1885 |
| deletion rate | 0.1973 | 0.1996 |
| insertion rate | 0.0732 | 0.0710 |
| question recall | 0.7500 | 0.7500 |

Decision: do not promote. The wider LoRA target and multisource teacher set
finally moved the window metric in the right direction, mainly by reducing
substitutions and insertions, but the short AMI gate regressed slightly and
deletion rate did not improve. The next attempt should keep the configurable
runner, but use a promotion objective that explicitly penalizes deletions on
long windows and either:

1. train from the accepted v13 teacher data with lower overlap/noise pressure;
2. add a small curated hard-negative set from the worst validation windows; or
3. improve segmentation/channel handling before additional LoRA capacity.

## Eval Suite: `bddy-asr-eval/v14`

Run directory:

```text
/data/model/bddy-asr-eval/v14
```

Purpose: create a fixed held-out Parquet evaluation suite so every ASR
checkpoint is judged against the same meeting cases.

Artifacts:

| Artifact | Rows | Seconds | Words |
| --- | ---: | ---: | ---: |
| `ami_sdm_validation_short.parquet` | 120 | 480.15 | 1469 |
| `ami_sdm_validation_windows.parquet` | 60 | 626.39 | 1898 |
| `ami_ihm_validation_short.parquet` | 120 | 448.08 | 1436 |
| `ami_ihm_validation_windows.parquet` | 60 | 648.69 | 2000 |

Smoke eval on `ami_sdm_validation_windows.parquet`, first 3 cases, with
`distil-whisper/distil-small.en`:

| Metric | Value |
| --- | ---: |
| mean WER | 0.4370 |
| median WER | 0.3953 |
| p90 WER | 0.6000 |
| substitution rate | 0.2051 |
| deletion rate | 0.2222 |
| insertion rate | 0.0513 |

Decision: use this suite for v14+ promotion gates. The SDM window set is the
primary room-mic conversational gate. The IHM window set is a secondary clean
meeting gate that helps identify whether a change improved language handling or
only overfit room acoustics.
