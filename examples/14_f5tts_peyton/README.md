# F5TTS Peyton Voice Smoke Test

This example verifies the existing Peyton voice clone service from
`/data/resumebot` and writes a generated WAV into this example directory.

Run from `/data/transformer_10`:

```bash
/home/peyton/miniconda3/envs/ai/bin/python examples/14_f5tts_peyton/run.py \
  --text "This is a quick local test of my F5 TTS voice running from transformer ten."
```

Default assets:

- Checkpoint: `/data/resumebot/checkpoints/final_finetuned_model.pt`
- Voice profile: `/data/resumebot/voice_profiles/Peyton`
- Output dir: `/data/transformer_10/examples/14_f5tts_peyton/out`

This uses the original PyTorch F5TTS path, not the Q4 WASM path. The Q4 bundle
is a compact weight export for the upcoming runtime; the complete F5TTS graph
and vocoder still need to be wired before it can synthesize audio on its own.
