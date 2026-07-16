# Agent Kernel Lite Training Examples

This directory contains source-only smoke examples copied from
`/data/transformer_10` for migration into model-stack.

Included examples:

- `14_f5tts_peyton`: Python F5TTS Peyton Q4 smoke
- `15_f5tts_q4_wasm_smoke`: browser/WASM Q4 smoke
- `16_f5tts_q4_dit_smoke`: F5 DiT smoke
- `17_f5tts_q4_end_to_end_smoke`: F5 plus Vocos end-to-end smoke
- `18_f5tts_q4_peyton_ref_smoke`: Peyton reference-conditioning smoke

Generated `.wav`, `.f32`, checkpoints, and model bundles are intentionally not
copied. Point these examples at local artifacts under `/data/model`,
`/data/resumebot`, or an ignored model-stack artifact directory when running
them.
