# Agent Kernel Lite Training Import

This directory is a source-only import of the Agent Kernel Lite training scripts
that fit model-stack's training and runtime boundaries. It is intentionally
namespaced while the code is being migrated so existing model-stack scripts are
not overwritten.

Imported lanes:

- Tiny diffusion student training and analysis:
  - FLUX flow distillation
  - live-teacher and rollout distillation
  - SANA latent distillation
  - BitDiT/browser export helpers
  - bridge/refiner/trajectory probes
- F5TTS Q4 distillation and browser validation:
  - Q4 bundle export
  - 12-to-4 and streaming teacher distillation
  - WASM forward/profile/hardening checks
  - Vocos Q4/FP16 export helpers
- Tiny seq2seq and PocketPal-style controller training:
  - encoder-decoder dataset builders
  - repair/curriculum builders
  - active-agent and policy-head gates
  - retrieval and controller evaluators

What was deliberately not copied:

- `tmp/` experiment runs and generated JSONL buffers
- checkpoints and model bundles
- generated web model binaries
- audio quality samples
- mobile app build artifacts

Run scripts from this directory when they use flat sibling imports:

```bash
cd scripts/agent_kernel_lite
python train_agentkernel_lite_encdec.py --help
python train_agentkernel_lite_image_flux_flow_distill.py --help
python distill_f5tts_q4_teacher.py --help
```

Some shell scripts still contain Agent Kernel Lite-era paths such as
`scripts/<name>.py`. Treat those as migration templates until the path references
are rewritten for model-stack.

The matching docs are in `docs/agent_kernel_lite_training/`. Source-only example
smokes are in `examples/agent_kernel_lite_training/`.
