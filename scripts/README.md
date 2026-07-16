# Scripts

Repository scripts are operational entry points for export, training,
distillation, evaluation, and migration work.

## Agent Kernel Lite Import

`scripts/agent_kernel_lite/` contains the source-only migration import from
`/data/agent_kernel_lite`.

Use it for:

- tiny diffusion student experiments
- F5TTS Q4 distillation and browser validation
- tiny seq2seq/PocketPal curriculum and evaluation work

It is intentionally namespaced while the code is being adapted. Do not assume
all scripts are model-stack-native yet; some still reference Agent Kernel
Lite-era paths and artifact layouts.

Preferred launcher:

```bash
python -m train.agent_kernel_lite list
python -m train.agent_kernel_lite run seq2seq -- --help
python -m train.agent_kernel_lite run diffusion-flow -- --help
python -m train.agent_kernel_lite run f5tts-distill -- --help
```

Use direct script execution only when debugging an individual imported file.
