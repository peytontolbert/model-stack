## interpret.py — Lightweight Activation Tracing

Helpers for grabbing block outputs and accessing the output projection weights for interpretability.

### Key APIs
- `is_block(name, module)` -> `bool` for LLaMA‑style `model.layers.N`
- `block_out_hook(key, module, inputs, output)` -> `Tensor|None` best‑effort extractor
- `get_W(m)` -> output projection (lm_head) tensor

### Usage
Register forward hooks on blocks of interest to capture hidden states for simple analyses.


