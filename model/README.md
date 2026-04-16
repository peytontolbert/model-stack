Model stack: Causal LM, Prefix LM, Encoder, and Encoder-Decoder (seq2seq).

Quickstart

```python
from specs.config import ModelConfig
from model.factory import build_model

cfg = ModelConfig(d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=32000)
lm = build_model(cfg, task="causal-lm", block="llama")

import torch
ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = lm(ids, attn_mask=None)
```

APIs

- Causal LM: `model.causal.CausalLM` (also available as `model.lm.TransformerLM`)
- Prefix LM: `model.prefix_lm.PrefixCausalLM`
- Encoder: `model.encoder.EncoderModel`
- Seq2Seq: `model.seq2seq.EncoderDecoderLM`
- Heads: `model.heads.{SequenceClassificationHead, TokenClassificationHead}`
- Factory: `model.factory.build_model(cfg, task=..., block=...)`
  Compatibility shim over `runtime/factory.py`
- Registry: `model.registry.build(name, cfg, **kwargs)`
  Compatibility shim over `runtime/factory.py`
- Top-level package: `model.{build, build_model, load_model_dir, load_runtime_model, ensure_snapshot, build_local_llama_from_snapshot, ...}`
  Compatibility reexports over `runtime/factory.py`, `runtime/modeling.py`, `runtime/loader.py`, and `runtime/checkpoint.py`
- Generation: `model.generate.{greedy_generate, sample_generate}`
  Compatibility shims over `runtime.generation` that now pass through penalties, sliding-window policy, cache-backend selection, and `attention_mask` / `attn_mask`.
- `model.causal.CausalLM.generate`
  Compatibility wrapper over the same runtime-owned generation config, including inferred sampling mode from sampling knobs and `cache_backend` passthrough.
- HF import/bootstrap: `model.hf_snapshot`, `model.hf_llama_loader`, and `model.llama_bootstrap`
  Compatibility shims over `runtime/checkpoint.py`

Notes

- Blocks are built via `blocks.factory.build_block_stack` and integrate attention from `attn.factory`.
- Attention masks may be boolean (True = masked) or additive; both are supported.
- Use `tensor.masking` helpers to build causal/prefix/local/banded masks when needed.
