Model stack: Causal LM, Prefix LM, Encoder, and Encoder-Decoder (seq2seq).

Quickstart

```python
from specs.config import ModelConfig
from model import build_model

cfg = ModelConfig(d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=32000)
lm = build_model(cfg, task="causal-lm", block="llama")

import torch
ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = lm(ids, attn_mask=None)
```

APIs

- Causal LM: `model.causal.CausalLM` (also available as `model.TransformerLM` and `model.lm.TransformerLM`)
  Compatibility alias over `runtime.causal.CausalLM`
- Prefix LM: `model.prefix_lm.PrefixCausalLM`
  Compatibility alias over `runtime.prefix_lm.PrefixCausalLM`
- Encoder: `model.encoder.EncoderModel`
  Compatibility alias over `runtime.encoder.EncoderModel`
- Seq2Seq: `model.seq2seq.EncoderDecoderLM`
  Compatibility alias over `runtime.seq2seq.EncoderDecoderLM`
- Heads: `model.heads.{SequenceClassificationHead, TokenClassificationHead}`
  Compatibility alias over `runtime.heads`
- Factory: `model.factory.build_model(cfg, task=..., block=...)`
  Compatibility alias over `runtime.factory`
- Registry: `model.registry.build(name, cfg, **kwargs)`
  Compatibility alias over `runtime.factory`
- Top-level package: `model.{build, build_model, load_model_dir, load_runtime_model, ensure_snapshot, build_local_llama_from_snapshot, ...}`
  Compatibility reexports over the runtime-owned factory, loader, checkpoint, prep, inspect, compile, and utility surfaces
- Generation: `model.generate.{greedy_generate, sample_generate}`
  Compatibility alias over `runtime.model_generate`, which delegates to `runtime.generation` and passes through penalties, sliding-window policy, cache-backend selection, and `attention_mask` / `attn_mask`.
- `model.causal.CausalLM.generate`
  Compatibility wrapper over the same runtime-owned generation config, including inferred sampling mode from sampling knobs and `cache_backend` passthrough.
- HF import/bootstrap: `model.hf_snapshot`, `model.hf_llama_loader`, and `model.llama_bootstrap`
  Compatibility aliases over `runtime.checkpoint`
- Compile helper: `model.compile.maybe_compile` and `model.maybe_compile`
  Compatibility alias over `runtime.compile`
- Legacy export wrapper: `model.export.{export_onnx, export_torchscript}`
  Compatibility alias over `runtime.model_export`

Notes

- Blocks are now built via `runtime.block_factory.build_block_stack`; `blocks.factory` is a compatibility alias over that runtime-owned construction surface.
- Core block classes now live in `runtime.block_modules`; `blocks.transformer_block`, `blocks.llama_block`, `blocks.gpt_block`, `blocks.encoder_block`, `blocks.decoder_block`, `blocks.cross_attn_block`, `blocks.parallel_block`, and `blocks.moe_block` are compatibility aliases.
- Block config/init/target helpers now live under `runtime.block_config`, `runtime.block_init`, `runtime.block_schedules`, `runtime.block_policies`, and `runtime.block_targets`; the corresponding `blocks.*` modules are compatibility aliases.
- Attention masks may be boolean (True = masked) or additive; both are supported.
- Use `tensor.masking` helpers to build causal/prefix/local/banded masks when needed.
