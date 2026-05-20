# Attention Compatibility Package

`attn/` is a compatibility package for attention-related APIs. Most
implementation now lives under `runtime/`; this package keeps older imports
working while exposing the same attention, KV cache, decoding, quantization,
MoE, and optimizer utilities.

Prefer adding new implementation in `runtime/` first. Add or update an `attn/`
shim only when an existing public import path needs to remain stable.

## Compatibility Map

| `attn` module | Runtime implementation |
| --- | --- |
| `attn.backends` | `runtime.attention` |
| `attn.decoding` | `runtime.decoding` |
| `attn.eager` | `runtime.attention_modules` |
| `attn.factory` | `runtime.attention_factory` |
| `attn.flash` | `runtime.attention_modules.FlashAttention` |
| `attn.gqa` | `runtime.gqa` |
| `attn.interfaces` | `runtime.attention_interfaces` |
| `attn.kv_cache` | `runtime.kv_cache` |
| `attn.moe` | `runtime.moe` |
| `attn.optim_utils` | `runtime.optim_utils` |
| `attn.quant` | `runtime.quant` |
| `attn.reference` | `runtime.attention_reference` |
| `attn.triton` | `runtime.attention_modules.TritonAttention` |
| `attn.xformers` | `runtime.attention_modules.XFormersAttention` |

## Package Exports

`attn.__init__` re-exports:

- attention backend selection and scaled dot-product attention
- eager, Flash, Triton, and XFormers attention module wrappers
- contiguous, paged, and INT3 KV cache helpers
- beam search, Mirostat, and regex decode constraints
- MHA/GQA/MQA reshaping helpers
- MoE routing/load-balance helpers
- FP8/NF4/INT8 quantization helpers
- gradient clipping and decay-mask utilities

## Native And Backend Policy

Backend selection is owned by `runtime.attention` and the native extension
metadata in `runtime.native`. The compatibility package should not introduce a
second policy layer. Environment flags and CUDA kernel details are documented in
[`docs/custom-kernel-architecture.md`](../docs/custom-kernel-architecture.md).
