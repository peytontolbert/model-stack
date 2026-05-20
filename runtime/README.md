# Runtime Package

`runtime/` is the main execution surface for Model Stack. It owns model
construction, attention selection, KV cache state, generation loops, quantized
runtime calls, native extension dispatch, and compatibility exports used by
older package paths.

The package is deliberately layered: Python APIs remain usable without the
native extension, while CUDA/C++ paths are selected when the extension is built
and the requested shape, dtype, device, and policy are supported.

## Main Surfaces

| Area | Files | Responsibility |
| --- | --- | --- |
| Package exports | `__init__.py`, `model_surface.py` | Stable public names and compatibility imports. |
| Model construction | `factory.py`, `model_registry.py`, `loader.py`, `prep.py`, `checkpoint.py` | Build, register, load, and prepare runtime models. |
| Model classes | `causal.py`, `encoder.py`, `seq2seq.py`, `prefix_lm.py`, `modeling.py` | Causal LM, encoder, seq2seq, prefix LM, and shared model wrappers. |
| Block system | `block_*.py`, `blocks.py`, `attention_modules.py` | Transformer block construction, policies, masks, initialization, schedules, and native-friendly execution. |
| Attention | `attention.py`, `attention_factory.py`, `attention_interfaces.py`, `attention_reference.py` | Backend selection, eager/reference attention, module wrappers, and attention interfaces. |
| KV cache | `cache.py`, `kv_cache.py` | Contiguous, paged, native-paged, and low-bit KV cache ownership. |
| Generation | `generation.py`, `sampling.py`, `decoding.py`, `model_generate.py` | Greedy/sample/beam/speculative decoding, sampling policies, token append, decode positions. |
| Quantization | `quant.py`, `ops.py` | Low-bit linear APIs, BitNet APIs, sampling/native op wrappers, Python fallbacks. |
| Native extension | `native.py`, `csrc/` | Lazy native module loading, runtime metadata, C++ ABI, CUDA backends. |
| Utilities | `runtime_utils.py`, `hardware.py`, `inspect.py`, `optim_utils.py` | Device/dtype helpers, hardware metadata, target-shape inspection, optimizer utilities. |

For CUDA/C++ backend details, see
[`docs/custom-kernel-architecture.md`](../docs/custom-kernel-architecture.md).

## Dispatch Rules

Runtime dispatch follows this order:

1. Normalize the Python API inputs.
2. Check runtime policy and environment flags.
3. Use `runtime.native.has_native_op(name)` before calling native symbols.
4. Use CUDA/C++ only when the native extension is available and the inputs are
   eligible.
5. Fall back to eager PyTorch behavior when native execution is unavailable,
   disabled, unsupported, or inappropriate under autograd.

Public functions should not require `_model_stack_native` at import time.
`runtime.native.native_module()` is lazy and may return `None`.

## Native Runtime Metadata

`runtime.native.runtime_info()` normalizes metadata from `_model_stack_native`.
Important query helpers:

- `native_available()`
- `has_native_op(name)`
- `cuda_kernel_ops()`
- `cuda_inference_ops()`
- `cuda_composite_ops()`
- `full_cuda_inference_available()`
- `resolve_linear_backend(requested)`

Set `MODEL_STACK_DISABLE_NATIVE=1` to force Python fallbacks even when the
extension is built.

## Generation Path

`generation.py` owns the high-level decode loop. `GenerationConfig` controls:

- max new tokens
- greedy vs sampling mode
- temperature, top-k, top-p
- no-repeat ngram and repetition/presence/frequency penalties
- beam search
- sliding-window cache eviction
- prefill chunking
- speculative decoding policy

The generation path uses `runtime.ops` wrappers for native-eligible operations
such as token append, decode positions, token counts, sampling policies, and
speculative acceptance. KV cache movement should stay in `cache.py` or
`kv_cache.py`, not in model-specific decode loops.

## KV Cache Backends

The runtime supports:

- contiguous KV cache
- paged KV cache
- native-paged KV cache state when available
- INT3 packed KV helpers
- row reordering, splitting, concatenation, eviction, and prefix truncation

Cache APIs must preserve batch-row semantics. Beam search and speculative
decoding depend on row reordering and split/concat behavior remaining stable.

## Model Loading And Preparation

The model loading surface is split by responsibility:

- `checkpoint.py`: Hugging Face snapshot/config conversion and local checkpoint
  IO.
- `loader.py`: runtime model directory/factory loading.
- `prep.py`: resolve config/device/dtype and return `RuntimeModelArtifacts`.
- `factory.py`: construct causal, encoder, seq2seq, prefix, or registered
  models.
- `model_registry.py`: lightweight registry entry point.

`model_surface.py` re-exports the model-loading API for callers that want a
smaller import surface than `runtime.__init__`.

## Compatibility Modules

Several top-level packages are thin compatibility shims over runtime modules.
For example, `attn/` re-exports attention, KV cache, quantization, decoding,
MoE, and optimizer helpers from `runtime/`. New implementation should generally
land in `runtime/` first, then be re-exported only when compatibility requires
it.

## Adding Runtime APIs

When adding a new runtime API:

1. Add an eager fallback first.
2. Decide whether it belongs in a focused module or in `ops.py`.
3. If native-backed, add a `has_native_op()` gate and preserve fallback parity.
4. Add source-surface tests for native registration when applicable.
5. Add numeric tests against the eager fallback for supported dtypes/devices.
6. Add the public name to `runtime.__init__` only when it is intended as a
   stable package-level API.

Keep runtime modules import-light. Optional packages and native extensions
should be imported lazily inside the code paths that need them.

