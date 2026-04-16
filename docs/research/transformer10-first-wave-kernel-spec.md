# transformer_10 First-Wave Kernel Spec

This document is the first implementation-facing spec for replacing the current torch-heavy hot path.

It is intentionally narrower than a full migration plan. The goal is to define the first wave of kernels and wrappers well enough that implementation can start without re-deciding fundamentals every day.

## Scope

The first wave covers the single-GPU forward serving path:

1. RMSNorm
2. RoPE apply
3. `cuBLASLt` linear wrappers
4. SwiGLU activation
5. KV cache append/read
6. prefill attention
7. decode attention
8. embedding lookup
9. sampling hot path

## Common Rules

Apply these rules to every item below:

- inputs and outputs must accept explicit stream handles
- shapes, strides, dtype, and device must be validated at the boundary
- BF16/FP16 execution should accumulate in FP32 when numerically required
- every op needs:
  - reference parity test
  - edge-shape test
  - benchmark

## 1. RMSNorm

Current local code:

- `tensor/norms.py`
- `blocks/transformer_block.py`
- `model/causal.py`

Recommended entrypoint:

```text
Status rmsnorm_forward(
  TensorView x,           // (B, T, D)
  TensorView weight,      // (D)
  float eps,
  MutableTensorView out,
  CudaStream stream);
```

Tensor contract:

- `x`: contiguous or last-dim contiguous
- `weight`: contiguous `(D)`
- `out`: same shape and dtype as `x`

Notes:

- accumulate variance in FP32
- specialize common hidden sizes first:
  - 2048
  - 3072
  - 4096
  - 5120
  - 6144
  - 7168
  - 8192

Best references:

- `TransformerEngine/common/normalization`
- `/data/transformer_10/other_repos/flash-attention/csrc/layer_norm/`
- `/data/transformer_10/other_repos/ThunderKittens/kernels/layernorm/`

Validation:

- compare against `tensor.norms.rmsnorm`
- odd batch and sequence sizes
- BF16/FP16 parity against FP32 reference

## 2. RoPE Apply

Current local code:

- `tensor/positional.py`
- `attn/eager.py`

Recommended entrypoint:

```text
Status rope_apply_forward(
  TensorView q,           // (B, Hq, T, Dh)
  TensorView k,           // (B, Hk, T, Dh)
  TensorView cos,         // (T, Dh)
  TensorView sin,         // (T, Dh)
  MutableTensorView q_out,
  MutableTensorView k_out,
  CudaStream stream);
```

Tensor contract:

- `Dh` must be even
- `cos` and `sin` must be contiguous
- input and output may alias only if the kernel explicitly supports in-place operation

Notes:

- keep the first version separate from attention for easier validation
- once stable, fuse with attention prep if profiling proves it worthwhile

Best references:

- `TransformerEngine/common/fused_rope`
- `/data/transformer_10/other_repos/flash-attention/csrc/flash_attn/src/rotary.h`
- `Fuser/tests/cpp/test_rope.cpp`

Validation:

- compare against `tensor.positional.apply_rotary`
- verify untouched behavior outside the rotary span if partial-rotary variants are added later

## 3. Linear Wrappers

Current local code:

- `attn/eager.py`
- `tensor/mlp.py`
- `model/causal.py`

Recommended entrypoint:

```text
Status linear_forward(
  LinearPlanHandle plan,
  TensorView x,           // usually (B, T, D_in) or flattened
  TensorView weight,
  OptionalTensorView bias,
  MutableTensorView out,
  Workspace workspace,
  CudaStream stream);
```

Required first wrappers:

- Q projection
- K projection
- V projection
- output projection
- MLP up/gate projection
- MLP down projection
- LM head

Notes:

- use one wrapper family with per-call plans rather than separate ad hoc code paths everywhere
- flatten `(B, T, D)` to `(B*T, D)` inside the wrapper or a nearby helper
- own weight-layout conversion policy in the wrapper

Best references:

- `CUDALibrarySamples/cuBLASLt`
- `TransformerEngine/common/gemm`
- `nvMatmulHeuristics`

Validation:

- compare against `torch.nn.Linear`
- cover bias/no-bias
- cover BF16/FP16/FP32
- benchmark both small decode batches and larger prefill shapes

## 4. SwiGLU

Current local code:

- `tensor/mlp.py`
- `tensor/activations.py`

Recommended entrypoint:

```text
Status swiglu_forward(
  TensorView a,           // (B, T, F)
  TensorView b,           // (B, T, F)
  MutableTensorView out,
  CudaStream stream);
```

Notes:

- this can be authored in Triton first if it accelerates delivery
- keep the runtime surface neutral so the backing implementation can change later

Best references:

- local Triton notes
- `TensorRT-LLM/triton_kernels/swiglu.py`

Validation:

- compare against `silu(a) * b`
- verify no dtype promotion bugs

## 5. KV Cache

Current local code:

- `attn/kv_cache.py`
- `serve/runtime.py`

Required runtime objects:

- `CacheHandle`
- `LayerCacheView`
- page allocator metadata

Recommended entrypoints:

```text
Status kv_cache_append(
  CacheHandle cache,
  int layer_idx,
  TensorView k,           // (B, Hk, T_new, Dh)
  TensorView v,           // (B, Hk, T_new, Dh)
  CudaStream stream);

Status kv_cache_read(
  CacheHandle cache,
  int layer_idx,
  int start,
  int end,
  MutableTensorView k_out,
  MutableTensorView v_out,
  CudaStream stream);
```

Notes:

- keep the first version simple and correct
- page tables and eviction policy can remain conservative in phase 1
- Python should no longer own lists of page tensors in the hot path

Best references:

- `TensorRT-LLM` KV cache docs and runtime
- `TransformerEngine` fused-attention cache support
- current semantics in `attn/kv_cache.py`

Validation:

- append then read equals original tokens
- multi-batch correctness
- multiple append rounds preserve earlier tokens exactly

## 6. Prefill Attention

Current local code:

- `attn/eager.py`
- `attn/backends.py`

Recommended entrypoint:

```text
Status attention_prefill_forward(
  AttentionPlanHandle plan,
  TensorView q,           // (B, Hq, Tq, Dh)
  TensorView k,           // (B, Hk, Tk, Dh)
  TensorView v,           // (B, Hk, Tk, Dh)
  OptionalTensorView mask,
  MutableTensorView out,  // (B, Hq, Tq, Dh)
  Workspace workspace,
  CudaStream stream);
```

Phase-1 requirements:

- causal mask
- no-dropout inference path
- MHA plus GQA/MQA
- BF16/FP16

Notes:

- first stable version can be less feature-rich than the current eager path as long as the serving path requirements are covered
- own mask/layout/scale policy here instead of routing back through Python backends

Best references:

- `TransformerEngine/common/fused_attn`
- `TensorRT-LLM/cpp/kernels/fmha_v2`
- `/data/transformer_10/other_repos/flash-attention/csrc/flash_attn`
- `/data/transformer_10/other_repos/ThunderKittens/kernels/attention/`

Validation:

- compare against current `attn/eager.py`
- causal-mask correctness
- GQA expansion correctness
- longer-sequence stability

## 7. Decode Attention

Current local code:

- `attn/eager.py`
- `serve/engine.py`
- `attn/kv_cache.py`

Recommended entrypoint:

```text
Status attention_decode_forward(
  AttentionPlanHandle plan,
  TensorView q,           // (B, Hq, T_new, Dh), usually T_new=1
  CacheHandle cache,
  int layer_idx,
  OptionalTensorView mask,
  MutableTensorView out,
  Workspace workspace,
  CudaStream stream);
```

Phase-1 requirements:

- causal decode
- MQA/GQA
- paged cache read
- stable small-batch latency

Best references:

- `TensorRT-LLM/cpp/kernels/xqa`
- `TensorRT-LLM` attention and KV cache docs
- `TransformerEngine` decode-oriented attention code where applicable

Validation:

- parity against eager path with cache enabled
- repeated token-by-token append/read correctness
- benchmark on `T_new=1`

## 8. Embedding Lookup

Current local code:

- `model/causal.py`
- `model/encoder.py`
- `model/seq2seq.py`

Current implementation status:

- native `embedding_forward` entrypoint exists in `runtime/csrc/model_stack_native.cpp`
- CUDA backend exists in `runtime/csrc/backend/cuda_embedding.cu`
- `model/causal.py`, `model/encoder.py`, and `model/seq2seq.py` call the runtime embedding path on the intended hot path
- validation is still blocked on rebuilding `_model_stack_native` in an environment with a full PyTorch extension toolchain
- `setup.py` now accepts repo-local CUDA build controls via `MODEL_STACK_CUDA_ARCH_LIST`, `MODEL_STACK_MAX_JOBS`, and `MODEL_STACK_USE_NINJA`

Recommended entrypoint:

```text
Status embedding_lookup_forward(
  TensorView token_ids,   // (B, T), int32 or int64
  TensorView table,       // (V, D)
  MutableTensorView out,  // (B, T, D)
  CudaStream stream);
```

Notes:

- this is a bandwidth-bound gather op
- do not treat it like a GEMM

Best references:

- `cuEmbed`
- `TensorRT-LLM` runtime kernels

Validation:

- compare against `torch.nn.Embedding`
- cover repeated indices and uneven vocab access

## 9. Sampling Hot Path

Current local code:

- `tensor/sampling.py`
- `serve/engine.py`

Current implementation status:

- the runtime exposes sampling entrypoints through `runtime/ops.py` and `runtime/csrc/model_stack_native.cpp`
- CUDA kernels now cover temperature scaling, token-count accumulation, presence/frequency penalties, repetition penalties, and greedy next-token selection
- the current implementation still relies on ATen/Torch ops for top-k, top-p, and multinomial sampling
- sampler ownership is still incomplete, but the remaining Level-1 serving hot-path gap is now narrower than it was after embeddings

Recommended entrypoint:

```text
Status sampling_forward(
  TensorView logits,              // (B, V)
  SamplingParams params,
  OptionalTensorView token_counts,
  MutableTensorView next_token,
  Workspace workspace,
  CudaStream stream);
```

Phase-1 requirements:

- temperature
- top-k
- top-p
- repetition/presence/frequency penalties

Notes:

- Triton is acceptable for the first backing implementation
- keep API neutral so it can later move to handwritten CUDA/C++ if needed

Best references:

- `TensorRT-LLM` sampling docs
- local Triton notes
- current semantics in `tensor/sampling.py`

Validation:

- compare masks and chosen tokens against eager reference on fixed RNG seeds
- cover extreme `k`, `p`, and low-temperature cases

## Recommended Landing Order

Implement in this order:

1. RMSNorm
2. RoPE apply
3. linear wrappers
4. KV cache
5. prefill attention
6. decode attention
7. SwiGLU
8. embedding lookup
9. sampling

That ordering minimizes risk because:

- the first three are narrow and easy to validate
- KV cache must stabilize before decode attention
- attention depends on earlier layout and runtime decisions

## Minimal Test Matrix

Every first-wave op should have:

- correctness test against eager reference
- dtype matrix:
  - FP32 reference
  - BF16
  - FP16 where supported
- shape matrix:
  - small sanity case
  - odd dimensions
  - realistic model shape
- benchmark:
  - latency
  - achieved bandwidth or throughput depending on the op

## Bottom Line

The first-wave implementation should not aim for full feature closure. It should aim for a stable forward-serving path whose primitives are:

- explicit
- benchmarked
- numerically checked
- structured so the remaining migration can build on them instead of replacing them again
