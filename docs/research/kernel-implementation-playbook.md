# Kernel Implementation Playbook

This document is the practical playbook for building and integrating CUDA/C++ kernels into `transformer_10`.

## 1. Runtime Architecture

Recommended structure for the new runtime layer:

```text
runtime/
  cuda/
    include/
      t10_cuda/
        tensor.h
        stream.h
        status.h
        gemm.h
        norm.h
        rope.h
        attention.h
        kv_cache.h
        collectives.h
    src/
      common/
      gemm/
      norm/
      rope/
      attention/
      kv_cache/
      collectives/
    tests/
    bench/
    python/
```

The important rule is separation of concerns:

- tensor/layout descriptors
- launch/configuration logic
- kernel bodies
- library-backed wrappers like cuBLASLt and NCCL
- tests and benchmarks

Do not bury launch policy inside Python if the goal is a direct runtime.

## 2. API Shape

Use explicit, low-ambiguity APIs.

Bad:

- functions that infer too much from raw pointers and loose shape lists

Good:

- typed descriptors with:
  - dtype
  - rank
  - shape
  - strides
  - device pointer
  - stream

Recommended baseline types:

- `TensorView`
- `MutableTensorView`
- `CudaStream`
- `Workspace`
- `Status`

This becomes important as soon as we support:

- packed KV cache
- GQA/MQA
- non-contiguous layouts
- tensor parallel shards

## 3. Build Rules

Use CMake and keep CUDA compilation explicit.

Recommended choices:

- C++17
- CUDA 12.x+
- separate compilation units per domain
- architecture list configured in one place
- tests and benchmarks as first-class targets

Minimum build requirements inferred from local NVIDIA references:

- CUDA Toolkit 12.x or newer
- cuBLASLt
- NCCL for distributed phases
- pybind11 if Python bindings are used

Keep architecture flags intentional:

- do not compile for every architecture by default
- match the actual fleet

## 4. Implementation Rules

### Rule 1: do not handwrite GEMMs

Use cuBLASLt for:

- QKV projections
- output projection
- MLP projections
- lm_head

What we own:

- tensor/layout canonicalization
- workspace management
- epilog selection
- fallback behavior

### Rule 2: specialize common transformer sizes

TransformerEngine's norm kernels are the right precedent:

- common hidden sizes get tuned launch configs
- uncommon shapes get a general path

Apply the same principle to:

- RMSNorm
- RoPE
- KV append
- softmax helpers

### Rule 3: own memory movement, not just math

A lot of speed comes from:

- fewer temporary allocations
- fewer transpose copies
- reusable workspaces
- stream-aware lifetime control

Use:

- `cudaMallocAsync`
- `cudaFreeAsync`
- memory pools

The `cuda-samples` stream-ordered allocation example is the baseline reference.

### Rule 4: keep layout contracts stable

Pick a small number of canonical layouts and enforce them.

Recommended canonical layouts:

- hidden states: `(B, T, D)`
- Q: `(B, Hq, T, Dh)`
- K/V: `(B, Hk, T, Dh)`
- cache pages or contiguous KV blocks: runtime-defined but fixed

Do not let every kernel invent its own interpretation.

### Rule 5: treat the stream as part of the API

Every runtime op should accept a stream or stream handle.

That matters for:

- overlap
- memory-pool semantics
- graph capture
- NCCL interop

## 5. Validation Strategy

Every migrated op needs three levels of validation.

### 1. Numerical unit tests

Compare against current torch implementations for:

- FP32 reference
- BF16/FP16 execution
- edge shapes
- odd sequence lengths
- odd head counts where supported

### 2. Property tests

Examples:

- RMSNorm preserves expected scaling behavior
- RoPE leaves untouched trailing dimensions outside rotary span
- KV append preserves prior tokens exactly
- attention with causal mask never reads future tokens

### 3. End-to-end parity

Compare entire block/model outputs against the current runtime for:

- fixed seeds
- small batch/sequence sizes
- longer sequence buckets
- cache and non-cache paths

## 6. Benchmarking Strategy

Do not measure only end-to-end generation.

Measure:

- per-op latency
- achieved bandwidth for memory-bound ops
- achieved TFLOPs for GEMM-backed paths
- graph capture/replay speedup
- KV-cache scaling with sequence length

Minimum per-op benchmark set:

1. RMSNorm
2. RoPE
3. QKV GEMM
4. attention
5. MLP up/gate/down
6. embedding lookup
7. sampler
8. NCCL all-reduce and all-gather

## 7. Fused Attention Guidance

Do not start by writing a fully custom FlashAttention clone from scratch unless the team explicitly wants that investment.

Safer progression:

1. direct GEMMs + custom mask/softmax helpers
2. custom KV append/update
3. adopt a fused attention path once the surrounding runtime is stable

Reference sources:

- `TransformerEngine/common/fused_attn/*`
- `TransformerEngine/common/fused_softmax/*`
- `Megatron-LM` inference and context-parallel code

The hard part is not only the math. It is:

- packed sequence support
- cache layout
- masking semantics
- GQA/MQA
- backward behavior if training is required

## 8. Distributed Guidance

For distributed replacement work:

- do not keep `torch.distributed` on the hot path if the target is direct CUDA/C++
- wrap NCCL directly behind runtime collectives

The model-level invariant from Megatron-LM is useful:

- linears and norms are sequence-local
- attention is the place where context-parallel KV exchange becomes necessary

That means the runtime should separate:

- local compute ops
- tensor-parallel collectives
- context-parallel KV exchange

## 9. Recommended Rollout Path

### Step 1

Add a feature-flagged CUDA runtime path next to the current torch path.

### Step 2

Replace only:

- RMSNorm
- RoPE
- GEMM wrappers

### Step 3

Replace attention and KV cache.

### Step 4

Replace embedding and sampler paths.

### Step 5

Replace distributed collectives.

This keeps the system debuggable at each stage.

## 10. Risks To Avoid

### Risk 1: over-customizing too early

If we handwrite GEMMs, allocator logic, and collectives before the runtime boundary is stable, the project will slow down badly.

### Risk 2: no parity harness

Without automated parity checks, every kernel bug becomes a model-level mystery.

### Risk 3: unstable layout contracts

If Q/K/V/cache layouts are not fixed early, every new kernel multiplies integration cost.

### Risk 4: benchmark theater

A kernel that is faster in isolation but forces extra transposes or allocations can still make the model slower.

## 11. First Concrete Implementation Slice

The best first implementation slice for this repo is:

1. cuBLASLt wrapper for linear layers.
2. RMSNorm forward kernel.
3. RoPE apply kernel.
4. Benchmarks and parity tests for those ops.
5. Feature-flagged integration into `attn/eager.py`, `tensor/norms.py`, and `tensor/positional.py`.

That slice is small enough to finish and large enough to validate the overall direction.
