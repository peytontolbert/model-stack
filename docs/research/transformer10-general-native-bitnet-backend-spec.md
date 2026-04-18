# transformer_10 General Native BitNet Backend Spec

This document defines the target design for a general native BitNet backend in `transformer_10`.

The goal is not a decode-only demo kernel. The goal is a first-class runtime backend that works for:

- batched prefill
- batched decode
- existing `runtime_linear_module(...)` call sites
- attention QKV / output projections
- MLP up/gate / down projections
- export, checkpoint, and compression flows

## 1. Scope

This backend must support the current repository execution model, not a separate BitNet-only model implementation.

That means it must integrate with:

- `runtime/csrc/model_stack_native.cpp`
- `runtime/ops.py`
- `runtime/native.py`
- `runtime/attention_modules.py`
- `tensor/mlp.py`
- `compress/quantization.py`
- `compress/export.py`
- `compress/apply.py`

It must also remain graph-capturable and preserve the current dense fallback behavior for unsupported situations.

## 2. Current Repo Reality

### 2.1 Native linear backends today

The native runtime currently resolves only dense linear backends in `runtime/csrc/model_stack_native.cpp`:

- `aten`
- `cublaslt`

The packed fast path in `runtime/attention_modules.py` is also explicitly tied to `cublaslt`.

Important implication:

- BitNet cannot be integrated as "just another packed weight format" unless the linear backend system itself is extended.

### 2.2 Existing packed quantization work

There is already an in-flight INT4 runtime path:

- `runtime/csrc/backend/cuda_int4_linear.cu`
- `runtime.quant.int4_linear(...)`
- `QuantizedLinearInt4` in `compress/quantization.py`

This is useful scaffolding, but it is not yet the right architecture for BitNet:

- the current INT4 CUDA kernel is a scalar unpack-and-dot implementation, not a tiled high-throughput backend
- `QuantizedLinearInt4.runtime_supports_packed_backend(...)` currently returns `False`
- the packed attention fast path still only accepts `cublaslt`
- `tensor/mlp.py` still sends raw weight tensors into `runtime_mlp(...)` rather than module-aware packed objects

The BitNet backend should reuse the integration pattern, not the kernel strategy.

### 2.3 Local BitNet repo limitations

The local `other_repos/BitNet` GPU path is valuable as a format and semantics reference, but not as the final backend design.

Useful parts:

- ternary weight convention `{-1, 0, 1}` packed into 2-bit storage
- activation quantization to INT8
- fused checkpoint conversion for `wqkv`, `w13`, `wo`, and `w2`
- 16x32 block permutation and interleave logic in `gpu/pack_weight.py`

Limitations:

- the CUDA entrypoint in `gpu/bitnet_kernels/bitnet_kernels.cu` hard-codes `M == 1`
- only a short whitelist of `(N, K)` shapes is supported
- the kernel is compiled for `sm80` in `gpu/bitnet_kernels/compile.sh`
- activation scaling is effectively decode-oriented, not a general per-row batched path
- the runtime boundary is a Python `ctypes` call, not an integrated C++/CUDA extension path

Conclusion:

- we can reuse BitNet's packing semantics and checkpoint conversion ideas
- we should not reuse its dispatch model as the general backend

### 2.4 Model-stack compatibility boundary

The general native backend described here targets `transformer_10`'s existing attention and MLP module boundaries.

That is not identical to the full BitNet reference model in `other_repos/BitNet/gpu/model.py`, which also introduces:

- `attn_sub_norm`
- `ffn_sub_norm`
- `squared_relu(x1) * x3` style feed-forward behavior

V1 decision:

- treat BitNet as a packed linear execution scheme that can serve the current model stack
- do not bake BitNet-specific block semantics into the linear kernel ABI

Implication:

- the backend must work with current `EagerAttention`, `MLP`, and `runtime_linear_module(...)` call patterns
- a future native BitNet-model architecture can reuse the backend, but it is a separate integration step

## 3. NVIDIA Capability Review

### 3.1 `__dp4a` is INT8, not INT2

NVIDIA's CUDA Math API documents `__dp4a` as four-way INT8 dot product with INT32 accumulation.

Implication:

- the baseline portable BitNet compute path must decode packed 2-bit ternary weights into INT8 fragments before dot-product
- `dp4a` is still useful on Ampere and as a simple correctness / fallback kernel family

Reference:

- CUDA Math API, integer intrinsics: <https://docs.nvidia.com/cuda/archive/12.5.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html>

### 3.2 `cuBLASLt` does not give us native INT2 BitNet GEMM

Current cuBLAS documentation lists regular INT8 matmul support for `cublasLtMatmul()` but does not expose a native INT2 / ternary matrix format for the path we need here.

Implication:

- `cuBLASLt` remains the correct dense fallback and reference baseline
- it is not the core execution engine for the BitNet path

Reference:

- cuBLAS 13.2 `cublasLtMatmul()` data-type table: <https://docs.nvidia.com/cuda/cublas/index.html>

### 3.3 CUDA sub-byte WMMA is not the right foundation

NVIDIA's CUDA C++ Programming Guide documents `u4`, `s4`, and `b1` WMMA operations as experimental / preview APIs. The same section also notes deprecation of these variants on `sm_90` via that WMMA path.

Implication:

- we should not build the main BitNet backend on top of `nvcuda::wmma::experimental`
- for Hopper / Blackwell optimization lanes, use lower-level MMA / WGMMA-capable kernel frameworks or handwritten PTX/CuTe-style kernels instead

Reference:

- CUDA C++ Programming Guide, sub-byte WMMA: <https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html>

### 3.4 CUTLASS is the right reference class for the general kernel family

NVIDIA CUTLASS documents:

- 4-bit and 8-bit narrow integer support
- grouped GEMM scheduling
- persistent scheduling patterns
- reusable GEMM decomposition across architectures

Implication:

- the BitNet backend should follow CUTLASS-style kernel decomposition and scheduler patterns
- even if we do not vendor CUTLASS directly, that is the design model we should emulate

References:

- CUTLASS overview: <https://docs.nvidia.com/cutlass/4.3.4/index.html>
- CUTLASS grouped scheduler: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html>

### 3.5 Local `cuda-samples` references that should shape v1

The local `cuda-samples` checkout adds several concrete kernel- and runtime-level references that are directly useful for BitNet.

Most relevant samples:

- `Samples/3_CUDA_Features/immaTensorCoreGemm`
  - use as the reference for CTA tiling, shared-memory staging, skewed layouts, and dynamic shared-memory carveout
  - do not treat it as a direct BitNet solution because it is still an INT8/WMMA path
- `Samples/3_CUDA_Features/globalToShmemAsyncCopy`
  - use as the primary reference for `cuda::memcpy_async`, multi-stage pipelines, and producer/consumer warp partitioning on `sm80+`
- `Samples/0_Introduction/simpleAWBarrier`
  - use as the reference for explicit arrive/wait synchronization between kernel phases
- `Samples/0_Introduction/simpleCooperativeGroups`
  - use as the reference for group-scoped partitioning and subgroup-local reductions
- `Samples/0_Introduction/simpleOccupancy`
  - use to seed shape-bucket launch policy and algorithm selection
- `Samples/6_Performance/cudaGraphsPerfScaling`
  - use to define graph instantiation, upload, and replay policy for decode and stable prefill shapes
- `Samples/6_Performance/transpose`
  - use as the reminder that shared-memory padding and CTA traversal order materially affect throughput
- `Samples/6_Performance/alignedTypes`
  - use to justify explicit alignment and padded-layout metadata in the packed BitNet format

Implication:

- the BitNet backend should combine CUTLASS-style decomposition with `cuda-samples`-style implementation details for staging, synchronization, and runtime policy

## 4. Design Decision

We will build BitNet as a custom packed linear backend with dense fallback, not as:

- a Python-side wrapper around `ctypes`
- a small variation of the current `cublaslt` path
- a decode-only GEMV special case
- a `wmma::experimental` project

The backend will have three layers:

1. format layer
2. kernel layer
3. runtime integration layer

## 5. Format Layer

### 5.1 Canonical stored format

Each BitNet weight tensor will be stored as:

- `packed_weight`
  - unsigned byte or word storage containing packed 2-bit ternary values
- `scale_values`
  - explicit dequantization metadata
- `layout_meta`
  - tile shape, permutation mode, padding, logical widths, and segment metadata

The local BitNet repo's 16x32 block permutation and interleave should be treated as the baseline wire format for v1 because it already matches a known working decode path.

### 5.2 Logical value convention

Stored values represent ternary weights:

- `0 -> -1`
- `1 -> 0`
- `2 -> +1`

The backend must not assume a single scale per matrix.

It must support at least:

- per-matrix scale
- per-segment scale
- per-output-channel-group scale

Segment support is required because the local BitNet conversion path already treats:

- fused `wqkv`
- fused `w13`
- standalone `wo`
- standalone `w2`

as different scaling domains.

### 5.3 Activation quantization metadata

The general backend must use explicit per-row or per-tile activation scaling, not the decode-only `s[0]` assumption from the demo kernel.

Minimum support:

- one activation scale per logical input row of the GEMM
- optional vectorized scale storage for grouped / batched execution

For prefill:

- row means `M = batch * seq`

For decode:

- row means `M = batch * beam` or another compact decode batch shape

### 5.4 New runtime object

Add a first-class runtime-owned module:

- `QuantizedLinearBitNet`

Required state:

- `packed_weight`
- `scale_values`
- `layout_meta`
- optional `bias`
- optional cached dense reference materialization for debugging only

Required methods:

- `runtime_signature()`
- `runtime_bias(...)`
- `runtime_linear(...)`
- `runtime_supports_packed_backend(...)`
- `from_float(...)`
- checkpoint/export load/save helpers

### 5.5 Packed layout invariants

V1 should define one canonical on-device layout and version it explicitly.

Recommended baseline invariants:

- logical weight shape is always `out_features x in_features`
- v1 tile shape matches the local BitNet path's `16 x 32` logical block structure
- logical `K` is padded to a multiple of `32`
- logical `N` is padded to the tile-grid requirement of the selected kernel family
- packed payload stores four 2-bit values per byte before any architecture-specific vectorization
- any permutation or interleave step is described by `layout_meta`, not by hidden kernel assumptions

Minimum `layout_meta` fields:

- `format_version`
- `tile_n`
- `tile_k`
- `logical_out_features`
- `logical_in_features`
- `padded_out_features`
- `padded_in_features`
- `segment_offsets`
- `scale_granularity`
- `interleave_mode`
- `arch_min`

Rule:

- every packer, exporter, importer, and native kernel must accept the same versioned layout contract
- no runtime path may infer the layout from shape alone

### 5.6 Runtime metadata lowering

For storage and export, `layout_meta` can remain a structured metadata object.

For the hot native call path, it should be lowered into fixed tensors rather than a Python dictionary.

Recommended runtime representation:

- `layout_header`
  - rank-1 contiguous `int32` tensor
- `segment_offsets`
  - rank-1 contiguous `int32` tensor of length `segment_count + 1`
- `scale_values`
  - contiguous floating-point tensor, preferably `float32` in v1

Recommended `layout_header` slots:

1. `format_version`
2. `tile_n`
3. `tile_k`
4. `logical_out_features`
5. `logical_in_features`
6. `padded_out_features`
7. `padded_in_features`
8. `scale_granularity`
9. `scale_group_size`
10. `interleave_mode`
11. `arch_min`
12. `segment_count`
13. `flags`

Rules:

- runtime kernels should consume fixed tensors, not inspect Python dicts
- `segment_offsets` should use prefix-sum semantics, starting at `0` and ending at `logical_out_features`
- `scale_values` interpretation is determined by `scale_granularity`, `scale_group_size`, and `segment_offsets`

## 6. Kernel Layer

### 6.1 Kernel families

We need two primary kernel families, not one:

- `bitnet_linear_decode`
  - optimized for very small `M`
  - decode / generation focused
  - persistent CTA scheduling
- `bitnet_linear_prefill`
  - optimized for larger `M`
  - batched prefill / training-style forward inference shapes
  - split-K / grouped / persistent scheduling as needed

Both kernels share:

- the same packed weight format
- the same activation quantization contract
- the same epilogue semantics

### 6.2 Baseline architecture support

V1 support target:

- `sm80+`

Reason:

- the local BitNet kernel and most current repo CUDA assumptions are already Ampere-forward
- Ampere gives us a reliable `dp4a` baseline and aligns with the current extension build expectations

Future optimization lanes:

- Hopper: WGMMA-oriented mainloops and better persistent scheduling
- Blackwell: newer narrow-precision lanes and updated epilogue / scheduler tuning

### 6.3 Compute strategy

#### Baseline path

On Ampere and as the universal fallback kernel family:

1. load INT8 activations
2. decode packed INT2 weights into registers or shared-memory fragments
3. perform INT8 dot-products with INT32 accumulation
4. apply row activation inverse scale and weight scale metadata in the epilogue
5. write BF16 / FP16 / FP32 output

This path is simple, portable, and matches the actual hardware capability exposed by `dp4a`.

#### Optimized path

On Hopper / Blackwell:

- stage decoded weights into a tensor-core-friendly internal representation
- use architecture-specific tiled MMA / WGMMA style kernels where profitable
- keep the same external packed weight format and epilogue

Important rule:

- optimized architecture lanes may change internal decode strategy
- they may not change the public packed format or runtime contract

### 6.4 Epilogue

The epilogue must support:

- activation inverse scale application
- weight scale application
- optional bias
- output cast to BF16 / FP16 / FP32

It should be fused into the kernel path.

Do not materialize:

- a decoded dense weight matrix
- a dequantized activation matrix
- an intermediate INT32 output buffer unless a debug path explicitly requests it

### 6.5 Scheduler requirements

The general backend must own scheduling logic for both decode and prefill.

Decode scheduler requirements:

- persistent CTA mode
- good behavior for `M` in the low single digits
- multiple problems in-flight across heads / projections / experts when possible
- reuse of hot weights in L2 where practical

Prefill scheduler requirements:

- general `M x N x K` GEMM support
- grouped problem mode for many small GEMMs
- split-K option for large `K`
- host-side or device-side precomputation where it improves occupancy

The CUTLASS grouped scheduler and the `flash-attention` varlen scheduler code are the best local references for these patterns.

### 6.6 Problem mapping contract

The external linear contract must stay identical to the current runtime:

- input: `[..., K]`
- output: `[..., N]`

Kernel dispatch should flatten the prefix dimensions into logical rows:

- `M = prod(prefix_shape)`

Important consequences:

- prefill naturally becomes `M = batch * seq`
- decode naturally becomes `M = batch`, `batch * beam`, or another compact serving batch
- attention fused QKV packing is represented as one logical output space with segmented `N`
- grouped scheduling must support multiple logical problems without changing the public operator signature

## 7. Runtime Integration Layer

### 7.1 Native extension surface

Add new native entrypoints in `_model_stack_native`:

- `bitnet_linear_forward(...)`
- `pack_bitnet_weight_forward(...)`
- optional `bitnet_plan_info(...)`

Extend backend resolution:

- `aten`
- `cublaslt`
- `bitnet`

Add runtime info reporting:

- backend availability
- supported dtypes
- packed storage description
- supported architectures

Minimum operator contract:

- `bitnet_linear_forward(x, packed_weight, scale_values, layout_header, segment_offsets, bias=None, out_dtype=None, debug_dense_fallback=False)`
- `pack_bitnet_weight_forward(weight, scale_values=None, layout_header=None, segment_offsets=None)`

These names can still change, but the contract must preserve:

- packed weight as a first-class input
- explicit scale metadata
- explicit lowered layout metadata
- optional segmented output interpretation for fused projections

### 7.2 Python/native glue

Update:

- `runtime/native.py`
- `runtime/ops.py`
- `runtime/quant.py`

Required new helpers:

- `runtime.quant.bitnet_linear(...)`
- `runtime.ops.pack_bitnet_weight(...)`

`runtime_linear_module(...)` must be able to route a BitNet module directly to the native backend without first materializing a dense weight tensor.

### 7.3 Attention integration

The attention packed path in `runtime/attention_modules.py` must stop being `cublaslt`-only.

Required changes:

- allow `bitnet` as a packed backend
- add packed QKV support for BitNet layout metadata
- add packed output projection support for BitNet layout metadata
- preserve segment boundaries and scaling metadata through fused QKV packing

The current cublasLt packers are not enough because BitNet needs:

- a different packed format
- scale metadata
- per-segment interpretation

### 7.4 MLP integration

The current `tensor/mlp.py` path passes raw weight tensors into `runtime_mlp(...)`.

That is not sufficient for BitNet.

One of these must happen:

1. make `runtime_mlp(...)` module-aware, or
2. refactor `tensor/mlp.py` so `w_in` and `w_out` can expose packed module-owned forward paths

For BitNet, module-aware MLP integration is the correct long-term choice because packed formats are now part of the runtime contract.

### 7.5 Dense fallback

The backend must always be able to fall back to:

- dense `cublaslt`
- dense `aten`

Fallback triggers include:

- unsupported architecture
- unsupported shape bucket
- missing packed metadata
- training / autograd mode when the packed path is not yet owned
- explicit debug environment override

### 7.6 Dispatch and autotune policy

`bitnet` should behave like a real backend name, not a single kernel toggle.

Required dispatch dimensions:

- architecture
- `M`, `N`, `K`
- grouped vs single-problem execution
- decode vs prefill preference
- output dtype
- segment layout

Recommended policy:

- cache a lightweight plan or algorithm choice per `(device, dtype, layout version, M bucket, N bucket, K bucket, segment pattern)`
- allow an environment override to force dense fallback or force a reference BitNet kernel for debugging
- keep the plan cache above the kernel layer so attention and MLP can reuse it

## 8. Compression, Checkpoint, and Export

### 8.1 Compression API

Extend `compress/quantization.py` with:

- scheme: `bitnet`

The BitNet quantizer must produce:

- `packed_weight`
- `scale_values`
- `layout_meta`

It must also expose enough metadata for runtime packing reuse and export/import parity.

### 8.2 Export format

Extend `compress/export.py` and `compress/apply.py` with a new quantized type:

- `bitnet_w2a8`

Required export fields:

- type
- packed_weight
- scale_values
- layout_meta
- bias if present
- logical `in_features`
- logical `out_features`

### 8.3 Checkpoint conversion

The current local BitNet conversion scripts are a good starting point for:

- ternary quantization convention
- fused projection packing
- scale extraction rules

But the production runtime should own checkpoint conversion in-repo, not rely on a separate ad hoc script tree.

### 8.4 Versioning and portability

Packed BitNet artifacts must be explicitly versioned.

Minimum requirements:

- persist `format_version` in export artifacts
- preserve logical dense shape alongside padded shape
- preserve scale granularity and segment boundaries
- allow import from either dense float checkpoints or already-packed native artifacts

Recommended portability rule:

- packed artifacts are the fast path
- dense re-materialization remains available for validation, fallback, and forward compatibility

## 9. Suggested File Plan

Recommended native structure:

```text
runtime/csrc/backend/bitnet/
  bitnet_common.cuh
  bitnet_formats.h
  bitnet_pack.cu
  bitnet_linear_decode.cu
  bitnet_linear_prefill.cu
  bitnet_linear_dispatch.cu
  bitnet_epilogue.cuh
```

Recommended Python/runtime additions:

```text
runtime/quant.py
runtime/ops.py
runtime/native.py
compress/quantization.py
compress/export.py
compress/apply.py
```

Tests and benchmarks:

```text
tensor/tests/test_runtime_bitnet_*.py
tests/bench_bitnet_linear.py
tests/bench_bitnet_attention.py
```

## 10. Validation Requirements

Every BitNet path must be validated against dequantized dense reference execution.

Required correctness coverage:

- standalone linear parity for `M in {1, 2, 4, 8, 16, 32, 64, 128, 256, ...}`
- QKV fused projection parity
- output projection parity
- gated MLP parity
- full block parity
- full model parity
- export/import roundtrip parity

Required performance coverage:

- decode latency
- prefill latency
- graph capture / replay
- packed-weight memory footprint
- end-to-end generation latency

Required shape families:

- BitNet-local shapes from `other_repos/BitNet`
- current repo Llama-style shapes
- non-BitNet shapes to confirm clean fallback

## 11. Milestones

1. Land this design and finalize the public runtime contract.
2. Add `QuantizedLinearBitNet` and export/import metadata.
3. Add `pack_bitnet_weight_forward(...)`.
4. Add decode kernel family and backend dispatch.
5. Add prefill kernel family and grouped scheduling.
6. Integrate attention packed QKV / O-proj.
7. Integrate module-aware MLP execution.
8. Add autotune, benchmarking, and graph-capture validation.
9. Expand architecture-specific optimized lanes for Hopper / Blackwell.

## 12. Non-Goals

This spec does not require:

- training-time BitNet backward kernels
- replacing dense linears everywhere immediately
- full cross-architecture support below `sm80`
- turning `other_repos/BitNet` into a runtime dependency

## 13. Short Verdict

For `transformer_10`, a real BitNet backend is:

- a new packed linear backend
- a new quantized module type
- a new kernel family for prefill and decode
- attention and MLP integration work
- explicit checkpoint/export ownership

It is not:

- a `ctypes` wrapper
- a new `cublaslt` mode
- a one-off decode kernel

## 14. References

NVIDIA primary docs:

- CUDA Math API integer intrinsics: <https://docs.nvidia.com/cuda/archive/12.5.1/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html>
- CUDA C++ Programming Guide, WMMA / sub-byte operations: <https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html>
- cuBLAS / cuBLASLt documentation: <https://docs.nvidia.com/cuda/cublas/index.html>
- CUTLASS overview: <https://docs.nvidia.com/cutlass/4.3.4/index.html>
- CUTLASS grouped scheduler: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html>

Local code references:

- `runtime/csrc/model_stack_native.cpp`
- `runtime/csrc/backend/cublaslt_linear.cu`
- `runtime/csrc/backend/cuda_int4_linear.cu`
- `runtime/attention_modules.py`
- `tensor/mlp.py`
- `compress/quantization.py`
- `other_repos/BitNet/gpu/model.py`
- `other_repos/BitNet/gpu/pack_weight.py`
- `other_repos/BitNet/gpu/convert_checkpoint.py`
- `other_repos/flash-attention/hopper/flash_prepare_scheduler.cu`
- `other_repos/flash-attention/flash_attn/cute/tile_scheduler.py`
- `other_repos/ThunderKittens/README.md`
