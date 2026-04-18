# transformer_10 BitNet Backend Bring-Up Plan

This document turns the general BitNet backend spec into a repo-specific implementation checklist.

It is not a replacement for:

- `transformer10-general-native-bitnet-backend-spec.md`

It is the concrete answer to:

- what files do we touch first
- what should land in each phase
- what can be stubbed early
- what must be validated before moving on

## 1. Working Rule

The bring-up order should preserve one stable public runtime contract while allowing the internal kernel strategy to evolve.

That means:

- land the backend name and packed module type early
- land correctness and fallback before chasing performance
- keep decode and prefill as separate kernel tracks
- do not tie the public ABI to one kernel family, one architecture, or one sample implementation

## 2. Target Repo Surfaces

The minimum repo surfaces for the BitNet backend are:

- `runtime/csrc/model_stack_native.cpp`
- `runtime/native.py`
- `runtime/ops.py`
- `runtime/quant.py`
- `runtime/attention_modules.py`
- `tensor/mlp.py`
- `compress/quantization.py`
- `compress/export.py`
- `compress/apply.py`
- `setup.py`

The minimum native file set for v1 should be:

```text
runtime/csrc/backend/bitnet/
  bitnet_common.cuh
  bitnet_formats.h
  bitnet_pack.cu
  bitnet_linear_dispatch.cu
  bitnet_linear_decode.cu
  bitnet_linear_prefill.cu
  bitnet_epilogue.cuh
```

## 3. Phase Plan

### Phase 0: Lock The Public ABI

Goal:

- make `bitnet` a real backend name and reserve the operator surface before the fast kernels exist

Files:

- `runtime/csrc/model_stack_native.cpp`
- `runtime/native.py`
- `runtime/ops.py`
- `runtime/quant.py`

Required work:

- extend linear backend resolution to accept `bitnet`
- expose `bitnet_linear_forward(...)`
- expose `pack_bitnet_weight_forward(...)`
- expose runtime info fields for:
  - backend availability
  - packed storage type
  - layout version
  - supported architectures
  - supported output dtypes
- add Python wrappers:
  - `runtime.quant.bitnet_linear(...)`
  - `runtime.ops.pack_bitnet_weight(...)`

Recommended hot-path metadata contract:

- `layout_header`
  - fixed-width `int32` tensor
- `segment_offsets`
  - `int32` prefix-sum tensor
- `scale_values`
  - contiguous floating-point tensor

The Python and export layer may still keep a richer `layout_meta` object, but the native operator should receive lowered tensors.

Recommended v1 runtime-info keys:

- `bitnet_available`
- `bitnet_arches`
- `bitnet_linear_dtypes`
- `bitnet_weight_storage`
- `bitnet_layout_version`

Exit criteria:

- `resolve_linear_backend("bitnet")` is legal
- `runtime_info()` advertises the backend
- Python can call the backend entrypoints without shape inference or hidden metadata

Important rule:

- even in this phase, do not invent an ABI that omits `layout_meta`
- the layout contract must exist before optimized kernels do

### Phase 1: Add `QuantizedLinearBitNet`

Goal:

- land the module type and packed artifact schema

Files:

- `compress/quantization.py`
- `compress/export.py`
- `compress/apply.py`
- optionally `compress/tests/...` or `tensor/tests/...`

Required work:

- add `QuantizedLinearBitNet`
- add `scheme == "bitnet"` to `quantize_linear_modules(...)`
- implement `from_float(...)` using in-repo packing logic derived from `other_repos/BitNet/gpu/pack_weight.py`
- make `runtime_signature()` include:
  - packed weight
  - scale metadata
  - layout metadata
  - optional bias
- implement `runtime_supports_packed_backend("bitnet")`
- add export/import support using a stable type name such as `bitnet_w2a8`

Minimum module state:

- `packed_weight`
- `scale_values`
- `layout_meta`
- `layout_header`
- `segment_offsets`
- optional `bias`

Recommended module behavior in this phase:

- `runtime_linear(...)` may still use a correctness-first fallback path
- dequantized dense materialization is allowed for debug and parity, not as the intended steady state

Exit criteria:

- model replacement works through `apply_compression(..., quant={"scheme": "bitnet"})`
- delta export/import round-trips preserve packed state and metadata
- module signatures invalidate caches when any packed tensor changes

### Phase 2: Land The Packer And Reference Native Path

Goal:

- make the native extension own BitNet packing and correctness-first execution

Files:

- `runtime/csrc/backend/bitnet/bitnet_formats.h`
- `runtime/csrc/backend/bitnet/bitnet_pack.cu`
- `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu`
- `runtime/csrc/model_stack_native.cpp`
- `setup.py`

Required work:

- define a versioned `layout_meta` schema in native code
- define the lowered runtime form:
  - `layout_header`
  - `segment_offsets`
- implement native pack validation:
  - logical shape checks
  - padding checks
  - segment-offset checks
  - scale-granularity checks
- implement `pack_bitnet_weight_forward(...)`
- implement a correctness-first `bitnet_linear_forward(...)` path that:
  - validates the packed contract
  - chooses `decode` vs `prefill` mode from `M`
  - falls back to dense if the optimized kernel is unavailable

Recommended fallback behavior:

- unpack or dequantize only inside a dedicated debug/reference path
- never hide that fallback behind an ABI change

Recommended packer outputs:

- `packed_weight`
- `scale_values`
- `layout_header`
- `segment_offsets`

Exit criteria:

- native packing returns artifacts that match Python-side metadata expectations
- native linear execution is callable end-to-end from a `QuantizedLinearBitNet`
- unsupported shapes fall back cleanly to dense execution

### Phase 3: Decode Kernel Family

Goal:

- bring up the small-`M` kernel path for generation

Files:

- `runtime/csrc/backend/bitnet/bitnet_linear_decode.cu`
- `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu`

Primary references:

- `other_repos/BitNet/gpu/bitnet_kernels/bitnet_kernels.cu`
- `other_repos/flash-attention/hopper/flash_prepare_scheduler.cu`
- `docs/research/nvidia-cuda-samples-standalone-notes.md`
  - `simpleAWBarrier`
  - `simpleCooperativeGroups`
  - `cudaGraphsPerfScaling`

Required work:

- decode packed INT2 weights into INT8 fragments
- compute INT32 accumulators
- fuse weight/activation scale application in the epilogue
- optimize for:
  - `M in {1, 2, 4, 8}`
  - persistent CTA scheduling
  - hot-weight reuse
- support segmented outputs for fused projections where practical

Explicit non-goal for this phase:

- do not force the decode kernel to also be the general prefill kernel

Exit criteria:

- decode latency is benchmarkable through the native backend
- correctness holds for low-`M` batches and common llama-like widths
- graph capture does not regress versus dense fallback on stable shape buckets

### Phase 4: Prefill Kernel Family

Goal:

- add the general batched GEMM path

Files:

- `runtime/csrc/backend/bitnet/bitnet_linear_prefill.cu`
- `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu`

Primary references:

- `other_repos/flash-attention/flash_attn/cute/tile_scheduler.py`
- `docs/research/nvidia-cuda-samples-standalone-notes.md`
  - `globalToShmemAsyncCopy`
  - `transpose`
  - `alignedTypes`
  - `simpleOccupancy`

Required work:

- support general `M x N x K`
- add grouped scheduling or problem batching for many small linears
- add split-K or equivalent strategy for large `K`
- use multi-stage shared-memory pipelines on `sm80+` where profitable
- keep the same public packed format as the decode kernel

Exit criteria:

- prefill supports real transformer batch/sequence shapes
- unsupported shape buckets still fall back to dense
- plan selection distinguishes decode and prefill instead of pretending one kernel solves both

### Phase 5: Attention Integration

Goal:

- stop treating packed attention as `cublaslt`-only

Files:

- `runtime/attention_modules.py`
- `runtime/ops.py`
- `runtime/csrc/model_stack_native.cpp`

Required work:

- allow `_packed_backend(...)` to return `bitnet`
- cache packed QKV artifacts for BitNet modules
- cache packed output-projection artifacts for BitNet modules
- preserve:
  - `segment_offsets`
  - `scale_values`
  - `layout_meta`
- route `runtime_linear_module(...)` through module-owned forward methods where possible

Recommended design:

- add BitNet-specific packed QKV helpers instead of trying to overload the current cublasLt packer with incompatible semantics

Exit criteria:

- QKV and O-proj can run through the BitNet backend without materializing dense weights
- attention cache invalidation keys include layout/signature metadata, not just raw tensor pointers

### Phase 6: MLP Integration

Goal:

- remove the dense-only assumption in `tensor/mlp.py`

Files:

- `tensor/mlp.py`
- `runtime/ops.py`
- optionally `runtime/csrc/model_stack_native.cpp`

Current issue:

- `runtime_mlp(...)` currently takes raw weights and biases, which is incompatible with module-owned packed BitNet weights

Two acceptable directions:

1. make `runtime_mlp(...)` module-aware
2. refactor `MLP.forward(...)` to use `runtime_linear_module(...)` or module-owned `runtime_linear(...)` for both projections

Preferred direction:

- move toward module-aware execution instead of preserving a raw-weight-only MLP path

Exit criteria:

- `MLP` can execute BitNet `w_in` / `w_out` without dequantizing to dense tensors
- gated MLP variants still preserve activation semantics

### Phase 7: Plan Cache, Heuristics, And Graph Policy

Goal:

- keep dispatch decisions above the kernel layer and make them reusable

Files:

- `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu`
- `runtime/csrc/model_stack_native.cpp`
- optionally `runtime/native.py`

Required work:

- cache a plan key that includes:
  - device
  - architecture
  - layout version
  - output dtype
  - `M`, `N`, `K` bucket
  - segment pattern
  - decode vs prefill mode
- add debug overrides for:
  - force dense fallback
  - force reference BitNet kernel
  - dump plan selection
- keep CUDA Graph behavior explicit:
  - instantiation cost
  - upload timing
  - repeat-launch timing

Primary references:

- `docs/research/nvidia-cuda-samples-standalone-notes.md`
  - `simpleOccupancy`
  - `cudaGraphsPerfScaling`
- `docs/research/nvidia-cudalibrarysamples-standalone-notes.md`

Exit criteria:

- plan selection is deterministic per shape bucket
- benchmarking can separate graph upload overhead from steady-state kernel latency

### Phase 8: Validation And Benchmarks

Goal:

- make the backend safe to iterate on

Suggested tests:

- `tensor/tests/test_runtime_bitnet_linear.py`
- `tensor/tests/test_runtime_bitnet_attention.py`
- `tensor/tests/test_runtime_bitnet_mlp.py`
- `tensor/tests/test_runtime_bitnet_export.py`

Suggested benchmarks:

- `tests/bench_bitnet_linear.py`
- `tests/bench_bitnet_attention.py`
- `tests/bench_bitnet_decode.py`

Required parity coverage:

- linear parity vs dense dequantized reference
- fused-QKV parity
- output-projection parity
- gated MLP parity
- full block parity
- export/import roundtrip parity

Required shape buckets:

- BitNet-local demo shapes from `other_repos/BitNet`
- current repo llama-like widths
- deliberately unsupported shapes to verify fallback

Required performance coverage:

- decode latency
- prefill latency
- graph capture and replay
- packed-weight memory footprint
- dense-fallback regression comparison

Exit criteria:

- the backend has repeatable numerical parity checks
- the benchmark harness can compare:
  - BitNet decode vs dense fallback
  - BitNet prefill vs dense fallback
  - graph vs non-graph execution

## 4. Current Gap Matrix

This is the current repo reality that blocks BitNet today.

### Gap 1: Backend registry does not know `bitnet`

Current state:

- `runtime/csrc/model_stack_native.cpp` advertises dense linear plus the new INT4 helper path
- there is no `bitnet_linear_forward(...)`
- there is no `pack_bitnet_weight_forward(...)`
- runtime info has no BitNet capability fields

Consequence:

- even a correct packed BitNet module would have no native backend contract to call

### Gap 2: Python runtime surface has no BitNet helper

Current state:

- `runtime/native.py` only normalizes the current native op set
- `runtime/quant.py` exposes `int4_linear(...)`, but no `bitnet_linear(...)`
- `runtime/__init__.py` does not export a BitNet quant helper

Consequence:

- there is no stable Python-side API for a BitNet packed module

### Gap 3: Quantization layer only knows `int8`, `int4`, and `fp8`

Current state:

- `compress/quantization.py` replaces `nn.Linear` only for:
  - `int8`
  - `int4`
  - `fp8`
- `compress/export.py` and `compress/apply.py` only serialize/apply those types

Consequence:

- no model can be quantized into a repo-native BitNet representation

### Gap 4: Packed attention path is hard-coded around `cublaslt`

Current state:

- `runtime/attention_modules.py` only enables the packed fast path when the resolved backend is `cublaslt`
- `runtime_pack_qkv_weights(...)` assumes dense packed tensors and optional dense bias

Consequence:

- even with a BitNet linear kernel, attention would still fall back to the dense path

### Gap 5: MLP path is still raw-weight based

Current state:

- `tensor/mlp.py` routes inference through `runtime_mlp(...)`
- that call passes `self.w_in.weight`, `self.w_in.bias`, `self.w_out.weight`, `self.w_out.bias`

Consequence:

- a packed BitNet module would be forced back into dense tensor materialization

### Gap 6: Existing INT4 work is scaffolding, not a direct template

Current state:

- `QuantizedLinearInt4` already demonstrates:
  - quantized module replacement
  - packed export/import
  - native helper routing
- but `runtime_supports_packed_backend(...)` still returns `False`
- the CUDA kernel is scalar unpack-and-dot rather than a tiled backend

Consequence:

- the hook points are real and useful
- the kernel strategy is not sufficient for BitNet

## 5. First PR Sequence

The best way to land this work is as a sequence of narrow PRs that lock interfaces early.

### PR 1: Public ABI And Quantized Module

Goal:

- make `bitnet` a legal runtime/backend concept across Python and C++

Files:

- `runtime/csrc/model_stack_native.cpp`
- `runtime/native.py`
- `runtime/ops.py`
- `runtime/quant.py`
- `runtime/__init__.py`
- `compress/quantization.py`
- `compress/export.py`
- `compress/apply.py`
- `setup.py`

Contents:

- register backend name `bitnet`
- add stub entrypoints:
  - `bitnet_linear_forward(...)`
  - `pack_bitnet_weight_forward(...)`
- add `QuantizedLinearBitNet`
- add `scheme == "bitnet"`
- add export/import serialization for packed BitNet state
- allow correctness-first fallback behavior

What this PR should not do:

- it should not contain the fast decode or prefill kernels yet

Success condition:

- models can be converted to BitNet wrappers and call a stable runtime API

### PR 2: Native Packer And Reference Dispatch

Goal:

- move packing and packed-layout validation into the native extension

Files:

- `runtime/csrc/backend/bitnet/bitnet_formats.h`
- `runtime/csrc/backend/bitnet/bitnet_pack.cu`
- `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu`
- `runtime/csrc/model_stack_native.cpp`

Contents:

- versioned `layout_meta`
- native packer
- native validation
- reference `bitnet_linear_forward(...)` path with dense fallback
- runtime info reporting for BitNet capabilities

Success condition:

- the native module owns the packed contract even if performance is still fallback-heavy

### PR 3: Attention And MLP Contract Fixes

Goal:

- remove the two highest-level dense-only assumptions before kernel optimization

Files:

- `runtime/attention_modules.py`
- `runtime/ops.py`
- `tensor/mlp.py`

Contents:

- packed attention support for `bitnet`
- BitNet-aware packed QKV / O-proj caching
- module-aware MLP execution path

Success condition:

- BitNet modules can flow through the current model stack without implicit dense materialization

### PR 4: Decode Kernel Family

Goal:

- make small-`M` generation performance real

Files:

- `runtime/csrc/backend/bitnet/bitnet_linear_decode.cu`
- `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu`

Contents:

- low-`M` persistent kernel family
- decode-focused scheduling
- graph-capture validation on stable shapes

### PR 5: Prefill Kernel Family And Plan Cache

Goal:

- close the gap to a general backend

Files:

- `runtime/csrc/backend/bitnet/bitnet_linear_prefill.cu`
- `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu`
- `runtime/csrc/model_stack_native.cpp`

Contents:

- general batched prefill path
- grouped or split-K options
- shape-bucket plan cache
- benchmark harness integration

## 6. File Ownership Checklist

### Native runtime and ABI

- `runtime/csrc/model_stack_native.cpp`
  - add backend registration
  - add pybind entrypoints
  - add runtime info keys
  - add dispatch plumbing
- `setup.py`
  - compile the BitNet backend sources

### Python runtime glue

- `runtime/native.py`
  - expose backend info
  - expose native op discovery
- `runtime/ops.py`
  - add pack helpers
  - add module-aware dispatch where needed
- `runtime/quant.py`
  - add public `bitnet_linear(...)` helper

### Module and compression layer

- `compress/quantization.py`
  - add `QuantizedLinearBitNet`
  - add quant scheme selection
- `compress/export.py`
  - serialize packed artifacts
- `compress/apply.py`
  - wire the new scheme into public compression application

### Model-stack integration

- `runtime/attention_modules.py`
  - packed attention routing
  - cache keys
  - BitNet pack handling
- `tensor/mlp.py`
  - module-aware MLP execution

## 7. Early Decisions We Should Not Revisit Repeatedly

These should be treated as locked unless evidence appears that they are wrong:

- `bitnet` is a first-class backend name
- packed layout is versioned
- `layout_meta` is explicit and mandatory
- decode and prefill are separate kernel families
- dense fallback remains available
- the first stable hardware target is `sm80+`
- BitNet is integrated as a packed linear backend first, not as a new full-block architecture

## 8. Known Risks

### Risk 1: INT4 scaffolding looks tempting to reuse directly

Reality:

- the repo's INT4 path provides useful hook points, but not the right kernel structure

Mitigation:

- copy the integration pattern only

### Risk 2: Attention integration drifts into special cases

Reality:

- the current packed attention path is implicitly designed around dense packed weights

Mitigation:

- treat scale metadata and segment offsets as part of the core packed-attention contract

### Risk 3: MLP remains raw-weight based for too long

Reality:

- that would force hidden dense materialization and undercut the whole backend

Mitigation:

- prioritize module-aware MLP integration before performance tuning

### Risk 4: The ABI becomes tied to one optimized kernel

Reality:

- that will break as soon as Hopper/Blackwell lanes differ from Ampere

Mitigation:

- keep layout/version/metadata stable and let only dispatch/mainloops vary

## 9. Recommended Landing Order

If this work starts immediately, the best landing order is:

1. Phase 0: backend name and public ABI
2. Phase 1: `QuantizedLinearBitNet` and export/import
3. Phase 2: native packer and correctness-first dispatch
4. Phase 5: attention integration
5. Phase 6: MLP integration
6. Phase 3: decode kernel family
7. Phase 4: prefill kernel family
8. Phase 7: plan cache and graph policy
9. Phase 8: final benchmark and rollout gates

Reason:

- the repo can start exercising the real backend contract before the fast kernels are finished
- that reduces the risk of building a fast kernel behind the wrong ABI

## 10. Definition Of Done

The BitNet backend is not done when a demo kernel runs.

It is done when:

- `bitnet` is selectable through the normal runtime backend path
- BitNet packed modules survive export/import
- attention and MLP both execute without dense weight materialization
- decode and prefill both have native kernels or well-defined fallback behavior
- graph capture is validated
- benchmarks show where BitNet wins, where it falls back, and why
