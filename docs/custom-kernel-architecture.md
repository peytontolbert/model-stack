# Custom Kernel Architecture

This document describes how Model Stack's custom kernel system is organized, how
Python dispatch reaches C++/CUDA code, which kernel families exist today, and
what contracts new kernels must preserve.

Model Stack does not treat kernels as isolated benchmark files. A kernel is part
of a system that includes Python wrappers, native extension metadata, policy
selection, packed runtime formats, validation, tests, benchmarks, export paths,
and fallbacks. That integration is the main architectural rule.

## 1. System Overview

The runtime stack has five layers:

```text
model / blocks / compress
  high-level modules, quantized modules, trainable BitNet layers

runtime/ops.py and runtime/quant.py
  Python dispatch, eager fallbacks, native-op gates, packed spec execution

runtime/native.py
  lazy native extension loading, runtime_info normalization, has_native_op()

runtime/csrc/model_stack_native.cpp
  pybind11 ABI boundary, op registration, metadata, high-level native routing

runtime/csrc/backend/
  CUDA/C++ backend kernels and backend-specific launch policy
```

The Python layer should always remain usable without the native extension. Native
kernels are acceleration and systems integration paths, not import-time
requirements.

The native extension is intentionally discoverable at runtime:

- `runtime.native.native_module()` lazily imports the extension.
- `runtime.native.has_native_op(name)` checks `runtime_info()` and fallback
  metadata before Python dispatch selects a native call.
- `runtime.native.resolve_linear_backend()` normalizes `"auto"`, `"aten"`,
  `"cublaslt"`, and native linear choices through the extension when available.
- `runtime/csrc/model_stack_native.cpp` exposes `runtime_info()` with compiled
  CUDA status, supported ops, kernel families, architecture notes, and feature
  flags.

## 2. Build Architecture

Native compilation is opt-in. `setup.py` creates no extension unless
`MODEL_STACK_BUILD_NATIVE=1` is set.

Typical CUDA build:

```bash
MODEL_STACK_BUILD_NATIVE=1 \
MODEL_STACK_BUILD_CUDA=1 \
MODEL_STACK_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX" \
python setup.py build_ext --inplace
```

H100-only build:

```bash
MODEL_STACK_BUILD_NATIVE=1 \
MODEL_STACK_BUILD_CUDA=1 \
MODEL_STACK_CUDA_ARCH_LIST="9.0" \
MODEL_STACK_MAX_JOBS=4 \
python setup.py build_ext --inplace
```

Important build controls:

| Variable | Purpose |
| --- | --- |
| `MODEL_STACK_BUILD_NATIVE` | Enables the C++/CUDA extension. |
| `MODEL_STACK_BUILD_CUDA` | Enables CUDA sources when CUDA is available. |
| `MODEL_STACK_CUDA_ARCH_LIST` | Repo-local alias for PyTorch's CUDA arch list. |
| `TORCH_CUDA_ARCH_LIST` | PyTorch-native architecture override; wins if set. |
| `MODEL_STACK_MAX_JOBS` | Repo-local parallel build limit. |
| `MODEL_STACK_USE_NINJA=0` | Forces setuptools build fallback. |
| `MODEL_STACK_CUTLASS_PATH` | Adds optional CUTLASS include roots. |
| `MODEL_STACK_PYTORCH_SOURCE_PATH` | Allows source-tree PyTorch attention headers. |
| `MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL` | Allows experimental SM90a targets when toolchain supports them. |

The CUDA source list is centralized in `setup.py`. Current source families
include:

- norms and residuals: `cuda_rms_norm.cu`, `cuda_add_rms_norm.cu`,
  `cuda_layer_norm.cu`, `cuda_residual_add.cu`
- embeddings, sampling, tokens, decode positions
- attention and paged KV cache
- quantized linear: INT8, INT4, NF4, FP8, cuBLASLt, CUTLASS
- BitNet and ternary kernels
- QKV projection, activation, gated activation, MLP helpers

When adding a backend file, add it to `setup.py`, expose capability metadata in
`runtime_info()`, bind it in `model_stack_native.cpp`, and add tests that prove
the Python dispatch can see the new op.

## 3. Native ABI Boundary

`runtime/csrc/model_stack_native.cpp` is the single pybind11 boundary for the
native extension. It has three responsibilities:

1. Validate public native arguments before backend launch.
2. Route to CUDA backend code when compiled and eligible.
3. Fall back to ATen/reference implementations when native CUDA is absent or
   unsupported for the requested shape/dtype.

The extension exposes:

- `runtime_info()`
- `has_op(name)`
- `resolve_linear_backend(requested)`
- one `*_forward` binding per public op

The binding layer should not contain complex kernel bodies. Kernel bodies live
under `runtime/csrc/backend/`. The binding layer may contain cross-backend
routing, reference fallback calls, and module-state extraction.

`runtime_info()` is part of the contract. It reports:

- extension ABI version
- CUDA compilation status
- `native_ops`
- `cuda_backend_ops`
- supported linear backends
- attention kernel families and policy flags
- BitNet layout/kernel information
- quantized linear storage formats and architecture notes

Tests under `tests/test_runtime_quant_source_surface.py` and related source
surface tests assert that critical bindings and source files remain registered.

## 4. Python Dispatch Rules

Python dispatch lives mostly in:

- `runtime/ops.py`: general runtime ops, attention, KV cache, linear, QKV,
  activations, BitNet packed specs
- `runtime/quant.py`: quantized linear APIs, BitNet linear APIs, INT8/NF4/FP8
  helpers, custom compile ops
- `compress/quantization.py`: quantized modules, BitNet QAT layers, packed
  state construction, runtime packed specs

Dispatch rules:

1. Normalize dtype, device, shape, and packed metadata in Python before native
   calls when cheap.
2. Use `has_native_op()` plus `native_module()` before calling a native symbol.
3. Preserve eager PyTorch fallback behavior under autograd unless a custom
   autograd path explicitly owns backward.
4. Keep CPU fallback behavior semantically aligned with CUDA.
5. Validate bit widths and packed format metadata consistently in Python and
   native code.

Autograd policy is deliberate:

- Many inference kernels use native only when gradients are not required.
- Some quantized paths have custom `torch.autograd.Function` wrappers.
- Some training BitNet modes are gated by environment variables and shape
  policies.

When adding a native op, do not make import-time native extension availability a
requirement. The Python fallback remains the baseline correctness oracle.

## 5. Backend Source Layout

The backend directory is organized by kernel family:

```text
runtime/csrc/backend/
  cuda_device_arch.cuh
  cuda_*                  general CUDA kernels
  cublaslt_linear.cu
  cutlass_*               optional CUTLASS-backed kernels
  attention/              custom attention dispatch and kernels
  bitnet/                 BitNet and ternary kernels

runtime/csrc/descriptors/
  attention_desc.h

runtime/csrc/policy/
  attention_policy.h

runtime/csrc/reference/
  aten_reference.h
  aten_reference.cpp
```

Use descriptors and policy headers for reusable shape decisions. Keep kernel
launch selection close to the backend but keep repository-wide routing visible
from `model_stack_native.cpp` and `runtime_info()`.

## 6. Implemented CUDA Source Inventory

This inventory is intentionally explicit. Every CUDA translation unit currently
registered in `setup.py` should appear here, so documentation review can catch
new kernels that were added without architecture notes.

| Source | Kernel responsibility |
| --- | --- |
| `runtime/csrc/backend/cuda_rms_norm.cu` | RMSNorm forward path. |
| `runtime/csrc/backend/cuda_add_rms_norm.cu` | residual/update add plus RMSNorm fusion. |
| `runtime/csrc/backend/cuda_residual_add.cu` | residual add utility kernel. |
| `runtime/csrc/backend/cuda_layer_norm.cu` | LayerNorm and add-LayerNorm forward paths. |
| `runtime/csrc/backend/cuda_embedding.cu` | embedding gather with padding handling. |
| `runtime/csrc/backend/cuda_sampling.cu` | temperature, top-k/top-p, penalties, token counts, beam/speculative sampling helpers. |
| `runtime/csrc/backend/cuda_append_tokens.cu` | append sampled token IDs to decode sequences. |
| `runtime/csrc/backend/cuda_decode_positions.cu` | decode position ID construction. |
| `runtime/csrc/backend/cuda_attention.cu` | public CUDA attention launch surface and routing. |
| `runtime/csrc/backend/attention/cuda_attention_decode_dispatch.cu` | decode attention dispatch registration. |
| `runtime/csrc/backend/attention/cuda_attention_prefill_dispatch.cu` | prefill attention dispatch registration. |
| `runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cu` | SM80/Ada inference prefill lane. |
| `runtime/csrc/backend/attention/cuda_attention_pytorch_memeff_prefill.cu` | PyTorch mem-efficient prefill bridge when headers are available. |
| `runtime/csrc/backend/attention/cuda_attention_sm80_flash_prefill.cu` | local SM80 Flash-style prefill lane. |
| `runtime/csrc/backend/cuda_kv_cache.cu` | contiguous KV, paged KV, INT3 KV, paged decode bridge, and fused decode cache writes. |
| `runtime/csrc/backend/cuda_rope.cu` | rotary embedding application and rotary metadata helpers. |
| `runtime/csrc/backend/cuda_activation.cu` | scalar activation kernels such as GELU/SILU/ReLU-family paths. |
| `runtime/csrc/backend/cuda_gated_activation.cu` | gated activation kernels such as SwiGLU/GEGLU-style paths. |
| `runtime/csrc/backend/cuda_fp8_linear.cu` | FP8 storage decode and linear execution. |
| `runtime/csrc/backend/cuda_int4_linear.cu` | packed INT4 linear and grad-input helper. |
| `runtime/csrc/backend/cuda_nf4_linear.cu` | packed NF4 codebook linear. |
| `runtime/csrc/backend/bitnet/bitnet_pack.cu` | BitNet packed weight creation and runtime-row ternary quantization. |
| `runtime/csrc/backend/bitnet/bitnet_ternary_linear.cu` | ternary mask packing, ternary activation quantization, strict ternary matmul. |
| `runtime/csrc/backend/bitnet/bitnet_frontend.cu` | BitNet input transform, spin/pre-scale, and activation quantization frontends. |
| `runtime/csrc/backend/bitnet/bitnet_linear_decode.cu` | BitNet decode-persistent, row1 bitplane, and fused norm decode kernels. |
| `runtime/csrc/backend/bitnet/bitnet_linear_prefill.cu` | BitNet tiled prefill and split-K prefill kernels. |
| `runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu` | BitNet public CUDA dispatch, plan selection, and safety validation. |
| `runtime/csrc/backend/bitnet/bitnet_attention_decode_dispatch.cu` | BitNet QKV decode attention dispatch helpers. |
| `runtime/csrc/backend/bitnet/bitnet_attention_prefill_dispatch.cu` | BitNet QKV prefill attention dispatch helpers. |
| `runtime/csrc/backend/bitnet/bitnet_attention_dispatch.cu` | fused BitNet QKV projection and attention-facing routing. |
| `runtime/csrc/backend/cuda_int8_attention.cu` | INT8 attention and float-input quantized attention path. |
| `runtime/csrc/backend/cuda_int8_linear.cu` | INT8 linear, float-input INT8 linear, and grad-weight helper. |
| `runtime/csrc/backend/cuda_quant_int8_frontend.cu` | rowwise, transpose, columnwise, ReLU2, and leaky-ReLU-half2 INT8 activation quantization. |
| `runtime/csrc/backend/cutlass_int4_linear.cu` | optional CUTLASS BF16 INT4 linear and shuffled packing. |
| `runtime/csrc/backend/cutlass_int8_linear.cu` | optional CUTLASS INT8 linear support. |
| `runtime/csrc/backend/cublaslt_linear.cu` | dense/cuBLASLt and INT8 accumulation backends. |

Important policy/header files:

| Header | Role |
| --- | --- |
| `runtime/csrc/backend/cuda_device_arch.cuh` | device architecture helpers used by CUDA dispatch policy. |
| `runtime/csrc/backend/cuda_hopper_advanced.cuh` | Hopper/SM90 feature gates and experimental helpers. |
| `runtime/csrc/backend/attention/cuda_attention_common.cuh` | shared attention constants, utilities, and validation helpers. |
| `runtime/csrc/backend/attention/cuda_attention_dispatch.cuh` | attention dispatch declarations. |
| `runtime/csrc/backend/attention/cuda_attention_decode.cuh` | decode kernel implementations/templates. |
| `runtime/csrc/backend/attention/cuda_attention_prefill.cuh` | generic prefill implementations/templates. |
| `runtime/csrc/backend/attention/cuda_attention_cutlass_prefill.cuh` | CUTLASS-backed prefill hooks. |
| `runtime/csrc/backend/attention/cuda_attention_pytorch_memeff_prefill.cuh` | PyTorch mem-efficient prefill bridge declarations. |
| `runtime/csrc/backend/attention/cuda_attention_sm80_flash_prefill.cuh` | SM80 Flash-style prefill declarations. |
| `runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cuh` | SM80/Ada inference prefill declarations. |
| `runtime/csrc/backend/bitnet/bitnet_attention_common.cuh` | shared BitNet attention descriptors/helpers. |
| `runtime/csrc/backend/bitnet/bitnet_common.cuh` | BitNet plan, validation, qmax, device, and launch declarations. |
| `runtime/csrc/backend/bitnet/bitnet_epilogue.cuh` | BitNet fused epilogue helpers. |
| `runtime/csrc/backend/bitnet/bitnet_formats.h` | BitNet layout constants and format validation. |
| `runtime/csrc/descriptors/attention_desc.h` | normalized attention descriptor. |
| `runtime/csrc/policy/attention_policy.h` | attention backend selection policy. |

## 7. Public Native Operation Inventory

Every pybind symbol in `runtime/csrc/model_stack_native.cpp` should appear in
this section. The operation name is the ABI-facing contract used by Python
dispatch and source-surface tests.

| Family | Native operation names |
| --- | --- |
| Runtime metadata | `runtime_info`, `has_op`, `resolve_linear_backend` |
| Norms and residuals | `rms_norm_forward`, `add_rms_norm_forward`, `residual_add_forward`, `layer_norm_forward`, `add_layer_norm_forward` |
| Rotary and cache | `apply_rotary_forward`, `kv_cache_append_forward`, `kv_cache_write_forward`, `kv_cache_gather_forward`, `int3_kv_pack_forward`, `int3_kv_dequantize_forward` |
| Paged KV and decode | `paged_kv_assign_blocks_forward`, `paged_kv_reserve_pages_forward`, `paged_kv_read_range_forward`, `paged_kv_read_last_forward`, `paged_kv_append_forward`, `paged_kv_compact_forward`, `paged_kv_gather_forward`, `paged_kv_write_forward`, `paged_attention_decode_forward` |
| Attention prep and planning | `prepare_attention_mask_forward`, `resolve_position_ids_forward`, `create_causal_mask_forward`, `resolve_rotary_embedding_forward`, `attention_forward`, `attention_partitioned_reference_forward`, `attention_plan_info` |
| Sampling and decoding utilities | `temperature_forward`, `topk_mask_forward`, `topp_mask_forward`, `apply_sampling_mask_forward`, `sample_with_policies_forward`, `presence_frequency_penalty_forward`, `no_repeat_ngram_mask_forward`, `sample_next_token_forward`, `speculative_accept_forward`, `beam_search_step_forward`, `incremental_beam_search_forward`, `repetition_penalty_forward`, `token_counts_forward`, `append_tokens_forward`, `decode_positions_forward` |
| Activations and dense projections | `activation_forward`, `gated_activation_forward`, `linear_forward`, `linear_module_forward`, `pack_linear_weight_forward`, `mlp_forward`, `embedding_forward` |
| Low-bit and quantized linear | `nf4_linear_forward`, `fp8_linear_forward`, `int8_quantize_activation_forward`, `int8_quantize_activation_transpose_forward`, `int8_quantize_activation_columnwise_forward`, `int8_quantize_relu2_activation_forward`, `int8_quantize_leaky_relu_half2_activation_forward`, `int8_linear_forward`, `int8_linear_from_float_forward`, `int8_linear_grad_weight_from_float_forward`, `int8_linear_accum_forward`, `int4_linear_forward`, `int4_linear_grad_input_forward`, `cutlass_int4_bf16_linear_forward`, `cutlass_int4_pack_shuffled_forward` |
| Quantized attention | `int8_attention_forward`, `int8_attention_from_float_forward` |
| BitNet linear and frontends | `bitnet_transform_input_forward`, `bitnet_linear_from_float_forward`, `bitnet_int8_linear_from_float_forward`, `bitnet_linear_forward`, `bitnet_linear_compute_packed_forward`, `pack_bitnet_weight_forward`, `bitnet_runtime_row_quantize_forward` |
| BitNet ternary training/runtime | `bitnet_ternary_pack_masks_forward`, `bitnet_ternary_pack_masks64_forward`, `bitnet_ternary_quantize_activation_forward`, `bitnet_ternary_quantize_activation64_forward`, `bitnet_ternary_linear_forward`, `bitnet_strict_ternary_linear_forward`, `bitnet_strict_ternary_linear64_forward` |
| QKV and head helpers | `pack_qkv_weights_forward`, `qkv_projection_forward`, `qkv_packed_heads_projection_forward`, `bitnet_qkv_packed_heads_projection_forward`, `bitnet_fused_qkv_packed_heads_projection_forward`, `bitnet_int8_fused_qkv_packed_heads_projection_forward`, `qkv_heads_projection_forward`, `split_heads_forward`, `merge_heads_forward`, `head_output_projection_forward` |

## 8. Attention Kernel System

Attention is split into descriptors, policy, dispatch, and implementation.

Key files:

| File | Role |
| --- | --- |
| `runtime/csrc/descriptors/attention_desc.h` | Normalized attention shape/mask/head-mode descriptor. |
| `runtime/csrc/policy/attention_policy.h` | Kernel kind selection, SM80 Flash/prelude gates, split-KV policy. |
| `runtime/csrc/backend/attention/cuda_attention_dispatch.cuh` | Shared attention dispatch contract. |
| `runtime/csrc/backend/attention/cuda_attention_decode.cuh` | Decode-oriented kernels. |
| `runtime/csrc/backend/attention/cuda_attention_prefill.cuh` | Generic and specialized prefill kernels. |
| `runtime/csrc/backend/attention/cuda_attention_*_prefill.cu` | SM80, PyTorch mem-efficient, and Flash-style lanes. |
| `runtime/csrc/backend/cuda_attention.cu` | Public CUDA attention launch surface. |

Policy separates decode and prefill:

- decode kernels specialize for `q_len == 1`, head dimension buckets, MHA/GQA,
  and mask modes.
- prefill kernels specialize head dimensions and causal/no-mask cases.
- split-KV policy handles large effective KV lengths.
- optional SM80/Ada inference lanes are gated by architecture, dtype, shape, and
  environment flags.

Important attention controls:

| Variable | Effect |
| --- | --- |
| `MODEL_STACK_PREFER_NATIVE_SM80_INFERENCE_ATTENTION` | Opt into SM80/Ada native prefill policy. |
| `MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_INFERENCE` | Disable SM80/Ada inference prefill lane. |
| `MODEL_STACK_SM80_NATIVE_ATTENTION_MIN_SMS` | Minimum SM count for SM80/Ada native selection. |
| `MODEL_STACK_DISABLE_SM80_NATIVE_ATTENTION_SHAPE_GUARD` | Bypass shape guard for experimentation. |
| `MODEL_STACK_ENABLE_ATTENTION_PREFILL_SM80_FLASH` | Enable local SM80 Flash-style prefill lane. |
| `MODEL_STACK_DISABLE_ATTENTION_PREFILL_SM80_FLASH` | Disable local SM80 Flash-style prefill lane. |
| `MODEL_STACK_SM80_FLASH_PREFILL_MIN_SEQ` | Minimum sequence threshold for SM80 Flash prefill. |
| `MODEL_STACK_DISABLE_ATTENTION_PREFILL_TENSORCORE` | Disable the tensorcore prefill lane. |

The attention system also exposes `attention_plan_info` so policy decisions can
be inspected without launching a kernel.

## 9. KV Cache And Decode Systems

KV cache kernels live in `runtime/csrc/backend/cuda_kv_cache.cu` and are called
through `runtime/ops.py` wrappers.

Supported runtime concepts:

- contiguous KV append/write/read/gather
- paged KV assignment and reservation
- paged KV append, compact, gather, read range, read last, write
- packed low-bit KV helpers such as INT3 pack/dequantize
- paged decode attention bridge
- projected QKV rotary paged write for fused decode append paths

The serving path uses these pieces to avoid Python-side cache movement during
decode. Paged cache behavior is part of the performance contract for long
contexts.

Important paged decode controls:

| Variable | Effect |
| --- | --- |
| `MODEL_STACK_PAGED_DECODE_SDPA_MAX_LENGTH` | Max cache length for SDPA bridge. |
| `MODEL_STACK_DISABLE_PAGED_DECODE_SDPA_BRIDGE` | Disable paged decode SDPA bridge. |
| `MODEL_STACK_DISABLE_BITNET_PROJECTED_QKV_DECODE_FUSED_APPEND` | Disable BitNet projected QKV fused append path. |

## 10. Linear And Quantized Linear System

Linear dispatch begins in `runtime/ops.py` and `runtime/quant.py` and is routed
through `model_stack_native.cpp` when native support is available.

Major linear families:

| Family | Files | Storage / Compute |
| --- | --- | --- |
| Dense native/cuBLASLt | `cublaslt_linear.cu`, `cuda_*` helpers | FP32/FP16/BF16 dense. |
| INT8 | `cuda_int8_linear.cu`, `cuda_quant_int8_frontend.cu`, `cutlass_int8_linear.cu` | int8 weights, rowwise int8 activations, DP4A/WMMA/CUTLASS/cuBLASLt paths. |
| INT4 | `cuda_int4_linear.cu`, `cutlass_int4_linear.cu` | packed uint8 symmetric INT4, optional CUTLASS BF16 path. |
| NF4 | `cuda_nf4_linear.cu` | packed NF4 codebook decode. |
| FP8 | `cuda_fp8_linear.cu` | fake FP8 tensor plus scalar scale. |
| BitNet | `runtime/csrc/backend/bitnet/*` | uint8 packed 2-bit ternary and optional int8 activation frontend. |

Linear backend policy should preserve:

- stable Python APIs
- explicit `backend=` choices
- `"auto"` behavior through `resolve_linear_backend`
- eager fallback under gradients unless a custom backward path owns it
- dtype and shape validation before launch

Important linear controls:

| Variable | Effect |
| --- | --- |
| `MODEL_STACK_DISABLE_SM8X_ATEN_LINEAR_AUTO` | Changes automatic dense linear selection on SM8x. |
| `MODEL_STACK_CUDA_LINEAR_NATIVE_MIN_ROWS` | Shape threshold for native/eager CUDA linear. |
| `MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA` | Enable experimental INT8 WGMMA lane. |
| `MODEL_STACK_INT8_LINEAR_WGMMA_MIN_OPS` | Minimum work threshold for WGMMA lane. |
| `MODEL_STACK_DISABLE_INT4_IMMA_ACT_QUANT` | Disable INT4 IMMA activation quant path. |
| `MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED` | Enable selected CUTLASS fused INT8 paths. |
| `MODEL_STACK_INT8_TRANSPOSE_TILE_DIM` | INT8 grad-weight transpose tile selection. |

## 11. BitNet Kernel System

BitNet is a first-class runtime subsystem. It includes weight packing,
weight-only linear, int8 activation frontends, compute-packed layouts,
decode-specialized bitplane layouts, fused QKV projection, and trainable
runtime-row export.

Key Python files:

| File | Role |
| --- | --- |
| `compress/quantization.py` | `QuantizedLinearBitNet`, `TrainableBitNetLinear`, packed state cache, QAT/export helpers. |
| `runtime/ops.py` | `pack_bitnet_weight`, packed input transforms, packed spec execution, QKV projection wrappers. |
| `runtime/quant.py` | `bitnet_linear`, `bitnet_linear_compute_packed`, `bitnet_linear_from_float`, int8 and ternary helpers. |

Key CUDA files:

| File | Role |
| --- | --- |
| `bitnet_formats.h` | Layout header indexes, layout parsing, segment/scale validation. |
| `bitnet_common.cuh` | Shared plan, device helpers, quantization helpers, launch declarations. |
| `bitnet_pack.cu` | Native packer and runtime-row ternary quantization. |
| `bitnet_linear_dispatch.cu` | Public BitNet CUDA entry points and plan dispatch. |
| `bitnet_linear_decode.cu` | Decode persistent kernels, bitplane row1 kernels, fused norm decode. |
| `bitnet_linear_prefill.cu` | Prefill tiled and split-K kernels. |
| `bitnet_frontend.cu` | Activation transform/calibration/int8-code frontend kernels. |
| `bitnet_ternary_linear.cu` | Ternary mask and strict ternary training kernels. |
| `bitnet_attention_*.cu` | Fused BitNet QKV projection and attention dispatch helpers. |

### 11.1 Packed Weight Format

BitNet runtime weights use `uint8` storage with four 2-bit ternary codes per
byte:

```text
code 0 -> -1
code 1 ->  0
code 2 -> +1
code 3 ->  0 / padding-safe zero
```

The layout header has versioned metadata:

| Index | Meaning |
| ---: | --- |
| 0 | format version |
| 1 | tile N |
| 2 | tile K |
| 3 | logical output features |
| 4 | logical input features |
| 5 | padded output features |
| 6 | padded input features |
| 7 | scale granularity |
| 8 | scale group size |
| 9 | interleave mode |
| 10 | minimum architecture |
| 11 | segment count |
| 12 | flags |

Supported scale granularities:

- `0`: per-matrix scale
- `1`: per-segment scale using `segment_offsets`
- `2`: per-output-group scale using `scale_group_size`

Native validation rejects malformed layout metadata before kernel launch:

- unsupported format version
- non-positive tile sizes
- non-positive logical dimensions
- padded dimensions smaller than logical dimensions
- unsupported scale granularity
- invalid per-group scale size
- unsupported interleave mode
- segment offset shape/order mismatches
- non-floating scale value storage in direct native calls

### 11.2 Runtime Plans

`ResolvePlan` chooses one of:

- `decode_persistent`
- `prefill_tiled`
- `prefill_splitk`

Decode persistent is used for rows `1..8` when not disabled. Prefill split-K is
available for larger row counts and large input dimensions when the workspace
size remains bounded. Otherwise prefill tiled is used.

BitNet plan controls:

| Variable | Effect |
| --- | --- |
| `MODEL_STACK_DISABLE_BITNET_PERSISTENT_DECODE` | Disables persistent decode scheduling. |
| `MODEL_STACK_BITNET_DECODE_CTA_MULTIPLIER` | Scales persistent decode CTA target. |
| `MODEL_STACK_DISABLE_BITNET_SPLITK` | Disables split-K prefill scheduling. |
| `MODEL_STACK_BITNET_SPLITK_MAX_SLICES` | Caps split-K slices. |

### 11.3 Activation Frontends

BitNet activation modes are normalized as:

- `none` / `off`
- `dynamic_int8`
- `static_int8`
- `dynamic_int4` and `dynamic_a4` as aliases for `dynamic_int8` with `4` bits
- `static_int4` and `static_a4` as aliases for `static_int8` with `4` bits

Activation bit width is constrained to `[2, 8]` across Python, native, and CUDA
helpers. This matters because the activation codes are stored in int8 lanes and
larger qmax values would overflow or wrap.

Dynamic activation scale is row-local. Static activation scale must be scalar or
one value per flattened row.

### 11.4 Compute-Packed And Decode-Packed Backends

Python caches two derived layouts in `QuantizedLinearBitNet`:

- compute-packed words and row scales for general compute-packed kernels
- decode bitplane masks and row scales for row1 decode

The dispatch path attempts the decode bitplane row1 path first when the plan is
decode-persistent and the decode layout is valid. It falls back to
compute-packed kernels, then to base packed decode/prefill if compute layout is
not valid.

The Python fallback for `bitnet_linear_compute_packed` must remain correct even
without CUDA. It should return the same result as `bitnet_linear` and cast to
`out_dtype` after fallback execution.

### 11.5 Fused BitNet Attention

BitNet attention paths combine packed QKV projection with standard attention
layout handling. The important distinction is that BitNet kernels own the packed
projection, while attention kernels still own Q/K/V layout, cache, and SDPA
behavior.

The fused projected decode append path exists to avoid separately materializing
and copying Q/K/V during decode.

### 11.6 Trainable BitNet

`TrainableBitNetLinear` keeps a floating shadow weight and exports packed
runtime state into `QuantizedLinearBitNet`. Training-time optimized paths are
gated by environment variables and shape policy:

| Variable | Effect |
| --- | --- |
| `MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD` | Selects dense, dynamic-int8, dynamic-int4, or packed training forward mode. |
| `MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT` | Overrides activation quant mode. |
| `MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT_BITS` | Overrides activation bit width. |
| `MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT_PERCENTILE` | Percentile used by activation calibration. |
| `MODEL_STACK_TRAINABLE_BITNET_SPIN` | Enables spin/Hadamard-style transforms. |
| `MODEL_STACK_TRAINABLE_BITNET_SPIN_RANDOM` | Controls random spin signs. |
| `MODEL_STACK_TRAINABLE_BITNET_SPIN_SEED` | Spin sign seed. |
| `MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE` | Enables shape-gated optimized training paths. |
| `MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT` | Selects grad-input backend mode. |
| `MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT` | Selects grad-weight backend mode. |
| `MODEL_STACK_TRAINABLE_BITNET_STRICT_TERNARY_FORWARD` | Enables strict ternary training forward path. |
| `MODEL_STACK_TRAINABLE_BITNET_TERNARY_MASK_FORWARD` | Enables ternary-mask training forward path. |
| `MODEL_STACK_TRAINABLE_BITNET_CUTLASS_INT4_FORWARD` | Enables selected CUTLASS packed INT4 training path. |

Some experimental training modes intentionally reject bias or unsupported
dtypes. Those are not general-purpose module replacements unless the shape gate
and mode-specific constraints allow them.

## 12. Norm, Activation, Sampling, And Utility Kernels

General CUDA kernels are exposed through `runtime/ops.py`:

- `rms_norm`, `add_rms_norm`
- `layer_norm`, `add_layer_norm`
- `residual_add`
- `embedding`
- `rope`
- activations and gated activations
- sampling, top-k/top-p masks, penalties, beam helpers
- append tokens and decode positions

These kernels have a simpler architecture than attention/BitNet:

1. Python wrapper validates the high-level shape.
2. Native binding validates dtype and rank.
3. CUDA backend launches a specialized kernel or returns to ATen fallback.

Even simple kernels must preserve fallback parity and autograd behavior.

## 13. Runtime Flags And Policy

Environment variables are policy inputs. They should not silently alter tensor
semantics; they should select between semantically equivalent implementations.

Common rules:

- `MODEL_STACK_DISABLE_*` should force a fallback or simpler path.
- `MODEL_STACK_ENABLE_*` should opt into experimental lanes.
- threshold variables should be parsed defensively and clamped to safe values.
- feature flags should be visible in `runtime_info()` when they represent a
  public capability or benchmark-relevant policy.

Do not hide an experimental fast path behind default behavior unless the test
suite proves parity and the benchmark result justifies it.

## 14. Adding A New Kernel

Use this workflow for a new kernel:

1. Define the Python API and eager fallback.
2. Add or reuse a descriptor for shape/layout semantics.
3. Add a C++ declaration in `model_stack_native.cpp` or a backend header.
4. Add the backend CUDA source under the appropriate family directory.
5. Add the file to `setup.py`.
6. Add public native validation before launch.
7. Add capability metadata to `runtime_info()`.
8. Bind the op with `m.def(...)`.
9. Add `has_native_op` dispatch in Python.
10. Add source-surface tests for registration and build wiring.
11. Add numeric parity tests against the Python fallback.
12. Add benchmark coverage for the target shape buckets.

Kernel implementation rules:

- Validate rank, dtype, device, shape, and layout before launching.
- For CUDA tensors used directly by a kernel, require same device.
- Keep fallback behavior available and semantically aligned.
- Avoid CPU synchronization on hot paths unless explicitly requested for debug.
- Prefer structured layout headers or descriptors over ad hoc argument lists.
- Keep launch policy explicit and inspectable.
- Return errors with the public op name in the message.

## 15. Validation Strategy

Model Stack uses several classes of tests.

| Test type | Purpose |
| --- | --- |
| Python fallback tests | Prove semantics without native extension. |
| Native source-surface tests | Prove files, bindings, and feature gates remain wired. |
| Numeric parity tests | Compare native outputs to Python/ATen references. |
| Benchmark scripts | Measure targeted shapes and speedups. |
| Export tests | Prove packed specs serialize into browser/runtime bundles. |

The documentation inventory is also tested. `tests/test_runtime_custom_kernel_docs_surface.py`
parses `setup.py` for CUDA sources and `runtime/csrc/model_stack_native.cpp`
for pybind operation names, then requires every source and binding to appear in
this document. That test does not prove numerical correctness; it proves the
architecture documentation stays complete as new native surfaces are added.

Minimum parity matrix for a new CUDA op:

- CPU/Python fallback
- CUDA native path
- FP32 reference where meaningful
- FP16/BF16 execution where supported
- contiguous and non-contiguous input where the API claims support
- empty or zero-row edge cases
- odd sizes and padded sizes
- autograd fallback behavior if called under gradients

Minimum BitNet parity matrix:

- base packed linear
- compute-packed fallback and CUDA
- decode row buckets `1, 2, 4, 8`
- row1 bitplane decode
- prefill tiled
- prefill split-K
- static activation quant
- dynamic activation quant
- activation bit widths `2, 4, 8`
- bias/no-bias
- per-matrix, per-segment, and per-output-group scales
- fused RMSNorm/AddRMSNorm decode
- fused QKV projection

## 16. Benchmarking And Profiling

Benchmarks should report both correctness and speed. A benchmark that does not
compare against a reference can hide wrong fast paths.

Useful existing benchmark locations:

- `tests/bench_bitnet_linear.py`
- `tests/bench_bitnet_attention.py`
- `tests/bench_bitnet_decode.py`
- `tests/bench_bitnet_h100_serving.py`
- `tests/bench_bitnet_h100_sweep.py`
- `examples/13_parameter_golf_h100/`

Benchmark output should include:

- backend name
- dtype
- shape
- batch/context size
- median latency
- reference latency
- speedup
- max absolute error vs dense/reference
- per-run timings when possible
- relevant runtime flags

For H100/SM90 work, keep architecture-specific assumptions explicit. For SM80/Ada
work, verify that policy gates do not route H100-only kernels by accident.

## 17. Safety And Compatibility Rules

Packed formats are compatibility contracts. Treat changes to layout headers,
scale semantics, or code mappings as versioned changes.

Rules:

- Increment format versions for incompatible packed state changes.
- Preserve checkpoint load behavior or provide migrations.
- Keep browser/WebGPU/WASM format expectations aligned with runtime formats.
- Keep direct native entry points stricter than Python convenience wrappers.
- Reject invalid bit widths before arithmetic that can shift or overflow.
- Reject cross-device CUDA tensor mixtures before kernel launch.
- Avoid silently accepting integer scale tensors in direct native paths.
- Keep source-surface tests for every registered kernel family.

## 18. Documentation Completeness Audit

As of this document update, the custom kernel architecture docs cover:

- every CUDA source currently appended by `setup.py`
- every public native operation currently bound with `m.def(...)`
- the header and policy files that define attention, BitNet, device, and
  descriptor contracts
- runtime flags that materially change backend selection
- validation expectations for native wiring, fallback parity, numeric parity,
  and benchmarks

Completeness here means source and ABI coverage. It does not mean every kernel
has exhaustive algorithm commentary or proof-level numeric validation. For that,
the next useful documentation layer would be per-kernel implementation notes
with shape formulas, launch geometry, memory layout diagrams, and benchmark
tables linked to the relevant tests.

## 19. Current Known Limits

The custom kernel stack is broad but not finished.

Known limits:

- Some training-time BitNet ternary/packed paths intentionally reject bias.
- Strict ternary fast paths have alignment and row/output constraints.
- CUDA numeric parity coverage is not complete for every BitNet plan and fused
  norm path.
- Some optional CUTLASS/PyTorch mem-efficient paths depend on local headers.
- Experimental SM90a/WGMMA paths require explicit build flags and hardware.
- Source-surface tests prove wiring, not numeric correctness.

These limits are acceptable only when they are visible in policy, tests,
runtime metadata, and documentation.
