# transformer_10 BitNet PR1 Contract

This document defines the exact scope for the first implementation PR of the native BitNet backend.

PR1 is intentionally narrow.

Its job is to lock:

- public symbol names
- metadata lowering rules
- quantized module shape
- export/import schema
- baseline tests

Its job is not to deliver the fast decode or prefill kernels.

## 1. PR1 Goal

After PR1 lands, the repo should be able to say:

- `bitnet` is a legal native linear backend name
- `QuantizedLinearBitNet` is a legal compression replacement
- BitNet packed artifacts can be exported and reloaded
- the native module exposes stable BitNet entrypoints
- the Python/runtime surface can call those entrypoints

Performance may still rely on dense fallback or a correctness-first native path.

## 2. Exact Symbols To Add

### Python exports

Add to `runtime/__init__.py`:

- `bitnet_linear`

Add to `runtime.quant`:

- `bitnet_linear(...)`

Add to `runtime.ops`:

- `pack_bitnet_weight(...)`

### Native extension exports

Add to `_model_stack_native`:

- `bitnet_linear_forward(...)`
- `pack_bitnet_weight_forward(...)`

Optional in PR1:

- `bitnet_plan_info(...)`

### Runtime info keys

Add to `runtime_info()` output:

- `bitnet_available`
- `bitnet_arches`
- `bitnet_linear_dtypes`
- `bitnet_weight_storage`
- `bitnet_layout_version`

## 3. Backend Name Rules

`resolve_linear_backend(...)` must accept:

- `auto`
- `aten`
- `cublaslt`
- `bitnet`

PR1 behavior for `auto`:

- it may still resolve to the current dense backend by default
- it must not reject `bitnet` as an unknown backend

Error behavior:

- if `bitnet` is requested explicitly but unavailable, the error should say:
  - BitNet backend requested
  - whether the native module is present
  - whether CUDA support is compiled
  - whether the architecture is unsupported

## 4. Quantized Module Contract

Add a new module class:

- `QuantizedLinearBitNet`

Recommended constructor:

```python
QuantizedLinearBitNet(in_features: int, out_features: int, bias: bool = True)
```

Required stored state:

- `packed_weight`
- `scale_values`
- `layout_header`
- `segment_offsets`
- `bias` if present

Allowed auxiliary state:

- `layout_meta`
- debug-only cached dense weight
- quantization calibration fields

Required methods:

- `runtime_signature()`
- `runtime_bias(...)`
- `runtime_linear(...)`
- `runtime_supports_packed_backend(...)`
- `from_float(...)`

Required semantics:

- `runtime_supports_packed_backend("bitnet")` returns `True` once PR1 is wired
- `runtime_supports_packed_backend("cublaslt")` returns `False`
- `runtime_linear(...)` routes through `runtime.quant.bitnet_linear(...)`

## 5. Metadata Lowering Contract

PR1 should freeze the lowered runtime representation.

### `layout_header`

Type:

- rank-1 contiguous `torch.int32`

Length:

- fixed length `13` in PR1

Slot definitions:

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

### `segment_offsets`

Type:

- rank-1 contiguous `torch.int32`

Rules:

- prefix-sum form
- starts at `0`
- ends at `logical_out_features`
- length is `segment_count + 1`

### `scale_values`

Type:

- contiguous floating-point tensor

PR1 recommendation:

- `torch.float32`

Rules:

- interpretation depends on `scale_granularity`
- must stay explicit, never inferred from weight shape alone

## 6. Enum Values To Freeze In PR1

These enum values should be documented in Python and mirrored in C++.

### `format_version`

- `1`
  - local BitNet-compatible `16x32` packed format

### `scale_granularity`

- `0`
  - per-matrix
- `1`
  - per-segment
- `2`
  - per-output-group

### `interleave_mode`

- `0`
  - none
- `1`
  - BitNet local `16x32` interleave/permutation

### `arch_min`

- `80`
  - `sm80`

### `flags`

PR1 should treat `flags` as a bitfield even if only bit `0` is defined.

Recommended bit usage:

- bit `0`
  - bias present
- bit `1`
  - debug/reference packed artifact

## 7. Operator Signatures

### Python helper

Recommended public helper:

```python
def bitnet_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale_values: torch.Tensor,
    layout_header: torch.Tensor,
    segment_offsets: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

### Native op

Recommended native signature:

```python
bitnet_linear_forward(
    x,
    packed_weight,
    scale_values,
    layout_header,
    segment_offsets,
    bias=None,
    out_dtype=None,
    debug_dense_fallback=False,
)
```

### Native packer

Recommended native signature:

```python
pack_bitnet_weight_forward(
    weight,
    scale_values=None,
    layout_header=None,
    segment_offsets=None,
)
```

Recommended packer return tuple:

```python
(packed_weight, scale_values, layout_header, segment_offsets)
```

## 8. Export Schema

Extend `compress/export.py` and `compress/apply.py` with:

- `type == "bitnet_w2a8"`

Required exported fields:

- `type`
- `packed_weight`
- `scale_values`
- `layout_header`
- `segment_offsets`
- `bias` if present
- `in_features`
- `out_features`

Optional exported fields:

- `layout_meta`
- `quant_calibration`
- `quant_percentile`

## 9. Expected File Diff In PR1

### `runtime/csrc/model_stack_native.cpp`

Add:

- native op registration
- backend resolution support for `bitnet`
- runtime info fields
- pybind bindings for BitNet entrypoints

### `runtime/native.py`

Add:

- no special logic beyond normalized runtime-info passthrough if possible

### `runtime/quant.py`

Add:

- `bitnet_linear(...)`
- dense/reference fallback path when native execution is unavailable

### `runtime/ops.py`

Add:

- `pack_bitnet_weight(...)`

### `runtime/__init__.py`

Add:

- export mapping for `bitnet_linear`

### `compress/quantization.py`

Add:

- `QuantizedLinearBitNet`
- `"bitnet"` scheme branch in `quantize_linear_modules(...)`

### `compress/export.py`

Add:

- BitNet serialization branch

### `compress/apply.py`

Add:

- `QuantizedLinearBitNet` import/export in `__all__`

### `setup.py`

Add:

- BitNet native sources, even if some are still stub implementations in PR1

## 10. Minimum Tests To Add In PR1

### Runtime surface

Extend:

- `tensor/tests/test_runtime_cuda_inference_surface.py`

Add assertions for:

- `runtime_pkg.bitnet_linear is runtime_quant_mod.bitnet_linear`
- runtime-info normalization preserves BitNet keys when present

### Quant helper behavior

Extend:

- `tensor/tests/test_runtime_quant_cuda_helpers.py`

Add tests for:

- BitNet helper falls back to dense/reference execution when native op is absent
- BitNet helper forwards to native op when present
- metadata validation failures raise deterministic errors

### Compression behavior

Extend:

- `tensor/tests/test_runtime_compress_parallel_helpers.py`

Add tests for:

- `QuantizedLinearBitNet` uses `runtime_bitnet_linear`
- `runtime_supports_packed_backend("bitnet")` is `True`
- `runtime_supports_packed_backend("cublaslt")` is `False`

### Export/import behavior

Add a new test file or extend an existing one to cover:

- exported BitNet deltas include `layout_header` and `segment_offsets`
- apply/import roundtrip restores buffers and invalidates cached weights

## 11. What PR1 Must Not Do

PR1 must not:

- pretend the decode kernel is production-ready
- pretend prefill is solved
- introduce hidden Python-dict parsing on the hot native path
- invent a second incompatible metadata representation
- force attention integration into the same diff if that blocks ABI landing

## 12. Merge Gate For PR1

PR1 is good enough to merge when:

- the new public symbols exist
- the module replacement path works
- export/import works
- the native ABI is frozen enough for later kernel work
- tests cover the Python/runtime/compression contract

PR1 does not need to prove BitNet is fast.

It needs to prove the repo now has the right shape for BitNet to exist.
