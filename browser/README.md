# Browser Runtime

This directory contains browser-native runtime pieces for exported Model Stack
bundles.

The first target is BitNet linear execution on WebGPU. Safari cannot execute the
CUDA backend and ONNX Runtime Web does not expose a native ternary BitNet linear
operator, so browser BitNet bundles need a small runtime that understands Model
Stack's packed ternary format.

## BitNet WebGPU v1

The `browser/bitnet` runtime expects the existing `QuantizedLinearBitNet` v1
packing contract:

- `layout_header[0] == 1`
- `layout_header[1] == 16`
- `layout_header[2] == 32`
- `layout_header[9] == 1`
- packed ternary weights are row-major `uint8` bytes
- each byte stores four 2-bit codes
- code `0` means `-1`, code `1` means `0`, and code `2` means `+1`

The current WGSL kernel is deliberately correctness-first. It handles one output
element per invocation and supports prefill/decode shapes through the same
`[rows, in_features] x [out_features, in_features]` contract. Faster tiled and
split-K kernels can be added behind the same JavaScript API once the export
bundle and numerics are stable.

`browser/bitnet/encdec_runtime.js` provides a correctness-first encoder-decoder
runner for batch size 1. It executes BitNet linear projections through WebGPU and
keeps orchestration, embeddings, norms, softmax attention, and activations in
JavaScript typed arrays. This is intended to prove Safari compatibility and
numerics before replacing attention/norm/MLP orchestration with faster WebGPU
kernels.

Safari deployment notes:

- serve over HTTPS for WebGPU availability in production
- gate startup with `navigator.gpu`
- provide a WASM SIMD fallback for browsers or devices without WebGPU
- use single-thread WASM unless the site sends cross-origin isolation headers
- keep bundle loading memory-aware on iPhone and iPad
