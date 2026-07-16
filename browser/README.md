# Browser Runtime

This directory contains browser-native runtime pieces for exported Model Stack
bundles. The browser path is not an ONNX compatibility shim: it owns the packed
weight formats and executes them through WebGPU or WebAssembly so compact model
artifacts stay compact on device.

Current browser targets:

- packed BitNet encoder-decoder bundles
- Q4 F5TTS DiT bundles
- Q4/FP16 Vocos-style vocoder bundles
- shared WASM kernels used as the Safari/iPhone fallback when WebGPU is not
  available or not fast enough for a specific operator

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
runner for batch size 1. It executes BitNet linear projections through WebGPU
when available and through `browser/bitnet_wasm` when WebGPU is unavailable. The
WASM path keeps the compact packed ternary model files instead of expanding to
dense ONNX weights.

The v1 WASM fallback uses a packed-byte tiled kernel and builds with WASM SIMD
enabled. It validates packed layout metadata, scale metadata, bias length, and
activation quantization bits before execution. The generation runtime also
reuses encoder memory plus decoder self-attention and cross-attention caches
between generated tokens. WebGPU still remains the faster path.

Standalone model runners are class-based:

- `BitNetEncoderDecoderWebGPU.fromManifestUrl(device, manifestUrl, options)`
- `BitNetEncoderDecoderWASM.fromManifestUrl(manifestUrl, options)`
- `BitNetEncoderDecoderGenerationSession`

The runtime also exposes task heads when the export manifest includes them:

- `retrievalQueryEmbedding(inputIds, options)`
- `retrievalDocEmbedding(inputIds, options)`
- `agentIntentLogits(inputIds, options)`
- `agentPolicyLogits(inputIds, options)`

These heads run on pooled encoder states and keep policy/retrieval behavior out
of the core decoder loop.

## Q4 F5TTS and Vocos

`browser/bitnet/q4_wasm_runtime.js` owns Q4 tensor bundle loading and fused
speech kernels. It supports:

- `Q4TensorBundleWASM.fromManifestUrl(manifestUrl)`
- `Q4TensorBundleWebGPU.fromManifestUrl(manifestUrl, options)`
- chunked Q4 tensor buffers for large browser bundles
- `Q4LinearHandle` caching
- fused Q4 triple-linear and MLP calls
- F5TTS session preparation, forward, and `sample_mel`
- Vocos ISTFT-head execution through WASM

`browser/bitnet/f5tts_q4_dit_runtime.js` is the JavaScript orchestrator for F5
DiT. It delegates hot paths to WASM when the bundle exposes them and keeps only
shape validation and fallback math in JavaScript. The intended production path
for F5 is fused WASM or native Metal; the pure JavaScript DiT path is a
correctness/debug fallback and is not the performance target.

The browser Vocos bundle uses manifest-side tensor indexes:

- `tensor_q4_index.json`
- `tensor_fp16_index.json`
- `tensors.q4.bin`
- `tensors.fp16.bin`

That split lets the runtime load large Q4 matrices separately from small dense
FP16 tensors and keeps lookup by tensor name deterministic.

Safari deployment notes:

- serve over HTTPS for WebGPU availability in production
- gate startup with `navigator.gpu`
- use the packed WASM fallback for browsers or devices without WebGPU
- use single-thread WASM unless the site sends cross-origin isolation headers
- keep bundle loading memory-aware on iPhone and iPad
- prefer the native Metal backend on iPhone once parity gates pass; the browser
  WASM path remains the reference fallback
