# Model Stack BitNet WASM

This crate provides the browser WebAssembly fallback kernel for Model Stack's
packed BitNet linear format. It executes the same `bitnet_w2a8` packed tensor
layout used by the WebGPU runtime, so browser fallback keeps the compact BitNet
artifact instead of expanding to dense ONNX weights.

The crate exports the packed linear primitive used by
`browser/bitnet/bitnet_wasm_runtime.js`. The higher-level standalone model
runner lives in `browser/bitnet/encdec_runtime.js` as
`BitNetEncoderDecoderWASM` and `loadBitNetEncoderDecoderWASM`.

Build the SIMD package copied into browser bundles with:

```bash
RUSTFLAGS="-C target-feature=+simd128" \
  wasm-pack build --target web --release --out-dir ../bitnet/pkg
```

Run Rust-side kernel tests with:

```bash
cargo test
```

## Runtime Contract

`bitnet_linear_f32` accepts:

- row-major `Float32Array` input
- packed 2-bit ternary weight bytes
- float32 scale values
- int32 segment offsets
- optional float32 bias
- v1 BitNet layout header
- optional input quantization scales

The supported packed layout is intentionally narrow:

- `layout_header[0] == 1`
- `layout_header[1] == 16`
- `layout_header[2] == 32`
- `layout_header[9] == 1`
- padded input features are divisible by 4
- scale granularity is per-matrix, per-segment, or per-output-group
- activation quantization bits are constrained to `[2, 8]`

Malformed packed artifacts are rejected before arithmetic. The JS wrapper mirrors
the same checks so browser errors fail early with useful messages.

## Standalone Browser Use

For a full exported encoder-decoder bundle, use the JavaScript runtime:

```js
import { loadBitNetEncoderDecoderWASM } from "./runtime/encdec_runtime.js";

const model = await loadBitNetEncoderDecoderWASM("./manifest.json");
const logits = await model.forward([1, 2, 3], [0]);
```

`loadBitNetEncoderDecoder()` attempts WebGPU first and falls back to WASM unless
`disableWasmFallback` is set.
