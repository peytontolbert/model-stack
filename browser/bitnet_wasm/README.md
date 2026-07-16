# Model Stack BitNet WASM

This crate provides the browser WebAssembly fallback kernels for Model Stack's
packed browser formats. It started as the fallback for packed BitNet linear
layers and now also contains the Q4 speech kernels used by the standalone F5TTS
and Vocos browser runtimes.

The crate is the reference CPU/WASM path for browsers and iPhone fallback:

- packed BitNet `bitnet_w2a8` linear execution
- handle-backed BitNet encoder-decoder layers
- attention cache helpers for incremental decoding
- rowwise symmetric Q4 linears and fused Q4 linears
- F5TTS Q4 DiT session execution
- Vocos ISTFT-head reconstruction

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

`bitnet_linear_f32` is the low-level packed BitNet primitive. It accepts:

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

For repeated execution, prefer `BitnetLinearHandle`. It validates and stores the
packed buffers once, builds sparse row metadata, caches row scales, and exposes
`run(input, rows)`.

The encoder-decoder path additionally exports:

- `DecoderLayerHandle`: fused self-attention, cross-attention, MLP, norms, and
  decode cache state for one decoder layer
- `AttentionKvCache`: reusable KV cache for standalone attention calls
- `bitnet_linear2_f32` / `bitnet_linear3_f32`: fused independent projections
- `bitnet_mlp_f32`: fused BitNet MLP with activation
- `bitnet_sample_token_f32`: top-p sampling with repetition penalty support

The Q4 speech path exports:

- `Q4LinearHandle`
- `q4_symmetric_linear_f32`
- `q4_linear3_f32`
- `q4_mlp_f32`
- `f5_dit_block_f32`
- `F5Q4DiTSession`
- `q4_conv1d_f32`, `q4_depthwise_conv1d_f32`, and `q4_grouped_conv1d_f32`
- `vocos_istft_head_f32`

The current F5 attention path specializes common `head_dim == 64` shapes. WASM
SIMD helpers handle scaled dot products, weighted value accumulation, four-query
attention batches, cached rotary embeddings, and head-major K/V layouts. Scalar
fallbacks remain in place for non-wasm targets and unusual shapes.

## Standalone Browser Use

For a full exported encoder-decoder bundle, use the JavaScript runtime:

```js
import { BitNetEncoderDecoderWASM } from "./runtime/encdec_runtime.js";

const model = await BitNetEncoderDecoderWASM.fromManifestUrl("./manifest.json");
const logits = await model.forward([1, 2, 3], [0]);
```

For Q4 speech bundles, use `Q4TensorBundleWASM` and the F5/Vocos orchestration
helpers from `browser/bitnet/q4_wasm_runtime.js` and
`browser/bitnet/f5tts_q4_dit_runtime.js`. The JavaScript layer should select
coarse-grained calls such as `runF5SampleMel` or `runVocosIstftHead`; individual
Q4 kernels are implementation details unless a parity test is being written.
