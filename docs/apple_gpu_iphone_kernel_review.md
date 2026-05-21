# Apple GPU Kernel Review for Agent Kernel Lite

Date: 2026-05-21

## Summary

For iPhone, the reliable high-performance path is still WASM SIMD unless we add a native Metal bridge. iPhones have Apple GPUs, but a Capacitor/WKWebView app cannot assume stable WebGPU compute access. Desktop browsers can use benchmark-gated WebGPU, but iPhone TTS should keep WASM SIMD as the default until a native Metal runtime is wired.

## GitHub References

- MLC LLM: https://github.com/mlc-ai/mlc-llm
  - Supports iOS/iPadOS through Metal on Apple A-series GPUs.
  - Uses compiled runtime/model libraries rather than per-op JavaScript dispatch.
  - Relevant direction: native iOS Metal runtime for F5/Vocos, with model library built from `/data/transformer_10`.

- MLC iOS deployment docs: https://github.com/mlc-ai/mlc-llm/blob/main/docs/deploy/ios.rst
  - Shows the app-side pattern: build runtime and model libraries, optionally bundle weights, expose through Swift.
  - Relevant direction: Agent Kernel Lite should expose a Capacitor native plugin for the hot F5/Vocos path if we want Apple GPU acceleration on iPhone.

- llama.cpp: https://github.com/ggml-org/llama.cpp
  - Apple Silicon is a first-class backend through Metal, with quantized kernels for low-bit weights.
  - Relevant direction: separate matvec/matmul kernels by shape and quant format, keep weights resident, batch command encoding.
  - Kernel files to study:
    - `ggml/src/ggml-metal/ggml-metal.metal`
    - `ggml/src/ggml-metal/ggml-metal-device.cpp`
    - `ggml/src/ggml-metal/ggml-metal-impl.h`
  - Useful pattern for Agent Kernel Lite: expose a small native runtime that owns the Metal device, command queue, weight buffers, and shape-specialized Q4 kernels. Do not route each layer through a separate JS call.

- MLC/Metal and llama.cpp internal notes show a common pattern:
  - Keep tensors resident on GPU.
  - Avoid readback between operations.
  - Batch multiple operations into a command buffer.
  - Specialize kernels for quantized matmul shape families.

- Cider: https://github.com/Mininglamp-AI/cider
  - W8A8/W4A8 MLX custom primitives with Metal kernels and lazy graph composition.
  - Relevant but not directly portable to iPhone today: its INT8 TensorOps path is described as M5+/Metal 4 cooperative tensor oriented.
  - Useful idea: activation quantization can be fast, but for Peyton F5 we cannot enable activation quantization unless the audio checksum/quality gate passes.

- metalQwen3: https://github.com/BoltzmannEntropy/metalQwen3
  - Emphasizes command batching and buffer pooling for Apple GPU inference.
  - Relevant direction: if we build native Metal, avoid one command buffer/readback per layer.

- PMetal: https://github.com/Epistates/pmetal
  - Auto-detects Apple GPU family and tunes kernel parameters.
  - Relevant direction: native runtime should detect Apple GPU family/device tier and choose tile sizes accordingly.

- fp8-mps-metal: https://github.com/tashiscool/fp8-mps-metal
  - Not directly Q4/F5, but useful as a compact example of custom Metal compute kernels for low-precision matrix work on Apple GPUs.
  - Relevant direction: keep the bridge minimal, compile kernels explicitly, and avoid relying on PyTorch/MPS for unsupported quantized compute.

- Apple Metal shader converter docs: https://developer.apple.com/metal/shader-converter/
  - Useful if we ever generate kernels from another shader dialect, but for this project a handwritten `.metal` kernel is the simpler path.

- Apple Metal ML timeline docs: https://developer.apple.com/documentation/metal/running-a-machine-learning-model-on-the-gpu-timeline
  - Confirms the intended native pattern: command buffers, GPU-resident resources, and direct compute passes.

## Implications for F5TTS Q4

### Current Best iPhone Path

- Keep using `/data/transformer_10/browser/bitnet_wasm` as canonical.
- Keep F5 int4/Q4 weights and f32 activations for quality.
- Keep WASM SIMD as default on iPhone.
- Keep WebGPU benchmark-gated for desktop browsers only.

### Native Metal Target

To truly use iPhone GPU, add a native Capacitor plugin with a Metal runtime:

1. Load F5 Q4 and Vocos Q4 manifests/weights into native memory.
2. Keep packed Q4 weights and dense tensors resident in Metal buffers.
3. Implement shape-specialized kernels:
   - Q4 x f32 linear for rows around 300-700 and dims 512/1024/2048.
   - Fused QKV linear with head-major K/V output.
   - Attention for F5 sequence lengths around 256-700.
   - MLP linear + GELU + linear.
   - Grouped/depthwise conv1d for text/input embedding and Vocos.
   - Gated residual and layernorm affine.
4. Batch a whole DiT block, then a whole forward pass, with no CPU readback until final mel/audio.
5. Reuse the existing deterministic gates:
   - 1-step checksum `73.719445`.
   - 2-step CFG checksum `-590.153`.
   - Forward fixture parity thresholds.

## Proposed `/data/transformer_10` Integration

Keep this backend in the central model stack and vendor generated artifacts into Agent Kernel Lite:

```text
/data/transformer_10/native/apple_metal/
  Package.swift                         # optional local Swift package for kernel/runtime tests
  Sources/ModelStackMetal/
    ModelStackMetalRuntime.swift        # device, queue, command buffer orchestration
    F5TTSMetalSession.swift             # F5/Vocos session API
    Q4TensorBundleMetal.swift           # manifest + packed Q4 buffer loader
    Kernels/
      q4_linear_f32.metal               # first target
      q4_linear3_f5_qkv.metal           # second target
      attention_f5_head64.metal         # third target
      layernorm_gated.metal             # later fusion target
  Tests/
    F5TTSMetalParityTests.swift

/data/transformer_10/browser/bitnet/
  q4_wasm_runtime.js                    # keeps WASM fallback and dispatch metadata

/data/agent_kernel_lite/apps/mobile/ios/App/App/
  ModelStackMetalPlugin.swift           # thin Capacitor bridge, vendored from model stack
```

### First Kernel To Port

Start with `Q4LinearHandle::run_into` equivalent for F5 shapes:

- Activations: f32.
- Weights: existing packed Q4 layout.
- Output: f32.
- Initial shapes:
  - rows 256-700, in 1024, out 1024.
  - rows 256-700, in 1024, out 2048.
  - rows 256-700, in 2048, out 1024.
- Keep row scales/bias in Metal buffers.
- Decode int4 inside the shader and accumulate f32.

This lets the iPhone path accelerate QKV, attention output, FF in/out, input projection, and Vocos Q4 layers without changing the F5 architecture.

### Bridge Contract

The Capacitor bridge should expose coarse calls only:

```swift
loadF5Bundle(manifestUrlOrPath)
loadVocosBundle(manifestUrlOrPath)
sampleF5Mel(condMel, condSeqLen, textIds, duration, steps, cfgStrength, seed)
decodeVocos(mel)
```

Avoid per-layer JS calls. The whole diffusion loop should stay native once the Metal backend is selected.

### Do Not Do

- Do not enable i8/q4 activation kernels for F5 until the audio quality gate passes.
- Do not rely on WebGPU for iPhone TestFlight builds.
- Do not add per-op GPU readback. That usually loses to WASM SIMD.
