# Model Stack Apple Metal Backend

This directory is the canonical home for the native iPhone runtime. Agent Kernel Lite should vendor or reference this backend rather than keeping app-local kernel code.

The current production fallback remains `/data/transformer_10/browser/bitnet_wasm`. The Metal backend is intended to take over on iPhone when the native Capacitor bridge is present and the parity gates pass.

## Runtime Contract

Expose coarse-grained calls only:

- `loadF5Bundle(manifestUrlOrPath)`
- `loadVocosBundle(manifestUrlOrPath)`
- `sampleF5Mel(condMel, condSeqLen, textIds, duration, steps, cfgStrength, seed)`
- `decodeVocos(mel)`

Do not call individual layers from JavaScript. Once Metal is selected, the whole F5 diffusion loop should stay native and keep tensors GPU-resident until the final mel/audio result.

## First Kernel

`Sources/ModelStackMetal/Kernels/q4_linear_f32.metal` implements the first backend target: packed Q4 weights with f32 activations and f32 output. This maps to the WASM `Q4LinearHandle::run_into` path without changing the F5TTS architecture or enabling activation quantization.

The first app integration should accelerate F5 QKV, attention output, FF in/out, input projection, and later Vocos Q4 layers. WASM SIMD remains the fallback and the reference for parity checks.

## Backend Selection

On iPhone, prefer this backend only from the native app shell. Browser Safari
should continue to use WebGPU when available and WASM otherwise; it cannot call
Metal directly without a native bridge.

Selection order for the app should be:

1. Metal backend when the native bridge is present and parity gates pass.
2. Browser WASM Q4/BitNet runtime as the deterministic fallback.
3. WebGPU for browser-only BitNet linear paths where it is available and faster.

Keep selection coarse-grained. JavaScript should choose a backend for the whole
F5/Vocos request, not per-layer Metal calls.

## Required Gates

Before the app selects this backend by default:

- Forward fixture parity must stay within the existing F5 gate.
- 1-step Peyton checksum must remain `73.719445`.
- 2-step CFG checksum must remain `-590.153`.
- Generated audio must pass the existing quality sample review.
