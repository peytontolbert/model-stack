# F5TTS WASM/iPhone Feasibility Notes

## Local Assets

- Service: `/data/resumebot/tts_service.py`
- Checkpoint: `/data/resumebot/checkpoints/final_finetuned_model.pt`
- Vocab: `/data/resumebot/checkpoints/F5TTS_Base_vocab.txt`
- Reference voice: `/data/resumebot/voice_profiles/Peyton/sample_0.wav`
- Reference text: `/data/resumebot/voice_profiles/Peyton/samples.txt`

The current resume bot uses F5TTS with:

```python
DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
mel_dim = 100
vocoder = Vocos
nfe_step = 32
sample_rate = 24000
```

## Size Reality

The fine-tuned checkpoint is about 2.6 GB on disk. Inspecting the PyTorch ZIP
container shows about 2.51 GiB of tensor storage, which is roughly 674M FP32
parameters if all tensor storage is FP32. The upstream F5TTS params script lists
this exact DiT configuration at about 335.8M parameters, so this local checkpoint
almost certainly includes duplicated model state such as regular weights plus
EMA or other training state.

Approximate current checkpoint tensor-storage sizes:

- FP32: 2.51 GiB
- FP16: 1.26 GiB
- INT8: 643 MiB
- 4-bit: 321 MiB
- 2-bit ternary/BitNet packed: 161 MiB plus scale/bias tensors

Approximate inference-pruned model-weight-only sizes, assuming ~335.8M params:

- FP32: 1.25 GiB
- FP16: 640 MiB
- INT8: 320 MiB
- 4-bit: 160 MiB
- 2-bit ternary/BitNet packed: 80 MiB plus scale/bias tensors

That makes pure size compression possible, but not sufficient. F5TTS inference
also needs the mel generator runtime, text/audio preprocessing, ODE sampling
loop, and a vocoder. The current model uses 32 function evaluations, so even a
2-bit linear kernel still repeats the DiT forward pass many times per utterance.

## Existing BitNet Runtime

This repo already has a browser WASM BitNet path:

- `browser/bitnet_wasm/src/lib.rs`
- `browser/bitnet/bitnet_wasm_runtime.js`

It supports packed signed ternary weights with row scales and optional int8-ish
activation quantization for linear layers. This can be reused for F5TTS linear
projections, but it is not a complete F5TTS runtime. We still need graph support
for:

- DiT attention and feed-forward blocks
- ConvNeXt/text encoder convolution pieces
- timestep/text/audio conditioning
- ODE/sway sampling loop
- mel spectrogram construction for reference audio
- Vocos or a smaller replacement vocoder

## Practical Path

1. Export a dense inference-only checkpoint.
   Remove optimizer, EMA duplicates, and training-only state. The F5TTS docs
   mention pruning checkpoints for inference; this should be the first cleanup
   before quantization.

2. Export a 4-bit bundle for the large projection weights.
   Use `scripts/export_f5tts_q4_bundle.py` from the `ai` conda environment. It
   reads `model_state_dict`, keeps dense tensors in FP16, and packs large
   rowwise-symmetric 4-bit matrix weights into `tensors.q4.bin`.

   Current Peyton export:

   - Path: `/data/resumebot/checkpoints/f5tts_peyton_q4_v0`
   - Total tensor payload: 163.65 MiB
   - Q4 tensors: 173 tensors / 335,472,640 params
   - FP16 dense tensors: 191 tensors / 1,624,196 params

   Matching WASM primitive:

   - `browser/bitnet_wasm/src/lib.rs`
   - `q4_symmetric_linear_f32(input, packed_weight, row_scales_f16, bias, rows, in_dim, out_dim)`
   - JS bundle loader: `browser/bitnet/q4_wasm_runtime.js`

3. Export ONNX/Core ML baselines before BitNet.
   For iPhone, Core ML or MPS-backed native inference is likely a much better
   first target than browser WASM. For browser-on-iPhone, WebAssembly SIMD is
   available, but WebGPU support and sustained thermal headroom are the harder
   constraints.

4. Quantize selectively.
   Start with DiT linear layers only: Q/K/V/O and feed-forward projections.
   Keep norms, embeddings, time/text conditioning, final projection, and vocoder
   dense or FP16 until quality is measured.

5. Reduce sampling cost.
   Test `nfe_step=16` and then distill to fewer steps. This is probably more
   important for interactivity than reducing weights from INT8 to 2-bit.

6. Replace or separately export the vocoder.
   Vocos must be exported/ported too. A small neural vocoder in Core ML, ONNX
   Runtime Web, or a server-generated LPCNet-style path may be easier than
   hand-porting all Vocos modules to WASM.

7. Distill if the target is fully offline browser TTS.
   The likely viable browser/iPhone target is a smaller student, not the full
   1024x22 F5TTS base. A 768x18 or smaller DiT, fewer flow steps, and a compact
   vocoder are the realistic shape for interactive mobile.

## Recommendation

Do not try to make the current 2.6 GB checkpoint run directly in iPhone Safari
as the first milestone. The fastest engineering path is:

1. Produce an inference-pruned checkpoint and measure actual model-only size.
2. Export a CPU/Core ML or ONNX baseline and measure latency for one sentence.
3. Apply ternary BitNet packing to DiT linear layers and compare audio quality.
4. Distill step count and model width/depth only after the baseline runs.

The existing BitNet WASM kernel is useful, but it should be treated as the
linear-layer backend for a custom F5TTS runtime, not as a drop-in solution.
