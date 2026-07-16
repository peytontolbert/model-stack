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

## Current Browser Runtime Status

This repo now has two related browser WASM paths:

- `browser/bitnet_wasm/src/lib.rs`
- `browser/bitnet/bitnet_wasm_runtime.js`
- `browser/bitnet/q4_wasm_runtime.js`
- `browser/bitnet/f5tts_q4_dit_runtime.js`

The BitNet path supports packed signed ternary weights with row scales and
optional activation quantization for linear layers. The Q4 path supports
rowwise-symmetric 4-bit matrix weights, FP16/dense side tensors, chunked Q4
bundles, fused triple-linears, fused MLPs, fused DiT blocks, cached rotary
tables, head-major K/V attention, F5 session execution, and Vocos ISTFT-head
reconstruction.

The remaining browser/iPhone work is no longer "write the whole F5 graph from
scratch"; it is now mostly:

- improving F5/Vocos latency on long durations
- preserving quality at fewer steps
- deciding when to select WASM, WebGPU, or native Metal
- adding parity fixtures around promoted checkpoints
- keeping model loading memory-aware on iPhone

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

3. Keep ONNX/Core ML/Metal baselines beside WASM.
   For iPhone app deployment, the native Metal scaffold in `native/apple_metal`
   is the preferred long-term backend. Browser-on-iPhone should still use WASM
   as the deterministic fallback; Safari cannot call Metal directly.

4. Quantize selectively.
   Start with DiT linear layers only: Q/K/V/O and feed-forward projections.
   Keep norms, embeddings, time/text conditioning, final projection, and vocoder
   dense or FP16 until quality is measured.

5. Reduce sampling cost.
   Test `nfe_step=16` and then distill to fewer steps. This is probably more
   important for interactivity than reducing weights from INT8 to 2-bit.

6. Keep Vocos packaged separately.
   Vocos now has Q4/FP16 manifest indexes and tensor buffers under
   `browser/models/vocos_mel_24khz_q4_v0`. Continue to treat it as a separate
   bundle so F5 mel generation and waveform decoding can be profiled and
   replaced independently.

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

The existing WASM kernels are useful, but they should be treated as custom
runtime kernels for a controlled F5TTS/Vocos stack, not as a generic drop-in
model execution engine.
