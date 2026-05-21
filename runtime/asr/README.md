# ASR Runtime

`runtime.asr` is the model-stack speech recognition surface. It keeps audio
preprocessing, stream chunking, decode policy, and backend execution separate so
Whisper, Distil-Whisper, Parakeet-style encoders, ONNX Runtime, and native
custom kernels can share the same application-facing API.

Current surface:

- `AsrStreamState`: turns live mono PCM into fixed windows with overlap.
- `pcm_to_log_mel`: dependency-light PyTorch log-mel frontend for Whisper-like
  models.
- `AsrDecodeOptions` / `apply_asr_logit_policy`: token suppression and decode
  policy primitives.
- `AsrRuntime`: façade that feeds features into a pluggable backend.

Backend implementations should expose `transcribe_features(log_mel, options)`.
That keeps model-specific encoder/decoder kernels out of the application layer
and lets bddy switch between CPU, ONNX, CUDA, WebGPU, or custom kernels without
rewiring capture code.
