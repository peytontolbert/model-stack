# Streaming Diarization Runtime

`runtime.diarization` is the low-level online speaker diarization surface.

Current scope:

- `DiarizationStreamState`: overlapped PCM windowing for live diarization
- `ReferenceStreamingDiarizationRuntime`: CPU reference online path
- `OnlineDiarizationState`: stable session speaker tracking and commit horizon

This first implementation is intentionally model-light. It does not try to be a
final production diarizer. It exists to establish the runtime contract, online
session ownership, chunk overlap behavior, and stable speaker stitching before
CUDA / Metal optimized backends are introduced.
