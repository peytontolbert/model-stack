## peft.py — Minimal PEFT‑like Export

Exports adapter weights in a PEFT‑style layout for quick benchmarking and interop.

### Key APIs
- `infer_target_names(model_id)` -> `Dict[short->path]`
- `save_peft_like(out_dir, adapters, r, alpha, target_modules, bias="none", int8=False, target_paths=None)`
  - Writes `adapter_config.json` and `adapter_model.bin` (torch save).

### Notes
- Mapping for LLaMA‑like architectures is best‑effort; pass `target_paths` to override.
- `int8=True` applies per‑tensor affine quantization for compact transport.


