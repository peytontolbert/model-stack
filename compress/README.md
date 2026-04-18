Compression & deployment efficiency

This package provides building blocks to shrink, accelerate, and deploy transformer models.

- Quantization: INT8/INT4/BitNet/FP8 weight quant, activation fake-quant, AWQ/GPTQ-style optimization
- Low-rank adaptation: LoRA adapters with merge/unmerge and export
- Pruning: SNIP, movement, and magnitude scoring + mask application
- Distillation: KD loss and intermediate feature matching helpers
- KV-cache: Paged KV cache with LRU eviction and compaction
- Export: Save/apply deltas compatible with light-weight deployment artifacts

Quickstart

```python
import torch
import torch.nn as nn
from compress.lora import inject_lora, merge_lora, get_lora_state_dict
from compress.quantization import collect_linear_calibration_inputs, quantize_linear_modules
from compress.pruning import magnitude_scores, build_global_pruning_mask, apply_pruning_masks
from compress.distill import kd_loss
from compress.kv_cache import PagedKVCache
from compress.export import build_delta, export_delta, load_delta, apply_delta

# LoRA
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
model, reps = inject_lora(model, lora_rank=8, lora_alpha=16)
# ... train LoRA params ...
merge_lora(model)  # fold LoRA into base weights for inference
lora_sd = get_lora_state_dict(model)

# Quantization with calibration-aware optimization
calibration_inputs = collect_linear_calibration_inputs(model, torch.randn(8, 16))
quantized = quantize_linear_modules(
    model,
    calibration_inputs=calibration_inputs,
    scheme="int4",
    weight_opt="awq",
    activation_quant="static_int8",
)

# Pruning
scores = magnitude_scores(model)
masks = build_global_pruning_mask(scores, sparsity=0.5)
apply_pruning_masks(model, masks)

# KD loss
student_logits = torch.randn(4, 100)
teacher_logits = torch.randn(4, 100)
loss = kd_loss(student_logits, teacher_logits, temperature=2.0, alpha=0.7)

# KV cache
cache = PagedKVCache(page_size=128, num_heads=8, head_dim=64, dtype=torch.float16)

# Export deltas
delta = build_delta(model=model, pruning_masks=masks)
export_delta("model.delta.pt", delta)
loaded = load_delta("model.delta.pt")
apply_delta(model, loaded)
```

Modules

- `lora.py`: `LoRALinear`, `inject_lora`, `merge_lora`, `get_lora_state_dict`, `extract_lora_delta`
- `quantization.py`: quantized linear wrappers, calibration capture, AWQ/GPTQ-style weight optimization, activation fake-quant helpers
- `pruning.py`: `snip_scores`, `movement_scores`, `magnitude_scores`, pruning mask helpers
- `distill.py`: `kd_loss`, `mse_match`, `DistillHooks`
- `kv_cache.py`: `PagedKVCache` for paging/compaction
- `export.py`: `build_delta`, `export_delta`, `load_delta`, `apply_delta`

Notes

- INT8/FP8 activation quantization is fake-quantized back into floating-point compute unless a native integer activation kernel is available.
- AWQ/GPTQ-style optimization requires calibration inputs; use `collect_linear_calibration_inputs(...)` or pass a module-name -> tensor map directly.
- FP8/FP4 helpers are fake-quant for experimentation unless native FP8 is available in your PyTorch build.
- `extract_lora_delta` provides an SVD-based low-rank delta for shipping only updates.
