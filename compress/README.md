Compression & deployment efficiency

This package provides building blocks to shrink, accelerate, and deploy transformer models.

- Quantization: INT8 per-channel weight quant, FP8/FP4 fake-quant helpers
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
from compress.quantization import quantize_linear_modules
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

# Quantization (weight-only int8 per-channel)
quantized = quantize_linear_modules(model)

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
- `quantization.py`: `QuantizedLinearInt8`, `quantize_linear_modules`, FP8/FP4 fake-quant helpers
- `pruning.py`: `snip_scores`, `movement_scores`, `magnitude_scores`, pruning mask helpers
- `distill.py`: `kd_loss`, `mse_match`, `DistillHooks`
- `kv_cache.py`: `PagedKVCache` for paging/compaction
- `export.py`: `build_delta`, `export_delta`, `load_delta`, `apply_delta`

Notes

- INT8 path dequantizes on-the-fly for correctness and portability; swap for custom kernels as needed.
- FP8/FP4 helpers are fake-quant for experimentation unless native FP8 is available in your PyTorch build.
- `extract_lora_delta` provides an SVD-based low-rank delta for shipping only updates.