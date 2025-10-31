## Shapes and masks

### Conventions
- Batch-first everywhere.
- Attention tensors typically use `(B, H, T, D)` for queries/keys/values and `(B, H, T, S)` for attention scores.
- Masks are boolean with `True` meaning masked (i.e., excluded). Broadcasting helpers return masks shaped `(B, H, T, S)`.

### Positional structures
- RoPE caches are `(T, Dh)` cosine/sine tables.
- ALiBi bias has shape `(1, H, T, S)`.
- Relative position bias (RPB) table is `(H, 2*max_distance - 1)`.

### Example: build and apply a causal mask
```python
from tensor import build_causal_mask, broadcast_mask, safe_softmax
import torch

B, H, T, Dh = 2, 8, 16, 64
q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, T, Dh)
scores = q @ k.transpose(-2, -1)  # (B, H, T, S)

mask = broadcast_mask(
    batch_size=B,
    num_heads=H,
    tgt_len=T,
    src_len=T,
    causal_mask=build_causal_mask(T),
)

probs = safe_softmax(scores, mask=mask, dim=-1)
```


