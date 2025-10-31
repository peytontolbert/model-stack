## Tensor

Stateless numerics and building blocks for transformer stacks. Includes positional math (RoPE/ALiBi/RPB), masking, norms, MLPs, activations, dtype/shape helpers, numerics, residuals, init, regularization, sampling, metrics, compile/runtime helpers, and more.

### Highlights
- Pure tensor ops, small `nn.Module`s where ergonomic
- Batch-first shapes; clear mask semantics
- Drop-in utilities for training, sampling, metrics, and numerics stability

### Quick example
```python
from tensor import (
    build_rope_cache, apply_rotary,
    build_causal_mask, broadcast_mask, safe_softmax,
)
import torch

B, H, T, Dh = 2, 8, 16, 64
q = torch.randn(B, H, T, Dh)
k = torch.randn(B, H, T, Dh)
cos, sin = build_rope_cache(T, Dh)
q, k = apply_rotary(q, k, cos, sin)
scores = q @ k.transpose(-2, -1)
mask = broadcast_mask(batch_size=B, num_heads=H, tgt_len=T, src_len=T, causal_mask=build_causal_mask(T))
probs = safe_softmax(scores, mask=mask, dim=-1)
```

### Documentation
- Getting started: see `getting-started.md`
- Shapes and masks: see `shapes-and-masks.md`
- API overview: see `api-overview.md`
- Vision & philosophy: `vision.md`
- Roadmap / future development: `roadmap.md`
- Contributing guide: `contributing.md`

### Modules
- activations: `modules/activations.md`
- norms: `modules/norms.md`
- mlp: `modules/mlp.md`
- positional: `modules/positional.md`
- masking: `modules/masking.md`
- numerics: `modules/numerics.md`
- stability: `modules/stability.md`
- dtypes: `modules/dtypes.md`
- shape: `modules/shape.md`
- symbolic: `modules/shape_symbolic.md`
- layout: `modules/layout.md`
- residual: `modules/residual.md`
- init: `modules/init.md`
- regularization: `modules/regularization.md`
- segment & ragged: `modules/segment_ragged.md`
- scan: `modules/scan.md`
- fft/spectral: `modules/fft_ops.md`
- state-space: `modules/state_space.md`
- schedule/memory: `modules/schedule_memory.md`
- profile: `modules/profile.md`
- sparse: `modules/sparse.md`
- losses: `modules/losses.md`
- sampling: `modules/sampling.md`
- metrics: `modules/metrics.md`
- compile: `modules/compile.md`
- shard: `modules/shard.md`
- einsum: `modules/einsum.md`
- io: `modules/io.md`
- checkpoint: `modules/checkpoint.md`
- lowrank: `modules/lowrank.md`
- random: `modules/random.md`
- export_safe: `modules/export_safe.md`
- windows: `modules/windows.md`
- arena: `modules/arena.md`
- debug: `modules/debug.md`
- quant_utils: `modules/quant_utils.md`
- ragged: `modules/ragged.md`
- optim: `modules/optim.md`

### Future-oriented topics
- Streaming and online numerics: see roadmap sections 2 and 5
- Stateless shape proofs and symbolic shapes: see `vision.md` and `roadmap.md`

For a fuller narrative overview, also see the repository `README.md` in `tensor/`.


