Pluggable low-level kernels (FlashAttention, xFormers, Triton/CUDA ops)

Overview

- Provide a small registry so optional deps can be wired without hard imports.
- Ship safe fallbacks and lazy loaders so import-time remains dependency-free.

Registry API

```python
from kernel import register, register_lazy, get, has, available

# Register an eager implementation
@register("mlp.bias_gelu")
def bias_gelu(x, bias):
    return (x + bias).gelu()

# Register lazily from an optional package
def _load_flash2():
    from flash_attn import flash_attn_func
    return lambda q,k,v,*,is_causal=False,dropout_p=0.0: flash_attn_func(...)

register_lazy("attn.flash2", _load_flash2)

# Query at runtime
if has("attn.flash2"):
    flash2 = get("attn.flash2")
```

Included adapters

- `attn.flash2`: wrapper over `flash_attn.flash_attn_func` (lazy)
- `attn.xformers`: wrapper over `xformers.ops.memory_efficient_attention` (lazy)
- `rope.apply`: default to `tensor.positional.apply_rotary` (can be overridden)
 - `attn.triton`: placeholder Triton SDPA wrapper with torch fallback (lazy)

Guidelines

- Names are lowercase and can be namespaced with dots, e.g., `attn.flash2`.
- Prefer `register_lazy` for optional backends; raise only on actual use.
- Keep wrappers thin and shape-safe; avoid mixing policy with kernels.

Benchmark

```bash
python -m kernel.bench
```
Outputs average ms/iter for available backends (torch, flash2, triton, xformers).