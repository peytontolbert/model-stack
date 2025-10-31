## Getting started

### Prerequisites
- Python 3.9+
- PyTorch (install a version compatible with your environment)

### Using in your project
This repository contains a Python package named `tensor`. You can import it directly when your working directory is the project root, or ensure the project root is on `PYTHONPATH`.

```python
import tensor
```

If you maintain your own project, a common way is to include this repository as a submodule and add the root to your environment, or vendor the `tensor/` package into your source tree.

### Minimal usage
```python
from tensor import build_rope_cache, apply_rotary
import torch

T, Dh = 16, 64
q = torch.randn(1, 8, T, Dh)
k = torch.randn(1, 8, T, Dh)
cos, sin = build_rope_cache(T, Dh)
q, k = apply_rotary(q, k, cos, sin)
```

See `api-overview.md` for a tour of the available utilities.


