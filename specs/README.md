canonical types, shapes, config schemas, and serialization contracts

This package also provides a curated registry that maps canonical spec names to implementations in the tensor repository (stateless numerics for the model stack). Use it to select activations, norms, positional math, masking, numerics, residual wiring, and more by name in configs, while keeping model code decoupled from concrete implementations.

Example:

```python
from specs.ops import get_op, list_ops

# Discover available options
print(list_ops("activations"))

# Resolve an op by name and use it
activation = get_op("activations", "silu")
```

Available categories (curated subset):
- activations, norms, positional, masking, numerics
- residual, regularization, init, sampling
- shape, dtypes, metrics, losses
- windows, ragged, sparse, fft, state_space, scan
- quant, shard, compile, random, io, debug