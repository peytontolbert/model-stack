## adapter.py — Program‑Conditioned Variant

Canonical behavior and parameter semantics are documented in the [repo-grounded adapter module doc](../../../repo_grounded_adapters/docs/modules/adapter.md).

### What is different here
- The implementation contract is the same.
- The only practical difference is the module path used by this example tree:

```python
from examples.program_conditioned_adapter.modules.adapter import generate_lora_from_embedding
```

Use the canonical doc above for shapes, targets, rank behavior, mapping details, and IO semantics.
