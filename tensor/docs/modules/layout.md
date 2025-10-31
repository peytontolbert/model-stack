## Layout helpers

APIs: `bhdt_to_bthd`, `pack_heads`, `unpack_heads`, `rechunk`, `contiguous_lastdim`, `tile`, `reorder`.

### Examples
```python
from tensor import bhdt_to_bthd, pack_heads, unpack_heads, rechunk, tile
```

- `tile(x, ("B",2,"T",1))` duplicates channel dimensions while preserving symbolic slots.
- `rechunk(x, block=128, dim=-2)` partitions sequences into blocks.


