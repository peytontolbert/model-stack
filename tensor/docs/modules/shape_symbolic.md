## Symbolic shapes and broadcast planning

APIs: `S`, `unify`, `graph_shapes`, `infer`, `broadcast_plan`.

### Basics
- `S("B,H,T,D")` returns a tuple of symbolic names.
- `unify(a,b)` merges two signatures and emits notes on mismatches.
- `graph_shapes(*tensors)` returns a human-readable summary for debugging.

### Inference
```python
from tensor import infer
sol, deriv = infer("B*H==G*Q", {"B":2, "H":8, "G":4})
# sol -> {"Q": 4}
```

### Broadcast plan
```python
from tensor import broadcast_plan
plan, deriv = broadcast_plan("B,H,T,D", "1,1,T,D")
# plan["dim[0]"] == 'expand_b', plan["dim[1]"] == 'expand_b'
```


