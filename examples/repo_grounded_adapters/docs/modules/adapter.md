## adapter.py — LoRA Generation and IO

Generates deterministic LoRA A/B factors from repository embeddings and provides NPZ IO helpers.

### Key APIs
- `generate_lora_from_embedding(z, d_model, num_layers, rank=8, seed=0, targets=None, target_shapes=None, layer_gate="zmean", target_weights=None, learn_bias=False)`
  - Returns `{ layers: List[Dict[target->{A,B,gate(,bias)}]], rank, d_model, targets, gates }`.
  - Deterministic per‑layer/per‑target seeds; optional MLP seed coupling for `up_proj`/`gate_proj`.
- `generate_lora_from_embedding_torch(...)`
  - Torch‑native variant; supports `einsum_opt` for opt_einsum‑style planning.
- `save_npz(out_dir, embedding, adapters, manifest)` and `load_adapters_npz(path)`
  - Flatten/restore adapter tensors to/from NPZ.

### Parameters and Shapes
- `targets`: list of short names. Typical: `q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`.
- `target_shapes`: dict short->(d_out, d_in). If omitted, squares default to `d_model`.
- `rank`: LoRA rank per target (effective rank may be later trimmed).
- `layer_gate`: `zmean|cosine|hump|linear` gate per layer.
- `target_weights`: multiplicative scaling per target, e.g. boost `o_proj`/`up_proj`.

### Usage (NumPy path)
```python
from examples.repo_grounded_adapters.modules.adapter import generate_lora_from_embedding

adapters = generate_lora_from_embedding(
    z, d_model=4096, num_layers=32, rank=8,
    targets=["q_proj","o_proj","up_proj"], target_shapes=shapes,
    target_weights={"o_proj":1.1, "up_proj":1.1}
)
```

### Notes
- MLP pairing: `up_proj` and `gate_proj` share A/B to improve alignment; `down_proj` is separate.
- Optional `learn_bias=True` adds zero biases for downstream bias‑only finetuning.
- Rounding: performed by callers (`build`/`runner`) to keep this module universal.


