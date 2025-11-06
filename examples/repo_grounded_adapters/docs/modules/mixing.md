## mixing.py — Mixed Adapter Application

Applies base and subgraph adapter deltas to model weights with scaling, per‑target weights, and safety caps. Supports clean removal.

### Key API
- `register_hook_mixed_adapters(model, base_layers, sub_layers, alpha_star, g_sub, rank, beta, target_weights=None, backend="local")`
  - Returns a list with an object exposing `.remove()` to subtract applied deltas.

### Details
- Scale: `scale = alpha_star / rank`.
- Composition: if both base and subgraph exist for a target, uses `(1-g_sub)*base + g_sub*sub`.
- MLP fused handling: when applying deltas into `mlp.w_in`, splits gate/up across the appropriate half rows.
- Safety cap: `REPO_ADAPTER_DELTA_CAP` env (e.g., `0.05`) scales deltas to keep ||ΔW|| ≤ cap·||W||.

### Tips
- Always remove hooks after generation to leave the model clean.
- Combine with entropy‑aware scheduling for stable capacity under large repos.


