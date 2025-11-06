## embedding.py — Repository and Subgraph Embeddings

Builds feature‑hashed embeddings that mix multiple "knowledge factors" (symbols, docs, topology, types, calls, tests, optional text).

### Key APIs
- `build_repo_embedding(repo_root, dim=1536, include_text=False, text_max_bytes=0, max_text_tokens=0, text_weight=0.25, calls_weight=0.25, types_weight=0.20, tests_weight=0.15, graph_prop_hops=0, graph_prop_damp=0.85, ignore=None)`
  - Returns dict with `z` and family views: `z_sym,z_doc,z_mod,z_top,z_types,z_calls,z_tests,(z_text)` and sparsity diagnostics.
- `build_subgraph_embedding_from_graph(g, include_modules=None, include_files=None, include_text=False, ...)`
  - Same features restricted to modules/files subset.
- `auto_model_dims(model_id, cache_dir)` -> `(num_layers, d_model)` from HF config.

### Factor Views
- Symbols (`z_sym`): names, signatures, returns.
- Docs (`z_doc`): docstring heads.
- Modules/Topology (`z_mod`,`z_top`): module IDs and import graph (with optional propagation).
- Types/Calls/Tests (`z_types`,`z_calls`,`z_tests`).
- Text (`z_text`): n‑gram hashes from repo text under byte/token budgets.

### Tips
- Enable `include_text` with moderate budgets for better recall; keep per‑file read bounded via `text_max_bytes`.
- Use `graph_prop_hops>0` for smoothed topology; damp with `graph_prop_damp`.


