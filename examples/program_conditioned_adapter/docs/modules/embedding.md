## embedding.py — Program and Subgraph Embeddings

Builds feature‑hashed embeddings that mix multiple "knowledge factors" (symbols, docs, topology, types, calls, tests, optional text).

### Key APIs
- `build_program_embedding(pg, sources_root, dim=1536, include_text=False, text_max_bytes=0, max_text_tokens=0, text_weight=0.25, calls_weight=0.25, types_weight=0.20, tests_weight=0.15, graph_prop_hops=0, graph_prop_damp=0.85, ignore=None)`
  - Returns dict with `z` and family views such as `z_sym,z_doc,z_mod,z_top,(z_text)` and diagnostics; accepts an optional `ProgramGraph`.
- `build_subgraph_embedding_from_program(pg, sources_root, include_owners=None, include_artifact_paths=None, include_text=False, ...)`
  - Same features restricted to owners/files subset via `ProgramGraph`.
- `auto_model_dims(model_id, cache_dir)` -> `(num_layers, d_model)` from HF config (helper).

### Factor Views
- Symbols (`z_sym`): names, signatures, returns.
- Docs (`z_doc`): docstring heads.
- Modules/Topology (`z_mod`,`z_top`): module IDs and import graph (with optional propagation).
- Optional: calls/types/tests depending on backend emitters.
- Text (`z_text`): n‑gram hashes from program artifacts under byte/token budgets.

### Tips
- Enable `include_text` with moderate budgets for better recall; keep per‑file read bounded via `text_max_bytes`.
- Use `graph_prop_hops>0` for smoothed topology; damp with `graph_prop_damp`.


