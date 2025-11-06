## runner.py — End‑to‑End Orchestrator

Drives selection, subgraph embedding, LoRA generation, adapter mixing, context packing, citation enforcement, and generation.

### Key API
- `generate_answer(model_id, adapters_npz, repo_root, prompt, ..., verbose=False)` -> `str`
  - Selection: `of_sources {question|zoom}`, `zoom_symbol`, `zoom_radius`, `rerank`.
  - Capacity/mixing: `alpha`, `rank`, `gsub`, `beta`, `entropy_aware`, `rank_min/max`, `gsub_min/max`, `entropy_weights`, `target_weights`.
  - Context: `pack_context`, `pack_mode {heads,windows}`, `context_tokens`, `function_first` and its knobs.
  - Rounding: `round_lora`, `round_threshold`.
  - Citations: `require_citations`, `citations_per_paragraph`.
  - Generation: sampling flags, token budgets, device mapping.

### Behavior Highlights
- Shapes: Calls `infer_target_shapes(model)` to map short names to module paths.
- Subgraph: Builds `sub_z` with modules+files; optional cones overlay (`function_first`) merged to increase locality.
- Mixing: Applies `register_hook_mixed_adapters` with target weights and per‑target slices for fused MLPs.
- Context: Packs either heads or windows; function‑first uses model heuristics to score windows.
- Citations: Enforces path:line format; retries once with a stronger scaffold; the example path now mirrors the first selected file from `[ctx]`.

### Troubleshooting
- Use `verbose=True` to print selection/rerank and entropy scaling.
- If OOM: retry path moves to CPU; reduce `rank` or disable adapters (`alpha=0`) to isolate.


