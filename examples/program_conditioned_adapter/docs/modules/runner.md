## runner.py — End‑to‑End Orchestrator

Drives selection, subgraph embedding, LoRA generation, adapter mixing, context packing, citation enforcement, and generation.

### Key API (program‑agnostic core)
- `generate_answer(model_id, adapters_npz, program_root, prompt, ..., verbose=False)` -> `str`
  - Selection: `of_sources {question|zoom}`, `zoom_symbol`, `zoom_radius`, `rerank`.
  - Capacity/mixing: `alpha`, `rank`, `gsub`, `beta`, `entropy_aware`, `rank_min/max`, `gsub_min/max`, `entropy_weights`, `target_weights`, `layer_schedule` (opt‑in), `q_aware_weights` (opt‑in), `per_target_rank_schedule` (opt‑in).
  - Layer rank tiers (opt‑in): `layer_rank_tiers` applies per‑layer keeps by target group across top/mid/low thirds.
  - Context: `pack_context`, `pack_mode {heads,windows}`, `context_tokens`, `function_first` and its knobs.
  - Rounding: `round_lora`, `round_threshold`.
  - Citations: `require_citations`, `citations_per_paragraph`.
  - Generation: sampling flags, token budgets, device mapping.
  - Mixture bank (opt‑in): `mixture_m`, `adapters_bank`.
  - Rank budget and ablations: `rank_budget` (sum of keeps across targets), `ablate_attn`, `ablate_mlp`.
  - Alpha warmup (opt‑in): `alpha_warmup` uses a lighter alpha on first attempt (or first structured pass), then full alpha on retry/subsequent passes.
  - Adapter‑aware decoding (opt‑in): `adapter_aware_decoding` gently relaxes sampling when citations are required and prepends a pointer‑first scaffold to the prompt.

### Behavior Highlights
- Shapes: Calls `infer_target_shapes(model)` to map short names to module paths.
- Subgraph: Program‑agnostic core does not compute subgraph embeddings; example backends can supply subgraph adapters or mixtures.
- Adapter mapping: always‑on enhanced mapping (normalized z + per‑target segment scaling).
- Mixing: Applies `register_hook_mixed_adapters` with:
  - target weights (plus optional question‑aware reweighting),
  - optional per‑layer multipliers (`layer_schedule`),
  - optional per‑target effective rank trimming (`per_target_rank_schedule`),
  - delta cap via env.
- Context: Packs either heads or windows; function‑first uses model heuristics to score windows.
- Citations: Enforces path:line format; retries once with a stronger scaffold; the example path now mirrors the first selected file from `[ctx]`.
- Mixture bank: optionally concatenates A/B from top‑m per‑module adapters (uniform softmax as a first pass).
- Stage‑2 fallback: on citation failure, re‑registers hooks with stronger top‑layer emphasis and slightly higher `g_sub`, then retries decoding.
- Telemetry (structured path): emits `metrics` with `citations_total`, `citations_must`, `modules_selected`, `files_selected`, `elapsed_sec`, and `retries` when `telemetry_out` is provided.
- Delta norms: `metrics.delta_norms` reports approximate AB norms per target averaged across layers.

### Troubleshooting
- Use `verbose=True` to print selection/rerank and entropy scaling.
- If OOM: retry path moves to CPU; reduce `rank` or disable adapters (`alpha=0`) to isolate.


