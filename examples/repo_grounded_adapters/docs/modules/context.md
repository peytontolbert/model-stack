## context.py — Context Packing and Function‑First Windows

Builds context strings from repo files with `[ctx] path: file.py:A-B` headers and collects function‑local windows.

### Key APIs
- `pack_context_heads(repo_root, files, tok, budget_tokens)`
  - Adds file heads up to budget.
- `pack_context_windows(repo_root, files, tok, budget_tokens)`
  - Adds 1–2 windows per file around `def/class` anchors.
- `collect_function_windows(repo_root, files, lines_each, max_candidates=24)`
  - Returns `(rel_path, a, b, anchor_ln, lines_block)` tuples for window scoring.
- `model_prob_yes(tok, model, prompt_q, window_txt)`
  - Simple heuristic using model logits to score relevance vs noise.

### Tips
- Prefer windows for citation‑heavy tasks; pair with `function_first` in runner to boost on‑topic windows.
- Keep budgets 2–5k tokens for multi‑module answers.


