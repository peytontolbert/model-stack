## selection.py — Candidate Discovery and Reranking

Provides question‑aware file/module selection, symbol zoom, and a simple feature‑based reranker.

### Key APIs
- `question_aware_modules_and_files(repo_root, prompt, top_k=8, ignore=None)` -> `(modules, files)`
- `modules_from_symbols(repo_root, seeds, radius=1, top_k=8, ignore=None)` -> `(modules, files)`
- `rerank_modules_and_files(repo_root, prompt, modules, files, ignore=None, self_queries_path=None, weights=(...))`
  - Combines signature/doc overlaps, call neighborhood size, pytest proxies, and visibility; optional self‑queries boost.

### Tips
- Prefer `--of-sources question` for broad prompts; use `--zoom-symbol` for targeted flows.
- Keep `--rerank` on; ensure prompts include stable tokens to match file names and symbol signatures.


