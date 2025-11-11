## tune.py — Distillation and Exports

Implements a pragmatic loop to generate verifiable answers per module, log results, and (optionally) export tuned adapters.

### Key APIs
- `distill_repo(repo, model, adapters, out_dir, ignore=None, max_prompts=3, cache_dir=None, device="cpu", gpu_ids=None, context_tokens=5000, timeout_sec=None, resume=False, log_every=25, citations_per_paragraph=False)` -> `(verified_modules, buffer_path)`
  - Writes `distill.jsonl` and per‑prompt logs.
- `export_tuned_adapters(repo, model, verified_modules, out_dir, per_module_adapters=False, include_deps=False, max_deps=4, rank=8, cache_dir=None)`

### Tips
- Keep `--require-citations` on and wire optional pytest verification for stronger signals.
- Start with CPU or a single GPU to avoid contention when looping many prompts.


