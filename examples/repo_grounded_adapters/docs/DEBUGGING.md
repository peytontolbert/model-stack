## Debugging Guide (Repo-Grounded Adapters)

Use this checklist to diagnose selection, context packing, citations, devices, and adapter mixing.

### Quick checklist
1) Run with `--verbose` and inspect the prints:
   - `[debug] selection modules=[...] files=[...]`
   - `[debug] reranked modules=[...] files=[...]`
   - Retry scaffold messages when citations are missing
2) Increase `--context-tokens` and prefer `--pack-mode windows`.
3) Ensure the example citation path matches `[ctx]` paths (runner now uses the first selected file).
4) If outputs are truncated/empty, raise `--max-new-tokens` and `--min-new-tokens`.
5) If GPU OOMs, try `--device-map none` (CPU) or smaller ranks.

### Common symptoms and fixes

**No citations emitted**
- Enable `--pack-context` and raise `--context-tokens` to 3000–5000
- Use `--pack-mode windows` (function‑local windows) and/or `--function-first`
- Keep `--require-citations`; the runner will retry with a stronger scaffold
- Confirm selected files are relevant (see `[debug]` lines)

**Answer ends with long zeros or empty text**
- This is usually model output, not the runner. Disable sampling for a deterministic check:
  - Remove `--do-sample`; keep greedy decoding for debugging
  - Increase `--max-new-tokens` (e.g. 256) and set `--min-new-tokens` (≥64)

**OOM or CUDA allocation errors**
- Try CPU: `--device-map none` and unset `CUDA_VISIBLE_DEVICES`
- Or reduce capacity: lower `--rank`, or run without adapters first (`--alpha 0`)
- Free VRAM before retry: `torch.cuda.empty_cache()` is called automatically on retry, but a clean process helps

**Selection seems empty or irrelevant**
- Switch to question‑aware selection: `--of-sources question`
- Increase candidate budget with `--rerank` on; ensure your prompt contains stable keywords
- For targeted runs, use `--zoom-symbol my.module.fn --zoom-radius 1`

**Citations path mismatch**
- The runner now uses the first selected file directly for the example path. If you pass your own context, keep citation examples identical to `[ctx] path: ...` file names.

### Sanity checks

**Verify model snapshot availability**
- If using HF ID, ensure `HUGGINGFACE_HUB_TOKEN` is set and cache is writable
- Or point `LLAMA8B_LOCAL_SNAPSHOT` at a local snapshot dir with `config.json` and tensors

**Inspect CodeGraph quickly**
```bash
python -m examples.repo_grounded_adapters.code_graph /path/to/repo --dump
python -m examples.repo_grounded_adapters.code_graph /path/to/repo --defs-in modules.runner
```

**Smoke test**
```bash
python -m examples.repo_grounded_adapters.smoke_small_repo_test
```

### Useful environment variables
- `TRANSFORMER_CACHE_DIR` or `HF_HOME`: model/tokenizer cache root
- `CUDA_VISIBLE_DEVICES`: select GPUs, or set empty to force CPU
- `REPO_ADAPTER_DELTA_CAP`: e.g. `0.05` to cap per‑target delta norms during mixing
- `SMOKE_FORCE_CPU=1`: force CPU for the smoke test

### Artifacts to inspect
- `adapters.npz`: contains LoRA A/B (search for `L<i>.<name>.A` keys)
- `embedding.npz`: includes component views (e.g., `z_sym`, `z_doc`)
- `manifest.json`: shapes, targets list, priors, and selection summary
- `sources.jsonl`: repository files included during embedding


