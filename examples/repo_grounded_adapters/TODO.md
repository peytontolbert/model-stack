## Smart Citations — Task List

- [x] Create `modules/sim_index.py` with `SymbolIndex`, `build_symbol_index`, `choose_path_style`
- [x] Augment `modules/verify.py` with symbol mention extraction and span resolution
- [x] Wire smart citations into `modules/runner.py` (imports, index build, fallback spans)
- [x] Add optional path formatting and de‑dupe/merge for spans
- [ ] Document behavior in docs and README (optional)

Notes:
- Anchor-first via `CodeGraph` spans; similarity fallback via lightweight adapter-bank encoder hook.
- Paths remain basenames unless collisions; switch to repo‑relative only for conflicts.
- Keep behavior behind current CLI (no new flags needed initially).

