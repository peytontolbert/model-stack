## verify.py — Citations and Test Verification

Checks for `path:line` citations and optionally runs mapped pytest nodes.

### Key APIs
- `has_citations(text, per_para)` -> `bool`
  - Regex for `[ctx]`‑style `file.py:A-B` citations.
- `extract_citations(text)` -> `List[str]`
- `verify_with_tests(g, module, repo_root, env)` -> `bool`
  - Runs pytest on discovered node IDs for the target module; soft‑passes if none.

### Tips
- Keep `--require-citations` enabled in runner to surface evidence and enable retries.
- Use `--citations-per-paragraph` for stricter guardrails when appropriate.


