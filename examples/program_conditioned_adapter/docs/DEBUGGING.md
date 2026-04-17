## Debugging Guide (Program‑Conditioned Adapter)

Common debugging checklist: see the canonical [repo-grounded debugging guide](../../repo_grounded_adapters/docs/DEBUGGING.md).

### What is different here
- This core does not assume a Python-repo `code_graph.py` owner at the top level.
- If you are using a Python repo backend, inspect the example backend under:
  - `examples/program_conditioned_adapter/examples/python_repo_grounded_qa/`
  - `examples/program_conditioned_adapter/examples/python_repo_grounded_planning/`
- `sources.jsonl` may describe generic program artifacts rather than repository files, depending on the backend.

### Practical checks
- Run with `--verbose` and inspect selection, retries, and citation scaffolding first.
- If a backend provides graph tooling, use that backend’s `code_graph`/smoke helpers rather than assuming the repo-grounded paths.
- Keep the canonical repo-grounded debugging page as the baseline for context packing, citations, devices, and adapter-mixing issues.
