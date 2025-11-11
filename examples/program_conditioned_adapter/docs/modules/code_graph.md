## code_graph.py — Repository Index and Utilities

Parses Python files to build symbols, modules, imports, call edges, star import expansion, pytest node mapping, and optional coverage attachment.

### CLI Examples (example backend: Python repo)
```bash
python -m examples.program_conditioned_adapter.examples.python_repo_grounded_qa.code_graph /repo --dump
python -m examples.program_conditioned_adapter.examples.python_repo_grounded_qa.code_graph /repo --defs-in modules.runner
python -m examples.program_conditioned_adapter.examples.python_repo_grounded_qa.code_graph /repo --search attention_score
```

### Programmatic
- `CodeGraph.load_or_build(root, ignore=None)` -> `CodeGraph`
- Lookups: `defs_in(module)`, `find_symbol(name)`, `owners_of(symbol)`, `who_calls(fqn)`, `refs_of(fqn)`, `tests_for_module(module)`
- Exports: `export_json()`, `export_sqlite(path)`
- Incremental: loads/saves `.codegraph.json` with mtimes and hashes.

### Notes
- Honors `.gitignore` when available; supports additional ignore prefixes.
- Coverage XML can be attached to compute per‑symbol coverage estimates.


