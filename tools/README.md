# Tools

`tools/` contains maintenance scripts for repository health checks. These tools
are not runtime APIs and should avoid importing heavy model dependencies unless
the specific check requires them.

## Available Tools

| Tool | Purpose |
| --- | --- |
| `verify_migration_doc_coverage.py` | Verifies that migration documentation exists, is indexed from `docs/research/README.md`, and still covers the expected repository tree. |

## Migration Documentation Coverage

Run:

```bash
python tools/verify_migration_doc_coverage.py
```

The script checks:

- required migration docs exist
- `docs/research/README.md` indexes the required docs
- the core module matrix is valid
- documented subtree counts still match the repository tree

Use it before large migration work or after moving packages around. It is a
documentation readiness check, not a code correctness test.

## Adding Tools

When adding a tool:

1. Keep the command non-destructive by default.
2. Print actionable failures.
3. Return a non-zero status on failed checks.
4. Document required environment variables and optional dependencies here.
5. Add a focused test if the tool parses repository structure or docs.

