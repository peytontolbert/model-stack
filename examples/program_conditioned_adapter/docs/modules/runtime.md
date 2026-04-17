## runtime.py — Thin Convenience Layer

Canonical notes on the thin runtime helpers live in the [repo-grounded runtime module doc](../../../repo_grounded_adapters/docs/modules/runtime.md).

### What is different here
- In this variant, `run_repo_adapter(...)` should be read as a legacy convenience wrapper for example backends.
- New code should prefer calling `modules.runner.generate_answer(...)` directly.

Use the canonical doc for the remaining helper semantics.
