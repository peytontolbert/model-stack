## runner.py — Program‑Conditioned Core

Canonical end-to-end flow is documented in the [repo-grounded runner doc](../../../repo_grounded_adapters/docs/modules/runner.md).

### What is different here
- This core is program-agnostic and may receive backend-supplied region/subgraph inputs.
- `generate_answer(...)` should be read as operating over a program root plus optional backend selection/evidence support.
- The program-conditioned core also exposes the structured answer path used by some backends.
- Subgraph construction and region selection may be delegated to a backend instead of being intrinsically repo/file based.

Use the canonical doc for the shared mixing, retry, citation, telemetry, and generation behavior.
