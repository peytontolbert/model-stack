## Modules Overview

Canonical module walkthrough: see [repo-grounded modules overview](../../repo_grounded_adapters/docs/MODULES.md).

This tree keeps the same core adapter/mixing/capacity/provenance/tuning pieces, but documents them from the program-conditioned angle.

### Main differences in this variant
- The core is program-agnostic and can be paired with a backend-provided `ProgramGraph`.
- `embedding.py` is described in terms of program/subgraph embeddings rather than repo-only embeddings.
- `runner.py` documents the program-conditioned orchestration surface and the structured answer path.
- `selection.py` is framed around policy-driven region/entity scoring rather than only repo module/file heuristics.
- `runtime.py` is treated as a thin convenience layer; prefer the runner API in new code.

### Variant-specific module docs
- [adapter.py](modules/adapter.md)
- [code_graph.py](modules/code_graph.md)
- [embedding.py](modules/embedding.md)
- [runner.py](modules/runner.md)
- [runtime.py](modules/runtime.md)
- [selection.py](modules/selection.md)

For the unchanged shared modules, use the canonical repo-grounded docs referenced above.
