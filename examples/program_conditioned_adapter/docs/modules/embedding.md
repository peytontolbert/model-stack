## embedding.py — Program and Subgraph Embeddings

Canonical embedding mechanics are documented in the [repo-grounded embedding doc](../../../repo_grounded_adapters/docs/modules/embedding.md).

### What is different here
- This variant frames embeddings around a generic program plus optional backend-provided `ProgramGraph`, not only repositories.
- The corresponding conceptual entrypoints are:
  - `build_program_embedding(...)`
  - `build_subgraph_embedding_from_program(...)`
- Backends may contribute extra channels beyond the repo-grounded defaults.

Use the canonical doc for the shared hashing, feature-budget, and text/topology guidance.
