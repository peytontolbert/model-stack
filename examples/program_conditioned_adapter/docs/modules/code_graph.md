## code_graph.py — Example Backend Note

The canonical Python-repo `code_graph.py` walkthrough lives in the [repo-grounded code-graph doc](../../../repo_grounded_adapters/docs/modules/code_graph.md).

### What is different here
- `Program‑Conditioned Adapter` core does not require a built-in Python `code_graph.py` owner.
- Python repository indexing is provided by example backends, for example under:
  - `examples.program_conditioned_adapter.examples.python_repo_grounded_qa.code_graph`
  - `examples.program_conditioned_adapter.examples.python_repo_grounded_planning.code_graph`

Use the canonical doc for the concrete Python repo graph behavior; treat this page as the pointer explaining that the core expects a backend, not a fixed graph implementation.
