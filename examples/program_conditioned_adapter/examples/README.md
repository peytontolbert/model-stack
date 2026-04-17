## Program Examples

This directory is the example index for Program-Conditioned Adapter (PCA). It is intentionally narrower than the main design docs.

Read these first:
- [../README.md](../README.md): repository entry point for PCA in this tree
- [../program_adapter.md](../program_adapter.md): canonical long-form design writeup
- [../docs/BUILD_AND_RUN.md](../docs/BUILD_AND_RUN.md): concrete setup and run notes

## Maintained Examples

### `python_repo_grounded_qa/`

Grounded question-answering over a small Python repository.

Key files:
- `run_smoke_example.py`: end-to-end smoke path
- `emit_repository_knowledge.py`: emits repository knowledge artifacts
- `program_config.py`: central config for paths, model, and backend wiring
- `repo_state.py`: cached repository state helpers
- `modules/`: prompts, selection, verification, and graph helpers

### `python_repo_grounded_planning/`

Grounded implementation planning over a small Python repository.

Key files:
- `run_smoke_example.py`: end-to-end planning smoke path
- `emit_planning_knowledge.py`: emits planning-specific knowledge artifacts
- `program_config.py`: central config for paths, model, and backend wiring
- `smoke_plan.py`: alternate one-shot runner kept for parity
- `README.md`: example-specific usage notes

## Shared Support Scripts

`scripts/` contains reusable helpers for graph construction, embeddings, citation enforcement, adapter synthesis, and small benchmarks. Treat these as support utilities, not as the primary getting-started surface.

## Archived Material

`hide/` contains historical sketches, partial experiments, and idea inventory. Keep it for reference, but do not treat it as the maintained example catalog.
