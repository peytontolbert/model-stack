## Python Repo Grounded Planning

This example runs Program-Conditioned Adapter (PCA) over a small Python repository and asks for a citation-backed implementation plan.

Read these first:
- [../README.md](../README.md): maintained example index for PCA
- [../../README.md](../../README.md): PCA entry point
- [../../docs/BUILD_AND_RUN.md](../../docs/BUILD_AND_RUN.md): variant-specific setup and run notes

## Quick Start

From the repository root:

```bash
python -m examples.program_conditioned_adapter.examples.python_repo_grounded_planning.run_smoke_example
```

Alternate one-shot runner:

```bash
python examples/program_conditioned_adapter/examples/python_repo_grounded_planning/smoke_plan.py
```

## What It Does

1. Emits planning-oriented knowledge from the smoke repo.
2. Builds PCA adapters and caches.
3. Runs grounded planning with structured output and citations.

The default prompt asks for a plan to add a CLI subcommand that lists all modules and their public functions.

## Key Files

- `run_smoke_example.py`: canonical end-to-end runner
- `emit_planning_knowledge.py`: emits planning artifacts such as components, entrypoints, mutations, tests map, and dependency edges
- `program_config.py`: paths, retrieval contract, and backend wiring
- `smoke_plan.py`: alternate runner kept for parity

## Outputs

Artifacts are written under `artifacts/smoke_planning/`, including:
- `planning_components.jsonl`
- `planning_entrypoints.jsonl`
- `planning_mutations.jsonl`
- `planning_tests_map.jsonl`
- `planning_dependencies.jsonl`
- `planning_rerank_features.npz`
- `.program_state.json`

The grounded plan itself is printed to stdout.
