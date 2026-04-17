## Repo-Grounded Adapters

This tree is the repository-specialized adapter variant: build a repo embedding, synthesize LoRA-style deltas, select question-relevant files or symbols, and answer with citations anchored to repository artifacts.

## Start Here

Use the dedicated docs instead of treating this README as the full manual:
- [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md): concrete setup, build, and run commands
- [docs/DEBUGGING.md](docs/DEBUGGING.md): common failure modes and quick checks
- [docs/MODULES.md](docs/MODULES.md): module-level ownership map
- [ON_THE_FLY_ADAPTERS.md](ON_THE_FLY_ADAPTERS.md): mixing and subgraph-adapter notes

## Main Entry Points

- `build.py`: build base adapters and repo caches under `--adapters-dir`
- `run.py`: grounded generation with selection, context packing, and citations
- `smoke_small_repo_test.py`: fast validation against the tiny checked-in smoke repo
- `modules/runner.py`: programmatic orchestrator behind the CLI

## Minimal Flow

Build adapters once:

```bash
python -m examples.repo_grounded_adapters.build \
  --repo /path/to/repo \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapters-dir /path/to/outdir \
  --embed-dim 1536 \
  --include-text
```

Run grounded QA:

```bash
python -m examples.repo_grounded_adapters.run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo /path/to/repo \
  --adapters-dir /path/to/outdir \
  --prompt "Explain the core training loop. Cite path:line for every claim." \
  --of-sources question \
  --pack-context \
  --require-citations
```

## Notes

This README is intentionally short. The detailed CLI surface, troubleshooting, and module breakdown already live in `docs/`, and keeping that material there avoids carrying the same build/run narrative in two places.
