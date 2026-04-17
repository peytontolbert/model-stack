## Build and Run Guide (Program‑Conditioned Adapter Core)

Common workflow: see the canonical [repo-grounded build and run guide](../../repo_grounded_adapters/docs/BUILD_AND_RUN.md).

### What is different here
- This variant is program-agnostic: use `--sources` instead of `--repo`.
- The build and run entrypoints live under `examples.program_conditioned_adapter.*`.
- A backend may optionally provide `--pg-backend <module_path:Factory>` to supply a `ProgramGraph` and richer selection/evidence behavior.
- Example smoke and backend-specific code-graph tooling live under:
  - `examples/program_conditioned_adapter/examples/python_repo_grounded_qa/`
  - `examples/program_conditioned_adapter/examples/python_repo_grounded_planning/`

### Minimal commands
```bash
python -m examples.program_conditioned_adapter.build \
  --sources /path/to/program \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapters-dir /path/to/outdir \
  --embed-dim 1536 --base-rank 8 \
  --include-text --kbann-priors

python -m examples.program_conditioned_adapter.run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --sources /path/to/program \
  --adapters-dir /path/to/outdir \
  --prompt "Explain the core training loop. Cite path:line for every claim." \
  --of-sources question \
  --pack-context --pack-mode windows --context-tokens 3000 \
  --require-citations
```

### Program-conditioned specifics
- Per-owner/module banks are backend-defined rather than assumed by the core docs.
- Selection can be entity-driven through a backend `ProgramGraph`, not just file/module heuristics.
- New code should treat the repo-grounded guide as the common operational baseline and this page as the delta for the program-conditioned surface.
