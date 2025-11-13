## Python Repo Grounded Planning (Program-Conditioned Adapter example)\n
This example demonstrates grounded planning over a small Python repository using the Program‑Conditioned Adapter (PCA) pipeline and a Python repository program graph backend. It prepares lightweight “planning knowledge” artifacts, builds adapters, and then generates a citation‑enforced, step‑by‑step plan to implement a concrete change in the repo.\n
\n
### What it does\n
- Builds a program graph for a tiny Python repo (`smoke_repo`) and extracts planning knowledge:\n
  - components, entrypoints, candidate mutation sites, tests map, dependencies, and simple re‑rank features\n- Builds PCA adapters and caches for the repo\n- Prompts the model to produce a grounded, structured plan (with citations and explicit file/line ranges)\n\n
### Repo layout (this example)\n
- `program_config.py`: Centralized configuration (paths, contracts, PG backend)\n- `emit_planning_knowledge.py`: Emits planning knowledge artifacts from the program graph\n- `run_smoke_example.py`: Canonical end‑to‑end runner (emit → build → plan)\n- `smoke_plan.py`: Alternate one‑shot runner (kept for parity with other examples)\n- `artifacts/smoke_planning/`: Output directory for emitted knowledge, adapters, and program state\n\n
## Prerequisites\n
- Python 3.10+ and this repository’s dependencies installed\n- Ability to load the specified model (default: `meta-llama/Llama-3.1-8B-Instruct`). Configure your environment to point to your inference backend (local or remote) if required.\n- Run commands from the repository root so relative imports resolve correctly.\n- For global environment and model setup details, see `examples/program_conditioned_adapter/examples/README.md`.\n\n
## Quick start\n
From the repository root:\n
\n
```bash\n
python -m examples.program_conditioned_adapter.examples.python_repo_grounded_planning.run_smoke_example\n
```\n
\n
Alternatively, you can invoke the script path directly:\n
\n
```bash\n
python examples/program_conditioned_adapter/examples/python_repo_grounded_planning/run_smoke_example.py\n
```\n
\n
You should see three phases:\n
1) Planning knowledge emission\n
2) Adapter build over `smoke_repo`\n
3) Grounded planning run (structured output, citations enforced)\n\n
### Alternate runner (legacy/one‑shot)\n
\n
```bash\n
python examples/program_conditioned_adapter/examples/python_repo_grounded_planning/smoke_plan.py\n
```\n
\n
This variant performs a similar end‑to‑end flow by calling internal modules via `-m`; prefer `run_smoke_example.py` for consistency with other examples.\n\n
## Outputs\n
All artifacts are written under:\n
\n
```\n
examples/program_conditioned_adapter/examples/python_repo_grounded_planning/artifacts/smoke_planning/\n
```\n
\n
Key files emitted by `emit_planning_knowledge.py`:\n
- `planning_components.jsonl`: Modules with primary artifact spans\n
- `planning_entrypoints.jsonl`: Detected CLI‑like entrypoints (heuristic)\n
- `planning_mutations.jsonl`: Candidate edit sites and affordances\n
- `planning_tests_map.jsonl`: Naive mapping between owners and test files\n
- `planning_dependencies.jsonl`: Import/call edges at entity granularity\n
- `planning_rerank_features.npz`: Owner list and simple centrality/placeholder features\n
\n
Additional files used by the runner:\n
- `.program_state.json`: PCA program state checkpoint\n
- Adapter caches and logs created by `build.py` and `run.py`\n
\n
The grounded plan is printed to stdout; logs/caches and the above knowledge files are saved in the artifacts directory.\n\n
## Configuration knobs\n
Edit `program_config.py` to customize behavior:\n
\n
- `ProgramPaths`\n
  - `adapters_dir`, `knowledge_dir`: Where adapters and planning knowledge are stored\n
  - `program_state_path`: Path to persist PCA program state\n
- `ProgramContracts`\n
  - `require_citations`: Enforce citations in the plan\n
  - `citations_per_paragraph`: Strengthen citation cadence per paragraph\n
  - `retrieval_policy`: Relative weighting of similarity/structure/plan signals (e.g., `sim:0.6,struct:0.4,plan:0.2`)\n
  - `retrieval_temp`: Sampling temperature for retrieval blending\n
- `pg_backend`: Dotted path to the program graph backend constructor. Defaults to the Python repo graph backend used by other examples.\n\n
## How it’s prompted\n
`run_smoke_example.py` uses a focused prompt:\n
\n
> “Produce a grounded step‑by‑step plan to add a new CLI subcommand that lists all modules and their public functions. Include specific paths and line ranges to modify or create, referencing code entities with citations.”\n
\n
Flags enable structured output, window‑packing, citation enforcement, adapter‑aware decoding, and warmup/delta control.\n\n
## Troubleshooting\n
- Module import errors: Run from the repository root, or use `python -m ...` forms shown above.\n
- Model/backend issues: Ensure your environment can resolve `--model` to a working inference endpoint (see the top‑level examples README for setup).\n
- Permission errors: The example writes to `artifacts/smoke_planning/`; ensure the directory is writable.\n
- Resetting state: You can delete `artifacts/smoke_planning/` to re‑run from a clean slate.\n\n
## Customizing the task\n
You can change the planning target by editing the prompt in `run_smoke_example.py` or adjusting max context packing and retrieval weights in `program_config.py`. For larger or different repos, point `load_program_config(...)` to your project root and re‑run.\n
\n

