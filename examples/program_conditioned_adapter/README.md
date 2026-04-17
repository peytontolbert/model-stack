## Program-Conditioned Adapter (PCA)

Program-Conditioned Adapter (PCA) synthesizes a small LoRA delta from a program representation so a pretrained LM can answer against concrete program artifacts at inference time.

PCA is meant to stay backend-agnostic:
- repositories and code graphs
- APIs and schemas
- CLIs and configs
- DAGs, logs, and other executable systems

## Start Here

Use one canonical doc for each level of depth:
- [program_adapter.md](program_adapter.md): full design and theory writeup
- [Paper.md](Paper.md): short white-paper summary
- [examples/README.md](examples/README.md): checked-in example catalog and support scripts
- [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md): variant-specific setup and run notes
- [docs/DEBUGGING.md](docs/DEBUGGING.md): variant-specific debugging notes

## Core Loop

1. Build a `ProgramGraph` over the target system.
2. Embed channelized structure, contracts, topology, text, and traces into a compact vector.
3. Generate low-rank adapter weights for the model's projections.
4. Select a question-relevant region or subgraph and mix/apply the delta.
5. Pack evidence, enforce citations, and answer against real artifacts.
6. Optionally verify results and keep only the deltas that improve measurable outcomes.

## Maintained Examples In This Tree

- `examples/python_repo_grounded_qa/`: grounded QA over a small Python repository.
- `examples/python_repo_grounded_planning/`: grounded change-planning over a small Python repository.
- `examples/scripts/`: shared support code for graphs, embeddings, citation enforcement, and small benchmarks.
- `examples/hide/`: archived sketches and partial experiments; useful for reference, not the maintained starting point.

## Minimal CLI Surface

Build adapters:

```bash
python -m examples.program_conditioned_adapter.build \
  --sources <program-root> \
  --model <hf-id> \
  --adapters-dir <out> \
  --embed-dim 1536 \
  --include-text
```

Run grounded generation:

```bash
python -m examples.program_conditioned_adapter.run \
  --model <hf-id> \
  --sources <program-root> \
  --adapters-dir <out> \
  --prompt "<question>" \
  --pack-context \
  --require-citations
```

If you have a custom backend, add `--pg-backend <module_path:Factory>` to `build` and `run`.

## Practical Note

This README is intentionally short. The repository already had several PCA docs repeating the same pipeline; this file is now the entry page, while [program_adapter.md](program_adapter.md) remains the canonical long-form explanation.
