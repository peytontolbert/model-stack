## Build and Run Guide (Program‑Conditioned Adapter Core)

This guide shows how to build base adapters for an arbitrary program and then run grounded answering with context packing and citation enforcement. Backends can optionally provide a `ProgramGraph` plugin to enrich selection and evidence.

### Prerequisites
- Python 3.9+
- Optional GPU with recent CUDA; CPU works but is slower
- Model snapshot availability:
  - Either set `LLAMA8B_LOCAL_SNAPSHOT` to a local LLaMA snapshot dir containing `config.json` and `.safetensors`
  - Or set `HUGGINGFACE_HUB_TOKEN` to allow downloading `meta-llama/Llama-3.1-8B-Instruct`
- Recommended environment variables:
  - `TRANSFORMER_CACHE_DIR` (or `HF_HOME`) to control local HF cache
  - `CUDA_VISIBLE_DEVICES` to select GPUs
- Optional env: `TRANSFORMER_CACHE_DIR` (or `HF_HOME`) to control local HF cache

### Build base adapters (program‑agnostic)
Build a compact prior for your program. Outputs live under `--adapters-dir`.

```bash
python -m examples.program_conditioned_adapter.build \
  --sources /path/to/program \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapters-dir /path/to/outdir \
  --embed-dim 1536 --base-rank 8 \
  --include-text --text-max-bytes 250000 --max-text-tokens 200000 \
  --kbann-priors --verbose
# Optionally provide a ProgramGraph backend:
#   --pg-backend <module_path:Factory>
```

Code‑recall preset (opt‑in) emphasizing `o,v,up,down,gate`:

```bash
python -m examples.program_conditioned_adapter.build \
  --sources /path/to/program \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapters-dir /path/to/outdir \
  --embed-dim 1536 --base-rank 8 --include-text \
  --code-recall-preset
```

Key outputs in `--adapters-dir`:
- `adapters.npz` (flattened LoRA A/B per target/layer)
- `embedding.npz` (repo and family views)
- `manifest.json` (shapes, targets, priors, provenance)
- `sources.jsonl` (repo files: path, sha256, bytes)

Notes:
- Use `--cache-dir` to control HF model cache; defaults to `<project>/checkpoints`.
- If shapes cannot be inferred from config, add `--probe-full`.

### Run with citations
Generate an answer using the modular runner. This will select a region, pack context windows, and enforce per‑paragraph citations if requested. If a backend is provided, selection can use `ProgramGraph` entities.

Minimal example:
```bash
python -m examples.program_conditioned_adapter.run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --sources /path/to/program \
  --adapters-dir /path/to/outdir \
  --prompt "Explain the core training loop. Cite path:line for every claim." \
  --of-sources question \
  --pack-context --pack-mode windows --context-tokens 3000 \
  --require-citations --verbose
# Optional backend for ProgramGraph‑aware selection:
#   --pg-backend <module_path:Factory>
```

Useful flags:
- Selection: `--of-sources {question|zoom}`, `--zoom-symbol`, `--zoom-radius`
- Mixing: `--alpha`, `--rank`, `--gsub`, `--mix-beta`, `--target-weights`, `--layer-schedule` (opt‑in), `--q-aware-weights` (opt‑in), `--layer-rank-tiers` (opt‑in)
- Context: `--pack-context`, `--pack-mode {heads,windows}`, `--context-tokens`
- Citations: `--require-citations`, `--citations-per-paragraph`
- Sampling: `--do-sample`, `--temperature`, `--top-p`, `--min-new-tokens`, `--max-new-tokens`
- Devices: `--device-map {auto,none}`, `--gpu-ids`, `--cache-dir`
- Mixture bank (opt‑in): `--mixture-m <int>` (top‑m sub_adapters), `--adapters-bank <dir>` (bank root from build `--per-module`)
- Adapter mapping: always‑on enhanced mapping; no flag needed
- Alpha warmup & decoding (opt‑in): `--alpha-warmup`, `--adapter-aware-decoding`
- Telemetry (structured path): `--telemetry-tests` runs a few tests and records pass counts

Tips:
- Keep `--require-citations` on for verifiable answers.
- If outputs are short, increase `--min-new-tokens` and `--max-new-tokens`.
- If missing citations, increase `--context-tokens` and use `--pack-mode windows`.

Recommended recipe for code‑recall tasks:
```bash
python -m examples.program_conditioned_adapter.run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --sources /path/to/program \
  --adapters-dir /path/to/outdir \
  --prompt "Describe how inference batching is implemented. Cite path:line for all claims." \
  --of-sources question \
  --pack-context --pack-mode windows --context-tokens 3000 \
  --require-citations \
  --code-recall-preset \
  --layer-schedule --layer-rank-tiers --q-aware-weights \
  --mixture-m 3 --adapters-bank /path/to/outdir
```

### Per‑module bank (example backends)
Example backends may export per‑owner/module sub‑adapters. Refer to the example backend docs if needed.

### Smoke test on the tiny demo repo
The repository includes a small self‑contained repo for quick validation.

```bash
# See example backends under:
# examples/program_conditioned_adapter/examples/python_repo_grounded_qa/
# examples/program_conditioned_adapter/examples/python_repo_grounded_planning/
```

This runs a baseline and an adapted pass against `examples/repo_grounded_adapters/smoke_repo`. Use `SMOKE_FORCE_CPU=1` to force CPU.

### Citation scaffold behavior
The runner prefixes context with `[ctx] path: file.py:A-B` blocks and instructs the model to cite `file.py:START-END`. The example citation path is now taken from the first selected file to exactly match `[ctx]` paths.

### Where things live
- Build script: `examples/program_conditioned_adapter/build.py`
- Run script (CLI): `examples/program_conditioned_adapter/run.py`
- Orchestrator (core): `examples/program_conditioned_adapter/modules/runner.py`


