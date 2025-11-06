## Build and Run Guide (Repo-Grounded Adapters)

This guide shows how to build base adapters for a repository and then run grounded Q/A with context packing and citation enforcement.

### Prerequisites
- Python 3.9+
- Optional GPU with recent CUDA; CPU works but is slower
- Model snapshot availability:
  - Either set `LLAMA8B_LOCAL_SNAPSHOT` to a local LLaMA snapshot dir containing `config.json` and `.safetensors`
  - Or set `HUGGINGFACE_HUB_TOKEN` to allow downloading `meta-llama/Llama-3.1-8B-Instruct`
- Recommended environment variables:
  - `TRANSFORMER_CACHE_DIR` (or `HF_HOME`) to control local HF cache
  - `CUDA_VISIBLE_DEVICES` to select GPUs
  - `REPO_ADAPTER_DELTA_CAP` to bound mixed deltas during safety testing (e.g., `0.05`)

### Build base adapters
Build a compact prior for your repo. Outputs live under `--adapters-dir`.

```bash
python -m examples.repo_grounded_adapters.build \
  --repo /path/to/repo \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapters-dir /path/to/outdir \
  --embed-dim 1536 --base-rank 8 \
  --include-text --text-max-bytes 250000 --max-text-tokens 200000 \
  --kbann-priors --round-lora --verbose
```

Key outputs in `--adapters-dir`:
- `adapters.npz` (flattened LoRA A/B per target/layer)
- `embedding.npz` (repo and family views)
- `manifest.json` (shapes, targets, priors, provenance)
- `sources.jsonl` (repo files: path, sha256, bytes)

Notes:
- Use `--cache-dir` to control HF model cache; defaults to `<project>/checkpoints`.
- If shapes cannot be inferred from config, add `--probe-full`.

### Run grounded Q/A with citations
Generate an answer using the modular runner. This will optionally re‑rank files, select a subgraph on‑the‑fly, pack context windows, and enforce per‑paragraph citations if requested.

Minimal example:
```bash
python -m examples.repo_grounded_adapters.run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo /path/to/repo \
  --adapters-dir /path/to/outdir \
  --prompt "Explain the core training loop. Cite path:line for every claim." \
  --of-sources question --rerank \
  --pack-context --pack-mode windows --context-tokens 3000 \
  --require-citations --verbose
```

Useful flags:
- Selection: `--of-sources {question|zoom}`, `--zoom-symbol`, `--zoom-radius`
- Mixing: `--alpha`, `--rank`, `--gsub`, `--mix-beta`, `--target-weights`
- Context: `--pack-context`, `--pack-mode {heads,windows}`, `--context-tokens`
- Citations: `--require-citations`, `--citations-per-paragraph`
- Sampling: `--do-sample`, `--temperature`, `--top-p`, `--min-new-tokens`, `--max-new-tokens`
- Devices: `--device-map {auto,none}`, `--gpu-ids`, `--cache-dir`

Tips:
- Keep `--require-citations` on for verifiable answers.
- If outputs are short, increase `--min-new-tokens` and `--max-new-tokens`.
- If missing citations, increase `--context-tokens` and use `--pack-mode windows`.

### Smoke test on the tiny demo repo
The repository includes a small self‑contained repo for quick validation.

```bash
python -m examples.repo_grounded_adapters.smoke_small_repo_test
```

This runs a baseline and an adapted pass against `examples/repo_grounded_adapters/smoke_repo`. Use `SMOKE_FORCE_CPU=1` to force CPU.

### Citation scaffold behavior
The runner prefixes context with `[ctx] path: file.py:A-B` blocks and instructs the model to cite `file.py:START-END`. The example citation path is now taken from the first selected file to exactly match `[ctx]` paths.

### Where things live
- Build script: `examples/repo_grounded_adapters/build.py`
- Run script (CLI): `examples/repo_grounded_adapters/run.py`
- Orchestrator (core): `examples/repo_grounded_adapters/modules/runner.py`


