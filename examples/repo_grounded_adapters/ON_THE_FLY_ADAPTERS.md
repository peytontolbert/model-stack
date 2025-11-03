# On-the-fly Query Adapters (RCA + Subgraph Overlay)

This guide explains how to mix a base repo adapter (global prior) with a query-specific subgraph adapter (local evidence), and optionally prepend code snippets so answers are grounded and cite actual files/lines.

> Note: The modular production runner `examples/repo_grounded_adapters/run_repo_adapter.py` also supports KBANN-inspired options:
> - `--kbann-priors` for CodeGraph-derived per-target weights
> - `--function-first` for localized cones over function windows
> - `--round-lora` for interpretable, quantized LoRA factors
> See that runner's `--help` for details and parity with selection/packing/citation/sampling flags.

## Files
- `examples/repo_grounded_adapters/build.py`: build/save base repo adapters (and optional per-module exports).
- `examples/repo_grounded_adapters/run.py`: run with saved adapters; supports OTF subgraph selection, context packing, and citations.
- `examples/repo_grounded_adapters/modules/runner.py`: modular mixer/runner used by `run.py`.
- `examples/repo_grounded_adapters/code_graph.py`: repository indexing and utilities.

## Quick start
1) Generate base adapters (example dims for LLaMA-3.1-8B-Instruct):
```bash
python -m examples.repo_grounded_adapters.build \
  --repo /path/to/repo \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapters-dir /path/to/outdir \
  --embed-dim 1536 --base-rank 8 \
  --include-text --text-max-bytes 250000 --max-text-tokens 200000 \
  --kbann-priors --round-lora
```
2) Run on-the-fly with grounding + citations:
```bash
python -m examples.repo_grounded_adapters.run \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo /path/to/repo \
  --adapters-dir /path/to/outdir \
  --prompt "Explain the core training loop. Cite path:line for every claim." \
  --alpha 20 --rank 12 --gsub 0.75 --mix-beta 0.08 \
  --of-sources zoom --zoom-symbol modules.train,modules.train.sft_step --zoom-radius 1 \
  --pack-context --context-tokens 3000 --require-citations \
  --do-sample --temperature 0.7 --top-p 0.9 --repetition-penalty 1.1 \
  --max-new-tokens 512 --verbose
```

## How it works (brief)
- Base RCA adapter: per-layer LoRA with A/=√r and optional B=0 init, scaled by α/r and per-layer gate.
- On-the-fly subgraph adapter: LoRA generated from modules/files relevant to the question.
- Mixed hooks: ΔW = (α/r)[(1−g_sub)·ΔW_repo + g_sub·ΔW_sub], variance-matched to module weights.
- Context packing: prepend token-capped code blocks so attention can copy/ground; citation check enforces `path:line` patterns.

## Important flags (on-the-fly)
- Selection: `--of-sources {zoom,question}`, `--zoom-symbol`, `--zoom-radius`, `--dry-run`.
- Mixing: `--alpha`, `--rank`, `--of-alpha`, `--of-rank`, `--gsub`, `--mix-beta`, `--target-weights`.
- Grounding: `--pack-context`, `--context-tokens`, `--require-citations`.
- Sampling: `--do-sample`, `--temperature`, `--top-p`, `--repetition-penalty`, `--min-new-tokens`.
- Debug: `--verbose`.

## Tuning tips
- Hard questions: rank 12–32, alpha 20–32; `gsub≈0.7–0.85`.
- Emphasis: slightly upweight `o_proj`/`up_proj`, slightly downweight `v_proj`.
## KBANN domain-theory in practice (quick checklist)
- Build with priors and interpretability:
  - `--kbann-priors` (or explicit `--target-weights`)
  - `--round-lora` for small-value LoRA factors
- Run with local cones and citations:
  - `--function-first` plus window packing; keep `--require-citations`
  - Use `--of-sources {question|zoom}` to select the subgraph from the CodeGraph rules
- Context: 3000–5000 tokens for cross-module reasoning.
- Sampling: enable for richer/complete answers; keep `min-new-tokens ≥ 64`.

## Troubleshooting
- Empty output: increase `--max-new-tokens`/`--min-new-tokens`, use `--verbose`.
- No citations: increase `--context-tokens`, ensure selected files contain the logic; keep `--require-citations`.
- Zoom finds nothing: auto-fallback to question-aware; or set `--of-sources question`.
- OOM on probing: avoid full HF model loads during detection.

## Programmatic overlay (sketch)
```python
from examples.repo_grounded_adapters.code_graph import CodeGraph
from examples.repo_grounded_adapters.modules.embedding import build_subgraph_embedding_from_graph
from examples.repo_grounded_adapters.modules.adapter import generate_lora_from_embedding

g = CodeGraph.load_or_build(repo_root)
mods, files = ["modules.train"], ["modules/train.py"]
sub_z = build_subgraph_embedding_from_graph(
    g,
    include_modules=mods,
    include_files=[g.root + "/" + f for f in files],
    include_text=True,
    text_max_bytes=250000,
)
# Detect shapes however you prefer; runner handles mixing
sub = generate_lora_from_embedding(
    sub_z["z"], d_model=d_model, num_layers=n_layers, rank=12, targets=list(shapes.keys()), target_shapes=shapes
)
# Hand sub to your mixer or call modules.runner.generate_answer for end-to-end
```

## Planned enhancements (docs-first roadmap)

### Model-aware target discovery
- Enumerate `model.named_modules()` to detect split vs. fused QKV, `gate_proj`, and arch-specific Linear paths.
- Per-target `rank/alpha` and per-layer schedules; document suggested defaults per architecture.

### Retrieval and context packaging
- Replace file-head packaging with symbol-local windows (defs + callers/callees + key blocks).
- Add MMR/semantic re-ranking for files and windows; expose `--pack-mode {heads,windows}` and `--mmr-lambda`.
- Auto-increase `--context-tokens` when coverage is low; log coverage scores.

### Multi-pass workflow and reranking
- Pass 1 (outline with citations) → Pass 2 (section-wise zoom + draft) → N samples → rank by citation count/variety and module coverage.
- Expose a one-shot planner flag `--multi-pass` with `--candidates N` and `--rank-metric {citations,coverage,mixed}`.

### Alignment heads (no-code baseline + geometric init)
- Soft prompt head: map subgraph embedding `z` to a prefix `P∈R^{T×d_model}`; prepend to inputs (no weight edits).
- Spectral init: project LoRA A/B into top PCs of target weights (`W_q/o/up/down`) for better alignment; document one-time PC extraction.

### Mixing/controller improvements
- Two-tier overlays: base + question-aware + zoom; separate `g_sub` per tier; per-target variance/norm calibration.
- Document heuristics for `q/k/v/o/up/down` weighting per task (explanatory vs. algorithmic).

### Robust citation policy
- Require at least one citation per paragraph; retry with larger context or different zoom seeds when missing.
- Keep permissive regex, but log citation density and the distinct file count.

### Caching and persistence
- Disk-backed LRU for OTF adapters keyed by `(modules, files, rank)`; TTL-based invalidation on git tree changes.
- Optionally export OTF adapters (npz/peft) for reuse and offline analysis.

### Sampling controls
- Add `--length-penalty`, `--top-k`, and stop-strings; adaptive `--min-new-tokens` based on outline size.
- Document recommended settings for summarization vs. “step-by-step explain”.

### Quantization & compatibility
- Guidance for 4/8-bit base models (bitsandbytes): keep LoRA in fp16/fp32; allow int8 transport with dequant on load.
- Document device/dtype interplay and safe fallbacks.

### Telemetry and reproducibility
- Persist selection (modules/files), packed line ranges, generation args, and citation stats to a sidecar JSON.
- Record git `commit` and `tree` SHAs (already in manifest) alongside on-the-fly runs.

## FAQ

### Why use both adapters and context tokens?
Adapters encode a compact prior over the repo; tokens carry instance-specific evidence (exact lines). Together they reduce hallucination and improve specificity.

### I still get “INSUFFICIENT_EVIDENCE”
- Increase `--context-tokens` and verify `--dry-run` shows the right files.
- Switch `--of-sources` to `question` or tweak `--zoom-symbol/--zoom-radius`.
- Enable `--do-sample` and increase `--min-new-tokens` for longer, more complete outputs.

### Latency is high
- Cache on-the-fly adapters with a small LRU; reduce rank for OTF (e.g., 8–12) and keep base rank higher.
- Limit packaging to top-N windows; prefer symbol windows over full file heads.

## Classical inspirations and how to apply them

This section connects classic ideas (1940s–1990s) to concrete, actionable extensions of the on-the-fly adapter workflow.

### Schmidhuber (1991): Neural nets learn to program neural nets with fast weights
- Relevance: “Fast weights” are ephemeral, rapidly changing parameters programmed by a slower network—analogous to our subgraph adapters or soft prompts conditioned on retrieval.
- Additions to consider:
  - Fast-weight buffer per layer: derive a tiny, per-prompt outer‑product update ΔW_fast = u vᵀ from subgraph features; mix with LoRA for extra capacity without increasing rank much.
  - KV‑cache modulation: compute per-layer gates from retrieval to bias attention (keys/values) similarly to fast-weight programming.
  - Controller consistency: log which files/snippets programmed the fast weights for traceability/citations.

### Schmidhuber (1992): Learning to control fast‑weight memories (alternative to recurrent nets)
- Relevance: Temporal working memory via learnable control of fast weights without explicit recurrence.
- Additions to consider:
  - Working-memory tape: keep a short list of “applied” snippets during multi-step answering; adjust g_sub or per-target weights step‑by‑step.
  - Two‑net controller: a small controller predicts per-target scaling schedules across sections (outline → details) to manage when/where fast updates matter.

### Bishop (1991): A fast procedure for retraining the multilayer perceptron
- Relevance: Rapid adaptation of existing networks without full retraining.
- Additions to consider:
  - Micro‑calibration: optional few gradient steps on A/B (or soft prompt) against a tiny self‑supervised objective on retrieved code (e.g., next‑token on snippets) with strong regularization; bounded time budget.
  - Warm‑start schedules: initialize OTF rank and α from closed‑form or prior runs; update only a small subset of targets.

### Karayiannis (1991): Fast learning algorithms for neural networks
- Relevance: Criteria and strategies for fast adaptation.
- Additions to consider:
  - Robust objectives: use Huber/quantile losses for micro‑calibration to avoid overfitting spiky tokens.
  - Adaptive step sizes: per-target learning rates (q/k/v/o/up/down) for quick but stable updates.

### Poggio & Girosi (1990): Networks for Approximation and Learning (HBF/RBF)
- Relevance: Represent functions as weighted sums of basis functions.
- Additions to consider:
  - Basis‑mixing view: treat retrieved snippets as basis functions; compute weights by MMR/semantic similarity; build ΔW as Σᵢ wᵢ φᵢ(zᵢ) (either as LoRA mixtures or soft‑prompt mixtures).
  - Prototype library: cache per‑module prototypes to reduce per‑query latency.

### McCulloch & Pitts (1943): A logical calculus of ideas immanent in nervous activity
- Relevance: Logic as a complement to connectionist computation.
- Additions to consider:
  - Lightweight logical constraints: require each paragraph to end with a valid path:line citation; cross‑check against packed files.
  - Static checks: simple regex/AST validators ensuring claims mention functions that exist in selected files; retry otherwise.

### Actionable next steps (docs-first)
- Fast‑weight buffer: document an optional ΔW_fast from subgraph features and how it mixes with LoRA (config flags, safety guards).
- Working‑memory tape: document a multi‑pass loop where adapter weights/gates evolve with sections; log per‑section snippet usage.
- Micro‑calibration: outline a bounded, opt‑in few‑step objective (time + param caps) and rollback on divergence.
- Basis‑mix packaging: describe symbol‑window selection as basis functions and MMR weighting; compare to file‑head packaging.
- Logical guardrails: specify paragraph‑level citation constraints and structured retries when constraints fail.
