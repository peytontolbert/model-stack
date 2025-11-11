1) Core pipeline (end-to-end)

Index → CodeGraph

Walk repo, collect modules, symbols, signatures, docs, imports, star-imports, call edges, pytest node mapping, coverage stubs. This is your “ground truth graph.”

Embed → Feature-hashed repo vectors

Hash multiple “knowledge factors” into a single normalized vector z: symbols & signatures, doc heads, module IDs, import topology (with optional graph propagation), optional raw text n-grams under byte/token budgets. Expose knobs: graph_prop_hops, graph_prop_damp, include_text, text_max_bytes, max_text_tokens, text_weight.

Adapt → Generate LoRA from embedding

Infer target shapes from the model config and produce per-layer low-rank A/B for q/k/v/o and MLP up/down/gate. Provide “knowledge-preset” target weighting and mixing factors so you can favor recall layers (e.g., o/up/down) vs. attention projections.

Mix + Apply → Runners

Two paths:

On-the-fly: select subgraph by question or zoom(symbol, radius); build per-layer deltas; add directly to layer weights with scaling alpha/rank; cleanly remove later.

Modular runner: same mixing, plus context packing, question-aware file picking, target-weight parsing, optional delta caps (env-guard), designed as composable utilities.

Answer → Pack context, require citations

Build prompt from selected files (heads/windows), enforce per-paragraph citations, and generate the final grounded answer. (Runner CLI controls include --pack-context, --require-citations, budgets, and file ignore lists.)

Self-tune → Verify, keep only wins

For each module: auto-draft verifiable prompts, run the modular runner, extract citations, run mapped tests if available; export per-module subgraph adapters for the ones that pass (optionally include direct deps). Append a Q/A distill buffer for replay.

2) Scripts you want in the scaffold (with their one job)

code_graph.py — fast, incremental repo index + utilities for owners, callers, refs, pytest node mapping, export (JSON/SQLite). This is your curriculum backbone.

build.py — build base adapters for a given HF model from a repository embedding (symbols/docs/topology + optional text). Infers target shapes, supports knowledge presets, KBANN priors, optional rounding, per‑module export, and writes artifacts.

run.py — simple runner: ensures base adapters exist (or builds them), then generates an answer with optional context packing/citations and on‑the‑fly subgraph selection (question or zoom). Uses the modular runner under the hood.

modules/tune.py — optional distillation helpers (verify modules via Q/A + citations/tests; export tuned artifacts). Also available through build.py via --self-tune-out.

3) Grounded run loop (concise)

Select modules/files by question (or zoom), mix the base adapter with an on-the-fly subgraph adapter, pack context with citations, and generate the answer. Keep per-paragraph citations on.

4) Controls that make it robust

Capacity knobs: --rank, --alpha, --gsub (blend base vs subgraph); per-target weights (e.g., boost o/up/down for “knowledge recall”); `--layer-schedule` (opt‑in per-layer multipliers rising toward top third); optional delta-norm caps via REPO_ADAPTER_DELTA_CAP.
Layer tiers and rank budgets: `--layer-rank-tiers` (per-layer keeps by target group), `--rank-budget` (cap sum of per-target keeps per layer).
Adapter mapping: always-on enhanced mapping (normalized z + per-target segment scaling).

Selection knobs: --of-sources question|zoom, --zoom-symbol, --zoom-radius, and text feature control (--include-text, --text-max-bytes, --max-text-tokens, --text-weight).

Context knobs: packing mode (heads/windows), token budget, and per-paragraph citation enforcement.

Question-aware weights: `--q-aware-weights` (opt‑in) upweights target groups based on prompt intent (e.g., signatures → o,v; behavior → up/down/gate; definitions → light q).

Mixture bank: `--mixture-m` + `--adapters-bank` (opt‑in) mixes top‑m per‑module adapters (from build `--per-module`) into the subgraph delta.

Safety/determinism: keep your determinism checks and unsafe-code filters in the runner/verifier when you start executing any repo snippets (mirrors the “verifiable environment” pattern).

5) Minimal “gold” CLI surface (what you ship)

Make base adapters (once per model)
python -m examples.repo_grounded_adapters.build \
  --repo <path> --model <hf-id> --adapters-dir <out> \
  --embed-dim 1536 --base-rank 8 --include-text --kbann-priors --round-lora

# Optional code‑recall preset
python -m examples.repo_grounded_adapters.build \
  --repo <path> --model <hf-id> --adapters-dir <out> \
  --embed-dim 1536 --base-rank 8 --include-text --code-recall-preset

Answer with repo adapters (with OTF subgraph selection)
python -m examples.repo_grounded_adapters.run \
  --model <hf-id> --repo <path> --adapters-dir <out> \
  --prompt "<q>" --of-sources question --pack-context --require-citations --verbose

# Optional enhancements (opt‑in)
python -m examples.repo_grounded_adapters.run \
  --model <hf-id> --repo <path> --adapters-dir <out> \
  --prompt "<q>" --of-sources question --pack-context --require-citations \
  --code-recall-preset --layer-schedule --layer-rank-tiers --q-aware-weights \
  --mixture-m 3 --adapters-bank <out> \
  --alpha-warmup --adapter-aware-decoding \
  --verbose

Optional: per-module export
# Per-module embeddings/adapters (static)
python -m examples.repo_grounded_adapters.build \
  --repo <path> --model <hf-id> --adapters-dir <out> \
  --per-module --include-deps --max-deps 4 --sub-rank 8

Artifacts written by build
- adapters.npz, embedding.npz
- sources.jsonl (path, sha256, bytes)
- manifest.json (includes: commit, tree, created_at, schema_version, model/dims/layers/rank, targets/target_shapes, include_text + graph propagation knobs, target_weights, selection summary, CodeGraph counts, sources_file + sources_count)

6) Curriculum & “fast-weights” angle (why this works)

Treat each per-module (or subgraph) LoRA as a fast-weight memory bound to repo structure; mix it in when the question points to that slice. You’re effectively doing externalized, low-rank fast weights keyed by the CodeGraph—exactly the right place to deploy “fast-weights” style adaptation in modern LLMs. (Concept tie-in; your current on-the-fly mixer already implements the mechanic.)

The Absolute-Zero / self-play framing fits your self_tune.py: the model proposes tasks (module-specific Qs), solves them, and you keep only artifacts that your verifiers approve—your “verifiable reward.”

7) What to add next (surgical upgrades)

Entropy-aware capacity: scale rank/gsub by subgraph size (imports, indegree, call fan-in) or by question complexity (files touched by keyword search). (All features are already emitted by CodeGraph & embedding; just use them as heuristics.)

Replay buffer: keep a rolling distill.jsonl with input prompt, selected files/modules, citations, test result, and adapter stats (rank, targets). Your self_tune.py already writes logs and supports --resume; extend with simple scores for “keep/skip.”

Activation-guided selection: the enhanced runner already imports an activation tracer—wire it to select layers/files when context budget is tight.

Coverage hooks: feed coverage into the topology features and into the curriculum chooser (prefer modules with tests & higher call centrality first).

Delta safety: keep the global delta-norm cap (env) in production to prevent bad mixes from destabilizing base weights.

8) KBANN-inspired options (domain-theory → adapters)

-- Domain priors (rules → weights): enable `--kbann-priors` to derive per-target weights from CodeGraph structure (imports/calls/ownership). Prior boosts emphasize `o_proj`/`up_proj`/`down_proj`, lightly damp `v_proj`.
-- Cone-like capacity (localized ranks): at run-time, enable `--function-first` (with window packing) to add small, local adapters tied to top function windows and blend them with the subgraph adapter.
-- NOFM-style evidence: function-first packaging + citations surfaces the N-of-M evidence your answers relied on; export telemetry for downstream rule extraction.
-- Interpretability: add `--round-lora` (with `--round-threshold`) to round LoRA A/B to a tiny value set {−q, 0, +q}, improving later attribution of which files/features mattered.

9) Repository as Domain Theory (KBANN mapping)

- Use CodeGraph as the rule base (domain theory): treat imports, calls, ownership, and API signatures as approximately-correct symbolic rules.
- Initialize adapters from rules:
  - Choose targets: emphasize attention/MLP submodules that best express code relationships (`q/k/v/o`, `up/gate/down`).
  - Per-target priors: use `--kbann-priors` or `--target-weights` to bias layer deltas toward “defines/uses” edges; downweight mutually exclusive paths.
  - Sparse, light init: the base build seeds LoRA with small, structured A/B; `--round-lora` preserves interpretability.
- Local constructive induction (cones): during run, `--function-first` focuses small ranks on the exact function/file windows that matter for the question.
- Curriculum without training: even without tuning, the combination of (a) graph-derived priors in build and (b) question-aware subgraph mixing in run provides rule-guided, grounded answers with citations.

9) If you skip self‑tune (what you miss vs. tuned flow)

- Verified modules list and distill buffer (JSONL) of successful Q/A is not produced.
- No per‑module tuned adapter exports; you rely on base adapters + on‑the‑fly subgraph mixing.
- No test‑gated pruning of weak modules; selection remains heuristic (question/zoom) rather than verification‑driven.

What you still get without tune
- Full CodeGraph indexing and a repo embedding that projects tokenized sources (when `--include-text` is enabled) into the adapter prior pre‑run.
- A compact base adapter prior and the ability to mix on‑the‑fly subgraph adapters at run time for grounded, question‑specific answers.