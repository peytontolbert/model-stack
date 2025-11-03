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

repo_conditioned_adapter.py — repo & subgraph embeddings via stable feature hashing across: symbols, docs, modules, topology (+optional text); graph propagation; normalization; deterministic seeds.

run.py — make base adapters once per model ID; infer shapes from HF config; set capacity & “knowledge-preset” target weights; then run a single prompt with packing/citations if desired.

run_llama_with_repo_adapter_on_the_fly.py — minimal OTF mixer: resolve layer modules, build per-layer deltas, add to weight tensors, question/zoom selection knobs (zoom_radius, include_text, text_weight).

run_repo_adapter.py — production mixer: target map shims, question-aware module/file selection, context packing, target-weight parsing, delta-cap safety; built on modular utilities.

self_tune.py — curriculum/self-play loop for repo Q/A: build prompts per module, run enhanced runner, extract citations, run mapped tests, export per-module adapters (rank, deps), persist distill.jsonl + logs.

tensor_utils.py — safe tensor ops + (optional) opt_einsum contraction planning cache; handy for future perf passes.

3) “Ultimate” self-tune loop (grounded, verifiable)

Propose verifiable tasks per module (e.g., “Explain top functions in X with path:line” and “What does foo do?”).

Solve with adapters mixed from the subgraph most relevant to the prompt; require citations.

Verify with two cheap verifiers:

Citation verifier: extract path:line hits and check they exist in the indexed files.

Test verifier: run mapped pytest nodes for that module when present; accept soft-pass if no tests.

Select/Keep: only write per-module adapters for answers that pass verification; optionally include a bounded set of import deps when exporting.

This mirrors the Absolute Zero shape (single model both proposes & solves with verifiable rewards). Your verifiers are (a) path:line and (b) tests; rewards are pass/fail of those. Keep a rolling buffer (distill) and iterate.

4) Controls that make it robust

Capacity knobs: --rank, --alpha, --gsub (blend base vs subgraph); per-target weights (e.g., boost o/up/down for “knowledge recall”); optional delta-norm caps via REPO_ADAPTER_DELTA_CAP.

Selection knobs: --of-sources question|zoom, --zoom-symbol, --zoom-radius, and text feature control (--include-text, --text-max-bytes, --max-text-tokens, --text-weight).

Context knobs: packing mode (heads/windows), token budget, and per-paragraph citation enforcement.

Safety/determinism: keep your determinism checks and unsafe-code filters in the runner/verifier when you start executing any repo snippets (mirrors the “verifiable environment” pattern).

5) Minimal “gold” CLI surface (what you ship)

Make base adapters (once per model)
python run.py --model <hf-id> --adapters-dir <out> --base-rank 8 --embed-dim 1536 --knowledge-preset

Answer with OTF adapters
python run_llama_with_repo_adapter_on_the_fly.py --model <hf-id> --adapters <base/adapters.npz> --repo <path> --prompt "<q>" --of-sources question --include-text

Build adapters (base + optional per-module)
python -m examples.repo_grounded_adapters.build_repo_adapter \
  --model <hf-id> --repo <path> --adapters-dir <out> \
  --embed-dim 1536 --base-rank 8 --include-text --kbann-priors --round-lora

Answer with modular repo adapter runner
python -m examples.repo_grounded_adapters.run_repo_adapter \
  --model <hf-id> --adapters <out>/adapters.npz --repo <path> \
  --prompt "<q>" --pack-context --require-citations

Self-tune (per-module distill + adapters)
python self_tune.py --repo <path> --model <hf-id> --adapters <npz> --out <dir> --per-module-adapters --include-deps --rank 8 --context-tokens 5000 --resume

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

- Domain priors (rules → weights): enable `--kbann-priors` to derive per-target weights from CodeGraph structure (imports/calls/ownership). Prior boosts emphasize `o_proj`/`up_proj`/`down_proj`, lightly damp `v_proj`.
- Cone-like capacity (localized ranks): enable `--function-first` with `--cone-rank` and `--cone-weight` to add small, local adapters tied to top function windows and merge them with the subgraph adapter.
- Curriculum-first self-tune: start from module-scoped Q/A via `self_tune.py`, then expand to cross-module prompts; symbolic priors narrow the search while adapters learn the missing glue.
- NOFM-style evidence: function-first packaging + citations surfaces the N-of-M evidence your answers relied on; export telemetry for downstream rule extraction.
- Interpretability during tuning: add `--round-lora` (with `--round-threshold`) to round LoRA A/B to a tiny value set {−q, 0, +q}, improving later attribution of which files/features mattered.