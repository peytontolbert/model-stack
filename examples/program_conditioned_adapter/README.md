## Program‑Conditioned Adapter (PCA)

Program Conditioned Adapter is a small, parameterization (LORA) computed from a Program--an explicit, configuration of entities, edges, artifacts, contracts, and state-- so that a pretrained LLM is tuned at inference to a domain+task, without retraining the base model.

PCA provides:
- Reversible conditioning: synthesize a tiny Δ and apply/remove cleanly at run time.
- Grounded decoding: selection, context, and citations anchored to program artifacts.
- Backend‑agnostic core: works with repositories, APIs, CLIs, SQL, DAGs, configs, logs.


## Core pipeline (end‑to‑end)

1) Index → ProgramGraph
- Build an incremental ProgramGraph with entities and typed edges over the target system.
- For the Python repository example, we walk the repo and produce a code graph (modules, symbols, signatures, docs, imports, star‑imports, call edges, test nodes, coverage stubs).

2) Embed → Channelized program vectors
- Compute a compact, normalized feature vector z from channelized views:
  - symbols and signatures, contracts/schemas, topology (imports/calls/ownership), optional raw text and traces.
- Controls include: graph propagation hops/damp, include_text, text_max_bytes, max_text_tokens, text_weight.

3) Adapt → Generate LoRA from embedding
- Infer target shapes from the model config and produce per‑layer low‑rank A/B for q/k/v/o and MLP up/down/gate.
- Expose per‑target weights and named presets (e.g., “knowledge” or “code_recall”) to bias which projections express domain knowledge.

4) Mix + Apply → Runners
- Two paths:
  - On‑the‑fly subgraph: select by question or zoom(symbol, radius); synthesize small Δ and apply with scaling (alpha/rank); remove cleanly later.
  - Modular runner: same mixing, plus context packing, retrieval policy, and target‑weight parsing; optional delta caps via env guard.

5) Answer → Pack context, require citations
- Build prompt from selected files/windows and enforce per‑paragraph citations.
- The core `CitationManager` validates/repairs anchors via `ProgramGraph.resolve(uri)` and stamps provenance on each emitted unit.

6) Self‑tune (optional) → Verify, keep only wins
- Run verifiable prompts per module/region, collect citations, run mapped tests when available; export per‑module adapters that pass. Maintain a distill buffer (Q/A + telemetry) for replay.


## Key abstractions

- Program (modality‑agnostic): P = ⟨U, E, A, C, S, O, T, Θ⟩ where
  - U entities, E edges, A artifacts, C contracts, S state, O observables, T traces, Θ semantics (probes/evaluators).
  - Region R ⊆ (U ∪ E) is the unit of conditioning/citation for a query q.
- ProgramGraph (core interface):
  - entities(), edges(), search_refs(token), subgraph(seeds, radius), artifacts(kind), resolve(uri) → {artifact_uri, span, hash}.
- RetrievalPolicy (policy‑driven selection):
  - score_entities(q, pg) using a mix like sim={bm25,dense,symbol} and struct={graph_distance,call_proximity,ownership} with temperature.
- Channelized Embedder:
  - z = z_sym ⊕ z_contract ⊕ z_topo ⊕ z_text ⊕ z_trace with per‑channel budgets and normalization.
- Adapter Synthesis and Mixing:
  - Deterministic seeding; target weights; layer schedule; rank tiers; optional delta norm caps.
- CitationManager:
  - collect/enforce/repair citations; stamp per‑unit provenance {program_id, manifest_sha, commit, policy}.


## Build‑as‑Cache (artifacts)

Build writes adapters and a manifest alongside fast retrieval/anchoring caches:
- adapters.npz, embedding.npz
- symbol_index.jsonl
- windows_index.jsonl (function‑first windows with line ranges, byte offsets, hashes)
- facts.jsonl (crisp entity definitions with spans, URIs, checksums)
- rerank_features.npz (BM25/TF‑IDF and structural placeholders)
- self_queries.jsonl (optional, empty scaffold by default)
- sources.jsonl (path, sha256, bytes)
- manifest.json
  - commit/tree, created_at, schema_version
  - model/dims/layers/rank, targets/target_shapes, include_text + graph propagation knobs
  - target_weights, selection summary, ProgramGraph counts
  - caches: file paths + sha256 for all emitted artifacts


## Minimal CLI (current example: Python repo)

1) Make base adapters (once per model, program-agnostic)

```bash
python -m examples.program_conditioned_adapter.build \
  --sources <program-root> \
  --model <hf-id> \
  --adapters-dir <out> \
  --embed-dim 1536 \
  --include-text \
  --kbann-priors
# Optional: provide a ProgramGraph backend plugin
#   --pg-backend <module_path:Factory>   # e.g., a backend that implements ProgramGraph
```

- Optional code‑recall preset:

```bash
python -m examples.program_conditioned_adapter.build \
  --sources <program-root> \
  --model <hf-id> \
  --adapters-dir <out> \
  --embed-dim 1536 \
  --include-text \
  --code-recall-preset
```

2) Answer with PCA (on‑the‑fly region + citations, program‑agnostic runner)

```bash
python -m examples.program_conditioned_adapter.run \
  --model <hf-id> \
  --sources <program-root> \
  --adapters-dir <out> \
  --prompt "<q>" \
  --of-sources question \
  --pack-context \
  --pack-mode windows \
  --context-tokens 3000 \
  --require-citations \
  --retrieval-policy "sim:0.6,struct:0.4" --retrieval-temp 0.7 \
  --citations-enforce \
  # Optional: provide a ProgramGraph backend plugin for selection/evidence stamping
  # --pg-backend <module_path:Factory> \
  --verbose
```

- Optional enhancements (opt‑in):

```bash
python -m examples.program_conditioned_adapter.run \
  --model <hf-id> \
  --sources <program-root> \
  --adapters-dir <out> \
  --prompt "<q>" \
  --of-sources question \
  --pack-context \
  --require-citations \
  --code-recall-preset \
  --layer-schedule \
  --layer-rank-tiers \
  --q-aware-weights \
  --per-target-rank-schedule \
  --rank-budget 64 \
  --mixture-m 3 \
  --adapters-bank <out> \
  --alpha-warmup \
  --adapter-aware-decoding \
  --verbose
```


## Controls that make it robust

- Capacity knobs: --rank, --alpha, --gsub (blend base vs subgraph); per‑target weights (boost o/up/down for knowledge recall); `--layer-schedule` (per‑layer multipliers toward top third); optional delta‑norm caps via environment guard.
- Layer tiers and rank budgets: `--layer-rank-tiers` (per‑layer keeps by target group), `--rank-budget` (cap sum of per‑target keeps per layer).
- Adapter mapping: enhanced mapping (normalized z + per‑target segment scaling) is always on.
- Selection knobs: `--of-sources question|zoom`, `--zoom-symbol`, `--zoom-radius`, and text feature controls (`--include-text`, `--text-max-bytes`, `--max-text-tokens`, `--text-weight`).
- Context knobs: packing mode (heads|windows), token budget, and per‑paragraph citation enforcement.
- Question‑aware weights: `--q-aware-weights` (opt‑in) upweights target groups by prompt intent (e.g., signatures → o,v; behavior → up/down/gate; definitions → light q).
- Mixture bank: `--mixture-m` + `--adapters-bank` mixes top‑m per‑module adapters into the subgraph delta.
- Safety/determinism: keep determinism checks and unsafe‑code filters in the runner/verifier if executing snippets; maintain global delta‑norm caps in production.


## KBANN‑inspired options

- Domain priors (rules → weights): enable `--kbann-priors` or pass `--target-weights` to bias layer deltas toward defines/uses topology; downweight mutually exclusive paths.
- Cone‑like capacity (localized ranks): at run‑time, `--function-first` focuses small ranks on top function windows and blends them with the subgraph adapter.
- NOFM‑style evidence: function‑first packaging + citations surfaces the N‑of‑M evidence your answers relied on; export telemetry for downstream rule extraction.
- Interpretability: add `--round-lora` (with `--round-threshold`) to round LoRA A/B to a tiny value set {−q, 0, +q}, improving later attribution of which files/features mattered.


## Why this works (fast‑weights angle)

Treat each per‑module or subgraph LoRA as a fast‑weight memory bound to program structure; mix it in when the question points to that slice. This externalizes low‑rank fast weights keyed by the ProgramGraph—exactly the right place to deploy “fast‑weights” style adaptation in modern LLMs. The on‑the‑fly mixer implements this mechanic.


## What you get without self‑tune

- Full ProgramGraph indexing and a program embedding that projects sources/specs into the adapter prior pre‑run (when `--include-text` is enabled).
- A compact base adapter prior and the ability to mix on‑the‑fly subgraph adapters at run‑time for grounded, question‑specific answers with citations.


## Example backend and smoke test

- Two repo‑conditioned examples (separate from the core) demonstrate knowledge planning and Q/A:
  - Q/A: `examples/program_conditioned_adapter/examples/python_repo_grounded_qa/run_smoke_example.py`
  - Planning: `examples/program_conditioned_adapter/examples/python_repo_grounded_planning/run_smoke_example.py`
- You can also provide your own `ProgramGraph` via `--pg-backend <module_path:Factory>` to enrich build/run with structured artifacts and evidence stamping.


## Notes and roadmap

- Backend‑agnostic core; additional backends (OpenAPI/gRPC, CLI, SQL, DAG/orchestrators, configs/infra) follow the same ProgramGraph interface.
- Union graphs: a future `UnionProgramGraph` composes multiple ProgramGraphs with namespace isolation and de‑dup; mixture bank weights across repos with per‑repo capacity budgets.
- Evidence/provenance schema and acceptance predicates are documented in `examples/program_conditioned_adapter/refactor.md`.