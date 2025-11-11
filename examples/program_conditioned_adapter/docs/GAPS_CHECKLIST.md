## Repo‑Grounded Adapters: Gaps and Backward‑Compatible Improvements

This checklist enumerates gaps versus the desired design and proposes additive, opt‑in improvements. Changes MUST preserve current behavior by default (no removals; defaults remain backward compatible).

### Legend
- [ ] pending
- [ ] in progress
- [ ] done


## 1) Target prioritization and effective ranks

- [x] Include `gate_proj` in default target weights and presets
  - Current: presets omit `gate_proj`; code supports it in mixing and adapter gen.
  - Implemented: added `--code-recall-preset` including `gate_proj`; legacy `--knowledge-preset` unchanged.
  - Files: `examples/repo_grounded_adapters/run.py`, `build.py`.
  - Backward‑compat: CLI overrides preserved.

- [x] Per‑target effective rank at run‑time (slice A/B by target, per layer)
  - Implemented: `--per-target-rank-schedule` (global keeps) + `--rank-budget` (cap sum per layer); layer‑tiered keeps via `--layer-rank-tiers`.
  - Files: `modules/mixing.py` (`per_target_keep`, `per_target_keep_layers`), `modules/runner.py`, `run.py`.
  - Backward‑compat: opt‑in.

- [x] Default weights align with code‑recall bias
  - Proposed defaults: o=1.15, v=1.10, up=1.10, down=1.05, gate=1.0, q=0.95, k=0.90.
  - Implemented via `--code-recall-preset` in build/run; remains opt‑in.


## 2) Per‑layer specialization (layerwise schedule)

- [x] Layerwise rank schedule across thirds (low/mid/top)
  - Proposed (per target group):
    - q,k: 2 / 3 / 4
    - v,o: 8 / 12 / 16
    - up,down: 12 / 16 / 24
    - gate: min(8, 0.5·up)
  - Implemented: `--layer-rank-tiers` computes per‑layer keeps and applies via mixing (`per_target_keep_layers`).
  - Files: `modules/runner.py`, `modules/mixing.py`.
  - Backward‑compat: opt‑in; off by default.

- [x] Layerwise weight multiplier `g_l` (rises toward top third)
  - Implement `weight_l(target) = base_target_weight * g_l(layer)` in mixing.
  - Implemented: optional `layer_multipliers` in `mixing.register_hook_mixed_adapters`, wired from runner when `--layer-schedule` is set.
  - Backward‑compat: default `g_l = 1.0`.


## 3) Subgraph mixture bank (cache + π‑gating)

- [x] Build/export a bank of subgraph adapters (per package/module/community)
  - Files: `build.py` already supports `--per-module`; extend manifest and loader utility.

- [x] At run‑time, retrieve top‑m subgraphs and mix with π weights
  - Δθ = Σ_i π_i · Δθ_i; support `--mixture-m` (default 0 → disabled).
  - Implemented: `--mixture-m` + `--adapters-bank` load and concatenate A/B from bank for selected modules (uniform π).
  - Files: `modules/runner.py`.
  - Backward‑compat: disabled by default; falls back to current behavior.


## 4) Mapping z → (A,B): normalization and constraints

- [x] Normalize repository/subgraph vector z (layer‑norm or cosine scale)
  - Implemented: normalize z before per‑target segmenting (always‑on).
  - Files: `modules/adapter.py` (NumPy/Torch paths).

- [x] Disjoint z segments per target to reduce interference
  - Implemented: per‑target segment scalar (deterministic by target/layer).
  - Files: `modules/adapter.py` (NumPy/Torch).

- [ ] Tiny per‑target mapping (affine or 1‑hidden‑layer MLP) to generate A,B scalings
  - Keep low‑cost; remain deterministic with seed.

- [ ] Spectral‑norm cap on A and B (Frobenius proxy)
  - Status: available as an internal option in adapter code; not exposed in CLI and currently disabled by default.

- [ ] Cross‑target orthogonalization within a layer (columns of A)
  - Optional; guard by `--orthogonalize-targets` (default off).

- Backward‑compat: Mapping is always-on (no dual simple/enhanced paths or flags).


## 5) Entropy‑aware capacity: two‑stage

- [x] Stage 1 (pre‑decode): already present
  - Files: `modules/runner.py` uses `entropy_score` + `scale_capacity`.

- [x] Stage 2 (mid‑decode fallback): bump rank/gsub for top layers and retry on failure
  - Trigger when citations absent or confidence low.
  - Implemented: Non‑structured path re‑registers hooks with stronger top‑layer multipliers and higher `g_sub` on retry.
  - Files: `modules/runner.py` retry branch.
  - Backward‑compat: only executed on citation failure.


## 6) Question‑aware reweighting of targets

- [x] Heuristic classifier over the prompt to reweight target groups
  - Examples:
    - Signature/params → upweight o,v
    - Behavior/“why fails” → upweight up,down,gate slightly
    - “Where is X defined” → light bump q; avoid heavy k unless retrieval is noisy
  - Implemented: optional `--q-aware-weights` adjusts target_weights before mixing.
  - Files: `modules/runner.py`.
  - Backward‑compat: default off.


## 7) Constrained decoding hooks (adapter‑aware)

- [x] Local decode adjustments near retrieved spans (baseline)
  - Lower repetition penalty and relax top‑p in windows that align with retrieved context.

- [x] Enforce citation pointer before prose for path:line claims (baseline)
  - Lightweight template or pointer token emission.

- Files: `modules/runner.py` (decoding scaffolding and prompt adjustments).
  - Backward‑compat: guard by `--adapter-aware-decoding` (default off).


## 8) Stabilization & safety rails

- [x] Per‑layer rank budget cap: Σ r_target(l) ≤ R_max(l)
  - Enforce during concat/mixture to avoid parameter blow‑up.

- [x] Warmup mixing: α schedule over first N decode steps (baseline)
  - Interpolate 0 → α across first 24 tokens (default off).

- [x] Ablation toggles for evaluation (group‑wise off: q/k, v/o, MLP)
  - Implemented: `--ablate-attn`, `--ablate-mlp`.
  - Files: `run.py`, `modules/runner.py`.

- Existing: delta cap via `REPO_ADAPTER_DELTA_CAP` (keep).
  - Files: `modules/mixing.py`.


## 9) Self‑tuning loop (no base finetune)

- [ ] Add inner‑loop adapter update with composite loss and Δ budget
  - Loss: citation precision/recall, tests, hallucination penalty, norm regularizer.
  - Early stop + rollback on metric degrade; only keep improved subgraph adapter.

- Files: `modules/tune.py` (extend), `modules/verify.py` (reuse), `modules/adapter.py` (export updated A/B).
  - Backward‑compat: guard by `--enable-adapter-tuning` in tuning script; default off.


## 10) Telemetry and metrics per question

- [x] Add metrics:
  - Citation@1/@k, line‑level accuracy
  - Compile/test pass rate (if runnable)
  - Edit distance from retrieved definitions (signatures)
  - Latency and retry counts
  - Delta norms per group/layer
  - Ablation sensitivity

- Implemented (baseline set) in structured path: `metrics` includes `citations_total`, `citations_must`, `modules_selected`, `files_selected`, `elapsed_sec`, `retries`. Extend in future for compile/tests and edit distance.
- Files: `modules/runner.py` (structured path). Backward‑compat: result enriched; sidecar written only when `--telemetry-out` is provided.
  - Tests: optional `--telemetry-tests` records `tests_checked` and `tests_passed` (best‑effort).
  - Edit distance: `metrics.signature_edit_mean` and `metrics.signature_edit_min` (best‑effort Levenshtein over cited snippets).


## 11) Concrete default set (opt‑in preset)

- [x] Add a `--code-recall-preset` that sets:
  - Targets: `o,v,up,down,gate,q,k`
  - Weights: o=1.15, v=1.10, up=1.10, down=1.05, gate=1.0, q=0.95, k=0.90
    - Implemented via `--code-recall-preset` in build/run (adds `gate_proj`).
  - Ranks (top/mid/low thirds):
    - q,k: 4 / 3 / 2
    - v,o: 16 / 12 / 8
    - up,down: 24 / 16 / 12
    - gate: min(8, 0.5·up)
    - Implemented via `--layer-rank-tiers` (per‑layer keeps by target group), capped by effective `rank`.
  - Δ cap τ ≈ 0.06
    - Baseline delta cap available via env `REPO_ADAPTER_DELTA_CAP=0.06` (applied at mix time).
    - Optional mapping cap via `build.py --map-cap <float>` (caps ||A||_F, ||B||_F per target).
  - Warmup α over first steps
    - Implemented baseline warmup via `--alpha-warmup` (lighter alpha on first attempt/pass).
  - Mixture bank: m=3 subgraphs (convex mix)
    - Implemented via `--mixture-m 3` + `--adapters-bank <dir>`; concatenation/weighted modes supported.

- Backward‑compat: This preset is opt‑in and leaves existing `--knowledge-preset` intact.


## Non‑destructive constraints

- All new behaviors are opt‑in via flags or presets; defaults preserve current outputs.
- No removal of existing knobs or flows (selection, cones, entropy Stage‑1, rounding, citations).
- Minimal overhead when features disabled (avoid extra allocations or large CPU/GPU work).


## Implementation breadcrumbs (where to plug each change)

- Runner core: `examples/repo_grounded_adapters/modules/runner.py`
  - Capacity schedules, q‑aware weights, mixture bank, Stage‑2 fallback, decoding hooks, telemetry.

- Mixing: `examples/repo_grounded_adapters/modules/mixing.py`
  - Per‑layer weight multipliers, per‑target rank slicing, rank budget cap, alpha warmup integration.

- Adapter gen: `examples/repo_grounded_adapters/modules/adapter.py`
  - z normalization, per‑target segmenting, small MLP/affine, spectral caps, orthogonalization.

- Build: `examples/repo_grounded_adapters/build.py`
  - Bank export and manifests; keep existing per‑module/files‑only exports.

- Tuning & Verify: `examples/repo_grounded_adapters/modules/tune.py`, `examples/repo_grounded_adapters/modules/verify.py`
  - Adapter inner‑loop and evaluation signals.


