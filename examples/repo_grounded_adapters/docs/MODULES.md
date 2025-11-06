## Modules Overview

High‑level documentation for the core modules used by the repo‑grounded adapter pipeline.

### Top‑level tools
- `build.py`: Builds base adapters from a repository embedding. Infers target shapes, applies priors/rounding, and writes artifacts (`adapters.npz`, `embedding.npz`, `manifest.json`, `sources.jsonl`, optional per‑module exports).
- `run.py`: CLI wrapper around the modular runner. Ensures base adapters exist, selects a subgraph on‑the‑fly, packs context, enforces citations, and generates an answer.
- `code_graph.py`: Indexes a Python repo (symbols, modules, imports, call edges, pytest nodes). Provides lookup utilities and an optional incremental cache.

### `modules/` package

- `adapter.py`
  - Purpose: Deterministic generation of LoRA A/B from repository embeddings (NumPy or Torch paths), plus NPZ save/load helpers.
  - Key functions: `generate_lora_from_embedding`, `generate_lora_from_embedding_torch`, `save_npz`, `load_adapters_npz`.

- `embedding.py`
  - Purpose: Build hashed repo/subgraph embeddings by combining multiple “knowledge factors” (symbols/docs/topology/types/calls/tests, optional text).
  - Key functions: `build_repo_embedding`, `build_subgraph_embedding_from_graph`, `auto_model_dims`.

- `runner.py`
  - Purpose: End‑to‑end orchestrator used by `run.py`. Loads base adapters, selects modules/files, builds an on‑the‑fly subgraph adapter, mixes adapters into the model, packs context, enforces citations, and generates text.
  - Key function: `generate_answer` (many flags for selection, mixing, context, sampling, and retries).

- `mixing.py`
  - Purpose: Apply mixed adapter deltas to model weights with safety caps and per‑target weighting. Clean removal after generation.
  - Key function: `register_hook_mixed_adapters`.

- `selection.py`
  - Purpose: Question‑aware and zoom‑by‑symbol selection; simple reranker combining signature/doc/test/visibility proxies.
  - Key functions: `question_aware_modules_and_files`, `modules_from_symbols`, `rerank_modules_and_files`, `name_matches_prompt`, `prompt_token_set`.

- `context.py`
  - Purpose: Context packers for file heads or symbol‑local windows; function‑first candidate collection; simple prompting heuristics to score window relevance.
  - Key functions: `pack_context_heads`, `pack_context_windows`, `collect_function_windows`, `model_prob_yes`.

- `capacity.py`
  - Purpose: Entropy/capacity scoring and schedules for `rank`/`gsub` based on repo/selection complexity.
  - Key functions: `entropy_score`, `scale_capacity`.

- `verify.py`
  - Purpose: Lightweight citation detection and optional pytest verification hook for selected modules.
  - Key functions: `has_citations`, `extract_citations`, `verify_with_tests`.

- `provenance.py`
  - Purpose: Git commit/tree helpers for provenance in manifests and answer footers.
  - Key functions: `git_commit_sha`, `git_tree_sha`.

- `prompts.py`
  - Purpose: Small prompt generators per module for distillation/verification runs.
  - Key function: `build_prompts_for_module`.

- `tune.py`
  - Purpose: Simple distillation loop: generate answers, enforce citations/tests, log results, and optionally export verified adapters.
  - Key functions: `distill_repo`, `export_tuned_adapters`.

- `peft.py`
  - Purpose: Minimal PEFT‑like export helper (best‑effort mapping); target name inference via model inspection.
  - Key functions: `infer_target_names`, `save_peft_like`.

- `targets.py`
  - Purpose: Utility to parse explicit target shapes from CLI strings.
  - Key function: `parse_target_shapes`.

- `interpret.py`
  - Purpose: Lightweight activation tracing hooks and helpers for block outputs and output head access.
  - Key functions: `is_block`, `block_out_hook`, `get_W`.

- `model.py`
  - Purpose: Reserved for snapshot helpers (thin wrapper); intentionally minimal here.

- `runtime.py`
  - Purpose: Thin programmatic entry points to run the repo adapter and compose deltas per layer.
  - Key functions: `build_per_layer_deltas`, `run_repo_adapter`.

- `telemetry.py` / `priors.py`
  - Purpose: Placeholders for future metrics/KBANN priors utilities documented elsewhere.

### Scripts and tests
- `scripts/verify_local_vs_hf.py`: Side‑by‑side HF vs local model logits/generation comparison.
- `scripts/diff_hf_local_weights.py`: Numerical comparison between HF shard weights and local weights.
- `smoke_small_repo_test.py`: Runs a baseline and adapted pass over the tiny `smoke_repo` to validate selection/packing/citation flow.


