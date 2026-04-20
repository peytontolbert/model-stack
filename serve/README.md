Serve stack overview

This directory provides a simple but complete serving stack:

- generation APIs: utility functions and an engine with temperature/top-k/p, penalties, EOS handling
- paged KV cache: per-layer, batched, paged cache used during autoregressive decoding
- endpoint server: FastAPI app exposing /v1/generate and /healthz
- request scheduler: optional background scheduler that reuses cached prompt prefixes, applies a unified token budget across prefill and decode work, supports concurrent partial prefills, can use `fcfs` or per-request priority scheduling, and batches eligible speculative decode work across requests

Quickstart

1) Prepare a model directory saved via model.checkpoint.save_pretrained (contains config.json and model.safetensors).

2) Run the API server:

   export MODEL_DIR=/path/to/model_dir
   export MODEL_STACK_DRAFT_MODEL_DIR=/path/to/draft_model_dir  # optional
   uvicorn serve.api:app --host 0.0.0.0 --port 8000

3) Call the generate endpoint:

   POST /v1/generate
   {
     "input_ids": [[1, 2, 3]],
     "max_new_tokens": 64,
     "do_sample": true,
     "temperature": 0.8,
     "top_p": 0.9
   }

Response:

   { "output_ids": [[1, 2, 3, ...]] }

Implementation notes

- runtime/generation.py now owns `GenerationConfig`, the decode session, and the generation loop.
- serve/engine.py and serve/generate.py are compatibility wrappers that preserve the old API surface while delegating generation config construction and decode execution to runtime-owned helpers.
- model/generate.py is also a compatibility shim over runtime-owned generation helpers rather than a separate eager decode implementation.
- runtime/loader.py now owns model-directory and factory-spec model loading, while runtime/prep.py owns runtime model preparation and device/dtype resolution for runtime-facing callers.
- serve/runtime.py loads a model from MODEL_DIR through `runtime/loader.py` plus `runtime/prep.py` and owns request/config coercion, health payloads, and KV cache allocation through `runtime.cache`.
- serve/runtime.py can also load an optional draft model from `MODEL_STACK_DRAFT_MODEL_DIR`, set default speculative width via `MODEL_STACK_NUM_SPECULATIVE_TOKENS`, and choose default speculative/acceptance methods through `MODEL_STACK_SPECULATIVE_METHOD` and `MODEL_STACK_SPEC_DECODING_ACCEPTANCE_METHOD` (`rejection_sampler`, `strict`, or `typical_acceptance_sampler`).
- serve/api.py is now mostly transport glue around runtime-owned helpers, including request-side sampling-mode inference and attention-mask/cache-backend passthrough.
- serve/scheduler.py owns the request queue, unified token-budget scheduling, exact/partial prefix-cache lookup, concurrent partial-prefill throttling, batched prefill/decode orchestration, cross-request speculative verification for draft/ngram/suffix methods, and per-request fallbacks when a batched step cannot be executed safely.
- runtime/blocks.py now owns block-stack execution, generic and patterned block mask shaping, attention-bias composition, and the fused residual/norm branch helpers used by model forward paths.
- runtime/cache.py owns cache-spec derivation, backend resolution, native paged-cache construction, the per-layer `layer(i)` view, and runtime-level eviction helpers.
- runtime/kv_cache.py now owns the concrete paged and contiguous cache implementations.
- attn/kv_cache.py is only a compatibility shim that re-exports the runtime-owned cache APIs.
- blocks/native_fusion.py is only a compatibility shim that re-exports the runtime-owned block helpers.
- attn/eager.py uses the KV cache when provided: reads historical K/V, concatenates with new, and appends new pages.
