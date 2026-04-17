Serve stack overview

This directory provides a simple but complete serving stack:

- generation APIs: utility functions and an engine with temperature/top-k/p, penalties, EOS handling
- paged KV cache: per-layer, batched, paged cache used during autoregressive decoding
- endpoint server: FastAPI app exposing /v1/generate and /healthz

Quickstart

1) Prepare a model directory saved via model.checkpoint.save_pretrained (contains config.json and model.safetensors).

2) Run the API server:

   export MODEL_DIR=/path/to/model_dir
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
- serve/api.py is now mostly transport glue around runtime-owned helpers, including request-side sampling-mode inference and attention-mask/cache-backend passthrough.
- runtime/blocks.py now owns block-stack execution, generic and patterned block mask shaping, attention-bias composition, and the fused residual/norm branch helpers used by model forward paths.
- runtime/cache.py owns cache-spec derivation, backend resolution, native paged-cache construction, the per-layer `layer(i)` view, and runtime-level eviction helpers.
- runtime/kv_cache.py now owns the concrete paged and contiguous cache implementations.
- attn/kv_cache.py is only a compatibility shim that re-exports the runtime-owned cache APIs.
- blocks/native_fusion.py is only a compatibility shim that re-exports the runtime-owned block helpers.
- attn/eager.py uses the KV cache when provided: reads historical K/V, concatenates with new, and appends new pages.
