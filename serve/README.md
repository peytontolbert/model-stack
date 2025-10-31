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
     "temperature": 0.8,
     "top_p": 0.9
   }

Response:

   { "output_ids": [[1, 2, 3, ...]] }

Implementation notes

- serve/engine.py implements decoding with sampling policies using tensor.sampling utilities.
- serve/runtime.py loads a model from MODEL_DIR and allocates KV caches sized to batch.
- attn/kv_cache.py provides a per-layer batched paged KV cache with layer(i) accessor used by model.CausalLM.
- attn/eager.py uses the KV cache when provided: reads historical K/V, concatenates with new, and appends new pages.
