# transformer_10 Serving Engine C++ Spec

This document defines the C++ serving runtime required to replace `serve/`, the serving parts of `model/generate.py`, `serve/runtime.py`, and the decode ownership currently hidden in Python.

## 1. Scope

Serving replacement must own:

- runtime model load
- request admission
- batch formation and scheduling
- paged KV cache allocation and reuse
- decode loop
- sampler execution
- streaming output
- instrumentation and safety hooks

## 2. Core Objects

Required classes:

- `RuntimeConfig`
  - model dir, device, dtype, cache policy, graph policy
- `Request`
  - prompt tokens, generation config, trace id, cancel token, metadata
- `GenerationConfig`
  - max tokens, EOS policy, sampler parameters, stop criteria
- `BatchSlot`
  - per-request state inside a live decode batch
- `DecodeBatch`
  - active slots sharing one decode launch
- `GenerationSession`
  - per-request mutable state and output buffers
- `ServingEngine`
  - top-level runtime that owns model, scheduler, cache manager, graph plans
- `Scheduler`
  - admission, bucket selection, continuous batching, eviction policy
- `Streamer`
  - token emission and partial-output callbacks
- `InstrumentationSink`
  - NVTX, counters, latency histograms, request traces
- `SafetyHook`
  - prompt/response guard integration

## 3. Request Lifecycle

Canonical path:

1. tokenize prompt
2. create request and generation session
3. reserve cache pages and batch slot
4. prefill pass over prompt
5. enter decode loop
6. run sampler and stopping criteria
7. stream token(s) out
8. finalize session and return cache pages

Rules:

- batching and scheduling live in C++, not Python glue
- sampler sees raw logits on device and runs there
- cache ownership is explicit and independent of Python object lifetime

## 4. Scheduler Design

Required scheduler features:

- continuous batching
- prompt-length bucketing
- decode-shape bucketing
- graph-safe stable shapes where possible
- cache-aware admission control
- cancellation support
- timeout and max-length enforcement

Recommended scheduling split:

- `PrefillScheduler`
- `DecodeScheduler`

because prefill and decode have different kernel and memory behavior.

## 5. KV Cache Runtime

Cache runtime must own:

- page allocation from async memory pool
- per-layer and per-head layout metadata
- append and read kernels
- block table / page table metadata
- reorder and compact operations
- sliding-window truncation
- optional eviction policy

Required C++ objects:

- `CachePool`
- `CachePageTable`
- `LayerCacheView`
- `RequestCacheHandle`

This replaces both `attn/kv_cache.py` orchestration and the ad hoc cache handling in `serve/engine.py`.

## 6. Kernel Integration

Serving runtime calls these kernel families:

- `qkv_prepare`
- `attention_prefill`
- `attention_decode`
- `paged_kv_append`
- `paged_kv_gather`
- `norm_residual`
- `mlp_gated`
- `sampler`

Library wrappers:

- cuBLASLt for projections and MLP GEMMs
- NCCL only when multi-GPU serving is enabled

## 7. CUDA Graph Policy

Serving runtime should own:

- graph-capture eligibility checks
- graph buckets by batch, prompt length, and decode length
- static workspace reservation
- replay handles and invalidation policy

Graph integration belongs in C++ runtime, not Python wrappers.

## 8. Sampler Runtime

Sampler must run in device-owned code for:

- temperature
- top-k / top-p / min-p / typical / eta / TFS
- repetition and frequency penalties
- no-repeat-ngram and constraint masks
- grammar/schema/regex masking when feasible
- token selection

Host-side policy decisions are allowed, but logits transforms and token choice should be device-owned.

## 9. Instrumentation

Required instrumentation surfaces:

- request-level latency
- prefill latency
- decode tokens/sec
- cache occupancy
- graph hit-rate
- kernel choice / autotune plan ids
- sampler latency
- per-request cancellation and failure reasons

Required hooks:

- NVTX ranges around prefill, decode, sampler, cache, and NCCL overlap
- trace export for visualization and debugging

This is the owned replacement for `serve/instrumented_generate.py`.

## 10. Safety And Policy Hooks

Safety hooks should be explicit interfaces:

- prompt pre-check
- output post-check
- token-level veto or mask injection

The serving runtime should not depend on Python to remain correct, but a Python policy adapter can exist for non-hot-path deployments.

## 11. Public APIs

Required C++ entrypoints:

- `ServingEngine::load(RuntimeConfig cfg)`
- `ServingEngine::submit(Request req) -> RequestHandle`
- `ServingEngine::poll(RequestHandle handle) -> StreamChunk`
- `ServingEngine::cancel(RequestHandle handle)`
- `ServingEngine::step()`
- `ServingEngine::shutdown()`

Python binding layer can expose:

- `generate()`
- `stream_generate()`
- `runtime_from_dir()`

but those are wrappers, not the implementation substrate.

## 12. File Mapping

| Current file | Target implementation |
|---|---|
| `serve/runtime.py` | `t10::serve::RuntimeConfig`, `ServingEngine::load` |
| `serve/engine.py` | `t10::serve::ServingEngine`, `Scheduler`, `GenerationSession` |
| `serve/generate.py` | Python binding over `ServingEngine` |
| `serve/instrumented_generate.py` | Python instrumentation adapter over `InstrumentationSink` |
| `serve/api.py` | Python API server wrapper around the C++ engine |
| `model/generate.py` | merge into `t10::serve` or keep as compatibility wrapper only |

## 13. Migration Order

1. single-request prefill + decode
2. batch decode with paged KV cache
3. device-owned sampler
4. continuous batching scheduler
5. CUDA graph buckets
6. streaming output and instrumentation
7. safety and policy hooks

## 14. Definition Of Serving-Ready

The serving spec is complete when all of these are explicit:

- request lifecycle
- batch scheduler
- cache ownership
- graph policy
- sampler ownership
- instrumentation path
- public binding boundary

That is the minimum required to replace the current Python serving loop with a true C++ runtime.
