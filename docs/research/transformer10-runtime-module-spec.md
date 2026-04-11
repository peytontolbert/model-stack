# transformer_10 Runtime Module Spec

This document turns the migration from "operator choices" into a target runtime structure.

The target is not a monolithic kernel. The target is the same kind of modular model-stack structure the repository already has, but with implementation ownership moved from Python/PyTorch into C++/CUDA.

The working assumption is still the same:

- Python remains a convenience and compatibility layer in the early phases
- execution-heavy math and runtime state move into a self-owned CUDA/C++ runtime
- the long-term model-stack implementation no longer depends on eager PyTorch math or PyTorch module objects

## 1. Target Boundary

Keep these in Python for the earliest bridge only:

- model configuration and checkpoint loading
- top-level orchestration in `model/factory.py`
- user-facing serving entrypoints in `serve/api.py`
- high-level generation policy

Longer term, these should also have C++ equivalents so the repository is a complete model-stack and not merely a Python wrapper around CUDA kernels.

Move these into CUDA/C++:

- norms
- RoPE application
- GEMM wrappers
- attention
- KV cache
- embedding lookup
- sampling hot path
- NCCL collectives

That means the current hot-path Python files become wrappers, not math owners:

- `attn/eager.py`
- `attn/kv_cache.py`
- `tensor/norms.py`
- `tensor/positional.py`
- `tensor/mlp.py`
- `tensor/sampling.py`
- `serve/runtime.py`

## 2. Target Tree

Recommended runtime tree:

```text
runtime/
  cpp/
    include/
      t10/
        config/
        data/
        checkpoint/
        model/
        blocks/
        serve/
        train/
        dist/
        eval/
        compress/
    src/
  cuda/
    CMakeLists.txt
    include/
      t10_cuda/
        status.h
        types.h
        tensor_view.h
        stream.h
        workspace.h
        allocator.h
        graph.h
        gemm.h
        norm.h
        rope.h
        embedding.h
        kv_cache.h
        attention.h
        sampling.h
        collectives.h
        engine.h
    src/
      common/
      allocator/
      graph/
      gemm/
      norm/
      rope/
      embedding/
      kv_cache/
      attention/
      sampling/
      collectives/
      engine/
    python/
      bindings.cpp
    tests/
    bench/
```

This should be a real model-stack runtime tree, not a set of disconnected extensions.

The C++ layer owns object structure, state, lifecycle, validation, and orchestration. The CUDA layer owns GPU execution kernels and vendor-library wrappers.

## 3. Module Ownership

## `common/`

Owns:

- `Status`
- error handling
- dtype/layout enums
- shape/stride helpers
- device and stream wrappers

Best references:

- NVIDIA runtime helpers from `TensorRT-LLM`
- compact helper style from `tiny-cuda-nn`
- direct-runtime minimalism from `tinygrad`

Rule:

- every public runtime op should consume typed descriptors, not loose pointer lists

## `allocator/`

Owns:

- async allocation wrappers
- reusable workspaces
- pool-aware scratch storage
- graph-stable allocation policy

Best references:

- `cudaMallocAsync` and memory-pool direction from NVIDIA corpus
- graph-aware memory discipline from `tiny-cuda-nn/include/tiny-cuda-nn/cuda_graph.h`
- current local pressure points in `tensor/arena.py`

Rule:

- the runtime should own temporary allocation policy, not Python

## `graph/`

Owns:

- capture/update/replay helpers
- bucketed graph instances by shape
- graph-safe entrypoints

Best references:

- `tiny-cuda-nn/include/tiny-cuda-nn/cuda_graph.h`
- `TensorRT-LLM` executor/runtime layering
- `cuda-samples` graph examples from the NVIDIA corpus

Rule:

- graph capture is a runtime concern, not something every kernel call site reimplements

## `gemm/`

Owns:

- `cuBLASLt` handle lifecycle
- algorithm selection
- workspace management
- epilog selection where supported
- wrappers for:
  - QKV projections
  - output projection
  - MLP up/gate/down
  - LM head

Best references:

- `CUDALibrarySamples/cuBLASLt`
- `TransformerEngine/common/gemm`
- `tiny-cuda-nn/include/tiny-cuda-nn/cutlass_matmul.h` as a secondary utility reference

Rule:

- never handwrite these GEMMs as the default path

## `norm/`

Owns:

- RMSNorm forward
- LayerNorm forward
- optional fused residual/norm variants later

Best references:

- `TransformerEngine/common/normalization`
- `flash-attention/csrc/layer_norm`
- `ThunderKittens/kernels/layernorm`

Rule:

- specialized hidden-size buckets are acceptable and desirable

## `rope/`

Owns:

- RoPE apply kernels
- optionally cache build helpers later if needed
- possible fusion with Q/K preparation

Best references:

- `TransformerEngine/common/fused_rope`
- `flash-attention/csrc/flash_attn/src/rotary.h`
- `Fuser/tests/cpp/test_rope.cpp`

Rule:

- the production target is application, not cache construction

## `embedding/`

Owns:

- token embedding gather
- optional positional/bias helpers later

Best references:

- `cuEmbed`
- `TensorRT-LLM` runtime kernels

Rule:

- treat embedding as a runtime primitive because it sits on the serving hot path

## `kv_cache/`

Owns:

- page metadata
- page allocation
- append/read/evict/reorder
- layer-local and engine-level cache handles

Best references:

- `TensorRT-LLM/docs/source/features/kvcache.md`
- `TensorRT-LLM` executor/runtime sources
- current local semantics in `attn/kv_cache.py`

Rule:

- Python should not own the cache data structure once the runtime exists

## `attention/`

Owns:

- prefill attention
- decode attention
- mask application
- softmax/value path
- MQA/GQA handling
- launch specialization by dtype/head size/mode

Best references:

- `TransformerEngine/common/fused_attn`
- `TensorRT-LLM/cpp/kernels/fmha_v2`
- `TensorRT-LLM/cpp/kernels/xqa`
- `flash-attention/csrc/flash_attn`
- `ThunderKittens/kernels/attention`

Rule:

- this module owns layout contracts and cache interaction, not `attn/eager.py`

## `sampling/`

Owns:

- temperature scaling
- repetition/presence/frequency penalties
- top-k and top-p masking
- token selection

Best references:

- local Triton notes
- `TensorRT-LLM` sampling features
- current semantics in `tensor/sampling.py`

Rule:

- keep sampler math out of Python once the hot path moves over

## `collectives/`

Owns:

- NCCL communicator lifecycle
- all-reduce
- all-gather
- reduce-scatter
- future context-parallel KV exchange

Best references:

- `nccl`
- `TensorRT-LLM` runtime communicator structure
- `ThunderKittens/kernels/parallel` only as a secondary study source

Rule:

- do not leave `torch.distributed` on the intended long-term hot path

## `engine/`

Owns:

- runtime engine state
- module handles
- shared workspaces
- graph buckets
- cache manager
- per-request execution plan

Best references:

- `TensorRT-LLM` executor/runtime
- current orchestration points in `serve/runtime.py` and `serve/engine.py`

Rule:

- this is where serving policy meets kernels; individual kernels should not own global serving state

## `model_stack/`

Owns:

- C++ equivalents for model configuration
- layer/block/module construction
- weight binding
- forward execution plans
- training/inference mode policy

Best references:

- current Python package boundaries in `model/`, `blocks/`, `attn/`, and `tensor/`
- `TensorRT-LLM` runtime/executor separation

Rule:

- preserve the repository's configurable model-stack shape; replace implementation substrate, not flexibility

## 4. Python Integration Points

## `attn/eager.py`

Current role:

- owns QKV projection, RoPE, cache concat, backend dispatch, and output projection

Target role:

- construct an attention call descriptor
- pass tensors and cache handle to `runtime.cuda.attention_forward(...)`
- receive output tensor only

Everything below that boundary should move out.

## `attn/kv_cache.py`

Current role:

- Python list-of-pages cache implementation

Target role:

- lightweight Python wrapper around a runtime cache handle
- maybe keep a debug/reference implementation for tests

## `tensor/norms.py`

Current role:

- owns full eager implementations

Target role:

- reference fallback plus runtime dispatch wrapper

## `tensor/positional.py`

Current role:

- owns both cache build and apply

Target role:

- keep cache-building helpers if convenient
- move `apply_rotary` hot path into runtime

## `tensor/mlp.py`

Current role:

- eager `Linear -> activation -> Linear`

Target role:

- call runtime GEMM wrappers and activation kernel
- keep module wiring in Python initially

## `tensor/sampling.py`

Current role:

- eager logits transforms and selection helpers

Target role:

- semantics reference and fallback
- serving path routed into runtime sampler

## `serve/runtime.py`

Current role:

- builds torch model, resolves dtype/device, allocates Python KV cache

Target role:

- construct runtime engine
- own cache manager handle
- manage graph bucket selection

## 5. Canonical Tensor Contracts

Keep the runtime on a small number of layouts.

Recommended canonical layouts:

- hidden states: `(B, T, D)`
- Q: `(B, Hq, T, Dh)`
- K/V live path: `(B, Hk, T, Dh)`
- cache pages: runtime-defined page structure, but fixed across attention kernels
- logits: `(B, V)`

Rules:

- avoid implicit transposes at every boundary
- attach shape/stride metadata to descriptors
- keep weight-layout policy in the GEMM wrapper, not throughout the Python code

## 6. Public Runtime Surface

Recommended first surface:

```text
create_engine(config) -> EngineHandle
destroy_engine(handle)

create_kv_cache(config, batch, max_tokens) -> CacheHandle
destroy_kv_cache(handle)

rmsnorm_forward(...)
rope_apply_forward(...)
embedding_lookup_forward(...)
gemm_linear_forward(...)
swiglu_forward(...)
attention_prefill_forward(...)
attention_decode_forward(...)
sampling_forward(...)
nccl_collective(...)
```

Recommended principle:

- expose low-ambiguity ops first
- add fused/block-level entrypoints only after the primitive layer is stable

## 7. Transitional Bridge

Phase 1 bridge choices:

- pybind11 is acceptable
- a thin C ABI is also acceptable if we want easier non-Python reuse later

Good local references:

- `extension-cpp` for minimal extension packaging
- `cuda_ext` for strict validation and graph-capturable variants

Bad outcome:

- a pile of unrelated custom ops with no shared runtime ownership

## 8. Build And Test Structure

Required targets:

- runtime library
- Python bindings
- unit tests
- microbenchmarks

Required test grouping:

- per-op numerical tests against current eager path
- per-op shape/layout failure tests
- end-to-end block/model parity tests
- graph-capture execution tests

Recommended benchmark grouping:

- norm
- RoPE
- GEMM wrappers
- prefill attention
- decode attention
- KV cache append/read
- sampler

## 9. Migration Sequence

Recommended order:

1. create the runtime tree and binding skeleton
2. land allocator/workspace and graph helpers
3. land RMSNorm and RoPE
4. land `cuBLASLt` GEMM wrappers
5. land KV cache
6. land prefill attention
7. land decode attention
8. land embedding and sampler
9. land NCCL collectives

## 10. Bottom Line

The runtime should not be a collection of kernels. It should be a small, explicit execution system with:

- stable tensor contracts
- owned memory/workspace policy
- owned cache state
- owned graph policy
- library-backed GEMMs and collectives
- custom kernels only where they actually add value
