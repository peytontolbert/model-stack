# TensorRT-LLM Standalone Notes

Base repo:

- `/data/parametergolf/helpful_repos/NVIDIA/TensorRT-LLM`

Why this repo matters:

- This is the strongest local reference for inference-runtime architecture.
- It matters less for "how do I write my first simple CUDA kernel" and more for "what does a real LLM serving runtime have to own once PyTorch is no longer the runtime?"
- For `transformer_10`, it closes the biggest gap left by kernel-only research:
  - paged KV cache
  - generation-vs-context execution
  - scheduling
  - sampling
  - runtime memory pools
  - multi-GPU executor structure

## Highest-Value Areas

### 1. Runtime core

Start here:

- `cpp/tensorrt_llm/runtime/cudaMemPool.cpp`
- `cpp/tensorrt_llm/runtime/bufferManager.cpp`
- `cpp/tensorrt_llm/runtime/runtimeKernels.cu`
- `cpp/tensorrt_llm/runtime/gptDecoder.cpp`
- `cpp/tensorrt_llm/runtime/gptDecoderBatched.cpp`
- `cpp/tensorrt_llm/runtime/ncclCommunicator.cpp`
- `cpp/tensorrt_llm/runtime/tllmRuntime.cpp`
- `cpp/tensorrt_llm/runtime/layerProfiler.cpp`

What it teaches:

- stream- and pool-aware runtime memory management
- decode/runtime workspace ownership
- runtime helper-kernel boundaries
- NCCL integration at runtime level
- profiler hooks after leaving framework-managed execution

### 2. Executor layer

Start here:

- `cpp/tensorrt_llm/executor/executor.cpp`
- `cpp/tensorrt_llm/executor/executorImpl.cpp`
- `cpp/tensorrt_llm/executor/dynamicBatchConfig.cpp`
- `cpp/tensorrt_llm/executor/dynamicBatchTuner.cpp`
- `cpp/tensorrt_llm/executor/kvCacheConfig.cpp`
- `cpp/tensorrt_llm/executor/samplingConfig.cpp`
- `cpp/tensorrt_llm/executor/speculativeDecodingConfig.cpp`
- `cpp/tensorrt_llm/executor/parallelConfig.cpp`

What it teaches:

- request/executor separation
- dynamic batching and scheduling knobs
- where KV cache policy belongs
- how sampling and speculative decoding become runtime configuration instead of ad hoc Python logic

### 3. Attention kernels

Start here:

- `cpp/kernels/fmha_v2`
- `cpp/kernels/xqa`
- `docs/source/features/attention.md`

Strong files:

- `cpp/kernels/xqa/mha.cu`
- `cpp/kernels/xqa/mha_sm90.cu`
- `cpp/kernels/xqa/tma.h`
- `cpp/kernels/xqa/specDec.h`
- `cpp/kernels/fmha_v2/README.md`

What it teaches:

- serving-oriented MQA/GQA generation kernels
- multi-block decode attention
- TMA/QGMMA-driven generation kernels on newer architectures
- heuristics that decide which attention path to use at runtime

### 4. KV cache system

Read:

- `docs/source/features/kvcache.md`
- `docs/source/features/paged-attention-ifb-scheduler.md`
- `docs/source/features/kv-cache-connector.md`
- `examples/cpp/executor/executorExampleKvEvents.cpp`

What it teaches:

- fixed-token paged block pools
- reuse across requests
- radix-tree-style reuse metadata
- retention and eviction policy
- offload and event surfaces

This is directly relevant to replacing:

- `attn/kv_cache.py`
- decode handling in `serve/engine.py`

### 5. Sampling, speculative decode, serving

Read:

- `docs/source/features/sampling.md`
- `docs/source/features/speculative-decoding.md`
- `docs/source/features/disagg-serving.md`
- `docs/source/features/overlap-scheduler.md`
- `examples/cpp/executor/executorExampleFastLogits.cpp`
- `examples/cpp/executor/executorExampleDisaggregated.cpp`

What it teaches:

- runtime batching of heterogeneous requests
- sampler configuration boundaries
- speculative decode runtime structure
- disaggregated serving patterns

## What To Copy Into `transformer_10`

- a real paged KV cache runtime, not Python lists of tensors
- a decode/runtime workspace layer with owned buffers and memory pools
- clear separation between:
  - model math kernels
  - runtime helper kernels
  - executor/scheduler code
- runtime-configurable sampling and batching policies

## What Not To Copy Blindly

- TensorRT engine ownership and plugin machinery are not required for the first direct CUDA/C++ pass
- the full executor system is bigger than `transformer_10` needs initially
- copy the runtime structure and policies first, not the whole product surface

## Where It Maps To `transformer_10`

Most direct mappings:

- `attn/eager.py`
- `attn/kv_cache.py`
- `serve/engine.py`
- `tensor/sampling.py`
- `dist/parallel/tensor_parallel.py`

Secondary mappings:

- `blocks/moe_block.py`
- `compress/quantization.py`

## Bottom Line For `transformer_10`

- if `TransformerEngine` is the best kernel-library reference, `TensorRT-LLM` is the best inference-runtime reference
- it should heavily influence:
  - KV cache architecture
  - decode scheduling
  - sampling runtime
  - generation-phase attention selection
- this repo is one of the strongest arguments against treating the migration as "just replace torch ops one by one"
