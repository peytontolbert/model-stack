# NVIDIA Kernel Reference Map

This document maps the most useful local NVIDIA repositories to the parts of `transformer_10` we need to replace.

Base path:

- `/data/parametergolf/helpful_repos/NVIDIA`

The key point is that not all "CUDA" repos are equally useful. For replacing PyTorch in this model stack, the highest-value references are:

1. `TransformerEngine`
2. `cutlass`
3. `CUDALibrarySamples`
4. `cuda-samples`
5. `cccl`
6. `nccl`
7. `TensorRT-LLM`
8. `Fuser`
9. `cuEmbed`
10. `Megatron-LM`
11. `cuda-tile`
12. `nvmath-python`
13. `triton` (adjacent external)

## 1. TransformerEngine

Best use: production transformer kernels, fused numerics, cuBLASLt integration, RoPE, norms, attention, comm+GEMM overlap.

Top files:

- `TransformerEngine/transformer_engine/common/fused_attn/flash_attn.cu`
- `TransformerEngine/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu`
- `TransformerEngine/transformer_engine/common/fused_attn/fused_attn_fp8.cu`
- `TransformerEngine/transformer_engine/common/fused_attn/kv_cache.cu`
- `TransformerEngine/transformer_engine/common/fused_softmax/scaled_masked_softmax.cu`
- `TransformerEngine/transformer_engine/common/fused_softmax/scaled_aligned_causal_masked_softmax.cu`
- `TransformerEngine/transformer_engine/common/fused_rope/fused_rope.cu`
- `TransformerEngine/transformer_engine/common/normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu`
- `TransformerEngine/transformer_engine/common/normalization/layernorm/ln_fwd_cuda_kernel.cu`
- `TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu`
- `TransformerEngine/transformer_engine/common/transpose/cast_transpose_fusion.cu`
- `TransformerEngine/transformer_engine/common/comm_gemm_overlap/userbuffers/userbuffers.cu`

Why it matters:

- It shows what a production transformer kernel library actually looks like.
- It demonstrates which ops are worth custom kernels and which should defer to cuBLASLt.
- `rmsnorm_fwd_cuda_kernel.cu` is especially relevant because it uses shape-specialized launch registration for common hidden sizes and falls back to more general paths for odd shapes.
- `fused_rope.cu` is relevant because it shows shared-memory sin/cos staging, support for interleaved and non-interleaved layouts, packed sequences, and context-parallel position handling.
- `cublaslt_gemm.cu` is relevant because it is a direct example of how to canonicalize row-major tensor layouts into cuBLASLt execution while preserving fusion and low-precision support.

What to copy conceptually:

- Shape-specialized launch tables for common transformer hidden sizes.
- A clean split between:
  - kernel launch/configuration logic
  - low-level kernel bodies
  - tensor/layout canonicalization
- Use of cooperative kernels only when cross-CTA coordination justifies it.

What not to copy blindly:

- FP8 and Blackwell/Hopper-specific branches are useful references, but they will complicate a first pass.
- Start with BF16/FP16/FP32 correctness and layout stability first.

## 2. cuda-samples

Best use: official API patterns, small isolated examples, graphs, async execution, memory pools, WMMA/Tensor Core usage, runtime compilation.

Top files:

- `cuda-samples/Samples/0_Introduction/asyncAPI/asyncAPI.cu`
- `cuda-samples/Samples/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu`
- `cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu`
- `cuda-samples/Samples/6_Performance/cudaGraphsPerfScaling/cudaGraphPerfScaling.cu`
- `cuda-samples/Samples/0_Introduction/matrixMul/matrixMul.cu`
- `cuda-samples/Samples/0_Introduction/matrixMul_nvrtc/matrixMul_kernel.cu`
- `cuda-samples/Samples/3_CUDA_Features/ptxjit/ptxjit_kernel.cu`
- `cuda-samples/Samples/0_Introduction/simpleP2P/simpleP2P.cu`

Why it matters:

- These samples are the cleanest baseline for correct CUDA API usage.
- `asyncAPI.cu` is the simple reference for overlapping host and device execution with streams and events.
- `streamOrderedAllocation.cu` is directly relevant to KV-cache and temporary workspace allocation because it demonstrates `cudaMallocAsync`, `cudaFreeAsync`, and memory-pool tuning.
- `cudaGraphPerfScaling.cu` is relevant for decode loops where launch overhead matters.
- `bf16TensorCoreGemm.cu` is useful when reasoning about tile sizes, shared-memory staging, padding/skew, and Tensor Core constraints, even if the final production path uses cuBLASLt instead of handwritten GEMM.

What to copy conceptually:

- Event timing patterns.
- Stream-ordered alloc/free.
- CUDA Graph capture and replay discipline.
- Shared-memory layout care for Tensor Core paths.

## 3. CCCL

Best use: reusable CUDA C++ building blocks instead of rewriting block- and warp-level mechanics by hand.

Top files:

- `cccl/cub/cub/block/block_reduce.cuh`
- `cccl/cub/cub/block/block_load.cuh`
- `cccl/cub/cub/block/block_store.cuh`
- `cccl/cub/cub/warp/warp_reduce.cuh`
- `cccl/cub/examples/block/example_block_reduce.cu`
- `cccl/libcudacxx/include/cuda/__atomic/atomic.h`
- `cccl/libcudacxx/include/cuda/__barrier/barrier.h`
- `cccl/libcudacxx/include/cuda/__fwd/pipeline.h`
- `cccl/libcudacxx/include/cuda/__memcpy_async/memcpy_async_barrier.h`
- `cccl/libcudacxx/include/cuda/__launch/launch.h`
- `cccl/libcudacxx/include/cuda/__mdspan/shared_memory_mdspan.h`

Why it matters:

- It gives us battle-tested primitives for reductions, scans, atomics, warp exchange, and staged shared-memory flows.
- `example_block_reduce.cu` is a minimal reference for how to structure a block-wide reduction kernel without inventing custom synchronization logic.
- libcudacxx gives safer access to barriers, pipelines, atomics, and launch abstractions.

What to copy conceptually:

- Use CUB for norm reductions, softmax reductions, and sampler reductions unless a custom warp path is materially better.
- Use libcudacxx primitives when shared-memory staging and async copy are part of the kernel design.

## 4. NCCL

Best use: replacing `torch.distributed` collectives for tensor parallel, sequence/context parallel, and overlap work.

Top files:

- `nccl/src/device/all_reduce.h`
- `nccl/src/device/all_gather.h`
- `nccl/src/device/reduce_scatter.h`
- `nccl/src/device/primitives.h`
- `nccl/src/device/prims_ll128.h`
- `nccl/src/device/common.cu`
- `nccl/src/include/strongstream.h`
- `nccl/docs/examples`

Why it matters:

- `tensor/shard.py` is still wrapping `torch.distributed`.
- A direct CUDA/C++ stack needs NCCL-backed collectives and explicit stream management.
- NCCL is also the right foundation for sequence/context-parallel attention and overlap with compute.

What to copy conceptually:

- Stream-aware collective scheduling.
- Separation between topology/transport selection and device-side primitive execution.
- The fact that collectives are part of the runtime design, not an afterthought.

## 5. cuEmbed

Best use: compact, readable, bandwidth-driven embedding kernels.

Top files:

- `cuEmbed/cuembed/include/embedding_lookup_kernels.cuh`
- `cuEmbed/cuembed/include/embedding_lookup_ops.cuh`
- `cuEmbed/cuembed/include/index_transforms_kernels.cuh`
- `cuEmbed/utils/src/embedding_gpu_forward.cu`
- `cuEmbed/utils/src/embedding_gpu_backward.cu`
- `cuEmbed/examples/pytorch/cuembed_embedding.cu`

Why it matters:

- `model/causal.py`, `model/encoder.py`, and `model/seq2seq.py` now route the hot-path embedding call through `runtime.ops.embedding`; `cuEmbed` remains a useful reference for improving that CUDA backend.
- `embedding_lookup_kernels.cuh` is a good example of a memory-level-parallelism design rather than a compute-heavy kernel.
- It is readable enough to adapt directly for a vocab embedding path.

What to copy conceptually:

- 2D CTA organization where `threadIdx.y` walks samples and `threadIdx.x` walks embedding width.
- Template specialization for weighted vs unweighted cases to lower register pressure.
- Clean separation of index loading, address generation, and combining logic.

## 6. CUTLASS

Best use: GEMM structure, fusion design, Tensor Core tiling, FMHA examples, architecture-specialized linear algebra.

Top files and dirs:

- `cutlass/include/cutlass`
- `cutlass/include/cute`
- `cutlass/examples/12_gemm_bias_relu`
- `cutlass/examples/35_gemm_softmax`
- `cutlass/examples/37_gemm_layernorm_gemm_fusion`
- `cutlass/examples/41_fused_multi_head_attention`
- `cutlass/examples/61_hopper_gemm_with_topk_and_softmax`
- `cutlass/examples/77_blackwell_fmha`
- `cutlass/examples/88_hopper_fmha`
- `cutlass/examples/93_blackwell_low_latency_gqa`

Why it matters:

- CUTLASS is one of the strongest local sources for studying how high-performance GEMM and fused tensor kernels are actually decomposed.
- It matters even if the final implementation starts with cuBLASLt, because it teaches the internal structure of the kernels you would otherwise treat as a black box.
- The examples cover exactly the kinds of fusions relevant to model stacks: bias epilogs, softmax, layernorm fusion, FMHA, and GQA.

What to copy conceptually:

- hierarchical tiling
- explicit epilog design
- architecture-specific policy selection
- use of `cute` layouts and tensor abstractions to control data motion

## 7. Megatron-LM

Best use: system design patterns for transformer parallelism, fused op selection, and production orchestration around TE/NCCL.

Top files:

- `Megatron-LM/megatron/core/tensor_parallel/layers.py`
- `Megatron-LM/megatron/core/tensor_parallel/mappings.py`
- `Megatron-LM/megatron/core/fusions/fused_bias_gelu.py`
- `Megatron-LM/megatron/core/fusions/fused_bias_swiglu.py`
- `Megatron-LM/megatron/core/fusions/fused_softmax.py`
- `Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py`
- `Megatron-LM/docs/user-guide/features/context_parallel.md`

Why it matters:

- It shows the runtime-level consequences of adopting fused kernels and parallel groups.
- The context parallel doc is directly relevant because it states a strong invariant: linears and norms are sequence-local, but attention requires KV exchange. That distinction should drive our distributed runtime design.
- It helps define the integration surface between custom kernels, cuBLASLt, NCCL, and higher-level model code.

What to copy conceptually:

- Keep tensor-parallel semantics explicit in the runtime API.
- Treat fused bias/activation/norm paths as first-class.
- Build context-parallel attention around KV exchange, not around generic tensor sharding.

## 8. CUDA Tile

Best use: future kernel generation and IR-driven optimization, not the first implementation pass.

Top files:

- `cuda-tile/include/cuda_tile/`
- `cuda-tile/lib/`
- `cuda-tile/tools/cuda-tile-optimize/`

Why it matters:

- Useful if we later want generated kernels for common layout variants or tile-specialized attention/MLP paths.
- Not the right place to start if the immediate goal is replacing PyTorch with dependable CUDA/C++ execution.

Recommendation:

- Treat this as phase-3 or phase-4 tooling.

## 9. CUDALibrarySamples

Best use: official library-backed runtime patterns for `cuBLASLt`, `cuBLASDx`, `cuTENSOR`, `cuSPARSELt`, `cuBLASMp`, and matmul heuristics.

Top files:

- `CUDALibrarySamples/cuBLASLt/LtSgemmCustomFind/main.cu`
- `CUDALibrarySamples/cuBLASLt/LtSgemmSimpleAutoTuning`
- `CUDALibrarySamples/cuBLASLt/LtFp8CustomFind/main.cu`
- `CUDALibrarySamples/MathDx/cuBLASDx`
- `CUDALibrarySamples/cuTENSOR/contraction.cu`
- `CUDALibrarySamples/cuTENSOR/contraction_jit.cu`
- `CUDALibrarySamples/cuTENSOR/contraction_plan_cache.cu`
- `CUDALibrarySamples/cuSPARSELt/matmul_advanced`
- `CUDALibrarySamples/cuSPARSE/graph_capture`
- `CUDALibrarySamples/nvMatmulHeuristics`
- `CUDALibrarySamples/cuBLASMp`

Why it matters:

- It is the strongest local reference for how library-backed GEMM/tensor runtime code should actually look.
- It closes the gap between "use cuBLASLt" as advice and actual algorithm-selection, workspace, heuristics, and planning code.
- `nvMatmulHeuristics` is particularly useful when launch and algo policy should be data-driven instead of ad hoc.

## 10. Fuser

Best use: generated fusion paths for pointwise, reduction, norm, RoPE, SDPA, and multidevice flows.

Top files:

- `Fuser/csrc/fusion.cpp`
- `Fuser/csrc/codegen.cpp`
- `Fuser/csrc/device_lower/analysis/tma.cpp`
- `Fuser/runtime/grid_reduction.cu`
- `Fuser/runtime/fused_reduction.cu`
- `Fuser/runtime/welford.cu`
- `Fuser/runtime/tma_copy.cu`
- `Fuser/benchmarks/cpp/rms_norm.cpp`
- `Fuser/benchmarks/cpp/softmax.cpp`
- `Fuser/tests/cpp/test_rope.cpp`
- `Fuser/tests/cpp/test_sdpa.cpp`
- `Fuser/examples/matmul_heuristic_plugin`

Why it matters:

- It shows what a fusion compiler needs around runtime helpers, device-lowering passes, and heuristic selection.
- It is especially relevant for deciding whether residual/activation/norm/reduction paths should stay handwritten or move behind a fusion/codegen layer.
- The multidevice and TMA material makes it more relevant than a simple pointwise-fusion repo.

## 11. TensorRT-LLM

Best use: inference runtime, executor, KV cache, batching, speculative decoding, plugins, and serving orchestration.

Top files:

- `TensorRT-LLM/cpp/tensorrt_llm/runtime/cudaMemPool.cpp`
- `TensorRT-LLM/cpp/tensorrt_llm/runtime/runtimeKernels.cu`
- `TensorRT-LLM/cpp/tensorrt_llm/runtime/ncclCommunicator.cpp`
- `TensorRT-LLM/cpp/tensorrt_llm/executor/executorImpl.cpp`
- `TensorRT-LLM/cpp/tensorrt_llm/executor/kvCacheConfig.cpp`
- `TensorRT-LLM/cpp/tensorrt_llm/executor/speculativeDecodingConfig.cpp`
- `TensorRT-LLM/docs/source/features/attention.md`
- `TensorRT-LLM/docs/source/features/kvcache.md`
- `TensorRT-LLM/docs/source/features/speculative-decoding.md`
- `TensorRT-LLM/examples/cpp/executor`
- `TensorRT-LLM/triton_kernels`

Why it matters:

- It is the best local source for what a high-throughput inference runtime actually has to own once framework execution is removed.
- KV cache, batching, memory pools, request scheduling, and distributed decode control are all first-class here.
- It also shows how Triton kernels and pluginized kernels can coexist inside a larger runtime.

## 12. nvmath-python

Best use: understanding how NVIDIA math libraries expose richer host-side planning APIs and epilog/prolog customization.

Top files:

- `nvmath-python/README.md`
- `nvmath-python/examples/linalg`
- `nvmath-python/examples/device`

Why it matters:

- Even though this is Python-facing, it reinforces an important design point: use library-backed planned execution for GEMM/FFT-like math when possible, and reserve custom kernels for operations that libraries do not handle well.
- The cuBLASLt-style epilog mindset is directly relevant for fused MLP and projection paths.

## 13. Triton

Base path:

- `/data/parametergolf/helpful_repos/statespace_101/triton`

Best use: DSL-authored kernels with JIT specialization and autotuning for matmul, flash attention, cross-entropy, and block-sparse operations.

Top files:

- `triton/ops/flash_attention.py`
- `triton/ops/matmul.py`
- `triton/ops/cross_entropy.py`
- `triton/ops/blocksparse/matmul.py`
- `triton/ops/blocksparse/softmax.py`
- `triton/runtime/jit.py`
- `triton/runtime/autotuner.py`
- `triton/compiler/compiler.py`
- `triton/compiler/code_generator.py`
- `triton/testing.py`

Why it matters:

- It is the cleanest adjacent reference for a non-CUDA-C++ authoring path that still targets GPU kernels directly.
- It is useful both as an alternative implementation route and as a benchmark for how much boilerplate direct CUDA/C++ is buying or costing us.

## Repo-to-Local-Code Mapping

| Local area | Replace with | Main references |
| --- | --- | --- |
| `tensor/norms.py` | custom RMSNorm/LayerNorm kernels | `TransformerEngine/common/normalization/*`, `cccl/cub/block/block_reduce.cuh` |
| `tensor/positional.py` | fused RoPE kernels | `TransformerEngine/common/fused_rope/fused_rope.cu` |
| `attn/eager.py` SDPA path | fused attention path + custom KV append | `TransformerEngine/common/fused_attn/*`, `Megatron-LM/.../fused_kv_append_kernel.py`, `cutlass/examples/41_fused_multi_head_attention` |
| QKV/O/MLP linears | cuBLASLt GEMMs and epilog fusion | `TransformerEngine/common/gemm/cublaslt_gemm.cu`, `CUDALibrarySamples/cuBLASLt/*`, `CUDALibrarySamples/nvMatmulHeuristics/*`, `cutlass/examples/12_gemm_bias_relu`, `cutlass/examples/37_gemm_layernorm_gemm_fusion`, `nvmath-python/README.md` |
| `model/causal.py` embedding | custom embedding lookup | `cuEmbed/cuembed/include/embedding_lookup_kernels.cuh` |
| `tensor/shard.py` | NCCL-backed collectives | `nccl/src/device/*`, `Megatron-LM/megatron/core/tensor_parallel/*` |
| decode loop / serving runtime | streams, async alloc, graphs, KV-cache ownership | `cuda-samples/.../asyncAPI.cu`, `streamOrderedAllocation.cu`, `cudaGraphPerfScaling.cu`, `TensorRT-LLM/cpp/tensorrt_llm/runtime/*`, `TensorRT-LLM/docs/source/features/kvcache.md` |
| pointwise/reduction fusion | generated fusion path or DSL alternative | `Fuser/runtime/*`, `Fuser/tests/cpp/test_sdpa.cpp`, `/data/parametergolf/helpful_repos/statespace_101/triton/ops/*` |

## Bottom Line

If we only keep one mental model from this survey, it should be this:

- `TransformerEngine` shows which transformer ops deserve custom CUDA kernels.
- `CUTLASS` shows how fused GEMM-heavy kernels are built when cuBLASLt alone is not the full answer.
- `CUDALibrarySamples` shows how NVIDIA math libraries should actually be driven from real code.
- `cuBLASLt` should carry GEMMs.
- `NCCL` should carry collectives.
- `CCCL` should carry block/warp mechanics.
- `cuda-samples` should carry API usage patterns.
- `TensorRT-LLM` should inform inference runtime and KV-cache design.
- `Fuser` and `triton` should be treated as serious alternatives for fusion-heavy kernels, not ignored as side projects.
- `cuEmbed` is the compact example to imitate for bandwidth-bound embedding work.
