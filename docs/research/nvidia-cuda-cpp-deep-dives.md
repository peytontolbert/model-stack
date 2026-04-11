# NVIDIA CUDA/C++ Deep Dives

This document goes deeper than the full inventory. It focuses on the repositories that matter most when studying CUDA kernels, C++ GPU runtimes, and buildable low-level systems.

Second-pass note:

- the first version emphasized kernel bodies
- this revision also pulls in the runtime, debug, benchmark, profiling, and validation surfaces needed for a full migration

## 1. CUTLASS

Why it matters:

- CUTLASS is one of the most important repos in the entire corpus.
- It is the best local source for understanding modern GEMM decomposition, epilog fusion, and architecture-specialized tensor compute.

What to read first:

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
- `cutlass/examples/65_distributed_gemm`

What it teaches:

- how to structure GEMM kernels hierarchically
- how to fuse epilogs without losing control of tiling
- how to think about FMHA and GQA as tiled data-movement problems, not just formulas
- how CuTe/CUTLASS abstractions encode layouts, tiles, and hardware atoms

Second-pass additions:

- `cutlass/tools/profiler/src`
- `cutlass/tools/library/src`
- `cutlass/tools/util/include/cutlass/util/reference`
- `cutlass/test/unit/pipeline`
- `cutlass/test/unit/cluster_launch`
- `cutlass/test/unit/cute/ampere`
- `cutlass/test/unit/cute/hopper`
- `cutlass/test/unit/nvrtc`

These paths matter because CUTLASS is not only an example repo:

- `tools/profiler` shows how to benchmark operation families systematically
- `tools/library` shows operation registration and selection machinery
- `test/unit/pipeline` and `test/unit/cute/hopper` are strong local references for async pipelines, TMA, and cluster-launch behavior
- `test/unit/nvrtc` is relevant if runtime specialization or JIT compilation becomes part of the stack

When to use it:

- when cuBLASLt is not enough
- when you need custom fused GEMM-heavy kernels
- when you need a real source-level education on Tensor Core kernel structure

## 2. TransformerEngine

Why it matters:

- This is the most directly relevant production transformer-kernel repo in the corpus.

What to read first:

- `TransformerEngine/transformer_engine/common/fused_attn`
- `TransformerEngine/transformer_engine/common/fused_softmax`
- `TransformerEngine/transformer_engine/common/fused_rope/fused_rope.cu`
- `TransformerEngine/transformer_engine/common/normalization/rmsnorm`
- `TransformerEngine/transformer_engine/common/normalization/layernorm`
- `TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu`
- `TransformerEngine/transformer_engine/common/transpose`
- `TransformerEngine/transformer_engine/common/comm_gemm_overlap`

Key takeaways:

- specialize common transformer shapes aggressively
- keep launch logic separate from kernel bodies
- use cuBLASLt as the GEMM backbone
- own rope, norm, cast, transpose, cache, and fused attention logic where libraries do not cover the problem cleanly

Specific patterns worth noticing:

- `rmsnorm_fwd_cuda_kernel.cu` uses tuned and general launchers rather than a single generic kernel
- `fused_rope.cu` stages sin/cos in shared memory and handles multiple sequence/layout modes
- `cublaslt_gemm.cu` does the hard layout canonicalization work so higher-level code does not have to

Second-pass additions:

- `TransformerEngine/docs/debug.rst`
- `TransformerEngine/docs/envvars.rst`
- `TransformerEngine/docs/installation.rst`
- `TransformerEngine/benchmarks/attention`
- `TransformerEngine/benchmarks/linear`
- `TransformerEngine/tests/cpp/operator`
- `TransformerEngine/tests/cpp_distributed`
- `TransformerEngine/transformer_engine/common/util/rtc.*`
- `TransformerEngine/transformer_engine/common/util/multi_stream.*`
- `TransformerEngine/transformer_engine/common/include/transformer_engine/*`

These paths matter because a direct migration needs more than kernels:

- TE documents build/runtime controls, backend selection, and debug hooks explicitly
- TE benchmarks and cpp tests are strong references for isolated kernel validation
- `rtc.*`, `multi_stream.*`, and the public C headers are useful models for the runtime boundary we will need

## 3. CCCL

Why it matters:

- This is the base toolkit for writing better kernels with less reinvention.

What to read first:

- `cccl/cub/cub/block/block_reduce.cuh`
- `cccl/cub/cub/block/block_load.cuh`
- `cccl/cub/cub/block/block_store.cuh`
- `cccl/cub/cub/warp/warp_reduce.cuh`
- `cccl/cub/examples/block/example_block_reduce.cu`
- `cccl/libcudacxx/include/cuda/__atomic/atomic.h`
- `cccl/libcudacxx/include/cuda/__barrier/barrier.h`
- `cccl/libcudacxx/include/cuda/__memcpy_async`
- `cccl/libcudacxx/include/cuda/__launch`
- `cccl/libcudacxx/include/cuda/__mdspan`

What it teaches:

- block- and warp-level reduction design
- memory movement helpers
- safer atomics and synchronization primitives
- how modern CUDA C++ is supposed to look

Second-pass additions:

- `cccl/docs/libcudacxx/runtime/memory_pools.rst`
- `cccl/docs/libcudacxx/runtime/launch.rst`
- `cccl/docs/libcudacxx/extended_api/asynchronous_operations.rst`
- `cccl/docs/libcudacxx/extended_api/synchronization_primitives/pipeline.rst`
- `cccl/docs/libcudacxx/extended_api/synchronization_primitives/barrier.rst`
- `cccl/docs/libcudacxx/extended_api/mdspan.rst`
- `cccl/docs/libcudacxx/ptx/instructions/*`
- `cccl/cudax/include/cuda/experimental/graph.cuh`
- `cccl/cudax/include/cuda/experimental/execution.cuh`
- `cccl/cudax/include/cuda/experimental/__execution/stream_context.cuh`
- `cccl/nvrtcc/README.md`

These are important because CCCL also covers:

- stream-ordered memory resources
- runtime launch wrappers
- graph and execution abstractions
- mdspan-based layout control
- PTX-level async-copy and barrier wrappers
- a smaller NVRTC-oriented toolchain surface

Most useful mental model:

- do not write custom warp and block plumbing unless profiling proves you need to

## 4. cuda-samples

Why it matters:

- Official small references for correct CUDA API usage.

What to read first:

- `cuda-samples/Samples/0_Introduction/asyncAPI/asyncAPI.cu`
- `cuda-samples/Samples/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu`
- `cuda-samples/Samples/6_Performance/cudaGraphsPerfScaling/cudaGraphPerfScaling.cu`
- `cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu`
- `cuda-samples/Samples/0_Introduction/matrixMul_nvrtc`
- `cuda-samples/Samples/3_CUDA_Features/ptxjit`
- `cuda-samples/Samples/0_Introduction/simpleP2P`

What it teaches:

- events and stream timing
- overlapping host/device work
- async allocators and memory pools
- graph capture and replay
- Tensor Core shared-memory tiling basics
- runtime compilation and PTX JIT

Second-pass additions:

- `cuda-samples/Samples/0_Introduction/simpleOccupancy/simpleOccupancy.cu`
- `cuda-samples/Samples/0_Introduction/clock_nvrtc/*`
- `cuda-samples/Samples/2_Concepts_and_Techniques/inlinePTX_nvrtc/*`
- `cuda-samples/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC/*`
- `cuda-samples/Samples/2_Concepts_and_Techniques/streamOrderedAllocationP2P/*`
- `cuda-samples/Samples/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`
- `cuda-samples/Samples/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu`
- `cuda-samples/Samples/3_CUDA_Features/jacobiCudaGraphs/*`
- `cuda-samples/Samples/3_CUDA_Features/graphConditionalNodes/*`
- `cuda-samples/Samples/3_CUDA_Features/graphMemoryNodes/*`
- `cuda-samples/Samples/3_CUDA_Features/graphMemoryFootprint/*`
- `cuda-samples/Samples/3_CUDA_Features/memMapIPCDrv/*`
- `cuda-samples/Samples/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu`

These are the right places to study:

- occupancy-guided launch selection
- async-copy into shared memory outside large production kernels
- graph memory-node and conditional-node behavior
- IPC and P2P allocation patterns
- NVRTC and inline PTX integration in small controlled examples

Use case:

- whenever an API question comes up, check here before reading large production repos

## 5. NCCL

Why it matters:

- Required if we want direct multi-GPU runtime control.

What to read first:

- `nccl/src/device/all_reduce.h`
- `nccl/src/device/all_gather.h`
- `nccl/src/device/reduce_scatter.h`
- `nccl/src/device/primitives.h`
- `nccl/src/device/prims_ll128.h`
- `nccl/src/device/common.cu`
- `nccl/src/include/strongstream.h`
- `nccl/src/transport`

What it teaches:

- collective device primitive structure
- transport-aware communication
- stream management and overlap concerns

Second-pass additions:

- `nccl/docs/examples/04_user_buffer_registration`
- `nccl/docs/examples/05_symmetric_memory`
- `nccl/docs/examples/06_device_api`
- `nccl/src/allocator.cc`
- `nccl/src/graph`
- `nccl/src/include/nvtx.h`
- `nccl/src/include/plugin/profiler`

These matter because a direct runtime eventually needs:

- user-managed communication buffers
- symmetric-memory exchange patterns
- device-side collective launch from inside kernels
- topology/tuning awareness
- explicit profiling and attribution hooks once framework traces disappear

What not to expect:

- this is not a beginner kernel tutorial repo
- it is a systems-performance repo

## 6. cuEmbed

Why it matters:

- A narrow, readable example of a real bandwidth-optimized kernel family.

What to read first:

- `cuEmbed/cuembed/include/embedding_lookup_kernels.cuh`
- `cuEmbed/cuembed/include/embedding_lookup_ops.cuh`
- `cuEmbed/cuembed/include/index_transforms_kernels.cuh`
- `cuEmbed/utils/src/embedding_gpu_forward.cu`
- `cuEmbed/utils/src/embedding_gpu_backward.cu`

What it teaches:

- 2D CTA organization for embedding width and sample batching
- compile-time specialization to reduce register waste
- separation of index loading, address generation, and combining

Why it is especially good:

- far smaller and easier to digest than a full framework

Second-pass additions:

- `cuEmbed/benchmarks/manual_benchmark.cu`
- `cuEmbed/tests/*`

These are worth reading because they show how to benchmark and correctness-check a compact kernel library without hiding behind a framework test stack.

## 7. CUDA Tile

Why it matters:

- Strong reference for IR-first and compiler-lowered CUDA workflows.

What to read first:

- `cuda-tile/include`
- `cuda-tile/lib`
- `cuda-tile/tools/cuda-tile-optimize`
- `cuda-tile/tools/cuda-tile-translate`
- `cuda-tile/test/Transforms`
- `cuda-tile/test/Bytecode`

What it teaches:

- how tile IR is represented
- how to build optimization passes and bytecode tooling
- how kernel generation can become part of the toolchain

When to use it:

- after direct handwritten/library-backed paths are understood
- when the team wants codegen or autotuned lowering

## 8. Tilus

Why it matters:

- A practical DSL-level view of GPU kernel design with examples that map closely to model-stack operations.

What to read first:

- `tilus/examples/attention/flash_attention_v1.py`
- `tilus/examples/attention/flash_attention_v2.py`
- `tilus/examples/attention_with_kvcache/attention_v1.py`
- `tilus/examples/flash_attention_decode/tilus_kernel.py`
- `tilus/examples/norm/layer_norm.py`
- `tilus/examples/softmax/softmax.py`
- `tilus/tests/instructions`

What it teaches:

- tile-level expression of attention, decode, norm, and softmax kernels
- async copy, barriers, and lower-level instruction testing in a more accessible form than raw C++

Second-pass additions:

- `tilus/python/tilus/ir/layout/inference/inference_rules/cp_async.py`
- `tilus/python/tilus/ir/layout/inference/validation_rules/*`

These matter because they encode legality and layout reasoning around async copy and tile transforms, which is easy to get wrong in handwritten CUDA.

Use case:

- useful for understanding alternative kernel-authoring workflows
- useful if the team wants a DSL rather than only C++

## 9. Megatron-LM

Why it matters:

- Important systems reference, even though the local raw CUDA source footprint is small.

What to read first:

- `Megatron-LM/megatron/core/tensor_parallel/layers.py`
- `Megatron-LM/megatron/core/tensor_parallel/mappings.py`
- `Megatron-LM/megatron/core/fusions`
- `Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py`
- `Megatron-LM/docs/user-guide/features/context_parallel.md`

What it teaches:

- how fused ops are chosen and wired into a large training system
- how TP, SP, and CP shape runtime design
- where the line is between compute kernels and distributed orchestration

Important caveat:

- read it for systems semantics and integration, not as the main raw-kernel source

## 10. nvmath-python

Why it matters:

- Not a kernel repo, but a very useful lens on NVIDIA math-library planning and epilog control.

What to read first:

- `nvmath-python/README.md`
- `nvmath-python/examples/linalg`
- `nvmath-python/examples/device`
- `nvmath-python/examples/tensor`

What it teaches:

- how to expose richer math-library controls than frameworks typically provide
- why planned execution and epilog control matter
- how device-side library APIs can be invoked from custom kernels

Second-pass additions:

- `nvmath-python/docs/sphinx/bindings/cublasLt.rst`
- `nvmath-python/docs/sphinx/distributed-apis/runtime.rst`
- `nvmath-python/examples/linalg/advanced/matmul`
- `nvmath-python/examples/tensor/contraction/example09_streams.py`
- `nvmath-python/examples/tensor/contraction/example11_memory_allocator.py`
- `nvmath-python/examples/tensor/contraction/example12_resource_mgmt.py`
- `nvmath-python/examples/distributed/linalg/advanced/matmul/example08_epilog_allreduce.py`
- `nvmath-python/tests/nvmath_tests/linalg/advanced/matmul`

These are useful because they expose:

- epilog and planning surfaces
- allocator/resource injection
- stream-aware execution
- distributed math-runtime patterns that PyTorch would otherwise hide

## 11. cuOpt

Why it matters:

- Domain-specific, but still a serious CUDA/C++ product codebase with graphs, utilities, helpers, and tests.

What to read first:

- `cuopt/cpp/src/routing`
- `cuopt/cpp/src/pdlp`
- `cuopt/cpp/src/utilities`
- `cuopt/cpp/src/routing/cuda_graph.cuh`
- `cuopt/cpp/tests`

What it teaches:

- how a large GPU-native C++ product organizes kernels, host code, utilities, and tests
- how CUDA Graphs can be treated as part of runtime design

## 12. CUDALibrarySamples

Why it matters:

- This repo closes a major practical gap: how NVIDIA expects its math and tensor libraries to actually be used from real code.

What to read first:

- `CUDALibrarySamples/cuBLASLt/README.md`
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

What it teaches:

- canonical `cuBLASLt` algorithm-selection, heuristics, workspace, and autotuning usage
- device-side math-library extension patterns through `MathDx`
- planned tensor contractions and plan caching
- sparse and distributed library usage
- matmul heuristic discovery as a runtime policy surface rather than hardcoded launch folklore

Why it matters for migration:

- a direct runtime should lean on library-backed execution where possible
- this repo is the cleanest local source for how to do that without framework indirection

## 13. Fuser

Why it matters:

- nvFuser is one of the strongest local sources for fusion compiler design targeting NVIDIA GPUs.

What to read first:

- `Fuser/csrc/fusion.cpp`
- `Fuser/csrc/codegen.cpp`
- `Fuser/csrc/device_lower/analysis/tma.cpp`
- `Fuser/csrc/host_ir/assign_streams.cpp`
- `Fuser/runtime/tma_copy.cu`
- `Fuser/runtime/grid_reduction.cu`
- `Fuser/runtime/fused_reduction.cu`
- `Fuser/runtime/welford.cu`
- `Fuser/benchmarks/cpp/rms_norm.cpp`
- `Fuser/benchmarks/cpp/softmax.cpp`
- `Fuser/tests/cpp/test_rope.cpp`
- `Fuser/tests/cpp/test_sdpa.cpp`
- `Fuser/examples/matmul_heuristic_plugin`
- `Fuser/doc/README.md`

What it teaches:

- how a fusion compiler lowers high-level tensor graphs into CUDA runtime helpers and generated kernels
- how TMA, reductions, Welford, and multidevice flows are represented in a compiler pipeline
- how fusion-specific benchmarking, heuristic caches, and validation are organized

When it matters:

- when deciding whether pointwise/reduction-heavy paths should be handwritten or compiler-generated
- when evaluating whether a direct CUDA/C++ runtime should still preserve a fusion/codegen tier

## 14. TensorRT-LLM

Why it matters:

- This is the strongest local runtime/serving reference in the corpus.

What to read first:

- `TensorRT-LLM/cpp/tensorrt_llm/runtime`
- `TensorRT-LLM/cpp/tensorrt_llm/executor`
- `TensorRT-LLM/cpp/tensorrt_llm/plugins`
- `TensorRT-LLM/cpp/tensorrt_llm/runtime/cudaMemPool.*`
- `TensorRT-LLM/cpp/tensorrt_llm/runtime/runtimeKernels.cu`
- `TensorRT-LLM/cpp/tensorrt_llm/runtime/ncclCommunicator.*`
- `TensorRT-LLM/docs/source/features/attention.md`
- `TensorRT-LLM/docs/source/features/kvcache.md`
- `TensorRT-LLM/docs/source/features/speculative-decoding.md`
- `TensorRT-LLM/examples/cpp/executor`
- `TensorRT-LLM/triton_kernels`

What it teaches:

- allocator, KV-cache, executor, scheduler, and request/response runtime design for high-throughput inference
- how pluginized kernels fit into a serving runtime
- how batching, decode scheduling, speculative decoding, and distributed execution interact
- how Triton-authored kernels can coexist with a C++ runtime and plugin stack

Important caveat:

- this is more of a serving/runtime reference than a first-principles kernel-tutorial repo
- for operator authoring, it complements rather than replaces `TransformerEngine`, `cutlass`, `cccl`, and `cuda-samples`

## 15. Triton

Base path:

- `/data/parametergolf/helpful_repos/statespace_101/triton`

Why it matters:

- This local checkout is not part of the NVIDIA corpus, but it materially changes the implementation trade space.

What to read first:

- `triton/ops/flash_attention.py`
- `triton/ops/matmul.py`
- `triton/ops/cross_entropy.py`
- `triton/ops/blocksparse/matmul.py`
- `triton/ops/blocksparse/softmax.py`
- `triton/runtime/jit.py`
- `triton/runtime/autotuner.py`
- `triton/runtime/cache.py`
- `triton/compiler/compiler.py`
- `triton/compiler/code_generator.py`
- `triton/language/core.py`
- `triton/language/semantic.py`
- `triton/testing.py`

What it teaches:

- how a kernel DSL exposes JIT specialization, autotuning, caching, and concise kernel definitions
- how matmul, flash attention, cross-entropy, and block-sparse kernels can be expressed above raw CUDA
- how benchmarking and CUDA-graph-aware testing can be packaged into the kernel authoring workflow

Important caveat:

- this checkout appears to be a package-style subset rather than a full top-level Triton repo clone
- it is still useful for kernel and runtime ideas, but it is not a substitute for the NVIDIA corpus itself

## 16. cuda-quantum / cudaqx / open-gpu-kernel-modules

These are worth knowing about, but not as first-line sources for model-stack CUDA compute work.

- `cuda-quantum`
  - strong C++ runtime/compiler code, but in the quantum domain
- `cudaqx`
  - extension libraries on top of CUDA-Q
- `open-gpu-kernel-modules`
  - Linux driver/kernel source, not CUDA compute-kernel source

## Suggested Study Sequence

If the team wants broad CUDA/C++ coverage before implementation, the best sequence is:

1. `cuda-samples`
2. `cccl`
3. `CUDALibrarySamples`
4. `cutlass`
5. `TransformerEngine`
6. `nccl`
7. `TensorRT-LLM`
8. `Fuser`
9. `cuEmbed`
10. `Megatron-LM`
11. `cuda-tile`
12. `tilus`
13. `nvmath-python`
14. `cuopt`
15. `triton`

This order moves from API fundamentals to reusable primitives to transformer production code to distributed and compiler-driven workflows.
