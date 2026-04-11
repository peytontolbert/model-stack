# NVIDIA Full-Migration Gap Audit

This is the second-pass answer to the question:

- is the current NVIDIA corpus research package comprehensive enough to fully migrate `transformer_10` off PyTorch?

## Short Answer

No, not fully.

The corpus itself is strong enough to support a real migration, but the first-pass docs were stronger on kernel exemplars than on the runtime, validation, debugging, and toolchain surfaces that make a full migration survivable.

More precisely:

- the current docs were already good enough to start operator-by-operator replacement work
- they were not yet comprehensive enough to claim full-stack migration coverage with confidence

After this second pass, the missing areas are documented more explicitly, but the remaining work is still practical engineering, not just more repo browsing.

## What The First Pass Underweighted

The first pass centered correctly on:

- `TransformerEngine`
- `cutlass`
- `cccl`
- `cuda-samples`
- `nccl`
- `cuEmbed`

But it underweighted five categories that matter for a full migration:

1. Runtime control surfaces
2. Build and JIT toolchain details
3. Debugging and diagnostics
4. Benchmark and validation harnesses
5. Communication-inside-kernel and graph-capture edge cases

That distinction matters. A stack can have enough kernel examples to write code, while still lacking enough operational knowledge to replace PyTorch safely.

## Second-Pass Missed Sources

### 1. TransformerEngine is also a runtime/debug reference, not just a kernel repo

Missed or underemphasized paths:

- `TransformerEngine/docs/debug.rst`
- `TransformerEngine/docs/debug/1_getting_started.rst`
- `TransformerEngine/docs/debug/4_distributed.rst`
- `TransformerEngine/docs/envvars.rst`
- `TransformerEngine/docs/installation.rst`
- `TransformerEngine/benchmarks/attention/benchmark_attention.py`
- `TransformerEngine/benchmarks/linear/benchmark_linear.py`
- `TransformerEngine/tests/cpp/operator/*`
- `TransformerEngine/tests/cpp_distributed/test_comm_gemm.cu`
- `TransformerEngine/transformer_engine/common/util/rtc.h`
- `TransformerEngine/transformer_engine/common/util/rtc.cpp`
- `TransformerEngine/transformer_engine/common/util/multi_stream.h`
- `TransformerEngine/transformer_engine/common/util/multi_stream.cpp`
- `TransformerEngine/transformer_engine/common/include/transformer_engine/*.h`

Why this matters:

- `docs/envvars.rst` documents build/runtime control that affects determinism, backend selection, architecture targeting, and workspace behavior.
- `docs/debug*` shows how a production transformer kernel stack exposes precision debugging and distributed debugging hooks.
- `benchmarks/*` and `tests/cpp/operator/*` show how NVIDIA validates kernels in isolation instead of only through framework integration.
- `common/util/rtc.*` and `multi_stream.*` matter because a direct runtime replacement needs explicit JIT, stream, and handle management.

Migration implication:

- full migration coverage requires documenting TE's runtime control and test structure, not only its `.cu` files

### 2. CCCL is also the missing runtime substrate

Missed or underemphasized paths:

- `cccl/docs/libcudacxx/runtime/memory_pools.rst`
- `cccl/docs/libcudacxx/runtime/launch.rst`
- `cccl/docs/libcudacxx/extended_api/asynchronous_operations.rst`
- `cccl/docs/libcudacxx/extended_api/asynchronous_operations/memcpy_async.rst`
- `cccl/docs/libcudacxx/extended_api/synchronization_primitives/pipeline.rst`
- `cccl/docs/libcudacxx/extended_api/synchronization_primitives/barrier.rst`
- `cccl/docs/libcudacxx/extended_api/mdspan.rst`
- `cccl/docs/libcudacxx/ptx/instructions/cp_async_bulk_tensor.rst`
- `cccl/docs/libcudacxx/ptx/instructions/mbarrier_arrive.rst`
- `cccl/docs/libcudacxx/ptx/instructions/barrier_cluster.rst`
- `cccl/docs/libcudacxx/ptx/instructions/multimem_ld_reduce.rst`
- `cccl/cudax/include/cuda/experimental/graph.cuh`
- `cccl/cudax/include/cuda/experimental/execution.cuh`
- `cccl/cudax/include/cuda/experimental/__execution/stream_context.cuh`
- `cccl/cudax/include/cuda/experimental/memory_resource.cuh`
- `cccl/nvrtcc/README.md`
- `cccl/nvrtcc/examples/demo.cu`

Why this matters:

- the first pass treated CCCL mostly as CUB plus barriers
- the second pass shows it also covers stream-ordered allocation, launch wrappers, mdspan layouts, execution abstractions, graph helpers, and PTX-level async-copy primitives

Migration implication:

- if we want a direct C++ runtime instead of "PyTorch with custom kernels", CCCL is part of the runtime design, not only the kernel utility layer

### 3. cuda-samples has more of the runtime edge cases than the first pass captured

Missed or underemphasized samples:

- `cuda-samples/Samples/0_Introduction/simpleOccupancy/simpleOccupancy.cu`
- `cuda-samples/Samples/0_Introduction/clock_nvrtc/*`
- `cuda-samples/Samples/2_Concepts_and_Techniques/inlinePTX_nvrtc/*`
- `cuda-samples/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC/*`
- `cuda-samples/Samples/2_Concepts_and_Techniques/streamOrderedAllocationP2P/*`
- `cuda-samples/Samples/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`
- `cuda-samples/Samples/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu`
- `cuda-samples/Samples/3_CUDA_Features/jacobiCudaGraphs/*`
- `cuda-samples/Samples/3_CUDA_Features/graphConditionalNodes/graphConditionalNodes.cu`
- `cuda-samples/Samples/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu`
- `cuda-samples/Samples/3_CUDA_Features/graphMemoryFootprint/graphMemoryFootprint.cu`
- `cuda-samples/Samples/3_CUDA_Features/memMapIPCDrv/*`
- `cuda-samples/Samples/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu`

Why this matters:

- occupancy estimation matters when replacing framework autotuning with direct launch choices
- IPC/P2P allocator examples matter for multi-process serving and shared KV/cache ownership
- graph memory-node and conditional-node examples matter once decode loops move under graph capture
- `globalToShmemAsyncCopy` is a compact reference for async shared-memory staging beyond basic GEMM examples
- NVRTC samples matter if we want runtime specialization instead of shipping every variant ahead of time

Migration implication:

- `cuda-samples` is not only a "basic API examples" repo; it is also the cleanest local source for awkward runtime features that a handwritten inference runtime eventually hits

### 4. NCCL coverage also needs device API, symmetric memory, and diagnostics

Missed or underemphasized paths:

- `nccl/docs/examples/04_user_buffer_registration/*`
- `nccl/docs/examples/05_symmetric_memory/*`
- `nccl/docs/examples/06_device_api/*`
- `nccl/src/allocator.cc`
- `nccl/src/graph/*`
- `nccl/src/include/nvtx.h`
- `nccl/src/include/plugin/profiler/*`

Why this matters:

- `docs/examples/06_device_api/README.md` is directly about communication inside user kernels
- user-buffer registration and symmetric memory are not academic side features; they are the kinds of mechanisms that make overlap and custom fused comm+compute practical
- `src/graph/*` matters for topology, tuning, and communication plan selection
- NVTX/profiler headers matter because once PyTorch is removed, profiling and attribution need to be owned directly

Migration implication:

- a direct distributed runtime needs more than `ncclAllReduce` wrappers
- it needs explicit stream, window, topology, and observability design

### 5. CUTLASS is also a profiling, reference, and pipeline repo

Missed or underemphasized paths:

- `cutlass/tools/profiler/src/*`
- `cutlass/tools/library/src/*`
- `cutlass/tools/util/include/cutlass/util/reference/*`
- `cutlass/test/unit/pipeline/*`
- `cutlass/test/unit/cluster_launch/*`
- `cutlass/test/unit/nvrtc/*`
- `cutlass/test/unit/cute/ampere/*`
- `cutlass/test/unit/cute/hopper/*`

Why this matters:

- the first pass captured the right examples, but not the harnesses that explain how to validate and benchmark them
- `tools/profiler` is one of the best local references for operation benchmarking and problem-space enumeration
- `tools/library` shows how kernels are registered and surfaced as selectable operations
- `test/unit/pipeline/*` and `test/unit/cute/hopper/*` are stronger references for TMA, async pipeline, and cluster-launch mechanics than many high-level examples
- `test/unit/nvrtc/*` matters if runtime kernel generation becomes part of the design

Migration implication:

- CUTLASS should inform not only kernel structure, but also benchmarking, operation registration, reference checking, and architecture-specific validation

### 6. nvmath-python is more relevant to planning and epilog control than the first pass stated

Missed or underemphasized paths:

- `nvmath-python/docs/sphinx/bindings/cublasLt.rst`
- `nvmath-python/docs/sphinx/distributed-apis/runtime.rst`
- `nvmath-python/examples/linalg/advanced/matmul/*`
- `nvmath-python/examples/tensor/contraction/example09_streams.py`
- `nvmath-python/examples/tensor/contraction/example11_memory_allocator.py`
- `nvmath-python/examples/tensor/contraction/example12_resource_mgmt.py`
- `nvmath-python/examples/distributed/linalg/advanced/matmul/example08_epilog_allreduce.py`
- `nvmath-python/examples/distributed/reshape/example07_sync_symmetric_memory.py`
- `nvmath-python/tests/nvmath_tests/linalg/advanced/matmul/test_epilog.py`
- `nvmath-python/tests/nvmath_tests/linalg/advanced/matmul/test_planning.py`
- `nvmath-python/tests/nvmath_tests/distributed/test_matmul.py`

Why this matters:

- this repo is not a handwritten kernel source, but it documents how NVIDIA exposes planning, epilogs, stream control, allocator injection, and distributed math-runtime choices
- those are exactly the policy surfaces a direct runtime must replace once PyTorch orchestration is removed

Migration implication:

- it is a design reference for the host/runtime API layer, not just a math-library curiosity

### 7. cuEmbed, cuOpt, and Tilus add validation and runtime habits that should not be skipped

Missed or underemphasized paths:

- `cuEmbed/benchmarks/manual_benchmark.cu`
- `cuEmbed/tests/*`
- `cuopt/ci/test_cpp.sh`
- `cuopt/ci/test_cpp_memcheck.sh`
- `tilus/tests/instructions/*`
- `tilus/python/tilus/ir/layout/inference/inference_rules/cp_async.py`
- `tilus/python/tilus/ir/layout/inference/validation_rules/*`

Why this matters:

- cuEmbed contributes a compact benchmark/test story for a bandwidth-bound op family
- cuOpt is valuable less for LLM kernels than for showing how a CUDA/C++ product repo treats CI, memcheck, and cpp-only validation
- Tilus adds a useful validation vocabulary around async copy and layout legality

Migration implication:

- the migration needs a stronger validation culture than "compare one output tensor and call it good"

## What Is Comprehensive Enough Now

After this second pass, the research package is comprehensive enough for:

- kernel-family discovery
- source-map coverage across the NVIDIA corpus
- identifying the right repos for norms, rope, softmax, attention, GEMM fusion, embeddings, collectives, async allocators, graphs, and runtime planning
- defining the runtime surfaces that PyTorch currently hides

That means:

- yes, there is enough corpus knowledge to begin a serious direct CUDA/C++ migration

## What Is Still Not Automatically Solved

Even with better doc coverage, a full migration still requires repo-local engineering decisions that the NVIDIA corpus cannot make for us:

- exact runtime ABI for `transformer_10`
- tensor-layout conventions across the whole stack
- acceptable numerical drift thresholds by op and dtype
- graph-capture boundaries for prefill versus decode
- allocator ownership rules for KV cache, temps, and distributed buffers
- distributed invariants for TP/CP in this repo specifically
- fallback policy across GPU architectures we plan to support

Those are implementation decisions, not missing corpus coverage.

## Practical Verdict

If the question is:

- "Can we fully migrate today just because we have looked at the NVIDIA repos once?"

the answer is:

- no

If the question is:

- "Do we now have enough documented NVIDIA-source coverage to begin the migration without flying blind?"

the answer is:

- yes

The real blocker was not missing CUDA kernel examples anymore. It was underdocumented runtime, validation, and observability surfaces. That is what this second pass closes.

## Third-Pass Additions

After the next sweep, four more repos changed the picture materially:

- `CUDALibrarySamples`
- `Fuser`
- `TensorRT-LLM`
- adjacent local `triton` at `/data/parametergolf/helpful_repos/statespace_101/triton`

### `CUDALibrarySamples` closes the library-usage gap

The earlier docs already said "use cuBLASLt" and "use library-backed execution where possible."

What was missing was:

- actual `cuBLASLt` autotuning and custom-find examples
- `cuTENSOR` contraction planning and plan-cache examples
- `cuSPARSELt` and `cuBLASMp` examples
- `nvMatmulHeuristics` examples for runtime algo/config policy

What this changes:

- the docs now cover not just low-level kernels, but also the most practical library-backed replacement route for GEMM/tensor-heavy paths

### `Fuser` closes the fusion-compiler gap

The earlier docs covered handwritten CUDA/C++, library-backed execution, and DSL/compiler-style repos like `cuda-tile` and `tilus`.

What was missing was a strong fusion compiler that directly targets NVIDIA GPUs and includes:

- a real lowering/codegen pipeline
- CUDA runtime helpers
- pointwise/reduction/norm/SDPA benchmarks
- TMA and multidevice material

What this changes:

- the docs now cover the serious alternative where some hot fused paths are compiler-generated instead of hand-maintained CUDA kernels

### `TensorRT-LLM` closes the inference-runtime gap

The earlier docs had production kernels and collective/runtime pieces, but not enough on:

- request scheduling
- KV-cache ownership and lifecycle
- decode-time batching
- speculative decoding runtime integration
- serving-oriented allocator and executor structure

What this changes:

- the docs now cover a strong end-to-end reference for what a non-framework inference runtime actually has to own

### `triton` closes the adjacent DSL/JIT comparison gap

This checkout is not in the NVIDIA corpus proper, but it still matters because it gives local reference code for:

- JIT specialization
- autotuning
- cached kernel builds
- Triton-authored matmul, flash-attention, block-sparse, and cross-entropy kernels

What this changes:

- the migration docs can now compare three serious implementation paths instead of only one:
  - handwritten CUDA/C++
  - library-backed runtime plus selective custom kernels
  - DSL/compiler-assisted kernels for some ops

## Revised Verdict

With `CUDALibrarySamples`, `Fuser`, `TensorRT-LLM`, and `triton` included, the research coverage is now strong enough to support:

- selecting an implementation strategy by operator class
- selecting a runtime architecture for serving/decode/KV-cache management
- selecting whether a fusion/codegen tier is necessary

That still does not mean the migration is "fully solved."

It means the remaining uncertainty is now mostly repo-local engineering and product choices, not missing source coverage.
