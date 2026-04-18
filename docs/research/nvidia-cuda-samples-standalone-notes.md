# CUDA Samples Standalone Notes

Base repo:

- `/data/parametergolf/helpful_repos/NVIDIA/cuda-samples`

Why this repo matters:

- `cuda-samples` is the best local reference for low-level CUDA execution patterns that sit below library wrappers.
- For `transformer_10`, it is especially useful when a design question has already moved past "should this be `cuBLASLt`?" and into "how should a custom kernel and runtime actually behave on NVIDIA hardware?"
- For the BitNet backend, this repo is not the answer by itself, but it is the strongest local source for:
  - integer GEMM staging patterns
  - shared-memory layout choices
  - async-copy pipelines
  - block and warp synchronization primitives
  - occupancy-guided launch policy
  - CUDA Graph upload and replay behavior

## Highest-Value Samples For BitNet

### 1. `immaTensorCoreGemm`

Start here:

- `Samples/3_CUDA_Features/immaTensorCoreGemm/immaTensorCoreGemm.cu`
- `Samples/3_CUDA_Features/immaTensorCoreGemm/README.md`

What it teaches:

- CTA-level integer GEMM decomposition with warp-level fragments
- staging A/B tiles through shared memory before MMA consumption
- using skewed shared-memory layouts to avoid bank conflicts during fragment loads
- vectorized global-to-shared copies with `int4`
- dynamic shared-memory opt-in through `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`

Important code anchors:

- skewed shared-memory layout and alignment constraint:
  - `immaTensorCoreGemm.cu:154-167`
- shared-memory tile staging and vectorized `int4` loads:
  - `immaTensorCoreGemm.cu:289-314`
- WMMA fragment loads and MMA mainloop:
  - `immaTensorCoreGemm.cu:320-343`
- dynamic shared-memory carveout for the high-performance kernel:
  - `immaTensorCoreGemm.cu:567-573`

Use in BitNet:

- copy the tiling mindset, not the exact WMMA path
- use it as the reference for:
  - how to stage packed weights into shared memory
  - how to skew or pad shared layouts for conflict reduction
  - how to gate high-shared-memory kernels by device capability

What not to copy literally:

- the WMMA integer API is still an INT8/INT accumulator path, not a native ternary/INT2 BitNet contract
- the public BitNet ABI should stay based on explicit packed-weight metadata, not on a WMMA fragment type

### 2. `globalToShmemAsyncCopy`

Start here:

- `Samples/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`
- `Samples/3_CUDA_Features/globalToShmemAsyncCopy/README.md`

What it teaches:

- `cuda::memcpy_async` and `cuda::pipeline` usage
- multi-stage producer/consumer pipelines
- block-scope shared pipeline state
- a producer warp feeding one or more consumer warps
- rotating shared-memory stage buffers

Important code anchors:

- block-scope shared pipeline state:
  - `globalToShmemAsyncCopy.cu:542-548`
- producer warp feeding staged shared-memory tiles:
  - `globalToShmemAsyncCopy.cu:554-570`
- consumer-side wait / compute / release loop:
  - `globalToShmemAsyncCopy.cu:579-590`

Use in BitNet:

- this is the best direct local reference for the prefill kernel family
- use it for:
  - async loading packed weight tiles and activation tiles
  - producer/consumer split when decode or small-batch kernels benefit from a dedicated loader warp
  - multi-stage buffering for larger `K`

Design implication:

- the BitNet backend should treat async-copy pipelines as an optimization lane for `sm80+`, not as a separate public operator contract

### 3. `simpleAWBarrier`

Start here:

- `Samples/0_Introduction/simpleAWBarrier/simpleAWBarrier.cu`
- `Samples/0_Introduction/simpleAWBarrier/README.md`

What it teaches:

- `cuda::barrier<cuda::thread_scope_block>`
- explicit `arrive()` / `wait()` synchronization
- combining warp-tile reductions with block-wide barrier coordination
- cooperative launch plus block/grid synchronization for structured reductions

Important code anchors:

- barrier arrive/wait in a reusable reduction helper:
  - `simpleAWBarrier.cu:46-79`
- barrier initialization and tiled partition use inside the kernel:
  - `simpleAWBarrier.cu:88-108`

Use in BitNet:

- use this when a kernel stage needs explicit phase boundaries that are cleaner than raw `__syncthreads()`
- especially relevant for:
  - decode kernels with staged unpack -> dot -> epilogue phases
  - partial reductions for per-row activation scale computation

### 4. `simpleCooperativeGroups`

Start here:

- `Samples/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu`
- `Samples/0_Introduction/simpleCooperativeGroups/README.md`

What it teaches:

- `thread_block`
- `thread_block_tile<N>`
- group-scoped workspace partitioning
- reduction helpers that are parameterized by group type instead of assuming one fixed warp/block shape

Important code anchors:

- generic group reduction helper:
  - `simpleCooperativeGroups.cu:52-81`
- block-wide and tile-wide partitioning:
  - `simpleCooperativeGroups.cu:89-137`

Use in BitNet:

- use cooperative-groups style APIs for kernels that need multiple logical subgroups inside one CTA
- strongest fit:
  - producer vs consumer partitioning
  - per-warp accumulator reductions
  - segmented epilogues for fused QKV and fused MLP projections

### 5. `simpleOccupancy`

Start here:

- `Samples/0_Introduction/simpleOccupancy/simpleOccupancy.cu`
- `Samples/0_Introduction/simpleOccupancy/README.md`

What it teaches:

- `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- `cudaOccupancyMaxPotentialBlockSize`
- occupancy as a launch-configuration starting point rather than a final answer

Important code anchors:

- occupancy reporting:
  - `simpleOccupancy.cu:62-83`
- occupancy-based launch suggestion:
  - `simpleOccupancy.cu:87-151`

Use in BitNet:

- use occupancy APIs to seed kernel-plan search for new shape buckets
- do not hardwire occupancy-maximizing launches as the final policy

Design implication:

- the BitNet runtime should cache shape-bucket decisions, but it still needs empirical benchmarking on top of occupancy-based first guesses

### 6. `cudaGraphsPerfScaling`

Start here:

- `Samples/6_Performance/cudaGraphsPerfScaling/cudaGraphPerfScaling.cu`
- `Samples/6_Performance/cudaGraphsPerfScaling/README.md`

What it teaches:

- stream capture to graph
- graph instantiation cost
- first-launch upload cost vs repeat launch cost
- explicit `cudaGraphUpload` off the critical path

Important code anchors:

- graph capture from stream work:
  - `cudaGraphPerfScaling.cu:128-157`
- first vs repeat launch measurement:
  - `cudaGraphPerfScaling.cu:164-197`
- off-critical-path upload before launch:
  - `cudaGraphPerfScaling.cu:242-274`

Use in BitNet:

- this is the right local reference for graph-capture policy in decode and prefill
- strongest fit:
  - pre-instantiating stable graph shapes
  - pre-uploading graph execs when decode shape buckets are known
  - separating graph-upload latency from actual kernel latency in benchmarks

### 7. `transpose`

Start here:

- `Samples/6_Performance/transpose/transpose.cu`
- `Samples/6_Performance/transpose/README.md`

What it teaches:

- coalescing-aware tile loads and stores
- shared-memory padding to avoid bank conflicts
- diagonal CTA traversal to improve memory-system behavior

Important code anchors:

- shared tile with bank conflicts:
  - `transpose.cu:141-164`
- padded shared tile removing conflicts:
  - `transpose.cu:168-190`
- diagonal block scheduling:
  - `transpose.cu:193-244`

Use in BitNet:

- use the padding lesson directly for packed-weight shared layouts
- use the diagonal scheduling idea as a reminder that CTA traversal order can matter once weight reuse and L2 locality become important

### 8. `alignedTypes`

Start here:

- `Samples/6_Performance/alignedTypes/alignedTypes.cu`
- `Samples/6_Performance/alignedTypes/doc/alignedTypes.txt`
- `Samples/6_Performance/alignedTypes/README.md`

What it teaches:

- alignment requirements for efficient vectorized global memory accesses
- why 4/8/16-byte aligned element groupings matter
- why structure-of-arrays layout often wins once records get too large

Important code anchors:

- aligned struct definitions:
  - `alignedTypes.cu:77-97`
- note about efficient native global-memory widths:
  - `alignedTypes.cu:99-108`

Use in BitNet:

- packed weights and scale metadata must be laid out so the kernel can use aligned vector transactions
- `layout_meta` should make alignment and padded shape explicit rather than letting kernels guess

## Secondary But Still Useful Samples

### 9. `matrixMul`

Why it still matters:

- it is the clearest baseline tiled shared-memory GEMM in the repo
- it is the best first correctness skeleton when bringing up a new custom kernel family

Start here:

- `Samples/0_Introduction/matrixMul/matrixMul.cu`

Use in BitNet:

- keep it as the simplest reference for "correct tile staging and accumulation before optimization"
- do not mistake it for a performance target

### 10. `matrixMulCUBLAS`

Why it still matters:

- it is the local reminder that library-backed dense GEMM remains the fallback and reference path

Start here:

- `Samples/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp`

Use in BitNet:

- keep dense `cuBLASLt` fallback and parity harnesses in the design
- do not try to replace dense fallback with a handwritten BitNet kernel for unsupported shapes

### 11. `bf16TensorCoreGemm`, `tf32TensorCoreGemm`, `dmmaTensorCoreGemm`

Why they matter:

- they show how NVIDIA keeps the same broad GEMM decomposition pattern while swapping the internal MMA lane by datatype and architecture

Use in BitNet:

- architecture-specific BitNet optimized lanes should follow that same principle:
  - stable external format and ABI
  - different inner mainloops on Ampere vs Hopper vs later GPUs

## Direct Implications For The Native BitNet Spec

From `cuda-samples`, the strongest design consequences are:

- the backend should keep one stable packed-weight ABI and let architecture-specific kernels change only the internal mainloop
- prefill kernels should use staged shared-memory pipelines and optionally producer/consumer warp partitioning on `sm80+`
- decode kernels should consider explicit barrier and cooperative-group structure instead of relying on only raw block-wide synchronization
- shared-memory skew and padding are mandatory design levers, not optional cleanup
- plan selection should start with occupancy APIs but end with benchmarked shape-bucket policy
- graph upload and replay should be part of the runtime design, not an afterthought added after kernels exist

## What Not To Overlearn From `cuda-samples`

- do not treat the samples as production templates
- do not copy WMMA-based integer GEMM as if it solves BitNet directly; it does not define the ternary packed format or row-wise activation scaling contract
- do not assume the sample launch geometry is correct for transformer-shaped workloads
- do not let sample clarity push the implementation toward scalar or toy kernels when the backend needs persistent, grouped, and batched behavior

## Bottom Line For `transformer_10`

- `cuda-samples` should directly influence the BitNet backend spec
- the most relevant references are:
  - `immaTensorCoreGemm`
  - `globalToShmemAsyncCopy`
  - `simpleAWBarrier`
  - `simpleCooperativeGroups`
  - `simpleOccupancy`
  - `cudaGraphsPerfScaling`
  - `transpose`
  - `alignedTypes`
- together, they strengthen the current conclusion that the native BitNet path should be:
  - a custom packed backend
  - async-copy and shared-memory aware
  - graph-ready
  - dense-fallback capable
