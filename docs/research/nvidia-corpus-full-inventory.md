# NVIDIA Corpus Full Inventory

This document covers every top-level repository currently present under:

- `/data/parametergolf/helpful_repos/NVIDIA`

It also tracks one adjacent non-NVIDIA repo because it materially changes the kernel-authoring design space:

- `/data/parametergolf/helpful_repos/statespace_101/triton`

The purpose is corpus coverage first:

- what each repo is
- whether it is directly useful for CUDA kernel and C++ work
- what kinds of source assets it contains
- where to start reading

## Inventory Snapshot

File-count signals were gathered locally with a simple extension scan:

- CUDA-ish: `*.cu`, `*.cuh`, `*.ptx`, `*.cubin`, `*.sass`
- C/C++ headers and sources: `*.cc`, `*.cpp`, `*.cxx`, `*.h`, `*.hpp`
- Python: `*.py`

These counts are directional signals, not quality scores.

| Repo | CUDA-ish files | C/C++ files | Python files | Classification |
| --- | ---: | ---: | ---: | --- |
| `CUDALibrarySamples` | 305 | 266 | 95 | primary |
| `Fuser` | 55 | 747 | 163 | compiler/IR primary |
| `Megatron-Energon` | 0 | 0 | 155 | adjacent |
| `Megatron-LM` | 1 | 1 | 920 | systems reference |
| `Model-Optimizer` | 4 | 5 | 786 | adjacent |
| `TensorRT-LLM` | 401 | 5730 | 2382 | primary |
| `TransformerEngine` | 129 | 127 | 353 | primary |
| `cccl` | 1678 | 5406 | 280 | primary |
| `cuEmbed` | 20 | 7 | 2 | primary |
| `cuEquivariance` | 0 | 0 | 152 | adjacent |
| `cuda-q-academic` | 0 | 0 | 54 | out of scope for CUDA compute kernels |
| `cuda-quantum` | 9 | 1347 | 362 | adjacent C++ runtime |
| `cuda-samples` | 227 | 508 | 2 | primary |
| `cuda-tile` | 0 | 64 | 5 | compiler/IR primary |
| `cudaqx` | 7 | 192 | 68 | adjacent C++ runtime |
| `cuopt` | 329 | 225 | 179 | domain-specific CUDA/C++ primary |
| `cuopt-examples` | 0 | 0 | 21 | adjacent |
| `cutlass` | 1081 | 1195 | 400 | primary |
| `gpu-operator` | 0 | 0 | 1 | deployment-adjacent |
| `nccl` | 28 | 423 | 28 | primary |
| `nvidia-container-toolkit` | 0 | 6 | 0 | deployment-adjacent |
| `nvmath-python` | 0 | 11 | 530 | adjacent math wrapper |
| `open-gpu-kernel-modules` | 0 | 2289 | 1 | driver/kernel, not CUDA compute |
| `tilus` | 0 | 8 | 511 | DSL/compiler primary |

## Classification Legend

- `primary`
  - directly useful for studying or building CUDA kernels and C++ GPU runtime code
- `compiler/IR primary`
  - directly useful for generated kernels, tiling, or IR/lowering workflows
- `systems reference`
  - mostly Python/framework code, but important for integration and distributed design
- `adjacent`
  - useful context, examples, or runtime patterns, but not a main handwritten-kernel source
- `deployment-adjacent`
  - useful for environment and packaging, not for compute-kernel authoring
- `out of scope for CUDA compute kernels`
  - present locally, but not meaningfully relevant to model-stack compute kernel work

## Per-Repo Notes

### `CUDALibrarySamples`

Role:

- official usage samples for CUDA math, tensor, sparse, FFT, compression, and related libraries

Why it matters:

- fills a major gap between low-level kernel repos and actual library-backed runtime code
- especially valuable for `cuBLASLt`, `cuBLASDx`, `cuTENSOR`, `cuSPARSELt`, `cuBLASMp`, `cuSPARSE` graph capture, and `nvMatmulHeuristics`

Notable dirs:

- `CUDALibrarySamples/cuBLASLt`
- `CUDALibrarySamples/MathDx/cuBLASDx`
- `CUDALibrarySamples/cuTENSOR`
- `CUDALibrarySamples/cuSPARSELt`
- `CUDALibrarySamples/cuBLASMp`
- `CUDALibrarySamples/nvMatmulHeuristics`

Classification:

- primary

### `Fuser`

Role:

- nvFuser fusion compiler and CUDA code generator for NVIDIA GPUs

Why it matters:

- strong source for generated pointwise, reduction, normalization, SDPA, and multidevice fusion design
- especially useful when comparing handwritten CUDA/C++ against compiler-authored fusion paths

Notable dirs:

- `Fuser/csrc`
- `Fuser/runtime`
- `Fuser/benchmarks`
- `Fuser/tests`
- `Fuser/examples`

Classification:

- compiler/IR primary

### `Megatron-Energon`

Role:

- multimodal data loading and dataset orchestration for Megatron

Why it matters:

- useful if runtime design eventually needs data/packing context
- not a CUDA or C++ kernel repo

Notable dirs:

- `Megatron-Energon/src`
- `Megatron-Energon/docs`

Classification:

- adjacent

### `Megatron-LM`

Role:

- distributed transformer training/reference framework with Megatron Core

Why it matters:

- important integration reference for tensor/context parallel semantics, fused-op selection, and TransformerEngine usage
- not rich in local `.cu` source itself; much of the low-level work is delegated or extension-backed

Notable dirs:

- `Megatron-LM/megatron/core/tensor_parallel`
- `Megatron-LM/megatron/core/fusions`
- `Megatron-LM/megatron/core/inference`
- `Megatron-LM/docs/user-guide/features`

Classification:

- systems reference

### `Model-Optimizer`

Role:

- optimization toolkit for quantization, pruning, distillation, export

Why it matters:

- useful around quantization/export decisions
- only a tiny amount of direct CUDA source locally

Notable paths:

- `Model-Optimizer/experimental/conv/implicit_gemm_kernel.cu`
- `Model-Optimizer/modelopt`
- `Model-Optimizer/docs`

Classification:

- adjacent

### `TransformerEngine`

Role:

- production transformer kernels and mixed-precision runtime

Why it matters:

- one of the strongest local sources for attention, norm, rope, cast, transpose, and cuBLASLt integration

Notable dirs:

- `TransformerEngine/transformer_engine/common/fused_attn`
- `TransformerEngine/transformer_engine/common/fused_softmax`
- `TransformerEngine/transformer_engine/common/fused_rope`
- `TransformerEngine/transformer_engine/common/normalization`
- `TransformerEngine/transformer_engine/common/gemm`

Classification:

- primary

### `TensorRT-LLM`

Role:

- high-performance LLM inference runtime with specialized kernels, executor/runtime layers, plugins, and serving infrastructure

Why it matters:

- one of the strongest local references for inference runtime architecture, KV cache ownership, batching, decode scheduling, speculative decoding, and plugin-based extension
- more valuable for runtime and serving design than for learning first-principles kernel authoring

Notable dirs:

- `TensorRT-LLM/cpp/tensorrt_llm/runtime`
- `TensorRT-LLM/cpp/tensorrt_llm/executor`
- `TensorRT-LLM/cpp/tensorrt_llm/plugins`
- `TensorRT-LLM/triton_kernels`
- `TensorRT-LLM/docs/source/features`

Classification:

- primary

### `cccl`

Role:

- CUDA Core Compute Libraries: CUB, Thrust, libcudacxx, and related layers

Why it matters:

- foundational source for block/warp/device primitives, barriers, atomics, launch helpers, async copy helpers

Notable dirs:

- `cccl/cub`
- `cccl/libcudacxx`
- `cccl/thrust`
- `cccl/examples`

Classification:

- primary

### `cuEmbed`

Role:

- focused embedding lookup kernel library

Why it matters:

- compact, readable reference for a bandwidth-bound kernel family

Notable dirs:

- `cuEmbed/cuembed/include`
- `cuEmbed/utils/src`
- `cuEmbed/tests`

Classification:

- primary

### `cuEquivariance`

Role:

- geometric deep-learning library with optimized kernels shipped externally

Why it matters:

- interesting if studying segmented polynomial or equivariant kernels
- local repo is mostly Python/frontend structure rather than raw CUDA source

Notable dirs:

- `cuEquivariance/cuequivariance`
- `cuEquivariance/cuequivariance_jax`
- `cuEquivariance/cuequivariance_torch`

Classification:

- adjacent

### `cuda-q-academic`

Role:

- educational quantum notebooks

Why it matters:

- not relevant for CUDA compute-kernel authoring in this model stack

Classification:

- out of scope for CUDA compute kernels

### `cuda-quantum`

Role:

- CUDA-Q compiler/runtime for hybrid quantum-classical systems

Why it matters:

- contains substantial C++ runtime/compiler infrastructure
- not a primary reference for LLM CUDA kernels

Notable dirs:

- `cuda-quantum/include`
- `cuda-quantum/lib`
- `cuda-quantum/runtime`

Classification:

- adjacent C++ runtime

### `cuda-samples`

Role:

- official CUDA Toolkit samples

Why it matters:

- canonical small references for streams, events, graphs, memory pools, WMMA, NVRTC, P2P, PTX JIT

Notable dirs:

- `cuda-samples/Samples`
- `cuda-samples/Common`

Classification:

- primary

### `cuda-tile`

Role:

- MLIR-based CUDA tile IR and optimization infrastructure

Why it matters:

- directly relevant if kernel generation or lowering becomes part of the toolchain

Notable dirs:

- `cuda-tile/include`
- `cuda-tile/lib`
- `cuda-tile/tools`
- `cuda-tile/test`

Classification:

- compiler/IR primary

### `cudaqx`

Role:

- libraries on top of CUDA-Q

Why it matters:

- C++/runtime oriented, but not central for CUDA compute-kernel work

Classification:

- adjacent C++ runtime

### `cuopt`

Role:

- GPU-native optimization engine

Why it matters:

- significant CUDA/C++ codebase with runtime, CUDA graphs, utilities, and solver kernels
- domain-specific, but still useful as a large product-quality CUDA/C++ codebase

Notable dirs:

- `cuopt/cpp/src`
- `cuopt/cpp/tests`
- `cuopt/benchmarks`

Classification:

- domain-specific CUDA/C++ primary

### `cuopt-examples`

Role:

- examples for cuOpt usage

Why it matters:

- not a kernel source repo

Classification:

- adjacent

### `cutlass`

Role:

- high-performance matrix and tensor-compute abstraction library

Why it matters:

- critical local source for GEMM and fusion patterns
- especially valuable for epilogs, fused attention examples, softmax/GEMM fusion, layernorm fusion, and architecture-specialized GEMM design

Notable dirs:

- `cutlass/include/cutlass`
- `cutlass/include/cute`
- `cutlass/examples`
- `cutlass/python`

Classification:

- primary

### `gpu-operator`

Role:

- Kubernetes GPU lifecycle/operator stack

Why it matters:

- useful for deployment environments
- not relevant for kernel authoring

Classification:

- deployment-adjacent

### `nccl`

Role:

- multi-GPU collective communication library

Why it matters:

- required reading for direct collective runtime design

Notable dirs:

- `nccl/src/device`
- `nccl/src/include`
- `nccl/src/transport`

Classification:

- primary

### `nvidia-container-toolkit`

Role:

- GPU container runtime setup

Why it matters:

- environment/deployment only

Classification:

- deployment-adjacent

### `nvmath-python`

Role:

- Python access to NVIDIA math libraries

Why it matters:

- reinforces library-backed planning, cuBLASLt exposure, device-side API patterns
- not itself a main handwritten kernel codebase

Notable dirs:

- `nvmath-python/examples/linalg`
- `nvmath-python/examples/device`
- `nvmath-python/nvmath`

Classification:

- adjacent math wrapper

### `open-gpu-kernel-modules`

Role:

- Linux kernel and driver source

Why it matters:

- important for driver internals
- not a CUDA compute-kernel reference

Classification:

- driver/kernel, not CUDA compute

### `tilus`

Role:

- tile-level GPU kernel DSL

Why it matters:

- strong source for kernel-shaping ideas and DSL-level examples
- especially relevant for attention, matmul, softmax, norm, and async/shared-memory instruction tests

Notable dirs:

- `tilus/examples`
- `tilus/python/tilus`
- `tilus/tests`

Classification:

- DSL/compiler primary

## Adjacent External Repo

### `triton`

Base path:

- `/data/parametergolf/helpful_repos/statespace_101/triton`

Role:

- Triton kernel DSL, JIT runtime, autotuning layer, and compiler frontend

Why it matters:

- not part of the NVIDIA corpus proper, but highly relevant when deciding whether some kernels should be handwritten CUDA/C++ or authored in a DSL
- especially relevant for matmul, flash attention, softmax, cross-entropy, autotuning, JIT specialization, and cache behavior

Notable dirs:

- `triton/ops`
- `triton/runtime`
- `triton/compiler`
- `triton/language`
- `triton/tools`

Classification:

- adjacent external DSL reference

## Priority Reading Order

If the goal is CUDA/C++ coverage for model-stack work, read in this order:

1. `cuda-samples`
2. `cccl`
3. `CUDALibrarySamples`
4. `cutlass`
5. `TransformerEngine`
6. `nccl`
7. `TensorRT-LLM`
8. `Fuser`
9. `cuEmbed`
10. `cuda-tile`
11. `tilus`
12. `Megatron-LM`
13. `nvmath-python`

Then read the adjacent repos only if the specific need appears:

- `Model-Optimizer` for quantization/export
- `cuopt` for large CUDA/C++ product structure
- `cuda-quantum` / `cudaqx` for C++ runtime design patterns
- `triton` for DSL-authored and JIT-specialized kernel alternatives

## Bottom Line

For full-corpus orientation:

- `TransformerEngine`, `cccl`, `CUDALibrarySamples`, `cutlass`, `cuda-samples`, `nccl`, `TensorRT-LLM`, and `cuEmbed` are the center of gravity.
- `Fuser`, `cuda-tile`, `tilus`, and `triton` matter if we want generated or DSL-authored kernels instead of only handwritten CUDA/C++.
- `Megatron-LM` matters more for systems integration than for raw local kernel source.
- several local repos are valuable context but not direct CUDA compute references.
