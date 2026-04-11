# CUDALibrarySamples Standalone Notes

Base repo:

- `/data/parametergolf/helpful_repos/NVIDIA/CUDALibrarySamples`

Why this repo matters:

- This is the best local reference for the question "how should we drive NVIDIA libraries correctly from our own runtime?"
- It is not a transformer repo, but it closes one of the biggest migration risks: replacing torch ops with badly-wrapped CUDA libraries.
- For `transformer_10`, this repo matters most for GEMMs, tensor contractions, sparse matmuls, grouped/batched execution, heuristics, and runtime-managed workspace.

## Highest-Value Directories

### 1. `cuBLASLt`

Start here:

- `cuBLASLt/LtSgemm/main.cpp`
- `cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu`
- `cuBLASLt/LtSgemmCustomFind/main.cu`
- `cuBLASLt/LtSgemmSimpleAutoTuning/sample_cublasLt_LtSgemmSimpleAutoTuning.cu`
- `cuBLASLt/LtHSHgemmStridedBatchSimple/sample_cublasLt_LtHSHgemmStridedBatchSimple.cu`
- `cuBLASLt/LtHSHgemmGroupedSimple/sample_cublasLt_LtHSHgemmGroupedSimple.cu`
- `cuBLASLt/LtFp8CustomFind/main.cu`
- `cuBLASLt/Common/LtMatmulCustomFind.h`

What it teaches:

- descriptor setup for `cublasLtMatmul`
- workspace handling
- preset vs heuristic algorithm choice
- grouped and strided-batched GEMM setup
- low-level custom-find loops instead of trusting a single default

Use in `transformer_10`:

- QKV projection GEMMs in `attn/eager.py`
- output projection GEMM in `attn/eager.py`
- MLP `w_in` and `w_out` GEMMs in `tensor/mlp.py`
- final LM head projection in `model/causal.py`
- tensor-parallel linears in `dist/parallel/tensor_parallel.py`

Primary decision:

- use `cuBLASLt` for nearly all dense linears before considering handwritten GEMM kernels

### 2. `cuTENSOR`

Start here:

- `cuTENSOR/contraction.cu`
- `cuTENSOR/contraction_jit.cu`
- `cuTENSOR/contraction_plan_cache.cu`
- `cuTENSOR/einsum.cu`
- `cuTENSOR/reduction.cu`
- `cuTENSOR/elementwise_binary.cu`
- `cuTENSOR/elementwise_trinary.cu`
- `cuTENSOR/elementwise_permute.cu`
- `cuTENSOR/blocksparse.cu`

What it teaches:

- contraction planning and plan cache usage
- JIT-backed contraction flow
- library-backed reductions and permutations
- block-sparse tensor algebra path

Use in `transformer_10`:

- only when an op is naturally a contraction/permutation problem rather than a plain GEMM
- strongest fit is optional block-sparse attention experimentation and odd einsum-style tensor contractions

Primary decision:

- `cuTENSOR` is secondary to `cuBLASLt` for this repo
- use it selectively for block-sparse or contraction-heavy variants, not the default dense transformer path

### 3. `cuSPARSELt`

Start here:

- `cuSPARSELt/matmul/matmul_example.cpp`
- `cuSPARSELt/matmul_advanced/matmul_advanced_example.cpp`

What it teaches:

- structured sparsity descriptors
- sparse Tensor Core matmul flow
- bias and activation around sparse matmul

Use in `transformer_10`:

- only if structured sparsity becomes a real model/runtime requirement
- relevant to `compress/pruning.py` follow-on work, not the base migration

Primary decision:

- not a phase-1 dependency

### 4. `nvMatmulHeuristics`

Start here:

- `nvMatmulHeuristics/1_gemm_heuristics.cpp`
- `nvMatmulHeuristics/2_discovery.cpp`
- `nvMatmulHeuristics/4_runtime_estimation.cpp`
- `nvMatmulHeuristics/5_get_configs.py`
- `nvMatmulHeuristics/6_get_configs_ex.py`
- `nvMatmulHeuristics/7_smem_carveout.py`

What it teaches:

- offline and runtime kernel-configuration ranking
- estimated runtime and energy-aware filtering
- how to narrow the search space before benchmarking

Use in `transformer_10`:

- shape-bucket tuning for recurring GEMMs
- selecting candidate matmul configs for common hidden sizes and vocab projections
- deciding when to keep cuBLASLt vs escalate to CUTLASS/custom paths

Primary decision:

- use it as a tuning aid, not an execution layer

### 5. `MathDx/cuBLASDx`

Start here:

- `MathDx/cuBLASDx/01_gemm_introduction/introduction_example.cu`
- `MathDx/cuBLASDx/09_gemm_custom_layout/simple_gemm_custom_layout.cu`
- `MathDx/cuBLASDx/14_gemm_fused/gemm_fusion.cu`
- `MathDx/cuBLASDx/15_gemm_nvrtc/nvrtc_gemm.cpp`
- `MathDx/cuBLASDx/reference/cublaslt_runner.hpp`

What it teaches:

- device-side GEMM building blocks
- custom layout and fused GEMM experiments
- NVRTC-based specialization

Use in `transformer_10`:

- useful if later phases need device-side or generated GEMM specialization
- not required for the first direct-runtime migration

## What To Copy Into Our Runtime

- a cuBLASLt wrapper layer that owns:
  - descriptor setup
  - workspace policy
  - layout canonicalization
  - heuristic selection
  - grouped/batched modes
- optional plan/config caches keyed by `(dtype, m, n, k, layout, epilog)`
- microbench harnesses that compare algorithms rather than assuming the default is good enough

## What Not To Do

- do not hand-write dense GEMMs just because we are leaving PyTorch
- do not hide algorithm selection behind an opaque Python call if the runtime needs repeatable performance
- do not treat `cuTENSOR` and `cuSPARSELt` as mandatory base dependencies for the dense LLM path

## Bottom Line For `transformer_10`

- `CUDALibrarySamples` is the strongest repo for library-backed operator execution
- it should directly shape the runtime for:
  - QKV/O/MLP/LM-head GEMMs
  - grouped and batched GEMM variants
  - future structured sparsity work
- if an op is contraction-dominated, this repo is the first place to check before writing a custom kernel
