# Triton Standalone Notes

Base repo:

- `/data/parametergolf/helpful_repos/statespace_101/triton`

Why this repo matters:

- This local checkout is not part of the NVIDIA corpus, but it is one of the most relevant adjacent repos for deciding when not to write handwritten CUDA/C++ first.
- It is the cleanest local reference for:
  - JIT kernel definition
  - autotune/config search
  - compact fused-kernel authoring
  - matmul, flash attention, and cross-entropy examples

## Highest-Value Areas

### 1. Runtime and compilation

Start here:

- `runtime/jit.py`
- `runtime/autotuner.py`
- `runtime/cache.py`
- `compiler/compiler.py`
- `compiler/code_generator.py`

What it teaches:

- source hashing and dependency invalidation for JIT kernels
- specialization keys and cache behavior
- autotune config search and pruning
- kernel launch wrapping and code generation boundaries

### 2. Language surface

Start here:

- `language/core.py`
- `language/semantic.py`

What it teaches:

- the real authoring surface behind Triton kernels
- how tile-level programs, block pointers, and reductions are expressed

### 3. High-value ops

Start here:

- `ops/flash_attention.py`
- `ops/matmul.py`
- `ops/cross_entropy.py`
- `ops/blocksparse/matmul.py`
- `ops/blocksparse/softmax.py`

What it teaches:

- fused tiled attention structure
- autotuned tiled matmul with split-K
- cross-entropy as a compact fused reduction kernel
- block-sparse attention and softmax sketches

## Where Triton Fits `transformer_10`

Best fits:

- standalone fused pointwise and reduction kernels
- cross-entropy and sampler helpers
- optional sparse/local attention variants
- quick-turnaround experiments before committing to a C++ ABI and handwritten CUDA runtime

Reasonable fits:

- SwiGLU and similar MLP middle-stage fusion
- top-k and masking helpers
- training-oriented fused ops

Weaker fits:

- paged KV cache ownership
- multi-GPU runtime/executor design
- final generation-phase attention runtime for a serving stack

Reason:

- Triton is excellent for stateless kernels
- it is much less compelling for stateful serving/runtime subsystems

## Decision For This Project

Use Triton when all of the following are true:

- the op is local and stateless
- the kernel benefits from fast iteration and autotuning
- the kernel is not defining the long-term runtime ABI

Typical `transformer_10` candidates:

- cross-entropy and tokenwise loss kernels
- top-k/top-p masking helpers
- SwiGLU-style elementwise fusion
- optional block-sparse/local attention variants

Avoid making Triton the primary answer for:

- paged decode attention runtime
- KV cache data structure management
- low-level multi-GPU communication

## Relationship To Handwritten CUDA

Triton is best treated here as:

- a fast prototyping path
- a strong implementation path for some stable stateless kernels
- a decision aid for which kernels deserve a later C++/CUDA rewrite

It is not a substitute for:

- a C++ runtime layer
- explicit memory-pool ownership
- NCCL/runtime integration

## Bottom Line For `transformer_10`

- Triton should be a selective tool, not the default architecture
- use it for compact fused kernels and optional attention variants
- keep handwritten CUDA/C++ for runtime-owned serving primitives
- keep cuBLASLt for dense GEMMs
