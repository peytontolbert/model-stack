# Fuser Standalone Notes

Base repo:

- `/data/parametergolf/helpful_repos/NVIDIA/Fuser`

Why this repo matters:

- `Fuser` is the strongest local reference for compiler-generated fusion on NVIDIA GPUs.
- It is especially valuable for reductions, norms, pointwise chains, schedule selection, and multi-device lowering.
- It is less aligned with the long-term "minimal torch dependency" goal than handwritten CUDA/C++, but it is still important research because it shows which fusion patterns are worth automating and what runtime support they need.

## Highest-Value Areas

### 1. Runtime kernels

Start here:

- `runtime/fused_reduction.cu`
- `runtime/grid_reduction.cu`
- `runtime/welford.cu`
- `runtime/fused_welford_impl.cu`
- `runtime/scan.cu`
- `runtime/topk.cu`
- `runtime/tma_copy.cu`
- `runtime/block_quantization_kernels.cu`

What it teaches:

- reusable reduction skeletons
- Welford-style statistics paths for norm-like ops
- top-k and scan runtime components
- async-copy/TMA-oriented runtime helpers
- how a fusion compiler still ends up owning a real low-level kernel runtime

### 2. Fusion IR and lowering

Start here:

- `csrc/fusion.cpp`
- `csrc/codegen.cpp`
- `csrc/device_lower/lower2device.cpp`
- `csrc/ops/normalization.cpp`
- `csrc/ops/composite.cpp`
- `csrc/fusion_segmenter.cpp`
- `csrc/cutlass/gemm.cpp`

What it teaches:

- graph-to-kernel lowering boundaries
- segmentation of graphs into fuseable regions
- normalization and composite op lowering
- when the compiler hands work to CUTLASS-backed GEMM generation instead of inventing a separate path

### 3. Docs for the hard parts

Read:

- `doc/dev/debug.md`
- `doc/dev/tma.md`
- `doc/dev/host_ir_jit.md`
- `doc/reading/tma-modeling-in-depth.md`
- `doc/reading/multigpu.md`

Why these matter:

- they cover the parts that usually make fusion systems brittle:
  - debug surfaces
  - host/runtime IR split
  - TMA modeling
  - multi-GPU reasoning

### 4. Benchmarks and tests

Start here:

- `benchmarks/cpp/rms_norm.cpp`
- `benchmarks/cpp/layer_norm.cpp`
- `benchmarks/cpp/softmax.cpp`
- `benchmarks/cpp/softmax_dropout.cpp`
- `benchmarks/cpp/matmul.cpp`
- `tests/cpp/test_rope.cpp`
- `tests/cpp/test_sdpa.cpp`
- `tests/cpp/test_moe.cpp`
- `tests/cpp/test_multidevice_transformer.cpp`

What they teach:

- which transformer-shaped fusions the project considers important
- how to validate correctness against reference implementations
- how the project exercises SDPA-, RoPE-, norm-, and MoE-adjacent logic

## Where It Maps To `transformer_10`

Best fits:

- pointwise and reduction-heavy chains around:
  - `tensor/norms.py`
  - `tensor/losses.py`
  - `tensor/sampling.py`
  - residual/dropout/norm glue in `blocks/transformer_block.py`
- training-oriented composite ops where staying in Python for orchestration is acceptable
- optional experimental MoE routing and reduction flows in `blocks/moe_block.py`

Weaker fits:

- paged KV cache runtime in `attn/kv_cache.py`
- inference decode attention in `serve/engine.py`
- serving-first attention kernels in `attn/eager.py`

Reason:

- the repo is excellent for fusion logic and codegen study
- it is not the cleanest direct model for a lean standalone inference runtime

## Decision For This Project

Use `Fuser` mainly for:

- deciding which pointwise/reduction chains should be fused
- validating whether a fusion-heavy op should stay compiler-driven or become a fixed handwritten kernel
- training-side or experimentation-side prototype work

Do not make it the primary long-term answer for:

- paged decode attention
- KV cache runtime ownership
- serving scheduler/runtime design

## Most Useful Mental Model

`Fuser` is not proof that every custom op should be a compiler problem.

It is proof that:

- many reduction and pointwise chains are worth fusing
- norms and softmax-family ops need explicit runtime support
- multi-device lowering and host/runtime separation are first-class concerns once kernels stop living inside PyTorch eager

## Bottom Line For `transformer_10`

- keep `Fuser` in the research set because it is one of the best local references for fusion strategy
- do not let it define the whole migration architecture
- for this repo, its highest value is in:
  - norms
  - softmax/reduction patterns
  - top-k and sampling-related utilities
  - training-side composite-op prototyping
