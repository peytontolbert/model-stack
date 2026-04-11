# Other Repos CUDA/C++ Reference Map

This document covers the local auxiliary corpus under `/data/transformer_10/other_repos`.

These repos are not the primary source of truth for CUDA library usage. That still comes from the NVIDIA corpus and the local `triton` checkout. The value here is different:

- concrete kernel organization
- lightweight extension boundaries
- runtime/JIT/graph wrappers
- didactic kernel patterns
- additional attention and norm implementations

## Short Verdict

Use the repos in this order of importance for `transformer_10`:

1. `flash-attention`
   - strongest non-NVIDIA reference for attention kernel decomposition, head-dimension specialization, and fused norm history
2. `ThunderKittens`
   - strongest educational source for modern Hopper/Blackwell kernel structure, TMA, warpgroup pipelines, and alongside-kernel test/benchmark layout
3. `tiny-cuda-nn`
   - strongest source here for small C++ runtime helpers: RTC wrappers, CUDA Graph capture/update, GPU memory abstractions, multi-stream support
4. `extension-cpp`
   - good minimal reference for a transitional Python binding boundary
5. `tinygrad`
   - useful only when we want to study a minimal direct CUDA-driver runtime or memory/launch plumbing
6. `cuda_ext`
   - useful local examples of strict binding contracts, workspace-heavy APIs, and graph-capturable extension entrypoints
7. `cuda-kernels`
   - good for reduction/prefix-sum/memory-coalescing basics, not for production transformer runtime design
8. `good-kernels`
   - useful for quick sanity checks on LayerNorm/Softmax/Matmul kernel structure, not for architecture decisions

## Repo-By-Repo Notes

## `ThunderKittens`

Best paths:

- `/data/transformer_10/other_repos/ThunderKittens/README.md`
- `/data/transformer_10/other_repos/ThunderKittens/kernels/gemm/educational_h100/README.md`
- `/data/transformer_10/other_repos/ThunderKittens/kernels/attention/mha_h100/`
- `/data/transformer_10/other_repos/ThunderKittens/kernels/layernorm/`
- `/data/transformer_10/other_repos/ThunderKittens/kernels/parallel/`

What it teaches well:

- how modern kernels are staged from simple loop to shared memory to tensor-core path to TMA and multi-warpgroup overlap
- how to package one kernel with its own `Makefile`, tests, and benchmark harness
- how to think in warp, warpgroup, block, and persistent-grid terms instead of generic CUDA threads only
- how to keep kernel-specific launch policy near the kernel rather than buried in Python

What to take into `transformer_10`:

- the kernel-local structure: source + benchmark + correctness test together
- the educational ladder when building our own attention and norm kernels
- the idea that bandwidth-bound ops and attention helpers should be architecture-specialized

What not to copy blindly:

- do not make ThunderKittens itself the new core dependency for the runtime
- do not use its GEMM kernels as a reason to replace `cuBLASLt`
- do not overfit the runtime to Hopper/Blackwell-only assumptions unless the fleet is actually limited to those GPUs

Direct mapping:

- attention kernel design
- LayerNorm/RMSNorm specialization
- future multi-GPU collectives experiments

## `flash-attention`

Best paths:

- `/data/transformer_10/other_repos/flash-attention/README.md`
- `/data/transformer_10/other_repos/flash-attention/csrc/flash_attn/flash_api.cpp`
- `/data/transformer_10/other_repos/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `/data/transformer_10/other_repos/flash-attention/csrc/flash_attn/src/flash_bwd_kernel.h`
- `/data/transformer_10/other_repos/flash-attention/csrc/flash_attn/src/softmax.h`
- `/data/transformer_10/other_repos/flash-attention/csrc/flash_attn/src/rotary.h`
- `/data/transformer_10/other_repos/flash-attention/csrc/layer_norm/README.md`
- `/data/transformer_10/other_repos/flash-attention/csrc/fused_dense_lib/README.md`

What it teaches well:

- production attention decomposition into API, launch templates, kernel traits, mask helpers, softmax helpers, and many compile units
- specialization by head dimension, dtype, and causal/non-causal mode
- how much real attention code is about mask/layout/dispatch details rather than the `QK^T` formula
- historical fused norm and fused dense patterns that can still inform our first-wave kernels even if the repo has since changed implementation style

What to take into `transformer_10`:

- the attention-specific organization for `runtime/cuda/src/attention`
- the idea of explicit specialization buckets for head size and dtype
- the utility split:
  - mask helpers
  - rotary helpers
  - softmax helpers
  - launch templates

What not to copy blindly:

- do not adopt its PyTorch packaging layout as the long-term runtime layout
- do not assume our first implementation needs full backward parity or all supported features at once
- do not carry over feature surface we do not use in `transformer_10`

Direct mapping:

- causal prefill attention
- decode attention
- rotary application structure
- RMSNorm/LayerNorm references

## `tiny-cuda-nn`

Best paths:

- `/data/transformer_10/other_repos/tiny-cuda-nn/README.md`
- `/data/transformer_10/other_repos/tiny-cuda-nn/include/tiny-cuda-nn/rtc_kernel.h`
- `/data/transformer_10/other_repos/tiny-cuda-nn/include/tiny-cuda-nn/cuda_graph.h`
- `/data/transformer_10/other_repos/tiny-cuda-nn/include/tiny-cuda-nn/gpu_memory.h`
- `/data/transformer_10/other_repos/tiny-cuda-nn/include/tiny-cuda-nn/multi_stream.h`
- `/data/transformer_10/other_repos/tiny-cuda-nn/include/tiny-cuda-nn/cutlass_matmul.h`

What it teaches well:

- a compact C++ runtime surface that still supports serious CUDA features
- runtime compilation wrappers through a small `CudaRtcKernel` API
- CUDA Graph capture/update and replay as a reusable utility, not a model-specific hack
- GPU memory and matrix abstractions that are explicit enough to survive optimization work

What to take into `transformer_10`:

- a small self-owned graph wrapper
- a small runtime-compiled kernel wrapper for experiments and fallback kernels
- explicit stream- and memory-aware helper types around kernel launches

What not to copy blindly:

- we do not need its network/optimizer framework
- we do not need to rebuild our model stack into a tiny-cuda-nn style object hierarchy
- do not use its fully fused MLP design as a reason to skip `cuBLASLt` for transformer GEMMs

Direct mapping:

- CUDA Graph utilities for serving
- optional RTC path for experiments
- memory and stream helper design for the new runtime

## `extension-cpp`

Best paths:

- `/data/transformer_10/other_repos/extension-cpp/README.md`
- `/data/transformer_10/other_repos/extension-cpp/extension_cpp/extension_cpp/csrc/muladd.cpp`

What it teaches well:

- minimal operator registration and extension packaging
- how to keep the binding layer thin and validation-heavy
- how to isolate CPU/CUDA registration from the rest of the code

What to take into `transformer_10`:

- only the general idea of a narrow transitional bridge
- explicit input validation at the language boundary

What not to copy blindly:

- this is a PyTorch-extension tutorial repo, not a runtime architecture
- do not make ATen the long-term dependency boundary if the goal is direct CUDA/C++

Direct mapping:

- temporary compatibility bridge while Python still orchestrates execution

## `tinygrad`

Best paths:

- `/data/transformer_10/other_repos/tinygrad/docs/runtime.md`
- `/data/transformer_10/other_repos/tinygrad/tinygrad/runtime/ops_cuda.py`
- `/data/transformer_10/other_repos/tinygrad/tinygrad/runtime/graph/cuda.py`
- `/data/transformer_10/other_repos/tinygrad/tinygrad/runtime/support/compiler_cuda.py`

What it teaches well:

- how little machinery is actually required to load a CUDA module and launch kernels
- allocator and peer-access setup with the CUDA driver API
- separation of launch encoding, module load, allocator, and graph execution

What to take into `transformer_10`:

- runtime minimalism
- small focused abstractions around launches and memory
- the discipline of treating device runtime as its own subsystem

What not to copy blindly:

- do not import its compiler/renderer stack into this repo
- do not rebuild the tensor IR or backend-selection machinery
- do not use Python driver-level launch code as the final serving runtime

Direct mapping:

- sanity reference for direct CUDA driver API usage
- inspiration for a minimal debug or fallback runtime harness

## `cuda_ext`

Best paths:

- `/data/transformer_10/other_repos/cuda_ext/causal_machine_scan.md`
- `/data/transformer_10/other_repos/cuda_ext/muon.md`
- `/data/transformer_10/other_repos/cuda_ext/causal_machine_scan.cpp`
- `/data/transformer_10/other_repos/cuda_ext/muon.cpp`

What it teaches well:

- strict tensor-contract validation before launch
- workspace-heavy extension APIs
- graph-capturable API variants that move scalar hyperparameters onto device tensors
- splitting binding logic from CUDA logic

What to take into `transformer_10`:

- the idea that our binding layer should fail early on shape/layout/device mismatches
- distinct API variants for simple use, workspace reuse, and graph-capturable serving

What not to copy blindly:

- these are specialized extension examples, not the runtime core
- do not mirror their exact PyTorch extension surface unless we actually need that compatibility path

Direct mapping:

- cache API validation
- future optimizer/training extensions
- graph-capturable decode/runtime entrypoints

## `cuda-kernels`

Best paths:

- `/data/transformer_10/other_repos/cuda-kernels/README.md`
- `/data/transformer_10/other_repos/cuda-kernels/parallel-prefix-sum/`
- `/data/transformer_10/other_repos/cuda-kernels/sum-reduction/`
- `/data/transformer_10/other_repos/cuda-kernels/sgemm-tiled/`

What it teaches well:

- basic reductions
- prefix-sum structure
- shared-memory tiling and coalescing
- small standalone CUDA project layout

What to take into `transformer_10`:

- only low-level didactic patterns

What not to copy blindly:

- not a production-quality transformer runtime reference
- not a reason to handwrite transformer GEMMs instead of using `cuBLASLt`

Direct mapping:

- reduction helpers
- scan-style utilities
- educational onboarding for contributors

## `good-kernels`

Best paths:

- `/data/transformer_10/other_repos/good-kernels/LayerNorm/src.py`
- `/data/transformer_10/other_repos/good-kernels/LayerNorm2/src.py`
- `/data/transformer_10/other_repos/good-kernels/Softmax/src.py`
- `/data/transformer_10/other_repos/good-kernels/MatmulFP32/src.py`

What it teaches well:

- compact reference implementations for a few standard kernels
- quick comparison between reference Python and generated kernel structure

What to take into `transformer_10`:

- quick inspection material for simple kernel decomposition

What not to copy blindly:

- not sufficient as a performance reference
- not sufficient as a runtime design reference

Direct mapping:

- quick sanity checks for LayerNorm/Softmax kernel structure

## How To Use This Corpus In The Migration

Recommended usage order by subsystem:

- attention and softmax:
  - first `TensorRT-LLM` and `TransformerEngine`
  - then `flash-attention`
  - then `ThunderKittens` for architecture-specific tuning ideas
- norms:
  - first `TransformerEngine`
  - then `flash-attention`
  - then `ThunderKittens` and `good-kernels` for alternative decomposition ideas
- runtime wrappers, graph capture, and small C++ utilities:
  - first `tiny-cuda-nn`
  - then `tinygrad`
  - then `cuda_ext`
- Python bridge during transition:
  - first `extension-cpp`
  - then `cuda_ext`

## Bottom Line

`other_repos` materially improves coverage, but not evenly.

The highest-value additions are:

- `flash-attention` for real attention implementation structure
- `ThunderKittens` for kernel construction patterns
- `tiny-cuda-nn` for runtime utilities
- `cuda_ext` and `extension-cpp` for boundary design

The smaller repos are best treated as didactic supplements rather than architecture drivers.
