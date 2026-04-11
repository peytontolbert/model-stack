# CUDA/C++ Research Index

This directory is a working research package for moving `transformer_10` from a mostly PyTorch/Python model-stack to a self-owned C++/CUDA model-stack.

Scope:

- Research is based on local repositories under `/data/parametergolf/helpful_repos/NVIDIA`.
- One adjacent non-NVIDIA repo is also tracked now because it materially affects kernel-authoring choices:
  - `/data/parametergolf/helpful_repos/statespace_101/triton`
- A second local corpus is also tracked for practical implementation patterns:
  - `/data/transformer_10/other_repos`
- The goal is not "learn CUDA in general". The goal is to identify what we need to replace the current torch-heavy model-stack implementation in this repo with reliable C++/CUDA components.
- CUDA is for GPU tensor execution. C++ is for runtime state, model construction, checkpointing, serving, training, distributed control, validation, and non-GPU model-stack systems.
- The intended product is still one repository where a user can configure, load, run, serve, train, quantize, distribute, benchmark, and validate models without depending on PyTorch as the implementation substrate.

## What Is Here

Corpus-first coverage:

- `nvidia-corpus-full-inventory.md`
  - Full top-level inventory for every repo under `/data/parametergolf/helpful_repos/NVIDIA`.
  - Includes the newly added `CUDALibrarySamples`, `Fuser`, and `TensorRT-LLM` repos.
  - Also tracks the adjacent local `triton` checkout separately.
  - Includes CUDA/C++/Python file-count signals, relevance classification, and notable paths.

- `nvidia-cuda-cpp-deep-dives.md`
  - Deeper notes for the repos with the strongest CUDA/C++ value.
  - This is the main "where do I study code?" document.

- `nvidia-kernel-reference-map.md`
  - A shorter task-oriented source map from NVIDIA repos to common model-stack needs.
  - Use this when looking for examples or production patterns after reading the corpus docs.

- `nvidia-full-migration-gap-audit.md`
  - Second-pass audit of what was still undercovered after the first corpus sweep.
  - This is the direct answer to "is the corpus documented enough to fully migrate yet?"

Standalone repo notes:

- `nvidia-cudalibrarysamples-standalone-notes.md`
  - Focused notes for `CUDALibrarySamples`.
  - Use this when deciding whether an op should stay library-backed instead of becoming a handwritten kernel.

- `nvidia-fuser-standalone-notes.md`
  - Focused notes for `Fuser`.
  - Use this when evaluating compiler-driven fusion for pointwise, reduction, norm, and SDPA-adjacent work.

- `nvidia-tensorrt-llm-standalone-notes.md`
  - Focused notes for `TensorRT-LLM`.
  - Use this when studying inference runtime structure, paged KV cache, scheduling, sampling, and serving-oriented attention paths.

- `triton-standalone-notes.md`
  - Focused notes for the adjacent local `triton` checkout.
  - Use this when deciding whether a kernel is a good Triton target instead of handwritten CUDA/C++.

Structured implementation docs:

- `other-repos-cuda-cpp-reference-map.md`
  - Structured notes for the local `/data/transformer_10/other_repos` corpus.
  - Covers what `ThunderKittens`, `flash-attention`, `tiny-cuda-nn`, `tinygrad`, `extension-cpp`, `cuda_ext`, `cuda-kernels`, and `good-kernels` are actually useful for in this migration.

- `transformer10-runtime-module-spec.md`
  - Target runtime/module architecture for replacing the current torch-heavy hot path.
  - Defines the proposed `runtime/cuda` tree, module ownership, ABI boundaries, canonical tensor layouts, and Python integration points.

- `transformer10-first-wave-kernel-spec.md`
  - First implementation-facing kernel spec.
  - Defines the first wave of kernels and wrappers, including recommended entrypoints, tensor contracts, validation targets, and best local references.

- `transformer10-full-model-stack-migration-blueprint.md`
  - Full-stack migration blueprint across `model/`, `blocks/`, `attn/`, `tensor/`, `serve/`, `dist/`, and `kernel/`.
  - Use this when planning the whole program, not just the first kernels.

- `transformer10-complete-model-stack-cuda-cpp-coverage.md`
  - Comprehensive coverage plan for converting the whole repository into a C++/CUDA model-stack.
  - Maps model configuration, tensor math, model layers, serving, training, data, checkpointing, compression, distributed, eval, autotune, and diagnostics.

- `transformer10-repository-coverage-audit.md`
  - Repo-wide audit of whether the current docs are enough to claim 100% migration coverage.
  - Tracks explicit coverage counts by package and lists modules still missing a target state.

- `transformer10-module-target-state-matrix.md`
  - One row per Python module with target state, target component, and migration priority.
  - This is the repository-wide target ledger for the migration.

- `transformer10-cpp-model-stack-api-spec.md`
  - Canonical C++ API boundary for the full model stack.
  - Defines runtime namespaces, object ownership, and the mapping from current Python modules to C++ objects.

- `transformer10-complete-tensor-math-function-inventory.md`
  - Function-level inventory for the tensor math surface.
  - Assigns every tensor responsibility to CUDA kernels, C++ runtime, library wrappers, or reference-only status.

- `transformer10-training-backward-cuda-spec.md`
  - Training and backward design for the owned C++/CUDA stack.
  - Covers explicit backward contracts, optimizer kernels, RNG, activation checkpointing, mixed precision, and distributed training.

- `transformer10-data-checkpoint-tokenizer-cpp-spec.md`
  - C++ ownership for tokenizers, shard readers, dataloaders, safetensors, checkpointing, and HF import.

- `transformer10-serving-engine-cpp-spec.md`
  - C++ serving runtime design.
  - Covers request lifecycle, scheduling, paged KV cache, sampler ownership, graph policy, and instrumentation.

- `transformer10-autotune-eval-benchmark-spec.md`
  - Owned autotune, parity, latency, memory, calibration, and benchmark systems.

- `transformer10-compression-quantization-runtime-spec.md`
  - Runtime design for quantization, LoRA, pruning, distillation, and compression deltas.

- `transformer10-noncore-systems-targets.md`
  - Explicit target-state classification for governance, corpus, interpretability, visualization, export, packaging, registry, RAG, RL, safety, examples, and legacy kernel wrappers.

- `transformer10-end-to-end-cpp-cuda-migration-runbook.md`
  - Start-to-finish migration runbook for replacing PyTorch execution with C++/CUDA runtime ownership while keeping Python as a thin extension-backed surface.
  - Defines the layered architecture, Python extension API boundary, phased migration order, and completion gates.

- `transformer10-full-repository-scope-closure.md`
  - Full-tree scope closure for the rest of the repository.
  - Explicitly classifies nested support packages, tests, examples, and `other_repos/` so migration does not get blocked by unowned subtrees outside the core module matrix.

- `transformer10-pytorch-decommission-checklist.md`
  - Explicit checklist for removing PyTorch from the hot path and later from the runtime boundary.
  - Use this to decide when a phase is actually complete.

Verification:

- `tools/verify_migration_doc_coverage.py`
  - Machine-checks that the required migration docs exist, the README indexes them, the core module matrix is valid, and the remaining Python files are covered by explicit subtree scope rules.
  - Run this before migration work to verify documentation coverage has not drifted.

- `transformer10-cuda-migration-plan.md`
  - Maps local `transformer_10` modules to concrete CUDA/C++ replacement work.
  - Defines priorities, phases, and the first kernel backlog.

- `transformer10-operator-decision-matrix.md`
  - Operator-by-operator decision matrix for this codebase.
  - Picks the primary path for each major op:
    - handwritten CUDA/C++
    - cuBLASLt/cuTENSOR
    - Fuser
    - Triton

- `kernel-implementation-playbook.md`
  - Build, runtime, testing, benchmarking, and rollout guidance.
  - Use this when actually implementing kernels and integrating them into the repo.

The migration docs remain here, but the intended reading order is now:

1. `nvidia-corpus-full-inventory.md`
2. `nvidia-cuda-cpp-deep-dives.md`
3. `nvidia-full-migration-gap-audit.md`
4. `nvidia-kernel-reference-map.md`
5. the standalone repo notes
6. `other-repos-cuda-cpp-reference-map.md`
7. `transformer10-runtime-module-spec.md`
8. `transformer10-first-wave-kernel-spec.md`
9. `transformer10-full-model-stack-migration-blueprint.md`
10. `transformer10-complete-model-stack-cuda-cpp-coverage.md`
11. `transformer10-repository-coverage-audit.md`
12. `transformer10-module-target-state-matrix.md`
13. `transformer10-cpp-model-stack-api-spec.md`
14. `transformer10-complete-tensor-math-function-inventory.md`
15. `transformer10-data-checkpoint-tokenizer-cpp-spec.md`
16. `transformer10-serving-engine-cpp-spec.md`
17. `transformer10-training-backward-cuda-spec.md`
18. `transformer10-autotune-eval-benchmark-spec.md`
19. `transformer10-compression-quantization-runtime-spec.md`
20. `transformer10-noncore-systems-targets.md`
21. `transformer10-end-to-end-cpp-cuda-migration-runbook.md`
22. `transformer10-full-repository-scope-closure.md`
23. `transformer10-pytorch-decommission-checklist.md`
24. then the remaining `transformer_10`-specific planning docs

## Current Repo Reality

The current model stack is still broadly Python/PyTorch-bound:

- `model/causal.py` owns embedding, block execution, final norm, and logits.
- `blocks/transformer_block.py` composes norm, attention, MLP, residual, and optional relative position bias.
- `attn/eager.py` performs QKV projections, RoPE application, KV cache concat, GQA expansion, masking, backend selection, and output projection.
- `tensor/norms.py`, `tensor/mlp.py`, and `tensor/positional.py` still implement important math in eager torch.
- `tensor/shard.py` still relies on `torch.distributed` wrappers for collectives.

That means replacing PyTorch/Python "directly" is not one task. It is at least eight:

1. Replace eager tensor math for transformer block internals.
2. Replace attention and KV-cache handling.
3. Replace GEMM-heavy linear layers with cuBLASLt-backed execution.
4. Replace distributed collectives and overlap paths with NCCL/CUDA-native paths.
5. Replace model construction/runtime state with C++ model-stack objects.
6. Replace serving/decode/generation control with runtime-owned C++ components.
7. Provide C++/CUDA equivalents for training, compression, checkpoint, data, eval, and autotune systems where they are part of the model-stack workflow.
8. Build a validation and benchmarking harness strong enough to catch numerical and performance regressions.

## Recommended Working Principle

Do not collapse the whole stack into one kernel and do not create disconnected one-off extensions.

Preserve the current modular model-stack shape, but move implementation ownership down into C++/CUDA modules:

- Python package today -> C++/CUDA module tomorrow
- Python class today -> C++ runtime/model-stack class tomorrow
- Python tensor function today -> CUDA kernel family or vendor-library wrapper tomorrow
- Python orchestration today -> C++ runtime orchestration tomorrow, with Python bindings kept as a convenience layer

Use the right layer for each operation:

- Use cuBLASLt for GEMMs and GEMM epilog fusion.
- Use NCCL for collectives.
- Use custom CUDA kernels for bandwidth-bound, reduction-heavy, or layout-transform ops:
  - RMSNorm
  - RoPE
  - KV cache append/update
  - masked softmax / attention utilities
  - sampling
  - embedding lookup
  - tensor transforms and fused residual/dropout/norm paths

This is the fastest path to "less PyTorch" without recreating a worse version of cuBLAS or NCCL ourselves.

## First Practical Outcome

If implementation started now, the recommended order would be:

1. CUDA/C++ runtime skeleton and benchmark harness.
2. RMSNorm, RoPE, KV cache append.
3. cuBLASLt-backed QKV/O/MLP GEMMs.
4. Attention kernel path.
5. Embedding and sampler kernels.
6. NCCL-backed tensor/context-parallel communication.

The rest of this directory explains why.
