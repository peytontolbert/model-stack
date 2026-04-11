# transformer_10 End-to-End C++/CUDA Migration Runbook

This document is the start-to-finish runbook for replacing PyTorch execution ownership in `transformer_10` with repository-owned C++ and CUDA code while keeping Python as a surface layer where it still helps usability.

The rule is simple:

- Python scripts may remain
- Python may remain the easiest user-facing interface
- PyTorch may remain as a temporary reference and compatibility path
- execution, runtime state, and model-stack ownership must move into C++/CUDA

That means the target is not "delete Python." The target is "Python stops being the implementation substrate."

## 1. End State

The intended final architecture is:

- C++ owns:
  - model-stack objects
  - config resolution
  - model/block/module construction
  - execution planning
  - serving engine state
  - cache ownership
  - checkpointing
  - data pipeline runtime
  - distributed runtime
  - training runtime
  - autotune and benchmark orchestration
- CUDA owns:
  - custom kernels
  - fused kernels
  - vendor-library wrappers
  - graph-safe launch paths
  - stream/workspace/allocator-aware execution
- Python owns only:
  - convenience APIs
  - CLI and scripting
  - compatibility wrappers
  - reference implementations
  - tests and parity harness entrypoints

The repository should still feel like one model-stack repository. Users should be able to configure, load, run, serve, train, benchmark, and validate models from this repo, but the underlying implementation should be ours rather than eager torch.

## 2. What This Migration Is Not

This migration is not:

- one monolithic kernel
- a pile of disconnected custom ops
- "rewrite every helper in CUDA immediately"
- "replace cuBLASLt with hand-written GEMM kernels"
- "delete Python first and figure the rest out later"

The correct target is a modular C++/CUDA model-stack with a narrow Python extension boundary.

## 3. Layered Architecture

The migration should produce four layers.

## Layer A: CUDA Execution Layer

Owns:

- handwritten kernels for norms, RoPE, cache updates, sampling, layout transforms, small reductions, and bandwidth-bound operations
- wrappers over `cuBLASLt`, `cuTENSOR`, NCCL, and other vendor components
- graph-safe launch functions
- device allocation and workspace helpers

This layer must not know about Python policy or user-facing CLI behavior.

## Layer B: C++ Runtime Layer

Owns:

- tensor descriptors
- typed execution contexts
- model/block/module classes
- serving engine
- tokenizer/data/checkpoint adapters
- training runtime and explicit backward contracts
- distributed runtime
- autotune plan selection

This is the real implementation owner for the repository.

## Layer C: Python Extension Layer

Owns:

- pybind11 or a thin CPython extension boundary
- opaque handles exposed to Python
- structured config marshaling between Python and C++
- minimal tensor exchange adapters

This layer exists so Python scripts can remain without continuing to own execution.

## Layer D: Python Surface Layer

Owns:

- `serve/api.py`
- convenience generation helpers
- CLI entrypoints
- debug scripts
- parity scripts
- migration-era reference wrappers

This layer should call the extension API, not re-implement math or runtime state.

## 4. Required Repository Shape

Recommended long-term source layout:

```text
runtime/
  cpp/
    include/t10/
    src/
  cuda/
    include/t10_cuda/
    src/
    tests/
    bench/
  python/
    bindings/
      module.cpp
      config_bindings.cpp
      tensor_bindings.cpp
      runtime_bindings.cpp
      serve_bindings.cpp
      train_bindings.cpp
      checkpoint_bindings.cpp
      dist_bindings.cpp
```

Recommended Python package shape:

```text
transformer_10/
  _ext/
    __init__.py
    runtime.py
    tensor.py
    serve.py
    train.py
    checkpoint.py
    dist.py
  model/
  blocks/
  attn/
  serve/
  train/
  eval/
```

Rules:

- `runtime/cpp` and `runtime/cuda` are authoritative
- `transformer_10/_ext` is the narrow Python bridge
- existing Python packages become orchestration and compatibility layers over `_ext`
- no new hot-path logic should be added above `_ext`

## 5. Python Extension API Boundary

The Python extension API is required. It is the correct way to keep Python surfaces while removing PyTorch as the implementation owner.

## 5.1 Binding Style

Preferred binding strategy:

- use `pybind11` for the main bridge
- expose opaque handle classes, not raw pointer tuples
- keep configuration marshaling explicit
- keep tensor ownership explicit
- avoid per-op Python dispatch logic in the hot path

Acceptable fallback:

- a thin C ABI wrapped by Python if pybind complexity becomes a blocker for some subsystems

## 5.2 Python-Visible Objects

The Python layer should expose objects like:

- `RuntimeConfig`
- `ModelHandle`
- `BlockHandle`
- `TensorHandle`
- `TensorView`
- `CheckpointHandle`
- `TokenizerHandle`
- `ServingEngine`
- `GenerationSession`
- `TrainSession`
- `DistSession`
- `AutotuneSession`

Python should not directly own:

- authoritative KV cache objects
- launch-time workspace policy
- algorithm-selection state
- graph capture state
- distributed communicator state

Those should be C++ objects behind handles.

## 5.3 Tensor Exchange Rules

The extension boundary needs clear tensor rules.

Required support:

- runtime-owned tensors exposed as opaque Python objects
- DLPack import/export for interop
- pinned host buffers for IO paths
- optional NumPy views for CPU-side inspection when safe

Transitional support only:

- accepting `torch.Tensor` through a compatibility adapter

Important rule:

- `torch.Tensor` may be accepted during transition
- `torch.Tensor` must not be the required runtime tensor type
- internal execution paths should convert once at the boundary, not route through torch ops internally

## 5.4 Suggested Python API Shape

Example high-level surface:

```python
from transformer_10._ext.runtime import RuntimeConfig, create_model
from transformer_10._ext.serve import ServingEngine
from transformer_10._ext.checkpoint import load_checkpoint

cfg = RuntimeConfig.from_dict(user_cfg)
model = create_model(cfg)
load_checkpoint(model, "/path/to/model.safetensors")

engine = ServingEngine(model, cfg.serving)
session = engine.start_session()
tokens = session.generate(prompt_ids=[1, 2, 3], max_new_tokens=64)
```

That is acceptable because Python is issuing commands to extension-backed objects rather than performing the implementation itself.

## 6. C++/CUDA Ownership Rules

Use these rules consistently.

## 6.1 Must Move Into CUDA/C++

- embedding lookup
- RMSNorm and LayerNorm
- RoPE application
- QKV/O/MLP/LM head execution
- attention prefill and decode
- KV cache append/read/evict/reorder
- sampler
- collectives
- training loss hot paths
- optimizer hot paths
- quant/dequant runtime paths

## 6.2 May Remain Python Surface Only

- CLI parsing
- debug scripts
- example scripts
- high-level experiment harnesses
- lightweight server wrappers
- compatibility adapters

## 6.3 May Remain Python Reference Only

- eager reference implementations for parity
- HF/PyTorch comparison scripts
- migration-era fallback paths guarded as non-authoritative

## 7. Operator Ownership Strategy

Not every operator should become a handwritten CUDA kernel.

Use:

- handwritten CUDA for:
  - norms
  - RoPE
  - sampling
  - cache operations
  - mask and reduction utilities
  - layout and pack/unpack transforms
  - fused residual/dropout/norm paths
- `cuBLASLt` for:
  - QKV projections
  - output projection
  - MLP up/gate/down
  - LM head
- NCCL for:
  - tensor, sequence, context, and data-parallel collectives
- Triton or fuser paths only when:
  - they are explicitly chosen as transitional or maintained codegen lanes
  - they sit behind the same runtime planner contract

The C++ runtime should own the dispatch decision either way.

## 8. Start-To-Finish Migration Sequence

The migration should be executed in phases. Do not start by scattering kernel files into the tree without the runtime boundary.

## Phase 0: Freeze The Target Contracts

Required outcomes:

- module target-state matrix is accepted as authoritative
- C++ runtime namespaces and objects are fixed
- Python extension API shape is fixed
- tensor descriptor conventions are fixed
- build system direction is fixed

Exit condition:

- new kernels can be added without re-deciding the public runtime boundary

## Phase 1: Build The Runtime Skeleton

Implement first:

- `Status`, `Expected<T>`, `TensorDesc`, `TensorView`, `DeviceBuffer`, `Workspace`
- stream and allocator wrappers
- runtime logging and profiling hooks
- pybind module bootstrap
- basic config marshaling

Exit condition:

- Python can create runtime objects and call no-op or reference-backed runtime entrypoints through `_ext`

## Phase 2: Replace Core Tensor Math

Implement:

- RMSNorm
- LayerNorm
- RoPE apply
- embedding
- sampler
- layout helpers
- cast/pack/unpack helpers

Use:

- custom CUDA kernels for norms, RoPE, sampling, and transforms
- library-backed paths where appropriate

Exit condition:

- `tensor/norms.py`, `tensor/positional.py`, and `tensor/sampling.py` stop owning hot-path math

## Phase 3: Replace GEMM-Centered Paths

Implement:

- `Linear`
- `QuantizedLinear`
- QKV projection wrappers
- MLP up/gate/down wrappers
- LM head wrapper

Use:

- `cuBLASLt` plan management in C++
- optional epilog fusion
- autotuned algorithm persistence

Exit condition:

- core linear layers no longer execute through `nn.Linear` on the intended runtime path

## Phase 4: Replace Attention And Cache Ownership

Implement:

- prefill attention
- decode attention
- paged KV cache
- append/read/evict/reorder
- mask policies
- MHA/MQA/GQA layout policies

Exit condition:

- `attn/eager.py` and `attn/kv_cache.py` become wrappers over runtime objects
- Python no longer owns authoritative cache state

## Phase 5: Rewire Blocks And Models

Implement:

- `t10::blocks::*`
- `t10::model::*`
- model factory
- block registry
- stack execution
- generate-path integration

Exit condition:

- `model/causal.py` and `blocks/transformer_block.py` are composition surfaces over runtime-backed modules

## Phase 6: Replace Serving Runtime

Implement:

- `ServingEngine`
- request scheduler
- generation session state
- graph bucket manager
- cache manager
- streaming output interface
- safety hook interface

Exit condition:

- `serve/engine.py` no longer performs decode math or owns serving cache state

## Phase 7: Replace Data, Checkpoint, And Tokenizer Ownership

Implement:

- checkpoint reader/writer
- HF import adapters
- tokenizer implementations and adapters
- data loader runtime
- distributed loader integration

Exit condition:

- Python may initiate loading, but authoritative loading and tokenization logic lives in C++

## Phase 8: Replace Training And Distributed Runtime

Implement:

- explicit backward-capable modules
- optimizer runtime
- activation checkpointing
- RNG policy
- NCCL communicator/runtime
- tensor/sequence/context parallel operations

Exit condition:

- intended training runtime no longer requires PyTorch autograd or `torch.distributed` for migrated paths

## Phase 9: Replace Compression, Eval, And Autotune Ownership

Implement:

- quantization runtime
- LoRA runtime
- pruning and distillation support
- parity and benchmark runners
- autotune search and persisted plans

Exit condition:

- the repo’s model-stack workflows no longer depend on PyTorch as the hidden runtime owner

## Phase 10: Remove Hot-Path PyTorch Ownership

Only after all prior phases:

- downgrade torch paths to reference-only or compatibility-only
- remove torch-only dispatch from intended runtime path
- enforce runtime-backed default execution

Exit condition:

- PyTorch is no longer authoritative for serving or the intended model execution path

## 9. File-Level Rewrite Pattern

Every existing Python module should migrate using one of four patterns.

## Pattern A: Wrapper

Keep the Python file, but make it call `_ext`.

Examples:

- `model/generate.py`
- `serve/generate.py`
- `serve/api.py`

## Pattern B: Reference

Keep the Python file for parity and tests only.

Examples:

- `attn/reference.py`
- HF parity scripts
- eager fallback helpers

## Pattern C: Replace With C++ Runtime Ownership

Move the module’s real logic into C++/CUDA and keep only a small Python shim if needed.

Examples:

- `model/causal.py`
- `blocks/transformer_block.py`
- `attn/eager.py`
- `attn/kv_cache.py`
- `serve/engine.py`
- `dist/engine.py`

## Pattern D: Remove Or Merge

Delete the file or merge it into a better-owned subsystem once the runtime exists.

Examples:

- legacy kernel registry wrappers
- duplicate compile helpers
- torch-only utility shims

## 10. Python Surface Rules During Migration

These rules prevent the repo from silently staying PyTorch-centered.

- Python may construct config objects and call extension handles
- Python may not remain the owner of cache mutation on the intended path
- Python may not perform authoritative tensor concatenation for serving
- Python may not perform real decode-step math on the intended path
- Python may not select kernels by importing a different eager backend directly
- Python may retain fallback and reference paths, but those must be clearly marked non-authoritative

If a Python file still performs the real tensor math for the intended runtime path, migration is not complete.

## 11. Build And Packaging Requirements

The build must support both direct runtime development and Python extension use.

Required:

- CMake as the authoritative build system
- shared libraries for runtime components
- one Python extension module or a small set of extension modules bound over those libraries
- explicit CUDA architecture configuration
- clear separation between runtime tests and Python tests

Recommended outputs:

- `libt10_core.so`
- `libt10_cuda.so`
- `libt10_serve.so`
- `libt10_train.so`
- `transformer_10/_ext/_runtime*.so`

The Python wheel should package bindings over repository-owned shared libraries rather than hide all implementation in a PyTorch custom-op package.

## 12. Validation Requirements

Each phase needs its own validation before moving on.

Required test lanes:

- per-op parity tests
- block parity tests
- model parity tests
- cache-state parity tests
- decode-step regression tests
- serving throughput and latency benchmarks
- graph-capture tests
- allocator/workspace reuse tests
- distributed correctness tests
- checkpoint round-trip tests

Required comparison baselines:

- current Python/PyTorch implementation
- known-good checkpoints
- deterministic seed-based regression cases

## 13. Completion Gates

The migration is only real when these are true.

## Gate 1: Serving Hot Path Is Runtime-Owned

- no eager torch math for embeddings, norm, linear, attention, cache, or sampling
- decode loop delegates to C++ runtime objects

## Gate 2: Model Construction Is Runtime-Owned

- model and block objects exist as C++ objects
- Python model classes are wrappers only

## Gate 3: Cache And Scheduling Are Runtime-Owned

- Python does not own the authoritative KV cache
- request scheduling and execution plan state live in C++

## Gate 4: Training And Distributed Paths Have Owned Runtime Contracts

- migrated training paths do not rely on PyTorch autograd internally
- intended distributed path does not require `torch.distributed`

## Gate 5: Python Is Optional For Core Serving

- direct runtime execution is possible without Python
- Python remains a surface, not a hidden dependency

## 14. Immediate Implementation Order

If code work starts now, the correct first build-out is:

1. runtime skeleton and extension bootstrap
2. tensor descriptors, allocator, workspace, and stream wrappers
3. RMSNorm, RoPE, embedding, sampler
4. `cuBLASLt` linear wrappers
5. attention and paged KV cache
6. block/model runtime rewiring
7. serving engine
8. checkpoint/data/tokenizer runtime
9. distributed and training runtime
10. compression, eval, and autotune integration

That order preserves momentum while preventing the repo from becoming a collection of unrelated extension entrypoints.

## 15. Definition Of Done

The migration counts as complete when:

- the intended runtime path is C++/CUDA-owned end to end
- Python surface scripts use extension-backed runtime objects
- PyTorch is not the implementation owner for serving, model execution, cache, or migrated training paths
- remaining PyTorch code is clearly compatibility-only, reference-only, or intentionally deferred

At that point, the repository is still a Python-friendly model stack, but it is no longer a PyTorch-centered one.
