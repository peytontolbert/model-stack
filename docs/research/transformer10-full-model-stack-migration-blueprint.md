# transformer_10 Full Model-Stack Migration Blueprint

This document is the full migration blueprint for moving `transformer_10` from a PyTorch/Python-centered model-stack to a self-owned C++/CUDA model-stack.

It is not the kernel spec. It is the program-level plan for migrating the whole repository as a model-stack:

- `model/`
- `blocks/`
- `attn/`
- `tensor/`
- `serve/`
- `dist/`
- `kernel/`
- `compress/`
- `data/`
- `train/`
- `eval/`
- `autotune/`
- `specs/`

## 1. Target End State

The intended end state is:

- the repository provides all necessary model-stack code itself
- C++ owns model-stack objects, runtime state, scheduling, serving, training, distributed control, checkpointing, and validation
- CUDA owns GPU tensor execution and fused kernel families
- Python remains an optional binding/convenience layer, not the implementation substrate
- PyTorch is no longer acceptable as the implementation owner for:
  - embedding
  - norms
  - RoPE
  - QKV/O/MLP/LM-head execution
  - attention
  - KV cache
  - sampler
  - collectives
  - model/layer execution
  - training losses and optimizer hot paths
  - quantization/dequantization runtime paths

That gives three progressively stricter milestones:

1. `Torch-light runtime`
   - Python still orchestrates, but hot ops execute in custom runtime code
2. `C++/CUDA model-stack runtime`
   - model construction, execution plans, cache, serving, and distributed runtime are owned by C++/CUDA
3. `Python-optional model stack`
   - users can still use Python bindings, but the repository's own C++/CUDA implementation is complete enough to run the stack directly

## 2. Current Stack Reality

The repo is currently split between model composition, helper math, serving glue, and distributed wrappers.

The most important torch-owned execution surfaces are:

- `/data/transformer_10/model/causal.py`
- `/data/transformer_10/blocks/transformer_block.py`
- `/data/transformer_10/attn/eager.py`
- `/data/transformer_10/attn/kv_cache.py`
- `/data/transformer_10/tensor/norms.py`
- `/data/transformer_10/tensor/positional.py`
- `/data/transformer_10/tensor/mlp.py`
- `/data/transformer_10/tensor/sampling.py`
- `/data/transformer_10/serve/engine.py`
- `/data/transformer_10/tensor/shard.py`
- `/data/transformer_10/dist/engine.py`

Those files collectively are the current runtime.

## 3. Migration Principle

Do not migrate by replacing Python file-by-file with ad hoc extensions.

Migrate by introducing a coherent C++/CUDA model-stack and pulling responsibilities into it in this order:

1. C++ model-stack foundation
2. CUDA tensor math primitives and vendor-library wrappers
3. stateful attention/KV/cache primitives
4. model/block/layer wiring
5. serving and generation engine
6. checkpoint/data/tokenization stack
7. collectives and distributed runtime
8. training, backward, optimizer, compression, eval, and autotune

If that order is reversed, the result will be an unstable mix of Python orchestration, custom extensions, and duplicated policy.

## 4. Workstreams

The full migration should run as eight workstreams.

## Workstream A: C++ Model-Stack Foundation

Owns:

- C++ config objects
- model/layer/block interfaces
- tensor descriptors
- weight metadata
- runtime handles
- module registry
- Python bindings as a convenience layer

Primary deliverables:

- C++ equivalents for `specs/`, `model/`, `blocks/`, `attn/`, and `tensor/` public contracts
- stable ABI/API for building model variants inside this repo

Success condition:

- a user can configure and instantiate the model-stack through repository-owned C++ objects, even if early execution still calls only a subset of CUDA kernels

## Workstream B: CUDA Runtime Foundation

Owns:

- `runtime/cuda` source tree
- build system
- typed tensor descriptors
- stream/workspace/allocator wrappers
- graph helpers
- pybind or C ABI boundary

Primary deliverables:

- stable runtime include tree
- stable launch API style
- benchmark/test harness
- graph-capturable runtime entrypoints

Success condition:

- kernels and wrappers can be landed without re-deciding ABI, build, or workspace policy

## Workstream C: Core Tensor Math Primitives

Owns:

- RMSNorm
- LayerNorm
- RoPE
- linear wrappers via `cuBLASLt`
- embedding
- sampler
- small layout/cast helpers

Primary deliverables:

- first-wave kernel set
- per-op parity and benchmark suite

Success condition:

- `tensor/` stops owning hot-path math

## Workstream D: Attention And KV Runtime

Owns:

- prefill attention
- decode attention
- KV cache
- GQA/MQA handling
- softmax and mask utilities
- attention-specialized layouts

Primary deliverables:

- runtime cache handle
- prefill and decode attention entrypoints
- serving-grade latency on incremental decode

Success condition:

- `attn/eager.py` and `attn/kv_cache.py` become wrappers rather than the real implementation

## Workstream E: Model And Block Rewiring

Owns:

- `model/`
- `blocks/`
- block execution boundary
- model forward path
- generate path integration

Primary deliverables:

- block-level integration with runtime ops
- model-forward path that no longer executes eager math
- runtime-backed generate/decode path

Success condition:

- `model/causal.py` and `blocks/transformer_block.py` become composition code only

## Workstream F: Serving Runtime

Owns:

- engine state
- request-time cache allocation
- graph bucket selection
- batching policy
- decode loop

Primary deliverables:

- runtime engine object
- graph-aware decode loop
- sliding-window or eviction policy in runtime

Success condition:

- `serve/engine.py` and `serve/runtime.py` no longer perform hot-path tensor math or cache ownership in Python

## Workstream G: Data, Checkpoint, Compression, Eval, And Autotune

Owns:

- tokenizer/data loader interfaces
- checkpoint and safetensor loading
- quantization/compression runtime paths
- evaluation metrics and benchmark harnesses
- autotune search and kernel-plan persistence

Primary deliverables:

- C++/CUDA-owned equivalents for model-stack workflows beyond forward execution

Success condition:

- a user can work inside this repository for model loading, quantization, benchmarking, and validation without PyTorch being the implementation owner

## Workstream H: Distributed And Training

Owns:

- collectives
- tensor/context/sequence parallelism
- backward path
- optimizer/runtime overlap later if needed

Primary deliverables:

- NCCL wrappers
- replacement for `tensor/shard.py` hot-path behavior
- distributed runtime integration

Success condition:

- `torch.distributed` is not required for the intended long-term runtime path

## 5. Package-By-Package Migration Map

## `tensor/`

Current role:

- owns many math helpers and eager reference implementations

Migration classification:

- `migrate now`:
  - `/data/transformer_10/tensor/norms.py`
  - `/data/transformer_10/tensor/positional.py`
  - `/data/transformer_10/tensor/mlp.py`
  - `/data/transformer_10/tensor/sampling.py`
  - `/data/transformer_10/tensor/shard.py`
- `keep as reference/test helpers for longer`:
  - numerics helpers
  - masking helpers
  - debug and export-safe helpers

Target state:

- `tensor/` becomes semantics/reference code, not execution ownership

## `attn/`

Current role:

- owns real attention execution, backend dispatch, and cache behavior

Migration classification:

- `migrate now`:
  - `/data/transformer_10/attn/eager.py`
  - `/data/transformer_10/attn/kv_cache.py`
  - `/data/transformer_10/attn/backends.py`
- `migrate later or keep as compatibility wrappers`:
  - `attn/reference.py`
  - optional backend wrappers like `attn/triton.py`, `attn/xformers.py`
- `training-side or auxiliary`:
  - `attn/moe.py`
  - `attn/optim_utils.py`

Target state:

- `attn/` chooses runtime entrypoints and preserves semantics
- it no longer performs actual tensor algebra on the hot path

## `blocks/`

Current role:

- owns transformer block composition and many optional attention variants

Migration classification:

- `migrate core first`:
  - `/data/transformer_10/blocks/transformer_block.py`
  - `/data/transformer_10/blocks/llama_block.py`
  - `/data/transformer_10/blocks/shared.py`
- `migrate optional variants second`:
  - local/prefix/banded/strided/window/segment-bidir/block-sparse blocks
  - cross-attention and MoE variants

Target state:

- block modules keep model semantics and policy
- runtime executes the math-heavy body

## `model/`

Current role:

- owns embedding, block execution, final norm, logits, generation utilities, export helpers, and some HF-loading glue

Migration classification:

- `migrate now`:
  - `/data/transformer_10/model/causal.py`
  - `/data/transformer_10/model/generate.py`
  - `/data/transformer_10/model/runtime_utils.py`
- `migrate later or keep torch-adjacent`:
  - `/data/transformer_10/model/encoder.py`
  - `/data/transformer_10/model/seq2seq.py`
  - `/data/transformer_10/model/heads.py`
- `keep torch-adjacent for compatibility`:
  - `/data/transformer_10/model/export.py`
  - `/data/transformer_10/model/compile.py`
  - `/data/transformer_10/model/hf_llama_loader.py`
  - `/data/transformer_10/model/hf_snapshot.py`

Target state:

- core LM execution becomes runtime-backed
- export and HF-interop code may remain torch-adjacent longer without blocking the runtime migration

## `serve/`

Current role:

- owns runtime object construction and generation loop behavior

Migration classification:

- `migrate now`:
  - `/data/transformer_10/serve/engine.py`
  - `/data/transformer_10/serve/runtime.py`
- `keep at API layer`:
  - `/data/transformer_10/serve/api.py`

Target state:

- serving API remains in Python if desired
- execution engine and cache manager move to runtime

## `dist/`

Current role:

- wraps DDP/FSDP/DeepSpeed and distributed setup

Migration classification:

- `migrate hot-path parts`:
  - `/data/transformer_10/dist/engine.py`
  - `/data/transformer_10/dist/parallel/tensor_parallel.py`
  - `/data/transformer_10/tensor/shard.py`
- `keep temporarily if they remain training-only wrappers`:
  - DDP/FSDP/DeepSpeed strategy wrappers

Target state:

- serving and intended direct runtime do not depend on `torch.distributed`
- training wrappers may remain until a later training-runtime phase

## `kernel/`

Current role:

- optional backend utilities and wrappers

Migration classification:

- `consolidate`
  - move runtime-critical kernel selection into `runtime/cuda`
  - keep `kernel/` only if it remains a lightweight compatibility surface

Target state:

- the authoritative implementation lives under `runtime/cuda`, not `kernel/`

## 6. Phased Program

## Phase 0: Stabilize The Runtime Boundary

Deliverables:

- `runtime/cuda` tree
- build/test/bench harness
- stable tensor descriptors
- runtime handle types

Exit criteria:

- at least one runtime op can be called from Python with clean validation and tests

## Phase 1: Replace Tensor Math

Deliverables:

- RMSNorm
- RoPE
- linear wrappers
- SwiGLU
- embedding
- sampling helpers

Exit criteria:

- `tensor/norms.py`, `tensor/positional.py`, `tensor/mlp.py`, and `tensor/sampling.py` are no longer the hot-path implementation

## Phase 2: Replace Attention And Cache

Deliverables:

- runtime KV cache
- prefill attention
- decode attention
- GQA/MQA handling

Exit criteria:

- `attn/eager.py` and `attn/kv_cache.py` are wrappers only

## Phase 3: Rewire Model And Blocks

Deliverables:

- runtime-backed `CausalLM` forward path
- runtime-backed block execution
- runtime-backed generate loop

Exit criteria:

- `model/causal.py` and `blocks/transformer_block.py` do not execute eager math on the hot path

## Phase 4: Rebuild Serving Runtime

Deliverables:

- runtime engine state
- graph capture/replay
- cache manager in runtime
- decode loop no longer concatenates torch tensors every step

Exit criteria:

- `serve/engine.py` is orchestration only

## Phase 5: Replace Collectives

Deliverables:

- NCCL wrappers
- tensor parallel collectives
- sequence/context exchange

Exit criteria:

- hot path no longer depends on `torch.distributed`

## Phase 6: Training And Backward

Deliverables:

- backward coverage for needed ops
- training-side fused paths where justified
- distributed training integration as needed

Exit criteria:

- training path has a clear runtime-backed story rather than relying on eager fallback everywhere

## 7. What Stays Torch-Adjacent The Longest

These areas should not block the serving migration:

- checkpoint loading
- Hugging Face import/export glue
- ONNX and TorchScript export utilities
- training wrappers for DDP/FSDP/DeepSpeed
- optional model families beyond the primary causal LM path

These are adjacent, but they are not the core hot path.

## 8. Validation Gates

The full-stack program needs four validation gates.

## Gate A: Per-Op Parity

For every runtime primitive:

- FP32 reference parity
- BF16/FP16 parity
- odd-shape tests
- error-path validation

## Gate B: Block Parity

For transformer blocks:

- same inputs, same mask, same cache state
- parity on outputs before and after residual paths

## Gate C: Model Parity

For model forward and decode:

- fixed seeds
- same token outputs
- same logits within tolerance
- same cache behavior

## Gate D: Serving Metrics

For serving runtime:

- latency improvement or at least parity
- fewer allocations
- graph replay works for stable buckets
- decode path no longer regresses with sequence growth due to Python-managed cache handling

## 9. Migration Risks

Main risks:

- rebuilding a second runtime in Python accidentally
- letting every op invent its own layout
- making GEMMs custom when libraries are better
- moving to runtime code without a parity harness
- replacing serving math while leaving Python-owned cache and decode structure untouched

The biggest structural risk is partial migration:

- custom norm kernels
- custom RoPE
- custom sampler
- but attention, cache, and decode loop still fundamentally torch-owned

That produces complexity without actually removing the bottleneck.

## 10. Immediate Implementation Backlog

If work starts now, the highest-value sequence is:

1. land runtime skeleton
2. land RMSNorm
3. land RoPE
4. land `cuBLASLt` linears
5. land KV cache
6. land prefill attention
7. land decode attention
8. rewire `CausalLM`
9. rewire `serve/engine.py`
10. land NCCL collectives

## 11. Bottom Line

The whole-stack migration is successful only when:

- `tensor/` no longer owns hot-path math
- `attn/` no longer owns hot-path execution
- `model/` and `blocks/` become composition layers
- `serve/` no longer owns cache and decode math
- `dist/` no longer supplies the intended long-term runtime collectives

Until then, the repo is still fundamentally PyTorch-centered even if some kernels have been added.
