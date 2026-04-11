# transformer_10 Torch-to-CUDA Migration Plan

This document ties the local codebase to a realistic replacement plan.

The current repo is not just "using torch". It is architected around torch tensors, torch modules, and torch-distributed control flow. Replacing that cleanly requires choosing the right boundary.

## Recommended Boundary

Do not try to replace Python orchestration, model config, and checkpoint logic in the first step.

Recommended first boundary:

- Keep Python for orchestration.
- Replace execution-heavy block internals with a C++/CUDA runtime library.
- Expose a narrow operator/runtime API through pybind11 or a thin C ABI.

That gives us:

- a stable place to land kernels quickly
- the ability to validate against existing eager torch implementations
- a path to a later pure C++ inference runtime if we want it

## Local Modules That Matter Most

### 1. `attn/eager.py`

This is the most important replacement target.

Today it does all of the following in Python/torch:

- Q/K/V linear projections
- head splitting and merge
- RoPE application
- cache read/append and concatenation
- GQA head expansion
- additive mask handling
- SDPA backend dispatch
- output projection

This file is effectively a mini runtime already. It should become a thin wrapper over a CUDA/C++ attention runner.

### 2. `blocks/transformer_block.py`

This is the second major boundary.

It composes:

- prenorm/postnorm policy
- attention call
- residual add/dropout
- MLP
- optional position bias paths

A future fused block runtime can reduce launch count and intermediate traffic here, but phase 1 should still expose these as separate kernels/operators.

### 3. `tensor/norms.py`

Direct replacement candidates:

- `RMSNorm.forward`
- `rmsnorm`
- `layer_norm`
- eventually `masked_rmsnorm`

These are reduction-heavy and memory-bandwidth-sensitive. Good custom-kernel targets.

### 4. `tensor/positional.py`

Direct replacement candidates:

- `build_rope_cache`
- `apply_rotary`
- eventually 2D and scaled variants

The important production target is not the cache builder. It is the application path, ideally fused into Q/K handling or attention prep.

### 5. `tensor/mlp.py`

This should not become "all handwritten CUDA".

Use:

- cuBLASLt for `w_in` and `w_out`
- fused bias/activation epilog where possible
- custom kernels only for residual glue or unsupported epilogs

The gated activations:

- `swiglu`
- `geglu`
- `reglu`

should be treated as fusion opportunities, not as standalone eager torch code.

### 6. `model/causal.py`

Primary replacements:

- embedding lookup
- final norm
- lm_head GEMM
- generate/decode runtime

The decode loop should later move toward:

- persistent stream usage
- memory-pool-backed temporaries
- CUDA Graph replay for stable shapes

### 7. `tensor/shard.py`

Everything here is a placeholder layer around `torch.distributed`.

If the target is direct CUDA/C++, this becomes:

- NCCL-backed all-reduce
- reduce-scatter
- all-gather
- sequence/context parallel exchange

## First Operator Backlog

These are the first operators worth building because they unlock most of the transformer forward path.

### Tier 1: mandatory

1. RMSNorm forward
2. RoPE apply for Q/K
3. KV cache append/update
4. QKV projection GEMM
5. output projection GEMM
6. MLP up/gate projection GEMM
7. SwiGLU activation/fusion
8. MLP down projection GEMM
9. masked softmax or fused attention
10. embedding lookup

### Tier 2: next

1. residual add + dropout fusion
2. logits projection and sampler helpers
3. transpose/cast/layout-transform kernels
4. quantize/dequantize helpers
5. vocab-parallel cross-entropy

### Tier 3: distributed

1. NCCL all-reduce
2. NCCL reduce-scatter
3. NCCL all-gather
4. context-parallel KV exchange
5. overlap of communication with GEMM or attention

## What Should Be Handwritten vs Library-Backed

### Use cuBLASLt

For:

- `nn.Linear` replacements
- QKV projections
- output projection
- MLP GEMMs
- lm_head

Reason:

- These are GEMM-dominated.
- cuBLASLt already handles the hard part: kernel selection, Tensor Core use, epilog support, and low-precision variants.
- TransformerEngine shows this clearly in `common/gemm/cublaslt_gemm.cu`.

### Use custom CUDA kernels

For:

- RMSNorm
- RoPE
- KV cache append/reorder
- masked softmax / attention helpers
- sampling
- embedding lookup
- layout transforms and fused residual paths

Reason:

- These are not generic GEMMs.
- They are usually memory-bound, reduction-heavy, or layout-sensitive.

### Use NCCL

For:

- tensor parallel reduction/gather
- sequence/context-parallel exchange
- collective overlap

Reason:

- Rebuilding collectives is wasted time and usually worse.

## Phased Migration

## Phase 0: infrastructure

Deliverables:

- `runtime/cuda` C++/CUDA library skeleton
- CMake/Ninja build
- pybind11 or C ABI bridge
- unit tests and microbench harness

Exit criteria:

- can call one CUDA op from Python
- can benchmark and compare against torch baseline

## Phase 1: single-GPU transformer forward

Deliverables:

- RMSNorm kernel
- RoPE kernel
- cuBLASLt GEMM wrappers
- KV cache append/update
- attention path
- embedding path

Exit criteria:

- a forward pass for `model/causal.py` can run without torch eager math for the hot path
- numerical parity passes on BF16/FP16/FP32 targets

## Phase 2: serving runtime

Deliverables:

- stream-ordered allocator usage
- decode-loop stream discipline
- CUDA Graph capture/replay for stable batch and shape buckets
- sampling kernels

Exit criteria:

- generate path is launch-efficient and uses reusable workspaces

## Phase 3: distributed runtime

Deliverables:

- NCCL-backed collectives
- TP and CP communication
- overlap design

Exit criteria:

- `tensor/shard.py` behavior is reproducible through direct runtime calls

## Phase 4: training and backward

Deliverables:

- backward kernels for norm/rope/attention where needed
- fused optimizer or optimizer-compatible tensor views
- activation checkpoint and workspace discipline

Exit criteria:

- training hot path no longer depends on PyTorch autograd for the migrated operators

## Local File-to-Work Mapping

| Local file | Current role | Replacement direction |
| --- | --- | --- |
| `attn/eager.py` | torch attention runtime | replace with `CudaAttentionRunner` |
| `tensor/norms.py` | eager norm math | replace with norm kernels |
| `tensor/positional.py` | eager rope math | replace with rope kernel or fuse into attention prep |
| `tensor/mlp.py` | eager linear + activation path | replace with cuBLASLt GEMM wrappers plus activation fusion |
| `model/causal.py` | embed/block/final norm/logits loop | call runtime ops instead of torch modules on the hot path |
| `tensor/shard.py` | `torch.distributed` wrappers | replace with NCCL runtime wrappers |

## Key Design Decisions

### Decision 1: inference-first

Reason:

- It avoids the autograd cliff.
- It lets us validate correctness against the current implementation quickly.

### Decision 2: operator runtime before full engine rewrite

Reason:

- The current repo structure already isolates kernels behind `kernel/`, `attn/`, and `tensor/`.
- We can exploit that architecture instead of replacing the whole stack at once.

### Decision 3: GEMMs via cuBLASLt, not custom kernels

Reason:

- The opportunity cost of handwriting GEMMs is too high.
- The real wins are in fusion, memory movement, and reduced launch count.

## Success Criteria

The migration is on the right track only if all of these become true:

1. The transformer forward hot path no longer depends on eager torch math.
2. Numerical parity against the current implementation is automated.
3. Benchmarks are per-op and end-to-end, not anecdotal.
4. The runtime API is narrow enough that model code does not become CUDA-specific everywhere.
5. Distributed behavior is designed explicitly rather than bolted on later.

## Immediate Next Build Targets

If implementation starts after this research pass, the first concrete milestone should be:

1. Build a CUDA runtime library with:
   - RMSNorm forward
   - RoPE apply
   - cuBLASLt GEMM wrapper
2. Integrate those three into the current block path behind a feature flag.
3. Measure parity and latency before touching fused attention.

That is the smallest step that materially reduces PyTorch dependency while keeping the project debuggable.
