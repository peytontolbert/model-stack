# transformer_10 PyTorch Decommission Checklist

This document defines what "migrated from PyTorch" means for `transformer_10`.

Without a checklist, it is too easy to land a few custom kernels and still keep the real runtime in eager torch.

Use this document as the completion gate for each migration phase.

## 1. Decommission Levels

There are three useful levels.

## Level 1: PyTorch No Longer Owns The Serving Hot Path

Required:

- no eager torch math for:
  - norms
  - RoPE
  - embedding
  - QKV/O/MLP/LM-head execution
  - attention
  - KV cache
  - sampler
- `serve/engine.py` does not perform per-step tensor concatenation and cache updates as the real implementation
- runtime ops are benchmarked and parity-tested

Allowed:

- Python orchestration
- checkpoint loading
- model config
- compatibility wrappers

## Level 2: PyTorch Is Only A Compatibility Layer

Required:

- all long-term runtime APIs are defined by self-owned CUDA/C++ code
- torch-specific backend dispatchers are gone from the hot path
- `torch.distributed` is not required for the intended runtime path
- PyTorch modules in `model/` and `blocks/` are wrappers only

Allowed:

- export helpers
- HF loader utilities
- training-only wrappers

## Level 3: Python And PyTorch Are Optional For Serving

Required:

- serving engine can run through a direct runtime interface
- cache manager, graph manager, and execution plan are runtime-owned
- Python API is optional rather than mandatory

This level is optional. Level 1 is the minimum real migration target.

## 2. File-Level Checklist

Mark a file complete only when its hot-path responsibilities have genuinely moved.

## `model/causal.py`

Complete when:

- embedding does not execute via `nn.Embedding` on the intended hot path
- final norm is runtime-backed
- logits projection is runtime-backed
- block loop delegates to runtime-backed blocks or runtime-backed ops
- generation path no longer relies on the eager full-recompute loop for the intended runtime

Not complete if:

- only attention changed but embedding, norm, and logits are still eager

## `blocks/transformer_block.py`

Complete when:

- prenorm/postnorm path calls runtime-backed norm
- attention call is runtime-backed
- MLP call is runtime-backed
- residual/dropout glue is either runtime-backed or intentionally left outside the hot path

Not complete if:

- the block still performs major tensor math through eager torch modules

## `attn/eager.py`

Complete when:

- it no longer computes QKV, RoPE, cache concat, head expansion, softmax, and output projection itself
- it becomes a wrapper that builds descriptors and calls runtime attention entrypoints

Not complete if:

- it still contains the real implementation with only some helper kernels offloaded

## `attn/kv_cache.py`

Complete when:

- Python lists of K/V pages are no longer the serving cache implementation
- append/read/evict are runtime-backed
- cache lifetime is managed by runtime handles

Not complete if:

- runtime kernels exist but Python still owns the authoritative cache state

## `tensor/norms.py`

Complete when:

- RMSNorm and LayerNorm hot-path calls dispatch into runtime
- file remains only as semantics reference and fallback

## `tensor/positional.py`

Complete when:

- `apply_rotary` hot path is runtime-backed
- file remains only as a compatibility shim over `runtime/positional.py`
- optional cache builders may remain if they are not performance-critical

## `tensor/mlp.py`

Complete when:

- `w_in` and `w_out` are runtime-backed linears
- gated activation is runtime-backed
- eager MLP is no longer the intended serving path

## `tensor/sampling.py`

Complete when:

- serving sampler is runtime-backed for temperature, token counting, penalties, top-k/top-p masking, and next-token selection on the intended path
- file remains only as a compatibility shim over `runtime/sampling.py`

## `serve/engine.py`

Complete when:

- it does not own the real decode loop math
- it does not append to Python-managed cache as the real implementation
- it does not recompute logits selection policies through eager torch in the intended runtime
- it delegates to runtime engine operations

## `serve/runtime.py`

Complete when:

- runtime object construction yields runtime handles, not a torch-model-plus-python-cache bundle
- model-directory and factory-spec config/build/load flow is runtime-owned rather than duplicated in serving or eval callers
- cache allocation is runtime-backed
- request/config coercion and serving health payloads are runtime-backed

## `tensor/shard.py`

Complete when:

- intended collectives route through NCCL-backed runtime calls
- tensor/sequence/context parallel helpers no longer rely on `torch.distributed` for the long-term path

## `dist/engine.py`

Complete when:

- long-term runtime does not depend on DDP/FSDP/DeepSpeed wrappers
- remaining file usage is explicitly training-only or compatibility-only

## `kernel/`

Complete when:

- runtime-critical implementation ownership has moved under `runtime/cuda`
- `kernel/` is either removed, reduced to compatibility wrappers, or clearly non-authoritative

## 3. Package-Level Completion Gates

## `tensor/` is decommissioned from hot path when:

- runtime owns norms
- runtime owns RoPE apply
- runtime owns sampler
- runtime owns collectives-facing execution helpers that matter to serving/training hot paths

Remaining runtime-adjacent `tensor/` ownership to remove:

- `tensor/masking.py` still backs parts of runtime/block mask preparation
- `tensor/norms.py` still owns the authoritative `RMSNorm` module type
- `tensor/mlp.py` still owns the eager MLP module used by block/model construction

## `attn/` is decommissioned from hot path when:

- runtime owns attention
- runtime owns cache
- runtime owns mask/layout-sensitive core logic

## `blocks/` is decommissioned from hot path when:

- runtime owns block-stack orchestration
- runtime owns shared mask shaping for encoder, cross-attention, and patterned self-attention block paths
- runtime owns shared attention-bias composition for block forward paths
- runtime owns fused residual/norm helpers used by block forward paths
- blocks no longer perform heavy eager tensor math internally

## `model/` is decommissioned from hot path when:

- it composes runtime-backed primitives instead of owning execution-heavy modules
- model-dir load/build entrypoints used by serving and eval have moved behind runtime-owned loaders or compatibility shims
- HF snapshot/bootstrap/import helpers have moved behind runtime-owned loaders or compatibility shims

## `serve/` is decommissioned from hot path when:

- it orchestrates requests and runtime handles rather than manipulating tensors as the real implementation

## `dist/` is decommissioned from hot path when:

- serving and intended direct runtime do not depend on torch-distributed wrappers

## 4. Hot-Path Questions To Ask Before Declaring Victory

If the answer to any of these is "yes", PyTorch still owns too much.

- Does a normal token decode step still call eager attention logic in Python?
- Does a normal token decode step still append or gather cache pages in Python?
- Does the main LM forward still use `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, or custom torch tensor math on the intended runtime path?
- Does the intended distributed runtime still rely on `torch.distributed` for core collectives?
- Does `serve/engine.py` still implement the real sampler instead of delegating to runtime?

## 5. Validation Checklist Before Removing Torch Fallbacks

Do not delete torch fallbacks until all of these are true:

- per-op parity tests pass
- block parity tests pass
- model parity tests pass
- decode cache parity tests pass
- serving latency is not worse in the intended deployment regime
- graph capture works where expected
- error handling at the runtime boundary is clear and deterministic

## 6. What Can Stay

These may remain PyTorch-adjacent without invalidating the migration:

- checkpoint and safetensor IO
- Hugging Face import glue
- ONNX and TorchScript export
- optional training wrappers during transition
- reference implementations used only for tests

These should not be confused with the serving hot path.

## 7. Definition Of Done

For the migration to count as real:

- serving path is runtime-backed
- cache is runtime-backed
- attention is runtime-backed
- linears are runtime-backed
- norms are runtime-backed
- sampler is runtime-backed
- collectives are runtime-backed where required

If those conditions are not met, the repo may contain CUDA kernels, but it is still PyTorch-centered.
