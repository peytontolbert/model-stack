# transformer_10 Training And Backward CUDA Spec

This document defines the training path required to replace `train/`, `tensor/optim.py`, `tensor/random.py`, `tensor/checkpoint.py`, and the training-relevant pieces of `tensor/losses.py`, `tensor/regularization.py`, and `dist/`.

## 1. Scope

Training replacement must own:

- forward execution in C++ over CUDA kernels and cuBLASLt/NCCL
- backward computation for all trainable model-stack modules
- optimizer updates
- mixed precision and loss scaling
- activation checkpointing
- gradient clipping and overflow detection
- EMA/SWA/SAM support
- distributed gradient reduction and sharded state handling

## 2. Core Decision

Do not rebuild a general-purpose dynamic autograd engine first.

Use a narrower model-stack training design:

- every C++ module provides explicit `forward()` and `backward()`
- forward records only the saved tensors and metadata needed by its own backward
- the trainer owns a simple tape of module invocations
- backward walks that tape in reverse order

This is the correct fit for this repository because the model stack is structured and known in advance.

## 3. Target Runtime Types

Required C++ objects:

- `TrainContext`
  - step id, microstep id, precision mode, RNG state, reduction policy
- `ModuleTape`
  - saved tensor handles, shape metadata, kernel-plan metadata
- `GradBucket`
  - contiguous FP32 or BF16 grad buffer for reduction and clipping
- `OptimizerState`
  - per-parameter state tensors
- `LossScaleState`
  - current scale, growth tracker, overflow tracker
- `ActivationCheckpointPolicy`
  - per-block or per-subgraph rematerialization policy
- `TrainStepResult`
  - loss, overflow flags, grad norm, timing counters

## 4. Backward Coverage By Module

## Embedding

Required:

- gather forward
- scatter-add backward into embedding grad buffer
- optional fused row-wise optimizer path later

## Linear / QKV / Output / MLP GEMMs

Use cuBLASLt or CUTLASS-backed wrappers for:

- forward GEMM
- dInput GEMM
- dWeight GEMM
- optional fused bias gradient

Required wrappers:

- `linear_fwd`
- `linear_bwd_input`
- `linear_bwd_weight`
- `linear_bwd_bias`

## Norms

Required custom backward kernels:

- RMSNorm backward
- LayerNorm backward
- fused residual+norm backward where forward path uses fusion

Saved data:

- input
- normalized output or inverse RMS statistics
- weight

## Activations And GLU

Required backward kernels:

- GELU
- SiLU
- SwiGLU
- GEGLU
- ReGLU
- bias+activation fused variants

## Residual And Dropout

Required:

- residual-add backward
- dropout mask replay from explicit RNG state, not global torch RNG
- fused residual+bias+dropout backward

## RoPE / Positional

Required:

- backward through RoPE application for Q and K
- cache generation itself is metadata-only and does not require gradient

## Attention

Required backward coverage:

- dense prefill attention backward
- decode attention backward for training-like single-step paths only if needed
- cross-attention backward
- mask-aware softmax backward
- Q/K/V and output projection grads

Notes:

- inference decode kernels are not the same as training backward kernels
- the training path can initially focus on dense attention backward and leave serving-only decode specialization separate

## Losses

Required backward kernels:

- cross entropy
- label-smoothed cross entropy
- NLL
- KL / JS for distillation
- focal loss if retained

## MoE

Required:

- router top-k or capacity backward
- expert gather/scatter combine backward
- auxiliary load-balance loss backward

This can be later than core dense transformer training, but it must have an explicit lane.

## 5. Optimizer Coverage

## First owned optimizer kernels

- AdamW
- Lion
- Adafactor

Minimum update kernel contract:

- consumes parameter, grad, optimizer state, hyperparameters
- supports FP32 master weights when parameters are stored in lower precision
- can run in-place over contiguous buckets

## Optimizer helpers in C++

- global grad norm
- per-parameter or unitwise clipping
- routed weight decay masks
- overflow detection
- loss-scale update
- EMA update
- SWA collect/finalize
- SAM/ASAM perturb and restore

`tensor/optim.py` should migrate as:

- update kernels -> CUDA
- scheduling, masks, routing, state bookkeeping -> C++

## 6. RNG And Determinism

Replace `tensor/random.py` and torch-global RNG dependence with:

- explicit Philox-like counter-based RNG state per stream or per step
- replayable dropout streams
- deterministic activation-checkpoint recompute
- graph-safe seed scopes

Requirements:

- no hidden `torch.manual_seed` dependence in train hot path
- forward and backward masks are reproducible under graph replay
- distributed ranks derive non-overlapping subsequences from global seed plus rank plus microstep

## 7. Activation Checkpointing

Target replacement for `tensor/checkpoint.py` and `train/trainer.py` checkpoint logic:

- per-block rematerialization policy owned by C++ trainer
- checkpoint boundaries at block granularity first
- saved state includes only boundary tensors and RNG metadata
- recompute uses same kernel-plan and RNG sequence as original forward

Do not depend on `torch.utils.checkpoint`.

## 8. Mixed Precision

Required modes:

- FP32
- BF16
- FP16 with loss scaling

Policy:

- parameters can live in BF16/FP16
- reductions for norms, softmax stats, loss accumulation, and grad norm use FP32 where required
- master weights optional for optimizer stability

Loss scaling:

- overflow check after backward and before optimizer step
- scale growth/backoff logic in C++
- unscale before clipping

## 9. Distributed Training

Training runtime must integrate with `t10::dist`:

- allreduce for DP gradients
- reduce-scatter/allgather for sharded optimizer or tensor-parallel weights
- overlap reduction with backward where profitable
- sharded checkpoint save/load

Do not retain DDP/FSDP/DeepSpeed as the long-term ownership boundary.

They may remain compatibility layers during transition only.

## 10. File Mapping

| Current file | Target implementation |
|---|---|
| `train/trainer.py` | `t10::train::Trainer`, `t10::train::TrainStep` |
| `train/run.py` | `t10::train::RunLoop` plus Python binding |
| `tensor/optim.py` | `t10::train::optim` plus `t10_cuda::kernels::optim` |
| `tensor/random.py` | `t10::core::RngState` and `t10::train::rng` |
| `tensor/checkpoint.py` | `t10::train::ActivationCheckpointPolicy` |
| `tensor/regularization.py` | `t10_cuda::kernels::regularization` plus `t10::train::schedule` |
| `tensor/losses.py` | `t10_cuda::kernels::loss` |
| `dist/engine.py`, `dist/checkpoint.py`, `dist/launch.py` | `t10::dist::*` |

## 11. Migration Order

1. Dense causal LM training only.
2. BF16 first, then FP16 loss-scaling path.
3. AdamW first.
4. Block-level activation checkpointing.
5. DP + TP communication.
6. Lion/Adafactor, EMA/SWA/SAM.
7. MoE backward and advanced regularization.
8. RL and distillation-specific training paths.

## 12. Definition Of Training-Ready

Training docs are complete only when all of the following have an owned design:

- explicit backward contract per module family
- optimizer state layout
- mixed-precision policy
- RNG replay policy
- checkpoint/recompute policy
- distributed reduction and state sharding policy
- parity and stability validation plan

Before implementation, this is the minimum defensible replacement for the current `train/` and torch-autograd-heavy path.
