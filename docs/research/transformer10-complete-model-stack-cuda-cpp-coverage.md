# transformer_10 Complete Model-Stack C++/CUDA Coverage

This document states the target clearly:

`transformer_10` should become a complete model-stack repository. A user should be able to configure, load, run, serve, train, quantize, distribute, benchmark, and validate models inside this repository, with the implementation provided by our C++/CUDA code rather than by PyTorch.

The target is not one monolithic kernel. The target is a modular C++/CUDA mirror of the current repository:

- Python package boundaries become C++/CUDA module boundaries.
- Python model classes become C++ model-stack classes.
- Python tensor functions become CUDA kernel families, fused kernels, or vendor-library wrappers.
- Python serving/training/distributed code becomes C++ runtime systems.
- Python remains useful as bindings and scripting, but not as the implementation substrate.

## 1. Product Boundary

The full model-stack includes:

- model configuration
- module registry and construction
- tensor descriptors and dtype/layout policy
- embeddings, linears, norms, activations, attention, cache, samplers
- transformer block variants
- causal LM, prefix LM, encoder, seq2seq, heads
- checkpoint loading and weight binding
- tokenization and data loading
- serving and generation
- training forward/backward/loss/optimizer paths
- distributed collectives and parallel layouts
- compression, quantization, LoRA, pruning, distillation
- evaluation, benchmarking, metrics, memory reporting
- autotuning and kernel-plan persistence
- diagnostics and visualization hooks where they affect model-stack work

## 2. Target Source Layout

Recommended long-term layout:

```text
runtime/
  cpp/
    include/t10/
      core/
      specs/
      tensor/
      nn/
      attention/
      blocks/
      model/
      serve/
      train/
      dist/
      data/
      checkpoint/
      compress/
      eval/
      autotune/
    src/
  cuda/
    include/t10_cuda/
      tensor/
      kernels/
      gemm/
      attention/
      cache/
      collectives/
    src/
      kernels/
      gemm/
      attention/
      cache/
      collectives/
  python/
    bindings/
```

The current Python tree can remain as a compatibility layer while the implementation moves under `runtime/`.

## 3. Tensor Math Coverage

## Core Transformer Math

| Current surface | Target implementation | Notes |
|---|---|---|
| `tensor/norms.py` | CUDA norm module | RMSNorm, LayerNorm, masked variants, backward later |
| `tensor/positional.py` | CUDA positional module + C++ cache policy | RoPE apply first; cache builders can be C++ CPU or CUDA depending on use |
| `tensor/activations.py` | CUDA activation/fusion module | GELU, SiLU, GLU family, bias+activation |
| `tensor/mlp.py` | C++ MLP module + cuBLASLt + CUDA activation kernels | Preserve configurable MLP structure |
| `tensor/residual.py` | CUDA residual/fusion module | residual add, bias/dropout/add, gated residual |
| `tensor/masking.py` | C++ mask policy + CUDA mask application | Avoid materializing giant masks where possible |
| `tensor/sampling.py` | C++ sampler policy + CUDA sampler kernels | top-k, top-p, penalties, constraints |
| `tensor/dtypes.py` | C++ dtype policy + CUDA cast kernels | dtype checks, casts, FP8 scale metadata |
| `tensor/shape.py` | C++ layout/shape descriptors + CUDA transforms | split/merge heads, views, transposes |

## Attention And Cache

| Current surface | Target implementation | Notes |
|---|---|---|
| `attn/eager.py` | C++ Attention module + CUDA kernels | QKV, RoPE, cache, GQA/MQA, masks, output |
| `attn/backends.py` | C++ attention planner | Runtime selects kernel family, not Python |
| `attn/kv_cache.py` | C++/CUDA cache manager | Page tables, append, read, evict, reorder |
| `attn/gqa.py` | C++ attention layout policy | Head sharing and layout metadata |
| `attn/reference.py` | Test/reference only | Keep as parity reference while useful |
| `attn/moe.py` | CUDA routing kernels + C++ MoE policy | top-k routing, combine, load balance |
| `attn/quant.py` | CUDA quant/dequant + quantized matmul wrappers | int8, NF4, FP8 glue |

## Numerics, Ragged, Losses, Metrics

| Current surface | Target implementation | Notes |
|---|---|---|
| `tensor/numerics/` | CUDA numerics kernels | stable softmax, logsumexp, chunked reductions |
| `tensor/ragged.py` | C++ ragged descriptors + CUDA segment kernels | pack/unpack, segment reductions, ragged gather/scatter |
| `tensor/losses.py` | CUDA loss kernels | CE, NLL, KL/JS, label smoothing, training path |
| `tensor/metrics.py` | C++/CUDA eval metrics | accuracy, entropy, ECE, Brier, sequence metrics |
| `eval/` | C++/CUDA eval harness plus Python bindings | benchmark, compare, calibration, memory |

## Distributed And Parallel Tensor Math

| Current surface | Target implementation | Notes |
|---|---|---|
| `tensor/shard.py` | C++ sharding planner + NCCL wrappers | TP/CP/SP partitioning and collectives |
| `dist/parallel/tensor_parallel.py` | C++ tensor-parallel modules | Column/row parallel linears, TP attention/MLP |
| `dist/parallel/pipeline.py` | C++ pipeline execution planner | Optional after core runtime |
| `dist/engine.py` | C++ distributed engine | DDP/FSDP/DeepSpeed wrappers become compatibility only |

## 4. Model And Block Coverage

## `blocks/`

Every block variant should become a C++ block class with CUDA-backed execution:

- `TransformerBlock`
- `LlamaBlock`
- `GPTBlock`
- `ParallelTransformerBlock`
- `MoEBlock`
- `CrossAttentionBlock`
- `EncoderBlock`
- `DecoderBlock`
- local, prefix, banded, strided, dilated, window-pattern, segment-bidir, block-sparse variants

The C++ block layer should preserve configurability:

- norm policy
- attention variant
- MLP activation
- residual/dropout policy
- positional bias policy
- MoE policy

CUDA should provide the execution kernels and fusions underneath those choices.

## `model/`

Every model family should have a C++ model class:

- causal LM
- prefix LM
- encoder
- seq2seq
- classification heads
- token heads

The C++ model layer owns:

- module construction
- weight binding
- forward execution plan
- train/inference mode
- cache compatibility
- generation compatibility

Python can expose the same API, but the implementation should not require `torch.nn.Module`.

## 5. Fused Kernel Families

The right unit is not "one kernel for everything". The right unit is a set of fused kernel families.

Required fusion families:

- `norm_residual`
  - RMSNorm/LayerNorm plus residual add where useful
- `qkv_prepare`
  - QKV projection wrapper output layout, split heads, RoPE, optional cache append
- `attention_prefill`
  - prefill attention, masks, softmax, value path
- `attention_decode`
  - token decode, paged KV read, MQA/GQA
- `mlp_gated`
  - up/gate projection wrapper, SwiGLU/GEGLU/ReGLU, down projection wrapper
- `sampler`
  - penalties, top-k/top-p, RNG/token selection
- `loss`
  - cross entropy, NLL, reduction/fused backward later
- `quant`
  - quantize, dequantize, scale update, packed weight helpers
- `ragged_segment`
  - packed sequence and segment reductions
- `collective_overlap`
  - NCCL launch/stream integration and overlap points

The C++ layer should decide which fusion family to call based on model config, shape, dtype, and runtime mode.

## 6. Non-Kernel C++ Coverage

Not everything becomes CUDA. Some parts should be C++ model-stack systems.

## `specs/`

Target:

- C++ config structs
- validation
- op registry
- model/block resolution

## `data/`

Target:

- tokenizer interfaces
- dataset/shard readers
- batching and pinned host staging
- async host-to-device transfer

## `checkpoint/` and model loading

Target:

- C++ checkpoint metadata
- safetensor reader/writer
- weight mapping and layout conversion
- HF-compatible import layer where needed

## `serve/`

Target:

- C++ serving engine
- request batching
- cache allocation
- decode loop
- graph bucket selection
- sampler policy

## `train/`

Target:

- C++ training loop primitives
- CUDA losses
- backward kernels
- optimizer kernels or vendor-backed optimizer paths
- gradient scaling and clipping

## `compress/`

Target:

- quantized runtime modules
- LoRA merge/apply
- pruning masks
- distillation losses
- export/import of deltas

## `autotune/`

Target:

- runtime shape collection
- kernel plan search
- cuBLASLt heuristic cache
- attention/norm launch policy cache

## `eval/`

Target:

- benchmark harness
- model comparison
- memory reporting
- calibration and metric kernels

## 7. Migration Tiers

Use tiers so the repo becomes useful at every stage.

## Tier 1: Core Causal LM Inference

Includes:

- config
- checkpoint loading
- causal model
- LLaMA/GPT block
- embeddings
- norms
- linears
- RoPE
- attention
- KV cache
- sampler
- serving engine

## Tier 2: Full Inference Model Stack

Adds:

- prefix LM
- encoder/seq2seq
- cross-attention
- optional attention variants
- compression/quantized inference
- eval/bench
- autotune

## Tier 3: Distributed Inference

Adds:

- NCCL collectives
- tensor parallel linears
- context/sequence exchange
- distributed serving plans

## Tier 4: Training

Adds:

- backward kernels
- losses
- optimizer support
- gradient clipping/scaling
- distributed training integration
- distillation and LoRA training paths

## Tier 5: Full Repository Replacement

Adds:

- all remaining Python utilities have C++ equivalents or are explicitly classified as binding/demo/compatibility only
- Python is optional for model-stack operation

## 8. Definition Of Complete Coverage

Coverage is complete when every current Python/PyTorch responsibility has one of these statuses:

- implemented in C++/CUDA
- intentionally retained as Python binding/demo code over a C++/CUDA implementation
- intentionally retained as compatibility/reference code with no runtime ownership
- removed because the C++/CUDA stack supersedes it

Coverage is not complete if a module still silently depends on PyTorch for model-stack execution.

## 9. Immediate Documentation Backlog

The next docs after this should be:

- C++ model-stack API spec
- complete tensor-math function inventory
- serving engine C++ spec
- training/backward CUDA spec
- data/checkpoint/tokenizer C++ spec
- compression/quantization runtime spec

Those docs should be written before large implementation begins, because they define the actual repository product surface.
