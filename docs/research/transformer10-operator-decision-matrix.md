# transformer_10 Operator Decision Matrix

This document picks the primary implementation path for the main operators in `transformer_10`.

Allowed choices in this matrix:

- handwritten CUDA/C++
- cuBLASLt/cuTENSOR
- Fuser
- Triton

Important exception:

- distributed collectives are not included as a primary row choice because they should use `NCCL`, not any of the four buckets above

Principle used here:

- if the op is dense linear algebra, prefer `cuBLASLt/cuTENSOR`
- if the op is runtime-owned, stateful, layout-sensitive, or serving-critical, prefer handwritten `CUDA/C++`
- if the op is stateless and fusion-heavy, `Triton` is often the best secondary lane
- `Fuser` stays mainly a training/prototyping/reference path for this repo, not the core long-term serving answer

## Core Serving Path

| Operator | Local code | Primary path | Why | Best references |
|---|---|---|---|---|
| Token embedding lookup | `model/causal.py`, `model/encoder.py`, `model/seq2seq.py` | handwritten CUDA/C++ | bandwidth-bound gather op, not a GEMM; should become a direct runtime primitive | `cuEmbed`, `TransformerEngine`, `TensorRT-LLM` |
| Q projection | `attn/eager.py` | cuBLASLt/cuTENSOR | dense GEMM, repeatable shapes, strong library coverage | `CUDALibrarySamples/cuBLASLt`, `TransformerEngine/common/gemm` |
| K projection | `attn/eager.py` | cuBLASLt/cuTENSOR | same reason as Q projection | `CUDALibrarySamples/cuBLASLt`, `TransformerEngine/common/gemm` |
| V projection | `attn/eager.py` | cuBLASLt/cuTENSOR | same reason as Q projection | `CUDALibrarySamples/cuBLASLt`, `TransformerEngine/common/gemm` |
| Output projection | `attn/eager.py` | cuBLASLt/cuTENSOR | dense GEMM; should stay library-backed | `CUDALibrarySamples/cuBLASLt`, `TransformerEngine/common/gemm` |
| RoPE apply to Q/K | `tensor/positional.py`, `attn/eager.py` | handwritten CUDA/C++ | small but hot layout-sensitive transform; good fusion target with attention prep | `TransformerEngine/common/fused_rope`, `Fuser/tests/cpp/test_rope.cpp` |
| Dense causal/context attention | `attn/eager.py`, `attn/backends.py` | handwritten CUDA/C++ | serving-critical, shape-sensitive, should own mask, layout, and fused softmax/value path directly | `TransformerEngine/common/fused_attn`, `TensorRT-LLM/cpp/kernels/fmha_v2` |
| Decode-phase MQA/GQA attention | `attn/eager.py`, `serve/engine.py` | handwritten CUDA/C++ | generation path needs specialized paged/MQA/GQA kernels and runtime heuristics | `TensorRT-LLM/cpp/kernels/xqa`, `TensorRT-LLM/docs/source/features/attention.md` |
| KV cache append/read/paging/eviction | `attn/kv_cache.py`, `serve/engine.py` | handwritten CUDA/C++ | runtime-owned stateful memory system; poor fit for Triton or Fuser | `TensorRT-LLM/docs/source/features/kvcache.md`, `TransformerEngine/common/fused_attn/kv_cache.cu` |
| Cast/transpose/layout helpers around attention | `attn/eager.py`, `tensor/shape.py` | handwritten CUDA/C++ | usually fused or tightly coupled to attention/GEMM boundaries | `TransformerEngine/common/transpose`, `TensorRT-LLM/runtimeKernels.cu` |
| RMSNorm | `tensor/norms.py`, `blocks/transformer_block.py`, `model/causal.py` | handwritten CUDA/C++ | reduction-heavy and hot; stable enough to justify fixed kernels | `TransformerEngine/common/normalization/rmsnorm`, `Fuser/benchmarks/cpp/rms_norm.cpp` |
| LayerNorm | `tensor/norms.py`, `model/encoder.py`, `model/seq2seq.py` | handwritten CUDA/C++ | same reasoning as RMSNorm | `TransformerEngine/common/normalization/layernorm`, `Fuser/benchmarks/cpp/layer_norm.cpp` |
| MLP up/gate projection | `tensor/mlp.py` | cuBLASLt/cuTENSOR | dense GEMM; should not be handwritten first | `CUDALibrarySamples/cuBLASLt`, `TransformerEngine/common/gemm` |
| SwiGLU/GEGLU/ReGLU activation middle stage | `tensor/mlp.py` | Triton | compact stateless fusion target; quicker to own than a bespoke C++ kernel on first pass | local `triton/ops`, `TensorRT-LLM/triton_kernels/swiglu.py` |
| MLP down projection | `tensor/mlp.py` | cuBLASLt/cuTENSOR | dense GEMM; same rule as other linears | `CUDALibrarySamples/cuBLASLt`, `TransformerEngine/common/gemm` |
| Final LM head projection | `model/causal.py`, `model/seq2seq.py` | cuBLASLt/cuTENSOR | large vocabulary GEMM with library heuristics and workspace support | `CUDALibrarySamples/cuBLASLt`, `nvMatmulHeuristics` |
| Sampling logits transforms and token selection | `tensor/sampling.py`, `serve/engine.py`, `model/causal.py` | Triton | mostly stateless top-k/top-p/mask/reduction work; good iterative kernel lane | local `triton/ops`, `TensorRT-LLM/docs/source/features/sampling.md`, `TensorRT-LLM/triton_kernels/topk.py` |

## Training And Loss Path

| Operator | Local code | Primary path | Why | Best references |
|---|---|---|---|---|
| Cross-entropy / tokenwise NLL | `tensor/losses.py` | Triton | classic fused reduction kernel; strong local Triton references exist | local `triton/ops/cross_entropy.py` |
| Label-smoothed CE / focal-like variants | `tensor/losses.py` | Triton | pointwise + reduction structure fits Triton better than fixed C++ first pass | local `triton/ops/cross_entropy.py`, `Fuser` reduction examples |
| Entropy / KL / JS helper reductions | `tensor/losses.py` | Fuser | mostly training-side reduction chains, not serving-critical | `Fuser/runtime/fused_reduction.cu`, `Fuser/benchmarks/cpp/reduction.cpp` |
| Residual add + dropout + simple pointwise glue | `blocks/transformer_block.py`, `blocks/parallel_block.py` | Fuser | strong training-oriented fusion candidate; not worth early handwritten kernels unless profiling forces it | `Fuser/benchmarks/cpp/many_pointwise_ops.cpp`, `softmax_dropout.cpp` |

## Optional Block Variants

| Operator | Local code | Primary path | Why | Best references |
|---|---|---|---|---|
| Prefix/local/banded/strided/windowed attention masks | `blocks/prefix_lm_block.py`, `blocks/local_attn_block.py`, `blocks/banded_attn_block.py`, `blocks/strided_attn_block.py`, `blocks/window_pattern_attn_block.py` | Triton | pattern-heavy attention variants are easier to prototype and iterate in Triton than fixed C++ immediately | local `triton/ops/flash_attention.py`, `triton/ops/blocksparse/*` |
| Block-sparse attention | `blocks/block_sparse_attn_block.py` | Triton | optional variant with direct local blocksparse examples | local `triton/ops/blocksparse/matmul.py`, `triton/ops/blocksparse/softmax.py`, `CUDALibrarySamples/cuTENSOR/blocksparse.cu` |
| Cross-attention | `blocks/cross_attn_block.py`, `model/seq2seq.py` | handwritten CUDA/C++ | operationally similar to dense attention; better to share the main attention runtime path | `TransformerEngine/common/fused_attn`, `TensorRT-LLM` attention docs |
| Segment-bidirectional attention | `blocks/segment_bidir_attn_block.py` | Triton | specialized mask-driven variant; easier to own as a DSL kernel if it remains a niche path | local `triton` attention references |

## MoE And Quantization

| Operator | Local code | Primary path | Why | Best references |
|---|---|---|---|---|
| MoE router top-k and combine | `blocks/moe_block.py`, `attn/moe.py` | Triton | top-k, gather, combine, and routing glue are fusion-heavy and irregular; Triton is the best first owned path | `TensorRT-LLM/triton_kernels/topk.py`, `Fuser/tests/cpp/test_moe.cpp` |
| Expert GEMMs | `blocks/moe_block.py`, `tensor/mlp.py` | cuBLASLt/cuTENSOR | still dense GEMMs once routing is decided | `CUDALibrarySamples/cuBLASLt`, `TransformerEngine/common/gemm` |
| Weight-only quant/dequant helpers | `compress/quantization.py`, `attn/quant.py` | handwritten CUDA/C++ | runtime-owned low-precision glue should be explicit and portable across serving paths | `TransformerEngine`, `TensorRT-LLM`, `Fuser/runtime/block_quantization_kernels.cu` |

## Collectives And Parallelism

Not assigned to one of the four implementation buckets:

- `dist/parallel/tensor_parallel.py`
- `tensor/shard.py`

Use:

- `NCCL`

Reason:

- collectives are not a Triton/Fuser/cuBLASLt problem
- they are runtime communication primitives

## Short Verdict

If this matrix is followed, the first stable non-torch hot path for `transformer_10` should look like:

1. `cuBLASLt` for dense linears
2. handwritten CUDA/C++ for attention, RoPE, norms, embedding, KV cache, and serving runtime helpers
3. Triton for selective fusion-heavy stateless kernels:
   - SwiGLU
   - sampling
   - cross-entropy
   - optional sparse attention variants
4. Fuser used mostly as a study/prototype lane, not the final serving architecture
