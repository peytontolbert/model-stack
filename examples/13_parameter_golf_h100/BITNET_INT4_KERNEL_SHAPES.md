# Parameter Golf BitNet INT4 Kernel Shapes

The packed INT4 tensor-core kernel target is the active `train_gpt.py` linear
surface, not the older exploratory SwiGLU harness.

For the current 8xH100 `runtime_row_1024x7_relu2_mlp2` recipe:

- Global training tokens per step: `524288`
- World size: `8`
- Per-rank GEMM rows: `M = 65536`
- Model width: `1024`
- Attention heads: `16`
- KV heads: `4`
- Head dim: `64`
- MLP hidden: `2048`

Training-critical linear GEMMs per layer:

| Name | M | K | N | Notes |
|---|---:|---:|---:|---|
| `attn_q` | 65536 | 1024 | 1024 | separate Q projection |
| `attn_k` | 65536 | 1024 | 256 | GQA K projection |
| `attn_v` | 65536 | 1024 | 256 | GQA V projection |
| `attn_out` | 65536 | 1024 | 1024 | attention output projection |
| `mlp_up_relu2` | 65536 | 1024 | 2048 | ReLU2 expansion |
| `mlp_down_relu2` | 65536 | 2048 | 1024 | ReLU2 down projection |

Kernel layout target:

- Activation operand: row-major BF16/FP16 input `A[M, K]`.
- Packed weight operand: signed INT4 weights `W[N, K]`, consumed logically as
  `B[K, N] = W.T`. The current `_pack_int4_signed` storage is row-major
  `[N, ceil(K / 2)]`; a true Hopper/CUTLASS narrow-operand kernel may require
  an offline reordered copy before launch.
- Output operand: BF16/FP16 `D[M, N]`.
- Weight scale: per-output-channel runtime-row BitNet scale, logically `N`
  scalars. In CUTLASS group-scale terms, this is group size `K`.

Current non-winning native path:

- `runtime/csrc/backend/cuda_int4_linear.cu` advertises an SM90 IMMA tile of
  `8x8x32`. That is the slow path that measured around `119 ms` on the two MLP
  shapes above. It is packed INT4 storage, but it is not the packed INT4
  tensor-core training kernel we need for Parameter Golf.
- The replacement target is a Hopper mixed-input GEMM tile in the CUTLASS
  family, with block-level shapes like `128x128x128` and the large per-rank
  MLP matrices above as the acceptance test.

Do not use the current `dynamic_int4_ste` path as proof of this kernel. That
path stores 4-bit-range activations in INT8 tensors and calls the INT8 backend;
it is not a packed INT4 tensor-core GEMM.

Do not use `int3_kv_pack` as training proof. That path is for KV-cache storage
and decode/eval plumbing. It only matters for Parameter Golf training speed if
we add a fused packed attention path that avoids dequantizing K/V before SDPA.
