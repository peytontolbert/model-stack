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

Direct shuffled CUTLASS pack result, measured on H100 2026-05-01 with
`bench_cutlass_direct_int4.py`:

| Shape | Pack | CUTLASS GEMM | Pack+GEMM | Dense BF16 | Result |
|---|---:|---:|---:|---:|---|
| `M=65536 K=1024 N=2048` | 0.0134 ms | 0.5786 ms | 0.5936 ms | 0.3838 ms | loses, 0.6466x |
| `M=65536 K=2048 N=1024` | 0.0140 ms | 0.5794 ms | 0.5836 ms | 0.3474 ms | loses, 0.5952x |

The row-major pack plus CUTLASS reorder overhead is no longer the bottleneck.
The direct packer is fast and exact for these shapes. The remaining loss is the
CUTLASS mixed BF16xINT4 GEMM itself versus cuBLAS BF16 dense GEMM on the PG MLP
matrices, so this path must remain opt-in until a different GEMM mainloop or
training fusion beats dense.

True ternary mask baseline, measured on H100 2026-05-01 with
`bench_ternary_mask_linear.py`.

| Strategy | Shape | Mask Pack | Ternary Linear | Pack+Linear | Dense BF16 | Result |
|---|---|---:|---:|---:|---:|---|
| shared activation staging | `M=65536 K=1024 N=2048` | 0.0075 ms | 97.9041 ms | 97.1964 ms | 0.3849 ms | loses, 0.0040x |
| shared activation staging | `M=65536 K=2048 N=1024` | 0.0073 ms | 97.8922 ms | 97.9303 ms | 0.3482 ms | loses, 0.0036x |
| 4-lane cooperative K reduction | `M=65536 K=1024 N=2048` | 0.0073 ms | 87.2686 ms | 87.1575 ms | 0.3852 ms | loses, 0.0044x |
| 4-lane cooperative K reduction | `M=65536 K=2048 N=1024` | 0.0070 ms | 87.5188 ms | 87.5237 ms | 0.3484 ms | loses, 0.0040x |
| 8-lane cooperative K reduction | `M=65536 K=1024 N=2048` | 0.0072 ms | 88.9800 ms | 88.9894 ms | 0.3854 ms | loses, 0.0043x |
| 8-lane cooperative K reduction | `M=65536 K=2048 N=1024` | 0.0074 ms | 90.1932 ms | 90.1952 ms | 0.3486 ms | loses, 0.0039x |

This path is true ternary in the sense that it consumes separate positive and
negative bitmasks and computes add/sub reductions, not generic INT4 multiply.
It is not a winning PG training kernel. The 4-lane cooperative reduction is the
best measured ternary mask layout so far, but the bottleneck remains the
per-output bitwalk/reduction algorithm: it cannot compete with H100 BF16 tensor
cores for large dense training GEMMs. Keep it behind
`MODEL_STACK_TRAINABLE_BITNET_TERNARY_MASK_FORWARD=1` as a correctness and
profiling baseline while developing a different ternary training algorithm.
