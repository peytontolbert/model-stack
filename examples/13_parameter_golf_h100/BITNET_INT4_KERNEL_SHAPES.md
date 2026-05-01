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
| shared masks, 4-row CTA, 16 cols | `M=65536 K=1024 N=2048` | 0.0075 ms | 85.0156 ms | 85.0245 ms | 0.3854 ms | loses, 0.0045x |
| shared masks, 4-row CTA, 16 cols | `M=65536 K=2048 N=1024` | 0.0071 ms | 84.8776 ms | 84.8824 ms | 0.3488 ms | loses, 0.0041x |
| shared masks, 2-row CTA, 16 cols | `M=65536 K=1024 N=2048` | 0.0072 ms | 84.5093 ms | 84.5147 ms | 0.3854 ms | loses, 0.0046x |
| shared masks, 2-row CTA, 16 cols | `M=65536 K=2048 N=1024` | 0.0070 ms | 84.2773 ms | 84.2811 ms | 0.3485 ms | loses, 0.0041x |
| flattened 1-row CTA, 16 cols | `M=65536 K=1024 N=2048` | 0.0074 ms | 84.1631 ms | 84.1539 ms | 0.3855 ms | loses, 0.0046x |
| flattened 1-row CTA, 16 cols | `M=65536 K=2048 N=1024` | 0.0072 ms | 83.8043 ms | 83.8090 ms | 0.3483 ms | loses, 0.0042x |
| flattened 1-row CTA, 32 cols | `M=65536 K=1024 N=2048` | 0.0074 ms | 83.0434 ms | 83.0431 ms | 0.3859 ms | loses, 0.0046x |
| flattened 1-row CTA, 32 cols | `M=65536 K=2048 N=1024` | 0.0070 ms | 82.6662 ms | 82.6721 ms | 0.3489 ms | loses, 0.0042x |
| flattened 1-row CTA, 64 cols | `M=65536 K=1024 N=2048` | 0.0073 ms | 82.6387 ms | 82.6528 ms | 0.3852 ms | loses, 0.0047x |
| flattened 1-row CTA, 64 cols | `M=65536 K=2048 N=1024` | 0.0073 ms | 82.7397 ms | 82.7457 ms | 0.3484 ms | loses, 0.0042x |
| flattened 1-row CTA, 128 cols | `M=65536 K=1024 N=2048` | 0.0075 ms | 85.6787 ms | 85.6855 ms | 0.3857 ms | loses, 0.0045x |
| flattened 1-row CTA, 128 cols | `M=65536 K=2048 N=1024` | 0.0074 ms | 85.0452 ms | 85.0648 ms | 0.3486 ms | loses, 0.0041x |

This path is true ternary in the sense that it consumes separate positive and
negative bitmasks and computes add/sub reductions, not generic INT4 multiply.
It is not a winning PG training kernel. The best measured ternary mask layout
so far is flattened 1-row CTAs with 64 output columns, shared mask staging, and
a 4-lane cooperative K reduction. The bottleneck remains the
per-output bitwalk/reduction algorithm: it cannot compete with H100 BF16 tensor
cores for large dense training GEMMs. Keep it behind
`MODEL_STACK_TRAINABLE_BITNET_TERNARY_MASK_FORWARD=1` as a correctness and
profiling baseline while developing a different ternary training algorithm.

Strict ternary activation + weight path, measured on H100 2026-05-01 with
`bench_strict_ternary_linear.py`:

| Strategy | Shape | Act Quant | Strict Ternary Linear | Full Strict | BF16 Ternary | CUTLASS INT4 | Dense BF16 | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 4-lane, 64 cols | `M=65536 K=1024 N=2048` | 0.2597 ms | 9.8623 ms | 10.1244 ms | 82.6440 ms | 0.5749 ms | 0.3850 ms | loses, 0.0380x |
| 4-lane, 64 cols | `M=65536 K=2048 N=1024` | 0.4014 ms | 9.5731 ms | 9.9209 ms | 82.1286 ms | 0.5474 ms | 0.3483 ms | loses, 0.0351x |
| 8-lane, 32 cols | `M=65536 K=1024 N=2048` | 0.2593 ms | 8.8436 ms | 9.1323 ms | 82.6473 ms | 0.5836 ms | 0.3850 ms | loses, 0.0422x |
| 8-lane, 32 cols | `M=65536 K=2048 N=1024` | 0.4009 ms | 6.8086 ms | 7.2784 ms | 82.7447 ms | 0.5477 ms | 0.3484 ms | loses, 0.0479x |
| 16-lane, 16 cols | `M=65536 K=1024 N=2048` | 0.2596 ms | 15.7242 ms | 16.0486 ms | 83.2635 ms | 0.5746 ms | 0.3853 ms | loses, 0.0240x |
| 16-lane, 16 cols | `M=65536 K=2048 N=1024` | 0.4054 ms | 8.7037 ms | 9.1692 ms | 82.7401 ms | 0.5448 ms | 0.3484 ms | loses, 0.0380x |
| 8-lane, 64 cols | `M=65536 K=1024 N=2048` | 0.2609 ms | 9.4156 ms | 9.7240 ms | 83.2736 ms | 0.5840 ms | 0.3852 ms | loses, 0.0396x |
| 8-lane, 64 cols | `M=65536 K=2048 N=1024` | 0.4046 ms | 7.1678 ms | 7.6155 ms | 82.1280 ms | 0.5451 ms | 0.3483 ms | loses, 0.0457x |
| 8-lane, 16 cols | `M=65536 K=1024 N=2048` | 0.2606 ms | 9.1641 ms | 9.3671 ms | 83.2650 ms | 0.5778 ms | 0.3854 ms | loses, 0.0411x |
| 8-lane, 16 cols | `M=65536 K=2048 N=1024` | 0.4013 ms | 6.5941 ms | 7.0901 ms | 82.1199 ms | 0.5494 ms | 0.3481 ms | loses, 0.0491x |
| static 32/64 words, 8-lane, 32 cols | `M=65536 K=1024 N=2048` | 0.2597 ms | 8.6689 ms | 8.9338 ms | 82.6476 ms | 0.5740 ms | 0.3855 ms | loses, 0.0432x |
| static 32/64 words, 8-lane, 32 cols | `M=65536 K=2048 N=1024` | 0.4020 ms | 6.3625 ms | 6.7956 ms | 82.7428 ms | 0.5512 ms | 0.3483 ms | loses, 0.0513x |
| static 32/64 words, 8-lane, 16 cols | `M=65536 K=1024 N=2048` | 0.2657 ms | 8.7135 ms | 8.9679 ms | 83.2688 ms | 0.5777 ms | 0.3853 ms | loses, 0.0430x |
| static 32/64 words, 8-lane, 16 cols | `M=65536 K=2048 N=1024` | 0.4013 ms | 6.2348 ms | 6.6879 ms | 82.7415 ms | 0.5415 ms | 0.3483 ms | loses, 0.0521x |
| static 32/64 words, 8-lane, 8 cols | `M=65536 K=1024 N=2048` | 0.2596 ms | 10.0981 ms | 10.3740 ms | 82.6468 ms | 0.5826 ms | 0.3852 ms | loses, 0.0371x |
| static 32/64 words, 8-lane, 8 cols | `M=65536 K=2048 N=1024` | 0.4092 ms | 6.2955 ms | 6.7461 ms | 82.7415 ms | 0.5467 ms | 0.3484 ms | loses, 0.0516x |
| unrolled static words, 8-lane, 16 cols | `M=65536 K=1024 N=2048` | 0.2600 ms | 7.2200 ms | 7.3801 ms | 82.6516 ms | 0.5834 ms | 0.3851 ms | loses, 0.0522x |
| unrolled static words, 8-lane, 16 cols | `M=65536 K=2048 N=1024` | 0.4030 ms | 5.9663 ms | 6.3808 ms | 82.7420 ms | 0.5447 ms | 0.3483 ms | loses, 0.0546x |
| unrolled static words, 8-lane, 32 cols | `M=65536 K=1024 N=2048` | 0.2594 ms | 7.1668 ms | 7.4261 ms | 83.2616 ms | 0.5809 ms | 0.3850 ms | loses, 0.0518x |
| unrolled static words, 8-lane, 32 cols | `M=65536 K=2048 N=1024` | 0.4009 ms | 5.9830 ms | 6.3610 ms | 82.1266 ms | 0.5458 ms | 0.3484 ms | loses, 0.0548x |
| unrolled static words, 8-lane, 8 cols | `M=65536 K=1024 N=2048` | 0.2595 ms | 10.1182 ms | 10.3520 ms | 82.6469 ms | 0.5736 ms | 0.3851 ms | loses, 0.0372x |
| unrolled static words, 8-lane, 8 cols | `M=65536 K=2048 N=1024` | 0.4001 ms | 6.0555 ms | 6.5034 ms | 82.7411 ms | 0.5476 ms | 0.3481 ms | loses, 0.0535x |
| corrected 16-lane reduction, 16 cols | `M=65536 K=1024 N=2048` | 0.2597 ms | 10.3753 ms | 10.6250 ms | 82.6592 ms | 0.5833 ms | 0.3851 ms | loses, 0.0362x |
| corrected 16-lane reduction, 16 cols | `M=65536 K=2048 N=1024` | 0.4051 ms | 6.8507 ms | 7.2224 ms | 82.1265 ms | 0.5469 ms | 0.3483 ms | loses, 0.0482x |
| full-warp 32-lane, 8 cols | `M=65536 K=1024 N=2048` | 0.2594 ms | 19.0425 ms | 19.3065 ms | 82.6473 ms | 0.5727 ms | 0.3853 ms | loses, 0.0200x |
| full-warp 32-lane, 8 cols | `M=65536 K=2048 N=1024` | 0.4017 ms | 10.5221 ms | 10.9368 ms | 82.7423 ms | 0.5402 ms | 0.3479 ms | loses, 0.0318x |
| unrolled static words, 4-lane, 32 cols | `M=65536 K=1024 N=2048` | 0.2605 ms | 9.8664 ms | 10.1346 ms | 83.2734 ms | 0.5763 ms | 0.3851 ms | loses, 0.0380x |
| unrolled static words, 4-lane, 32 cols | `M=65536 K=2048 N=1024` | 0.4013 ms | 9.4872 ms | 9.9180 ms | 82.7494 ms | 0.5487 ms | 0.3482 ms | loses, 0.0351x |
| two-output lane group, 8-lane, 32 cols | `M=65536 K=1024 N=2048` | 0.2592 ms | 7.6630 ms | 7.9371 ms | 83.2606 ms | 0.5742 ms | 0.3851 ms | loses, 0.0485x |
| two-output lane group, 8-lane, 32 cols | `M=65536 K=2048 N=1024` | 0.4063 ms | 6.7395 ms | 7.1276 ms | 82.7399 ms | 0.5460 ms | 0.3480 ms | loses, 0.0488x |
| two-popcount dot, aligned, 8-lane, 16 cols | `M=65536 K=1024 N=2048` | 0.2601 ms | 6.7145 ms | 6.9191 ms | 82.6398 ms | 0.5734 ms | 0.3850 ms | loses, 0.0556x |
| two-popcount dot, aligned, 8-lane, 16 cols | `M=65536 K=2048 N=1024` | 0.3998 ms | 5.8166 ms | 6.2773 ms | 82.7399 ms | 0.5465 ms | 0.3483 ms | loses, 0.0555x |
| canonical sign-mismatch dot, aligned, 8-lane, 16 cols | `M=65536 K=1024 N=2048` | 0.2606 ms | 6.8200 ms | 7.0420 ms | 83.2684 ms | 0.5776 ms | 0.3851 ms | loses, 0.0547x |
| canonical sign-mismatch dot, aligned, 8-lane, 16 cols | `M=65536 K=2048 N=1024` | 0.4006 ms | 5.8101 ms | 6.1783 ms | 82.1302 ms | 0.5454 ms | 0.3482 ms | loses, 0.0564x |
| paired 64-bit popcount, aligned, 8-lane, 16 cols | `M=65536 K=1024 N=2048` | 0.2618 ms | 6.6236 ms | 6.8650 ms | 82.6443 ms | 0.5715 ms | 0.3850 ms | loses, 0.0561x |
| paired 64-bit popcount, aligned, 8-lane, 16 cols | `M=65536 K=2048 N=1024` | 0.4019 ms | 5.7592 ms | 6.1906 ms | 82.1248 ms | 0.5489 ms | 0.3481 ms | loses, 0.0562x |
| paired 64-bit popcount, aligned, 8-lane, 32 cols | `M=65536 K=1024 N=2048` | 0.2589 ms | 6.7545 ms | 7.0139 ms | 82.6479 ms | 0.5722 ms | 0.3850 ms | loses, 0.0549x |
| paired 64-bit popcount, aligned, 8-lane, 32 cols | `M=65536 K=2048 N=1024` | 0.4018 ms | 5.8517 ms | 6.2661 ms | 82.7401 ms | 0.5466 ms | 0.3480 ms | loses, 0.0555x |

The selected strict ternary layout is paired 64-bit popcount over
static-specialized 32/64 mask words, 8 lanes per output, and 16 output columns
per CTA. It is the best measured overall MLP-pair tradeoff so far. Shared
activation-mask caching was also tested, but the extra CTA synchronization
regressed to 9.9031 ms and 7.8523 ms full strict on the same two shapes. Wider
lane groups were retested with complete reductions; 16-lane and full-warp
32-lane groups both regressed, so the winning lane split remains 8 lanes per
ternary dot. This is the first strict path that is meaningfully faster than the
BF16-activation ternary baseline, because compute is now bitset/popcount on
ternary activation masks and ternary weight masks. It is still slower than
dense BF16 and CUTLASS INT4.
