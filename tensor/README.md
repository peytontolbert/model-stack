## Stateless numerics for transformer stacks
### Future development
Tensor’s direction emphasizes stateless, composable numerics with strong shape/mask and numerical guarantees. For upcoming work and priorities, see:

- Vision & philosophy: `docs/vision.md`
- Roadmap: `docs/roadmap.md`
- Contributing: `docs/contributing.md`


Positional math (RoPE/ALiBi/RPB), masking, norms, MLPs, activations, dtype/shape helpers, numerics, residuals, init, regularization, sampling, metrics, and losses. Pure tensor ops with a few lightweight modules where ergonomic.

## Modules (high-level)
- **activations**: `gelu`, `silu`, `bias_gelu`, `bias_silu`, plus `fast_gelu`, `quick_gelu`, `mish`, `tanh_gelu`
- **norms**: `RMSNorm`, `ScaleNorm`
- **mlp**: `MLP` (standard, SwiGLU)
- **positional**: `build_rope_cache`/`apply_rotary`/`apply_rotary_scaled`, `build_alibi_bias`, `build_relative_position_indices`, `relative_position_bias_from_table`, buckets, 2D/NTK/YARN helpers
- **masking**: `build_causal_mask`, `build_sliding_window_causal_mask`, `build_prefix_lm_mask`, `build_padding_mask`, `broadcast_mask`, `attention_mask_from_lengths`, `lengths_from_attention_mask`, `invert_mask`, `as_bool_mask`
- **numerics**: `safe_softmax`, `masked_log_softmax`, `masked_logsumexp`, stability utils (`log1mexp`, `softplus_safe`, `safe_sigmoid`, `safe_tanh`, etc.), chunked/blocked softmax
- **dtypes**: `cast_for_softmax`, `cast_for_norm`, `restore_dtype`, `to_dtype_like`, precision helpers (`set_matmul_precision`, `with_logits_precision`, `maybe_autocast`), dtype checks (`is_fp16/bf16/int8/fp8`)
- **shape**: `ensure_even_last_dim`, `split_heads`/`merge_heads`, `split_qkv`/`merge_qkv`, GQA helpers, padding (`center_pad`, `pad_to_multiple`, `right_trim_to`), asserts
- **residual**: `residual_add`, gated/bias-dropout-add, `prenorm`/`postnorm`
- **init**: `xavier_uniform_linear`, `kaiming_uniform_linear`, DeepNet scaling and related helpers
- **regularization**: `drop_path`, `StochasticDepth`, z-loss, label smoothing, token/sequence drop
- **losses**: `masked_cross_entropy`, `sequence_nll` (+ label smoothing, z-loss, JS/KL, MSE/Huber, BCE variants)
- **sampling**: temperature, repetition/presence/frequency penalties, regex/no-repeat-gram constraints, min-p, top-k/top-p
- **metrics**: masked accuracy/top-k, token F1, ECE binning, sequence logprob
- **compile/runtime**: in-graph helpers, safe masked fill, shape inference, CUDA graphs seed/record guards
- **utils**: ragged sequence pack/unpack, sharding/TP helpers and FLOPs/bytes estimators, window partition/merge, optim (grad norm/clip), quant utils (int8 pack/scale), export-safe ops

## Shapes & conventions
- **Batch-first** everywhere; attention shapes typically `(B, H, T, D)` and scores `(B, H, T, S)`
- **Masks** are boolean with `True` meaning masked; broadcasting helpers return `(B, H, T, S)`
- **RoPE** caches are `(T, Dh)` cos/sin; **ALiBi** bias is `(1, H, T, S)`; **RPB** table is `(H, 2*max_distance-1)`

## Example
```python
from tensor import build_rope_cache, apply_rotary, build_causal_mask, broadcast_mask, safe_softmax
import torch

B,H,T,Dh = 2, 8, 16, 64
q = torch.randn(B,H,T,Dh)
k = torch.randn(B,H,T,Dh)
cos,sin = build_rope_cache(T, Dh)
q,k = apply_rotary(q, k, cos, sin)
scores = q @ k.transpose(-2, -1)
mask = broadcast_mask(batch_size=B, num_heads=H, tgt_len=T, src_len=T, causal_mask=build_causal_mask(T))
probs = safe_softmax(scores, mask=mask, dim=-1)
```

## Scope
- Pure, stateless tensor ops with small `nn.Module`s where ergonomic (e.g., norms, MLP)
- No attention kernels/caches (live in `attn/`) and no model wiring (live in `blocks/` or `model/`)