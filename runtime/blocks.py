from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn

from runtime.native import has_native_op, native_available
from runtime.ops import add_layer_norm as runtime_add_layer_norm
from runtime.ops import add_rms_norm as runtime_add_rms_norm
from runtime.ops import layer_norm as runtime_layer_norm
from runtime.ops import residual_add as runtime_residual_add
from runtime.ops import rms_norm as runtime_rms_norm
from tensor.masking import (
    broadcast_mask,
    build_banded_mask,
    build_block_sparse_mask,
    build_dilated_causal_mask,
    build_prefix_lm_mask,
    build_segment_bidir_mask,
    build_sliding_window_causal_mask,
    build_strided_mask,
    to_additive_mask,
    window_pattern_from_spans,
)
from tensor.norms import RMSNorm
from runtime.positional import build_alibi_bias, build_relative_position_indices, relative_position_bias_from_table


def can_apply_native_norm(norm, training: bool) -> bool:
    return (not training) and isinstance(norm, (RMSNorm, nn.LayerNorm))


def apply_native_norm(x: torch.Tensor, norm):
    if can_apply_native_norm(norm, norm.training):
        if isinstance(norm, RMSNorm):
            return runtime_rms_norm(x, weight=norm.weight, eps=float(norm.eps))
        return runtime_layer_norm(x, weight=norm.weight, bias=norm.bias, eps=float(norm.eps))
    return norm(x)


def fused_add_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    norm,
    residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if can_apply_native_norm(norm, norm.training):
        if isinstance(norm, RMSNorm):
            return runtime_add_rms_norm(
                x,
                update,
                norm.weight,
                residual_scale=float(residual_scale),
                eps=float(norm.eps),
            )
        return runtime_add_layer_norm(
            x,
            update,
            norm.weight,
            norm.bias,
            residual_scale=float(residual_scale),
            eps=float(norm.eps),
        )
    combined = x + (update * float(residual_scale))
    return combined, norm(combined)


def apply_residual_update(
    x: torch.Tensor,
    update: torch.Tensor,
    *,
    residual_scale: float,
    resid_dropout,
    drop_path,
) -> torch.Tensor:
    residual = drop_path(resid_dropout(update))
    return runtime_residual_add(x, residual, residual_scale=float(residual_scale))


def _merge_attention_bias(
    attn_mask: torch.Tensor | None,
    bias: torch.Tensor,
) -> torch.Tensor:
    if attn_mask is None:
        return bias
    if attn_mask.dtype == torch.bool:
        attn_mask = to_additive_mask(attn_mask)
    return attn_mask + bias


def apply_attention_biases(
    x: torch.Tensor,
    base_mask: torch.Tensor | None,
    *,
    num_heads: int,
    use_alibi: bool = False,
    rpb_table: torch.Tensor | None = None,
    rpb_max_distance: int | None = None,
) -> torch.Tensor | None:
    attn_mask = base_mask
    _, seq_len, _ = x.shape
    if use_alibi:
        bias = build_alibi_bias(int(num_heads), seq_len=seq_len, device=x.device).to(dtype=x.dtype)
        attn_mask = _merge_attention_bias(attn_mask, bias)
    if rpb_table is not None:
        if rpb_max_distance is None:
            raise ValueError("rpb_max_distance is required when rpb_table is provided")
        idx = build_relative_position_indices(seq_len, seq_len, int(rpb_max_distance), device=x.device)
        rpb = relative_position_bias_from_table(idx, rpb_table).to(dtype=x.dtype)
        attn_mask = _merge_attention_bias(attn_mask, rpb)
    return attn_mask


def prepare_attention_mask_for_heads(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_heads: int,
    tgt_len: int,
    src_len: int,
    padding_mask_is_1_for_token: bool = True,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        if tuple(mask.shape) == (int(batch_size), int(src_len)):
            return broadcast_mask(
                batch_size=int(batch_size),
                num_heads=int(num_heads),
                tgt_len=int(tgt_len),
                src_len=int(src_len),
                padding_mask=mask,
                padding_mask_is_1_for_token=padding_mask_is_1_for_token,
            )
        if tuple(mask.shape) == (int(tgt_len), int(src_len)):
            return mask.view(1, 1, int(tgt_len), int(src_len)).expand(
                int(batch_size),
                int(num_heads),
                int(tgt_len),
                int(src_len),
            )
        raise ValueError(
            f"2D attention mask must have shape (B,S) or (T,S); got {tuple(mask.shape)} "
            f"for batch={batch_size}, tgt_len={tgt_len}, src_len={src_len}"
        )
    if mask.ndim == 3:
        if tuple(mask.shape) != (int(batch_size), int(tgt_len), int(src_len)):
            raise ValueError(
                f"3D attention mask must have shape (B,T,S); got {tuple(mask.shape)} "
                f"for batch={batch_size}, tgt_len={tgt_len}, src_len={src_len}"
            )
        return mask.unsqueeze(1).expand(int(batch_size), int(num_heads), int(tgt_len), int(src_len))
    if mask.ndim == 4:
        if mask.shape[-2] != int(tgt_len) or mask.shape[-1] != int(src_len):
            raise ValueError(
                f"4D attention mask must end in (T,S); got {tuple(mask.shape)} "
                f"for tgt_len={tgt_len}, src_len={src_len}"
            )
        if mask.shape[0] not in (1, int(batch_size)):
            raise ValueError(f"4D attention mask batch dim must be 1 or {batch_size}; got {mask.shape[0]}")
        if mask.shape[1] not in (1, int(num_heads)):
            raise ValueError(f"4D attention mask head dim must be 1 or {num_heads}; got {mask.shape[1]}")
        return mask.expand(int(batch_size), int(num_heads), int(tgt_len), int(src_len))
    raise ValueError(f"unsupported attention mask rank: {mask.ndim}")


def combine_prepared_attention_masks(
    primary: torch.Tensor | None,
    secondary: torch.Tensor | None,
) -> torch.Tensor | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    if primary.dtype == torch.bool and secondary.dtype == torch.bool:
        return primary | secondary
    if primary.dtype == torch.bool:
        primary = to_additive_mask(primary)
    if secondary.dtype == torch.bool:
        secondary = to_additive_mask(secondary)
    return primary + secondary


def prepare_pattern_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    pattern_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, tgt_len, _ = x.shape
    src_len = int(pattern_mask.shape[-1])
    prepared_pattern = prepare_attention_mask_for_heads(
        pattern_mask,
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        tgt_len=int(tgt_len),
        src_len=int(src_len),
        padding_mask_is_1_for_token=False,
    )
    prepared_mask = prepare_attention_mask_for_heads(
        mask,
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        tgt_len=int(tgt_len),
        src_len=int(src_len),
    )
    combined = combine_prepared_attention_masks(prepared_pattern, prepared_mask)
    if combined is None:
        raise ValueError("pattern attention mask preparation unexpectedly produced None")
    return combined


def prepare_encoder_attention_mask(
    x: torch.Tensor,
    padding_mask: torch.Tensor | None,
    *,
    num_heads: int,
    rpb_table: torch.Tensor | None = None,
    rpb_max_distance: int | None = None,
) -> torch.Tensor | None:
    batch_size, tgt_len, _ = x.shape
    attn_mask = prepare_attention_mask_for_heads(
        padding_mask,
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        tgt_len=int(tgt_len),
        src_len=int(tgt_len),
        padding_mask_is_1_for_token=True,
    )
    return apply_attention_biases(
        x,
        attn_mask,
        num_heads=int(num_heads),
        rpb_table=rpb_table,
        rpb_max_distance=rpb_max_distance,
    )


def prepare_cross_attention_mask(
    x: torch.Tensor,
    memory: torch.Tensor,
    enc_mask: torch.Tensor | None,
    *,
    num_heads: int,
    padding_mask_is_1_for_token: bool = True,
) -> torch.Tensor | None:
    batch_size, tgt_len, _ = x.shape
    src_len = int(memory.shape[1])
    return prepare_attention_mask_for_heads(
        enc_mask,
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        tgt_len=int(tgt_len),
        src_len=src_len,
        padding_mask_is_1_for_token=padding_mask_is_1_for_token,
    )


def prepare_local_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    window_size: int,
) -> torch.Tensor:
    _, tgt_len, _ = x.shape
    pattern = build_sliding_window_causal_mask(
        int(tgt_len),
        int(window_size),
        device=x.device,
        dtype=torch.bool,
    )
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=pattern)


def prepare_prefix_lm_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    prefix_len: int,
) -> torch.Tensor:
    _, tgt_len, _ = x.shape
    pattern = build_prefix_lm_mask(
        int(tgt_len),
        int(prefix_len),
        device=x.device,
        dtype=torch.bool,
    )
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=pattern)


def prepare_strided_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    stride: int,
) -> torch.Tensor:
    _, tgt_len, _ = x.shape
    pattern = build_strided_mask(
        int(tgt_len),
        int(stride),
        device=x.device,
        dtype=torch.bool,
    )
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=pattern)


def prepare_banded_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    bandwidth: int,
) -> torch.Tensor:
    _, tgt_len, _ = x.shape
    pattern = build_banded_mask(
        int(tgt_len),
        int(bandwidth),
        device=x.device,
        dtype=torch.bool,
    )
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=pattern)


def prepare_dilated_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    window: int,
    dilation: int,
) -> torch.Tensor:
    _, tgt_len, _ = x.shape
    pattern = build_dilated_causal_mask(
        int(tgt_len),
        int(window),
        int(dilation),
        device=x.device,
        dtype=torch.bool,
    )
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=pattern)


def prepare_window_pattern_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    _, tgt_len, _ = x.shape
    pattern = window_pattern_from_spans(list(spans), int(tgt_len)).to(device=x.device, dtype=torch.bool)
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=pattern)


def default_block_sparse_pattern(
    seq_len: int,
    block_size: int,
    *,
    device=None,
) -> torch.Tensor:
    num_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
    pattern = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        pattern[i, i] = True
    return pattern


def prepare_block_sparse_attention_mask(
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
    block_size: int,
    pattern: torch.Tensor | None = None,
) -> torch.Tensor:
    _, tgt_len, _ = x.shape
    block_pattern = (
        default_block_sparse_pattern(int(tgt_len), int(block_size), device=x.device)
        if pattern is None
        else pattern.to(device=x.device)
    )
    sparse = build_block_sparse_mask(
        int(tgt_len),
        int(block_size),
        block_pattern,
        device=x.device,
        dtype=torch.bool,
    )
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=sparse)


def prepare_segment_bidir_attention_mask(
    x: torch.Tensor,
    segment_ids: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_heads: int,
) -> torch.Tensor:
    pattern = build_segment_bidir_mask(segment_ids.to(device=x.device))
    return prepare_pattern_attention_mask(x, mask, num_heads=int(num_heads), pattern_mask=pattern)


def execute_attention_mlp_block(
    x: torch.Tensor,
    *,
    attn_fn: Callable[[torch.Tensor], torch.Tensor],
    mlp_fn: Callable[[torch.Tensor], torch.Tensor],
    n1,
    n2,
    resid_dropout,
    drop_path,
    residual_scale: float = 1.0,
    norm_policy: str = "prenorm",
) -> torch.Tensor:
    if str(norm_policy).lower() == "prenorm":
        a = attn_fn(n1(x))
        if can_apply_native_norm(n2, n2.training):
            x, mlp_in = fused_add_norm(x, a, n2, residual_scale)
        else:
            x = apply_residual_update(
                x,
                a,
                residual_scale=residual_scale,
                resid_dropout=resid_dropout,
                drop_path=drop_path,
            )
            mlp_in = n2(x)
        m = mlp_fn(mlp_in)
        return apply_residual_update(
            x,
            m,
            residual_scale=residual_scale,
            resid_dropout=resid_dropout,
            drop_path=drop_path,
        )

    a = attn_fn(x)
    if can_apply_native_norm(n1, n1.training):
        _, x = fused_add_norm(x, a, n1, residual_scale)
    else:
        x = n1(
            apply_residual_update(
                x,
                a,
                residual_scale=residual_scale,
                resid_dropout=resid_dropout,
                drop_path=drop_path,
            )
        )
    m = mlp_fn(x)
    if can_apply_native_norm(n2, n2.training):
        _, x = fused_add_norm(x, m, n2, residual_scale)
    else:
        x = n2(
            apply_residual_update(
                x,
                m,
                residual_scale=residual_scale,
                resid_dropout=resid_dropout,
                drop_path=drop_path,
            )
        )
    return x


def execute_parallel_attention_mlp_block(
    x: torch.Tensor,
    *,
    attn_fn: Callable[[torch.Tensor], torch.Tensor],
    mlp_fn: Callable[[torch.Tensor], torch.Tensor],
    norm,
    resid_dropout,
    drop_path,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    y = apply_native_norm(x, norm)
    a = attn_fn(y)
    m = mlp_fn(y)
    out = apply_residual_update(
        x,
        a,
        residual_scale=residual_scale,
        resid_dropout=resid_dropout,
        drop_path=drop_path,
    )
    return apply_residual_update(
        out,
        m,
        residual_scale=residual_scale,
        resid_dropout=resid_dropout,
        drop_path=drop_path,
    )


def execute_block_stack(
    blocks: Iterable[nn.Module],
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    cache=None,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    for i, blk in enumerate(blocks):
        layer_cache = None if cache is None else cache.layer(i)
        if position_embeddings is None and position_ids is None:
            x = blk(x, mask, layer_cache)
        else:
            x = blk(
                x,
                mask,
                layer_cache,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )
    return x


def execute_encoder_stack(
    blocks: Iterable[nn.Module],
    x: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    for blk in blocks:
        x = blk(x, padding_mask)
    return x


def execute_decoder_stack(
    blocks: Iterable[nn.Module],
    x: torch.Tensor,
    memory: torch.Tensor,
    self_mask: torch.Tensor | None = None,
    memory_mask: torch.Tensor | None = None,
    cache=None,
) -> torch.Tensor:
    for i, blk in enumerate(blocks):
        layer_cache = None if cache is None else cache.layer(i)
        x = blk(x, memory, self_mask, memory_mask, layer_cache)
    return x


def block_native_execution_info(block) -> dict[str, object]:
    norm_1 = getattr(block, "n1", None)
    norm_2 = getattr(block, "n2", None)
    norm = getattr(block, "n", None)
    has_attn = hasattr(block, "attn")
    has_mlp = hasattr(block, "mlp")
    info = {
        "block_type": type(block).__name__,
        "native_runtime_available": native_available(),
        "attention_native": bool(has_attn and has_native_op("attention_prefill") and has_native_op("attention_decode")),
        "mlp_native": bool(has_mlp and has_native_op("mlp")),
        "residual_add_native": bool(has_native_op("residual_add")),
        "norm_1_native": bool(norm_1 is not None and can_apply_native_norm(norm_1, training=False)),
        "norm_2_native": bool(norm_2 is not None and can_apply_native_norm(norm_2, training=False)),
        "norm_native": bool(norm is not None and can_apply_native_norm(norm, training=False)),
    }
    info["fully_native_inference_path"] = bool(
        info["native_runtime_available"]
        and info["residual_add_native"]
        and (
            info["norm_native"]
            or (
                info["norm_1_native"]
                and info["attention_native"]
                and (not has_mlp or info["mlp_native"])
            )
        )
    )
    return info


def stack_native_execution_info(blocks: Iterable[nn.Module]) -> list[dict[str, object]]:
    return [block_native_execution_info(block) for block in blocks]


__all__ = [
    "apply_attention_biases",
    "apply_native_norm",
    "apply_residual_update",
    "block_native_execution_info",
    "can_apply_native_norm",
    "combine_prepared_attention_masks",
    "default_block_sparse_pattern",
    "execute_attention_mlp_block",
    "execute_block_stack",
    "execute_decoder_stack",
    "execute_encoder_stack",
    "execute_parallel_attention_mlp_block",
    "fused_add_norm",
    "prepare_attention_mask_for_heads",
    "prepare_banded_attention_mask",
    "prepare_block_sparse_attention_mask",
    "prepare_cross_attention_mask",
    "prepare_dilated_attention_mask",
    "prepare_encoder_attention_mask",
    "prepare_local_attention_mask",
    "prepare_pattern_attention_mask",
    "prepare_prefix_lm_attention_mask",
    "prepare_segment_bidir_attention_mask",
    "prepare_strided_attention_mask",
    "prepare_window_pattern_attention_mask",
    "stack_native_execution_info",
]
