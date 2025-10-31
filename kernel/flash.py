from __future__ import annotations

from typing import Optional
import torch

from .registry import register_lazy


def _load_flash2():
    def _flash2_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = False,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func  # type: ignore

        B, H, T, D = q.shape
        S = k.shape[2]
        out = flash_attn_func(
            q.reshape(B * H, T, D),
            k.reshape(B * H, S, D),
            v.reshape(B * H, S, D),
            causal=is_causal,
            dropout_p=dropout_p,
        )
        return out.reshape(B, H, T, D)

    return _flash2_attn


def _load_xformers():
    def _xformers_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        import xformers.ops as xops  # type: ignore

        return xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=dropout_p, op=None)

    return _xformers_attn


# Register lazily so import-time doesn't require optional deps
register_lazy("attn.flash2", _load_flash2)
register_lazy("attn.xformers", _load_xformers)


