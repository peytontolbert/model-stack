from __future__ import annotations

from typing import Optional
import torch

from .registry import register_lazy


def _load_triton_attn():
    """Return a callable implementing SDPA via Triton if available; fallback to torch.

    The returned function signature matches other backends and expects inputs shaped
    (B, H, T, D) with optional causal and dropout flags.
    """

    def _triton_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        # Try a Triton implementation if available; otherwise use torch SDPA
        try:
            # Placeholder for a real Triton kernel dispatch; fall back to torch
            import triton  # type: ignore  # noqa: F401
            import triton.language as tl  # type: ignore  # noqa: F401
            # For now, use torch SDPA until a specialized kernel is added
        except Exception:
            pass
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    return _triton_sdpa


# Register lazily so importing kernel doesn't require Triton
register_lazy("attn.triton", _load_triton_attn)


