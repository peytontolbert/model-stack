from typing import Optional, Literal
import os
import json
import torch


def _read_backend_from_env_or_file() -> Optional[str]:
    forced = os.getenv("ATTN_BACKEND")
    if forced:
        return forced.strip().lower()
    path = os.getenv("ATTN_BACKEND_FILE")
    if path and os.path.isfile(path):
        try:
            data = json.loads(open(path, "r").read())
            val = str(data.get("backend", "")).strip().lower()
            return val or None
        except Exception:
            return None
    return None


def select_attention_backend(is_causal: bool, dtype: torch.dtype, seq: int, heads: int, device: torch.device) -> Literal["flash2","triton","xformers","torch"]:
    # Heuristic router: prefer FlashAttention-2 when available, else xFormers, else PyTorch SDPA
    # First consult kernel registry (lazy, safe) then fall back to direct imports
    forced = _read_backend_from_env_or_file()
    if forced in ("flash2", "triton", "xformers", "torch"):
        return forced  # user-forced override takes precedence
    try:
        from kernel import has as kernel_has
    except Exception:
        kernel_has = lambda _name: False  # type: ignore

    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16) and seq >= 64:
        # Prefer flash2, then triton, then xformers, else torch
        if kernel_has("attn.flash2"):
            return "flash2"
        try:
            import flash_attn  # type: ignore  # noqa: F401
            return "flash2"
        except Exception:
            pass
        if kernel_has("attn.triton"):
            return "triton"
        try:
            import triton  # type: ignore  # noqa: F401
            return "triton"
        except Exception:
            pass

    if kernel_has("attn.xformers"):
        return "xformers"
    try:
        import xformers  # type: ignore  # noqa: F401
        return "xformers"
    except Exception:
        return "torch"


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    backend: Optional[str] = None,
    is_causal: Optional[bool] = None,
) -> torch.Tensor:
    # Prefer torch SDPA for portability; route when explicitly requested
    if backend is None or backend == "torch":
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    # Try kernel registry first for requested backends
    try:
        from kernel import has as kernel_has, get as kernel_get
    except Exception:
        kernel_has = lambda _name: False  # type: ignore
        def kernel_get(_name):  # type: ignore
            raise KeyError(_name)

    if backend == "flash2":
        if kernel_has("attn.flash2"):
            flash2 = kernel_get("attn.flash2")
            return flash2(q, k, v, is_causal=bool(is_causal), dropout_p=float(dropout_p))
        # fallback to direct import
        try:
            from flash_attn import flash_attn_func  # type: ignore
            B, H, T, D = q.shape
            S = k.shape[2]
            out = flash_attn_func(q.reshape(B * H, T, D), k.reshape(B * H, S, D), v.reshape(B * H, S, D), causal=is_causal or False, dropout_p=dropout_p)
            return out.reshape(B, H, T, D)
        except Exception:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    if backend == "triton":
        if kernel_has("attn.triton"):
            triton_attn = kernel_get("attn.triton")
            return triton_attn(q, k, v, attn_mask=attn_mask, is_causal=bool(is_causal), dropout_p=float(dropout_p))
        try:
            import triton  # type: ignore  # noqa: F401
            # Fallback: use torch SDPA if only triton is present without a kernel impl
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        except Exception:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    if backend == "xformers":
        if kernel_has("attn.xformers"):
            xme = kernel_get("attn.xformers")
            return xme(q, k, v, attn_bias=attn_mask, dropout_p=float(dropout_p))
        try:
            import xformers.ops as xops  # type: ignore
            return xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask, p=dropout_p, op=None)
        except Exception:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    raise ValueError("Unknown backend")


