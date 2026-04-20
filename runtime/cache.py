from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from specs.config import ModelConfig
from runtime.native import create_native_paged_kv_cache_state, has_native_op, native_paged_kv_cache_available


@dataclass(frozen=True)
class KVCacheSpec:
    batch: int
    n_layers: int
    n_kv_heads: int
    head_dim: int
    pagesize: int
    dtype: torch.dtype
    device: torch.device
    backend: str = "auto"


class RuntimeLayerCacheView:
    """Runtime-owned per-layer cache view used by model and attention code."""

    def __init__(self, parent, layer_idx: int) -> None:
        self.parent = parent
        self.layer_idx = int(layer_idx)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        assert k.dim() == 4 and v.dim() == 4, "KV must be (B,H,T,D)"
        assert k.shape == v.shape, "K and V shapes must match"
        self.parent.append_batch(self.layer_idx, k, v)

    def append_and_read(self, k: torch.Tensor, v: torch.Tensor, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.append(k, v)
        return self.read(int(start), self.length())

    def read(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.parent.read(self.layer_idx, int(start), int(end))

    def supports_paged_attention_decode(self) -> bool:
        return all(
            hasattr(self.parent, name)
            for name in ("layer_page_count", "layer_pages", "layer_block_table")
        )

    def paged_attention_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        from runtime.kv_cache import paged_attention_decode as runtime_paged_attention_decode

        return runtime_paged_attention_decode(
            self.parent,
            self.layer_idx,
            q,
            k,
            v,
            attn_mask=attn_mask,
            scale=scale,
        )

    def length(self) -> int:
        if hasattr(self.parent, "layer_max_length"):
            return int(self.parent.layer_max_length(self.layer_idx))
        native_cache = getattr(self.parent, "_native_cache", None)
        if native_cache is not None:
            return int(native_cache.max_length(self.layer_idx))
        lengths = self.parent.layer_lengths(self.layer_idx)
        return int(lengths.max().item()) if lengths.numel() > 0 else 0


class RuntimeKVCacheMixin:
    """Runtime-owned cache facade shared by compatibility cache implementations."""

    def layer(self, layer_idx: int) -> RuntimeLayerCacheView:
        return RuntimeLayerCacheView(self, int(layer_idx))


def _normalize_backend_name(requested: str | None) -> str:
    candidate = (requested or "auto").strip().lower()
    aliases = {
        "native": "native-paged",
        "native_paged": "native-paged",
    }
    candidate = aliases.get(candidate, candidate)
    valid = {"auto", "paged", "contiguous", "native-paged"}
    if candidate not in valid:
        raise ValueError(
            f"Unsupported KV cache backend '{requested}'. "
            f"Expected one of: {', '.join(sorted(valid))}."
        )
    return candidate


def resolve_kv_cache_backend(requested: str | None = None) -> str:
    requested_name = requested if requested is not None else os.getenv("MODEL_STACK_KV_BACKEND", "auto")
    candidate = _normalize_backend_name(requested_name)
    if candidate == "auto":
        if native_paged_kv_cache_available():
            return "native-paged"
        if has_native_op("kv_cache_append"):
            return "contiguous"
        return "paged"
    if candidate == "native-paged" and not native_paged_kv_cache_available():
        raise RuntimeError("MODEL_STACK_KV_BACKEND requested native-paged, but no native paged KV cache state is available")
    return candidate


def kv_cache_spec_from_config(
    cfg: ModelConfig,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    pagesize: int = 512,
    backend: str | None = None,
) -> KVCacheSpec:
    n_layers = int(cfg.n_layers)
    n_kv_heads = int(getattr(cfg, "n_kv_heads", cfg.n_heads))
    head_dim = int(getattr(cfg, "head_dim", None) or (int(cfg.d_model) // int(cfg.n_heads)))
    return KVCacheSpec(
        batch=int(batch_size),
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        pagesize=max(int(pagesize), 1),
        dtype=dtype,
        device=torch.device(device),
        backend=resolve_kv_cache_backend(backend),
    )


def kv_cache_spec_from_model(
    model,
    *,
    batch_size: int,
    pagesize: int = 512,
    backend: str | None = None,
) -> KVCacheSpec:
    try:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    except Exception as exc:
        raise ValueError("Could not infer model device/dtype for KV cache allocation") from exc
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("Model does not expose .cfg, so KV cache dimensions cannot be derived")
    return kv_cache_spec_from_config(
        cfg,
        batch_size=int(batch_size),
        dtype=dtype,
        device=device,
        pagesize=int(pagesize),
        backend=backend,
    )


def create_kv_cache(spec: KVCacheSpec):
    from runtime.kv_cache import ContiguousKVCache, PagedKVCache

    backend = resolve_kv_cache_backend(spec.backend)
    if backend == "contiguous":
        return ContiguousKVCache(
            spec.batch,
            spec.n_layers,
            spec.n_kv_heads,
            spec.head_dim,
            spec.pagesize,
            spec.dtype,
            spec.device,
            backend_name=backend,
        )
    if backend == "native-paged":
        native_cache_state, native_layer_states = create_native_paged_kv_cache_state(
            batch=spec.batch,
            n_layers=spec.n_layers,
            n_kv_heads=spec.n_kv_heads,
            head_dim=spec.head_dim,
            pagesize=spec.pagesize,
            dtype=spec.dtype,
            device=spec.device,
        )
        return PagedKVCache(
            spec.batch,
            spec.n_layers,
            spec.n_kv_heads,
            spec.head_dim,
            spec.pagesize,
            spec.dtype,
            spec.device,
            native_cache_state=native_cache_state,
            native_layer_states=native_layer_states,
            backend_name=backend,
        )
    return PagedKVCache(
        spec.batch,
        spec.n_layers,
        spec.n_kv_heads,
        spec.head_dim,
        spec.pagesize,
        spec.dtype,
        spec.device,
        backend_name=backend,
    )


def allocate_model_kv_cache(
    model,
    *,
    batch_size: int,
    pagesize: int = 512,
    backend: str | None = None,
):
    return create_kv_cache(
        kv_cache_spec_from_model(
            model,
            batch_size=int(batch_size),
            pagesize=int(pagesize),
            backend=backend,
        )
    )


def kv_cache_runtime_info(cache) -> dict[str, Any]:
    backend = getattr(cache, "backend", None) or type(cache).__name__.lower()
    info = {
        "backend": str(backend),
        "batch": int(getattr(cache, "batch", 0)),
        "n_layers": int(getattr(cache, "n_layers", 0)),
        "n_kv_heads": int(getattr(cache, "n_kv_heads", 0)),
        "head_dim": int(getattr(cache, "head_dim", 0)),
        "pagesize": int(getattr(cache, "pagesize", 0)),
        "native_cache_state": bool(getattr(cache, "_native_cache", None) is not None),
        "native_layer_states": bool(getattr(cache, "_native_layers", None) is not None),
    }
    return info


def evict_kv_cache(cache, max_tokens: int, policy: str = "fifo") -> None:
    cache.evict(int(max_tokens), str(policy))
