"""Compatibility shim for KV cache APIs that are now runtime-owned."""

from runtime.kv_cache import (
    ContiguousKVCache,
    PagedKVCache,
    init_kv_cache,
    kv_cache_append,
    kv_cache_evict,
    kv_cache_slice,
)

__all__ = [
    "ContiguousKVCache",
    "PagedKVCache",
    "init_kv_cache",
    "kv_cache_append",
    "kv_cache_evict",
    "kv_cache_slice",
]
