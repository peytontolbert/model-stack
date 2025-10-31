"""KV cache paging and compaction utilities.

This module provides a simple paged KV cache for autoregressive decoding
with support for LRU eviction and compaction.
"""

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import torch


@dataclass
class CachePage:
    keys: torch.Tensor  # [seq_len, num_heads, head_dim]
    values: torch.Tensor  # [seq_len, num_heads, head_dim]
    length: int  # number of valid tokens stored


class PagedKVCache:
    def __init__(
        self,
        page_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
        max_pages: Optional[int] = None,
    ) -> None:
        self.page_size = int(page_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.max_pages = max_pages
        self.pages: Deque[CachePage] = deque()
        self._lru: OrderedDict[int, None] = OrderedDict()

    def _new_page(self) -> CachePage:
        k = torch.empty(self.page_size, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        v = torch.empty_like(k)
        return CachePage(keys=k, values=v, length=0)

    def _touch(self, idx: int) -> None:
        self._lru.pop(idx, None)
        self._lru[idx] = None

    def _maybe_evict(self) -> None:
        if self.max_pages is None:
            return
        while len(self.pages) > self.max_pages:
            # Evict least recently used
            if not self._lru:
                self.pages.popleft()
                continue
            evict_idx, _ = self._lru.popitem(last=False)
            # Remove that page index from deque
            # Rebuild deque without the evicted page to keep indices consistent
            new_pages = deque()
            for i, p in enumerate(self.pages):
                if i != evict_idx:
                    new_pages.append(p)
            self.pages = new_pages
            # Rebuild LRU indices
            self._lru = OrderedDict({i: None for i in range(len(self.pages))})

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append a single token's KV across heads: [num_heads, head_dim]."""
        assert k.shape == (self.num_heads, self.head_dim)
        assert v.shape == (self.num_heads, self.head_dim)
        if len(self.pages) == 0 or self.pages[-1].length == self.page_size:
            self.pages.append(self._new_page())
        page = self.pages[-1]
        page.keys[page.length].copy_(k)
        page.values[page.length].copy_(v)
        page.length += 1
        self._touch(len(self.pages) - 1)
        self._maybe_evict()

    def get_range(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return stacked K, V for [start:end] tokens: [T, H, D]."""
        if start < 0 or end < start:
            raise ValueError("invalid range")
        out_k = []
        out_v = []
        offset = 0
        for i, page in enumerate(self.pages):
            page_start = offset
            page_end = offset + page.length
            if page_start >= end:
                break
            if page_end <= start:
                offset = page_end
                continue
            s = max(0, start - page_start)
            e = min(page.length, end - page_start)
            if s < e:
                out_k.append(page.keys[s:e])
                out_v.append(page.values[s:e])
                self._touch(i)
            offset = page_end
        if not out_k:
            return (
                torch.empty(0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device),
                torch.empty(0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device),
            )
        return torch.cat(out_k, dim=0), torch.cat(out_v, dim=0)

    def compact(self) -> None:
        """Remove fully-empty tail pages and pack partial pages to reduce fragmentation."""
        all_k = []
        all_v = []
        for page in self.pages:
            if page.length > 0:
                all_k.append(page.keys[: page.length])
                all_v.append(page.values[: page.length])
        if not all_k:
            self.pages.clear()
            self._lru.clear()
            return
        K = torch.cat(all_k, dim=0)
        V = torch.cat(all_v, dim=0)
        # Rebuild pages from concatenated tensors
        total = K.shape[0]
        new_pages: Deque[CachePage] = deque()
        pos = 0
        while pos < total:
            page = self._new_page()
            take = min(self.page_size, total - pos)
            page.keys[:take].copy_(K[pos : pos + take])
            page.values[:take].copy_(V[pos : pos + take])
            page.length = take
            new_pages.append(page)
            pos += take
        self.pages = new_pages
        self._lru = OrderedDict({i: None for i in range(len(self.pages))})

    def __len__(self) -> int:
        return sum(p.length for p in self.pages)


__all__ = [
    "CachePage",
    "PagedKVCache",
]


