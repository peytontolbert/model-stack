from __future__ import annotations

import os
from typing import Optional

import torch

from runtime.cache import RuntimeKVCacheMixin
from runtime.native import create_native_paged_kv_cache_state
from runtime.ops import kv_cache_write as runtime_kv_cache_write
from runtime.ops import paged_kv_append as runtime_paged_kv_append
from runtime.ops import paged_kv_compact as runtime_paged_kv_compact
from runtime.ops import paged_kv_read_last as runtime_paged_kv_read_last
from runtime.ops import paged_kv_read_range as runtime_paged_kv_read_range


class _NativeLayerTensorList:
    def __init__(self, parent: "PagedKVCache", attr: str) -> None:
        self.parent = parent
        self.attr = attr

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        layer_idx = int(layer_idx)
        if self.parent._native_cache is not None:
            return getattr(self.parent._native_cache, self.attr)(layer_idx)
        layer = self.parent._native_layers[layer_idx]
        return getattr(layer, self.attr)()

    def __len__(self) -> int:
        return self.parent.n_layers

    def __iter__(self):
        for layer_idx in range(self.parent.n_layers):
            yield self[layer_idx]


class _NativeLayerScalarList:
    def __init__(self, parent: "PagedKVCache", attr: str) -> None:
        self.parent = parent
        self.attr = attr

    def __getitem__(self, layer_idx: int) -> int:
        layer_idx = int(layer_idx)
        if self.parent._native_cache is not None:
            return int(getattr(self.parent._native_cache, self.attr)(layer_idx))
        layer = self.parent._native_layers[layer_idx]
        return int(getattr(layer, self.attr)())

    def __len__(self) -> int:
        return self.parent.n_layers

    def __iter__(self):
        for layer_idx in range(self.parent.n_layers):
            yield self[layer_idx]


class PagedKVCache(RuntimeKVCacheMixin):
    def __init__(
        self,
        batch: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        pagesize: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        native_cache_state=None,
        native_layer_states=None,
        backend_name: str = "paged",
    ):
        self.batch = batch
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.pagesize = max(int(pagesize), 1)
        self.dtype = dtype
        self.device = device
        self.backend = str(backend_name)
        self._native_cache = native_cache_state
        self._native_layers = native_layer_states
        if self._native_cache is None and self._native_layers is None:
            self._native_cache, self._native_layers = create_native_paged_kv_cache_state(
                batch=int(batch),
                n_layers=int(n_layers),
                n_kv_heads=int(n_kv_heads),
                head_dim=int(head_dim),
                pagesize=self.pagesize,
                dtype=dtype,
                device=device,
            )
            if self._native_cache is not None or self._native_layers is not None:
                self.backend = "native-paged"
        if self._has_native_state():
            self.K_pages = _NativeLayerTensorList(self, "k_pages")
            self.V_pages = _NativeLayerTensorList(self, "v_pages")
            self.block_tables = _NativeLayerTensorList(self, "block_table")
            self.lengths = _NativeLayerTensorList(self, "lengths")
            self.page_counts = _NativeLayerScalarList(self, "page_count")
        else:
            self.K_pages = [self._allocate_page_pool(0) for _ in range(n_layers)]
            self.V_pages = [self._allocate_page_pool(0) for _ in range(n_layers)]
            self.page_counts = [0 for _ in range(n_layers)]
            self.block_tables = [
                torch.empty(batch, 0, dtype=torch.long, device=self.device)
                for _ in range(n_layers)
            ]
            self.lengths = [
                torch.zeros(batch, dtype=torch.long, device=self.device)
                for _ in range(n_layers)
            ]

    def _allocate_page_pool(self, capacity_pages: int) -> torch.Tensor:
        return torch.empty(
            int(capacity_pages),
            self.n_kv_heads,
            self.pagesize,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def _has_native_state(self) -> bool:
        return self._native_cache is not None or self._native_layers is not None

    def layer_lengths(self, layer_idx: int) -> torch.Tensor:
        if self._native_cache is not None:
            return self._native_cache.lengths(layer_idx)
        if self._native_layers is not None:
            return self._native_layers[layer_idx].lengths()
        return self.lengths[layer_idx]

    def layer_page_count(self, layer_idx: int) -> int:
        if self._native_cache is not None:
            return int(self._native_cache.page_count(layer_idx))
        if self._native_layers is not None:
            return int(self._native_layers[layer_idx].page_count())
        return int(self.page_counts[layer_idx])

    def layer_block_table(self, layer_idx: int) -> torch.Tensor:
        if self._native_cache is not None:
            return self._native_cache.block_table(layer_idx)
        if self._native_layers is not None:
            return self._native_layers[layer_idx].block_table()
        return self.block_tables[layer_idx]

    def layer_pages(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._native_cache is not None:
            return self._native_cache.k_pages(layer_idx), self._native_cache.v_pages(layer_idx)
        if self._native_layers is not None:
            layer = self._native_layers[layer_idx]
            return layer.k_pages(), layer.v_pages()
        return self.K_pages[layer_idx], self.V_pages[layer_idx]

    def _reset_layer_storage(self, layer_idx: int) -> None:
        if self._native_cache is not None:
            self._native_cache.reset_layer(layer_idx)
            return
        if self._native_layers is not None:
            self._native_layers[layer_idx].reset()
            return
        self.K_pages[layer_idx] = self._allocate_page_pool(0)
        self.V_pages[layer_idx] = self._allocate_page_pool(0)
        self.page_counts[layer_idx] = 0
        self.block_tables[layer_idx] = torch.empty(self.batch, 0, dtype=torch.long, device=self.device)
        self.lengths[layer_idx] = torch.zeros(self.batch, dtype=torch.long, device=self.device)

    def _normalize_block_ids(self, block_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if block_ids is None:
            return torch.arange(self.batch, device=self.device, dtype=torch.long)
        block_ids_long = block_ids.to(device=self.device, dtype=torch.long).contiguous().view(-1)
        if block_ids_long.numel() == 0:
            return block_ids_long
        assert int(block_ids_long.min().item()) >= 0 and int(block_ids_long.max().item()) < self.batch, "block_ids out of range"
        assert int(torch.unique(block_ids_long).numel()) == int(block_ids_long.numel()), "block_ids must be unique"
        return block_ids_long

    def append(self, layer_idx: int, b: int, k_chunk: torch.Tensor, v_chunk: torch.Tensor):
        self.append_batch(
            layer_idx,
            k_chunk.unsqueeze(0),
            v_chunk.unsqueeze(0),
            block_ids=torch.tensor([int(b)], device=self.device, dtype=torch.long),
        )

    def append_batch(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        block_ids: Optional[torch.Tensor] = None,
    ) -> None:
        k = k.to(device=self.device, dtype=self.dtype).contiguous()
        v = v.to(device=self.device, dtype=self.dtype).contiguous()
        assert k.dim() == 4 and v.dim() == 4, "KV must be (B,H,T,D)"
        assert k.shape == v.shape, "K and V shapes must match"
        assert k.shape[1] == self.n_kv_heads and k.shape[3] == self.head_dim, "KV chunk shape mismatch"
        rows = int(k.shape[0])
        block_ids_long = self._normalize_block_ids(block_ids)
        if block_ids is None:
            assert rows == self.batch, "KV batch size mismatch"
        else:
            assert block_ids_long.numel() == rows, "block_ids must match KV batch size"

        total = int(k.shape[2])
        if total == 0:
            return

        if self._native_cache is not None:
            self._native_cache.append(layer_idx, k, v, block_ids_long)
            return
        if self._native_layers is not None:
            self._native_layers[layer_idx].append(k, v, block_ids_long)
            return

        self.K_pages[layer_idx], self.V_pages[layer_idx], self.block_tables[layer_idx], self.lengths[layer_idx], self.page_counts[layer_idx] = runtime_paged_kv_append(
            self.K_pages[layer_idx],
            self.V_pages[layer_idx],
            self.block_tables[layer_idx],
            self.lengths[layer_idx],
            int(self.page_counts[layer_idx]),
            k,
            v,
            block_ids_long,
        )

    def slice(self, layer_idx: int, b: int, start: int, end: int):
        k, v = self.read_batch(
            layer_idx,
            int(start),
            int(end),
            block_ids=torch.tensor([int(b)], device=self.device, dtype=torch.long),
        )
        if k.shape[2] == 0:
            return None, None
        return k[0], v[0]

    def read_batch(
        self,
        layer_idx: int,
        start: int,
        end: int,
        block_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = max(int(start), 0)
        e = max(int(end), s)
        gather_seq = e - s
        block_ids_long = self._normalize_block_ids(block_ids)
        if block_ids_long.numel() == 0:
            empty = torch.empty(
                0,
                self.n_kv_heads,
                0 if gather_seq == 0 else gather_seq,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty.clone()
        rows = int(block_ids_long.numel())
        if gather_seq == 0:
            empty = torch.empty(
                rows,
                self.n_kv_heads,
                0,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            return empty, empty.clone()

        if self._native_cache is not None:
            return self._native_cache.read_range(layer_idx, s, e, block_ids_long)
        if self._native_layers is not None:
            k, v = self._native_layers[layer_idx].read_range(s, e, block_ids_long)
            return k, v

        live_lengths = self.layer_lengths(layer_idx).index_select(0, block_ids_long)
        page_count = self.layer_page_count(layer_idx)
        block_table = self.layer_block_table(layer_idx).index_select(0, block_ids_long).contiguous().clone()
        k_pages, v_pages = self.layer_pages(layer_idx)
        k_pages = k_pages[:page_count]
        v_pages = v_pages[:page_count]
        k = runtime_paged_kv_read_range(k_pages, block_table, live_lengths, s, e)
        v = runtime_paged_kv_read_range(v_pages, block_table, live_lengths, s, e)
        return k, v

    def read(self, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.read_batch(layer_idx, start, end, block_ids=None)

    def read_last_batch(
        self,
        layer_idx: int,
        keep: int,
        block_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep = max(int(keep), 0)
        block_ids_long = self._normalize_block_ids(block_ids)
        if block_ids_long.numel() == 0:
            empty = torch.empty(0, self.n_kv_heads, 0, self.head_dim, dtype=self.dtype, device=self.device)
            empty_lengths = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty.clone(), block_ids_long, empty_lengths

        live_lengths = self.layer_lengths(layer_idx).index_select(0, block_ids_long)
        kept_lengths = torch.clamp(live_lengths, min=0, max=keep)
        max_keep = int(kept_lengths.max().item()) if kept_lengths.numel() > 0 else 0
        if max_keep == 0:
            empty = torch.empty(block_ids_long.numel(), self.n_kv_heads, 0, self.head_dim, dtype=self.dtype, device=self.device)
            return empty, empty.clone(), block_ids_long, kept_lengths

        if self._native_cache is not None:
            return self._native_cache.read_last(layer_idx, keep, block_ids_long)
        if self._native_layers is not None:
            k, v, ids_out, kept_lengths_out = self._native_layers[layer_idx].read_last(keep, block_ids_long)
            return k, v, ids_out, kept_lengths_out

        page_count = self.layer_page_count(layer_idx)
        block_table = self.layer_block_table(layer_idx).index_select(0, block_ids_long).contiguous().clone()
        block_table.clamp_min_(0)
        k_pages, v_pages = self.layer_pages(layer_idx)
        k_pages = k_pages[:page_count]
        v_pages = v_pages[:page_count]
        k, kept_lengths_out = runtime_paged_kv_read_last(k_pages, block_table, live_lengths, keep)
        v, kept_lengths_v = runtime_paged_kv_read_last(v_pages, block_table, live_lengths, keep)
        assert torch.equal(kept_lengths_out, kept_lengths_v), "kept lengths mismatch between K and V paged reads"
        return k, v, block_ids_long, kept_lengths_out

    def evict(self, max_tokens: int, policy: str = "fifo"):
        keep = max(int(max_tokens), 0)
        if policy not in ("fifo", "sliding-window", "lru"):
            return
        if self._native_cache is not None:
            self._native_cache.compact(keep)
            return
        for layer_idx in range(self.n_layers):
            layer_lengths = self.layer_lengths(layer_idx)
            if bool((layer_lengths <= keep).all().item()):
                continue
            if self._native_layers is not None:
                self._native_layers[layer_idx].compact(keep)
                continue
            k_pages, v_pages = self.layer_pages(layer_idx)
            k_pages, v_pages, next_table, next_lengths = runtime_paged_kv_compact(
                k_pages[: self.layer_page_count(layer_idx)],
                v_pages[: self.layer_page_count(layer_idx)],
                self.layer_block_table(layer_idx),
                layer_lengths,
                keep,
            )
            self.K_pages[layer_idx] = k_pages
            self.V_pages[layer_idx] = v_pages
            self.page_counts[layer_idx] = int(k_pages.shape[0])
            self.block_tables[layer_idx] = next_table
            self.lengths[layer_idx] = next_lengths


class ContiguousKVCache(RuntimeKVCacheMixin):
    def __init__(
        self,
        batch: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        pagesize: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        backend_name: str = "contiguous",
    ):
        self.batch = batch
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.pagesize = max(int(pagesize), 1)
        self.dtype = dtype
        self.device = device
        self.backend = str(backend_name)
        self.K: list[list[torch.Tensor | None]] = [[None for _ in range(batch)] for _ in range(n_layers)]
        self.V: list[list[torch.Tensor | None]] = [[None for _ in range(batch)] for _ in range(n_layers)]
        self.lengths = [[0 for _ in range(batch)] for _ in range(n_layers)]
        self.capacities = [[0 for _ in range(batch)] for _ in range(n_layers)]

    def layer_lengths(self, layer_idx: int) -> torch.Tensor:
        return torch.tensor(self.lengths[layer_idx], dtype=torch.long, device=self.device)

    def _allocate(self, capacity: int) -> torch.Tensor:
        return torch.empty(
            self.n_kv_heads,
            int(capacity),
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def _ensure_capacity(self, layer_idx: int, batch_idx: int, needed: int) -> None:
        cur_cap = int(self.capacities[layer_idx][batch_idx])
        if cur_cap >= int(needed):
            return
        new_cap = max(self.pagesize, cur_cap if cur_cap > 0 else self.pagesize)
        while new_cap < int(needed):
            new_cap *= 2
        k_new = self._allocate(new_cap)
        v_new = self._allocate(new_cap)
        cur_len = int(self.lengths[layer_idx][batch_idx])
        k_prev = self.K[layer_idx][batch_idx]
        v_prev = self.V[layer_idx][batch_idx]
        if k_prev is not None and cur_len > 0:
            runtime_kv_cache_write(k_new, k_prev[:, :cur_len, :].contiguous(), 0)
        if v_prev is not None and cur_len > 0:
            runtime_kv_cache_write(v_new, v_prev[:, :cur_len, :].contiguous(), 0)
        self.K[layer_idx][batch_idx] = k_new
        self.V[layer_idx][batch_idx] = v_new
        self.capacities[layer_idx][batch_idx] = int(new_cap)

    def append(self, layer_idx: int, batch_idx: int, k_chunk: torch.Tensor, v_chunk: torch.Tensor):
        k_chunk = k_chunk.to(device=self.device, dtype=self.dtype).contiguous()
        v_chunk = v_chunk.to(device=self.device, dtype=self.dtype).contiguous()
        cur_len = int(self.lengths[layer_idx][batch_idx])
        next_len = cur_len + int(k_chunk.shape[1])
        self._ensure_capacity(layer_idx, batch_idx, next_len)
        k_buf = self.K[layer_idx][batch_idx]
        v_buf = self.V[layer_idx][batch_idx]
        assert k_buf is not None and v_buf is not None
        runtime_kv_cache_write(k_buf, k_chunk, cur_len)
        runtime_kv_cache_write(v_buf, v_chunk, cur_len)
        self.lengths[layer_idx][batch_idx] = next_len

    def append_batch(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        block_ids: Optional[torch.Tensor] = None,
    ) -> None:
        assert k.dim() == 4 and v.dim() == 4, "KV must be (B,H,T,D)"
        assert k.shape == v.shape, "K and V shapes must match"
        assert k.shape[1] == self.n_kv_heads and k.shape[3] == self.head_dim, "KV chunk shape mismatch"
        if block_ids is None:
            assert int(k.shape[0]) == self.batch, "KV batch size mismatch"
            for batch_idx in range(self.batch):
                self.append(layer_idx, batch_idx, k[batch_idx], v[batch_idx])
            return
        block_ids_long = block_ids.to(device=self.device, dtype=torch.long).contiguous().view(-1)
        assert int(block_ids_long.numel()) == int(k.shape[0]), "block_ids must match KV batch size"
        for idx, batch_idx in enumerate(block_ids_long.tolist()):
            self.append(layer_idx, int(batch_idx), k[idx], v[idx])

    def slice(self, layer_idx: int, batch_idx: int, start: int, end: int):
        k = self.K[layer_idx][batch_idx]
        v = self.V[layer_idx][batch_idx]
        if k is None or v is None:
            return None, None
        s = max(int(start), 0)
        e = min(int(end), int(self.lengths[layer_idx][batch_idx]))
        if s >= e:
            return None, None
        return k[:, s:e, :], v[:, s:e, :]

    def read_batch(
        self,
        layer_idx: int,
        start: int,
        end: int,
        block_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = (
            torch.arange(self.batch, device=self.device, dtype=torch.long)
            if block_ids is None
            else block_ids.to(device=self.device, dtype=torch.long).contiguous().view(-1)
        )
        rows = int(ids.numel())
        s = max(int(start), 0)
        e = max(int(end), s)
        width = e - s
        if rows == 0:
            empty = torch.empty(0, self.n_kv_heads, width, self.head_dim, dtype=self.dtype, device=self.device)
            return empty, empty.clone()
        out_k = torch.empty(rows, self.n_kv_heads, width, self.head_dim, dtype=self.dtype, device=self.device)
        out_v = torch.empty(rows, self.n_kv_heads, width, self.head_dim, dtype=self.dtype, device=self.device)
        for row, batch_idx in enumerate(ids.tolist()):
            k_row, v_row = self.slice(layer_idx, int(batch_idx), s, e)
            if k_row is None or v_row is None:
                out_k[row].zero_()
                out_v[row].zero_()
                continue
            live = int(k_row.shape[1])
            out_k[row].zero_()
            out_v[row].zero_()
            out_k[row, :, :live, :].copy_(k_row)
            out_v[row, :, :live, :].copy_(v_row)
        return out_k, out_v

    def read(self, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.read_batch(layer_idx, start, end, block_ids=None)

    def evict(self, max_tokens: int, policy: str = "fifo"):
        keep = max(int(max_tokens), 0)
        for layer_idx in range(self.n_layers):
            for batch_idx in range(self.batch):
                k = self.K[layer_idx][batch_idx]
                v = self.V[layer_idx][batch_idx]
                if k is None or v is None:
                    self.lengths[layer_idx][batch_idx] = 0
                    continue
                live = int(self.lengths[layer_idx][batch_idx])
                if live <= keep:
                    self.lengths[layer_idx][batch_idx] = live
                    continue
                if policy in ("fifo", "sliding-window", "lru"):
                    if keep > 0:
                        tail_k = k[:, live - keep : live, :].contiguous()
                        tail_v = v[:, live - keep : live, :].contiguous()
                        runtime_kv_cache_write(k, tail_k, 0)
                        runtime_kv_cache_write(v, tail_v, 0)
                    else:
                        self.K[layer_idx][batch_idx] = None
                        self.V[layer_idx][batch_idx] = None
                        self.capacities[layer_idx][batch_idx] = 0
                    self.lengths[layer_idx][batch_idx] = keep


def init_kv_cache(batch: int, n_layers: int, n_kv_heads: int, head_dim: int, pagesize: int, dtype: torch.dtype, device: torch.device):
    from runtime.cache import KVCacheSpec, create_kv_cache

    return create_kv_cache(
        KVCacheSpec(
            batch=int(batch),
            n_layers=int(n_layers),
            n_kv_heads=int(n_kv_heads),
            head_dim=int(head_dim),
            pagesize=max(int(pagesize), 1),
            dtype=dtype,
            device=torch.device(device),
            backend=(os.getenv("MODEL_STACK_KV_BACKEND", "auto") or "auto"),
        )
    )


def kv_cache_append(cache: PagedKVCache, layer_idx: int, k_chunk: torch.Tensor, v_chunk: torch.Tensor, block_ids: Optional[torch.Tensor] = None):
    if hasattr(cache, "append_batch"):
        cache.append_batch(layer_idx, k_chunk, v_chunk, block_ids=block_ids)
        return
    batch = k_chunk.shape[0]
    ids = block_ids if block_ids is not None else torch.arange(batch, device=k_chunk.device)
    for idx, batch_idx in enumerate(ids.tolist()):
        cache.append(layer_idx, batch_idx, k_chunk[idx], v_chunk[idx])


def kv_cache_slice(
    cache: PagedKVCache,
    layer_idx: int,
    start: int,
    end: int,
    block_ids: Optional[torch.Tensor] = None,
):
    if hasattr(cache, "read_batch"):
        k, v = cache.read_batch(layer_idx, int(start), int(end), block_ids=block_ids)
        return [k[b] for b in range(k.shape[0])], [v[b] for b in range(v.shape[0])]
    ids = block_ids if block_ids is not None else torch.arange(cache.batch, device=cache.device)
    ks, vs = [], []
    for batch_idx in ids.tolist():
        k_row, v_row = cache.slice(layer_idx, batch_idx, start, end)
        ks.append(k_row)
        vs.append(v_row)
    return ks, vs


def kv_cache_evict(cache: PagedKVCache, max_tokens: int, policy: str = "fifo"):
    cache.evict(max_tokens, policy)


__all__ = [
    "ContiguousKVCache",
    "PagedKVCache",
    "init_kv_cache",
    "kv_cache_append",
    "kv_cache_evict",
    "kv_cache_slice",
]
