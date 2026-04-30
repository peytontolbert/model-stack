from __future__ import annotations

import os
from typing import Optional

import torch

from runtime.cache import RuntimeKVCacheMixin
from runtime.native import create_native_paged_kv_cache_state, has_native_op, native_module
from runtime.ops import kv_cache_write as runtime_kv_cache_write
from runtime.ops import paged_kv_append as runtime_paged_kv_append
from runtime.ops import paged_kv_compact as runtime_paged_kv_compact
from runtime.ops import paged_attention_decode as runtime_paged_attention_decode
from runtime.ops import paged_kv_read_last as runtime_paged_kv_read_last
from runtime.ops import paged_kv_read_range as runtime_paged_kv_read_range


_INT3_PACK_SHIFTS = torch.tensor([0, 3, 6, 9, 12, 15, 18, 21], dtype=torch.int32)


def _int3_shifts(device: torch.device) -> torch.Tensor:
    return _INT3_PACK_SHIFTS.to(device=device)


def quantize_pack_int3_lastdim(x: torch.Tensor, *, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack signed int3 values for the last dimension using per-vector scales.

    Values are quantized to [-3, 3], stored as unsigned codes [0, 6], and packed
    eight codes into three bytes. The returned scale has shape x.shape[:-1].
    """
    if x.ndim < 1:
        raise ValueError("int3 KV quantization expects rank >= 1")
    original_last_dim = int(x.shape[-1])
    if original_last_dim <= 0:
        raise ValueError("int3 KV quantization expects non-empty last dimension")
    if (
        x.is_cuda
        and os.environ.get("MODEL_STACK_DISABLE_NATIVE_INT3_KV", "0") != "1"
        and has_native_op("int3_kv_pack")
    ):
        module = native_module()
        if module is not None and hasattr(module, "int3_kv_pack_forward"):
            packed, scale = module.int3_kv_pack_forward(x)
            return packed, scale, original_last_dim
    x_f = x.float()
    scale = (x_f.abs().amax(dim=-1).clamp_min(float(eps)) / 3.0).to(dtype=torch.float32)
    q = torch.round(x_f / scale.unsqueeze(-1)).clamp_(-3, 3).to(dtype=torch.int16) + 3
    pad = (-original_last_dim) % 8
    if pad:
        q = torch.nn.functional.pad(q, (0, pad), value=3)
    groups = q.reshape(*q.shape[:-1], q.shape[-1] // 8, 8).to(dtype=torch.int32)
    words = (groups << _int3_shifts(q.device)).sum(dim=-1).to(dtype=torch.int32)
    packed = torch.stack(
        (
            (words & 0xFF).to(dtype=torch.uint8),
            ((words >> 8) & 0xFF).to(dtype=torch.uint8),
            ((words >> 16) & 0xFF).to(dtype=torch.uint8),
        ),
        dim=-1,
    ).reshape(*words.shape[:-1], words.shape[-1] * 3).contiguous()
    return packed, scale.contiguous(), original_last_dim


def unpack_dequantize_int3_lastdim(
    packed: torch.Tensor,
    scale: torch.Tensor,
    original_last_dim: int,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if packed.ndim < 1:
        raise ValueError("packed int3 tensor expects rank >= 1")
    if int(original_last_dim) <= 0:
        raise ValueError("original_last_dim must be positive")
    if int(packed.shape[-1]) % 3 != 0:
        raise ValueError("packed int3 last dimension must be a multiple of 3 bytes")
    if (
        packed.is_cuda
        and os.environ.get("MODEL_STACK_DISABLE_NATIVE_INT3_KV", "0") != "1"
        and has_native_op("int3_kv_dequantize")
    ):
        module = native_module()
        if module is not None and hasattr(module, "int3_kv_dequantize_forward"):
            return module.int3_kv_dequantize_forward(packed, scale, int(original_last_dim), dtype or torch.float32)
    byte_groups = packed.reshape(*packed.shape[:-1], packed.shape[-1] // 3, 3).to(dtype=torch.int32)
    words = byte_groups[..., 0] | (byte_groups[..., 1] << 8) | (byte_groups[..., 2] << 16)
    codes = ((words.unsqueeze(-1) >> _int3_shifts(packed.device)) & 0x7).to(dtype=torch.int16)
    signed = (codes.reshape(*packed.shape[:-1], -1)[..., : int(original_last_dim)] - 3).to(dtype=torch.float32)
    out = signed * scale.to(device=packed.device, dtype=torch.float32).unsqueeze(-1)
    return out.to(dtype=dtype or torch.float32)


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
        self._refresh_storage_views()

    def _allocate_page_pool(self, capacity_pages: int) -> torch.Tensor:
        return torch.empty(
            int(capacity_pages),
            self.n_kv_heads,
            self.pagesize,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def _refresh_storage_views(self) -> None:
        if self._has_native_state():
            self.K_pages = _NativeLayerTensorList(self, "k_pages")
            self.V_pages = _NativeLayerTensorList(self, "v_pages")
            self.block_tables = _NativeLayerTensorList(self, "block_table")
            self.lengths = _NativeLayerTensorList(self, "lengths")
            self.page_counts = _NativeLayerScalarList(self, "page_count")
            return
        self.K_pages = [self._allocate_page_pool(0) for _ in range(self.n_layers)]
        self.V_pages = [self._allocate_page_pool(0) for _ in range(self.n_layers)]
        self.page_counts = [0 for _ in range(self.n_layers)]
        self.block_tables = [
            torch.empty(self.batch, 0, dtype=torch.long, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.lengths = [
            torch.zeros(self.batch, dtype=torch.long, device=self.device)
            for _ in range(self.n_layers)
        ]

    def _has_native_state(self) -> bool:
        return self._native_cache is not None or self._native_layers is not None

    def _replace_from(self, other: "PagedKVCache") -> None:
        self.batch = other.batch
        self.n_layers = other.n_layers
        self.n_kv_heads = other.n_kv_heads
        self.head_dim = other.head_dim
        self.pagesize = other.pagesize
        self.dtype = other.dtype
        self.device = other.device
        self.backend = other.backend
        self._native_cache = getattr(other, "_native_cache", None)
        self._native_layers = getattr(other, "_native_layers", None)
        if self._has_native_state():
            self._refresh_storage_views()
            return
        self.K_pages = other.K_pages
        self.V_pages = other.V_pages
        self.page_counts = other.page_counts
        self.block_tables = other.block_tables
        self.lengths = other.lengths

    def layer_lengths(self, layer_idx: int) -> torch.Tensor:
        if self._native_cache is not None:
            return self._native_cache.lengths(layer_idx)
        if self._native_layers is not None:
            return self._native_layers[layer_idx].lengths()
        return self.lengths[layer_idx]

    def layer_max_length(self, layer_idx: int) -> int:
        if self._native_cache is not None:
            return int(self._native_cache.max_length(layer_idx))
        if self._native_layers is not None:
            layer = self._native_layers[layer_idx]
            if hasattr(layer, "max_length"):
                return int(layer.max_length())
        lengths = self.layer_lengths(layer_idx)
        return int(lengths.max().item()) if lengths.numel() > 0 else 0

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

    def reorder_rows_(self, row_ids: torch.Tensor):
        row_ids_long = row_ids.to(device=self.device, dtype=torch.long).contiguous().view(-1)
        if self._native_cache is not None:
            native_cache = self._native_cache
            if hasattr(native_cache, "reorder_rows_"):
                native_cache.reorder_rows_(row_ids_long)
            elif hasattr(native_cache, "clone_rows"):
                self._native_cache = native_cache.clone_rows(row_ids_long)
            else:
                self._replace_from(clone_kv_cache_rows(self, row_ids_long))
                return self
            self.batch = int(row_ids_long.numel())
            self._native_layers = None
            self.backend = getattr(self, "backend", "native-paged")
            self._refresh_storage_views()
            return self
        if self._native_layers is not None:
            native_layers = self._native_layers
            if all(hasattr(layer, "reorder_rows_") for layer in native_layers):
                for layer in native_layers:
                    layer.reorder_rows_(row_ids_long)
            elif all(hasattr(layer, "clone_rows") for layer in native_layers):
                self._native_layers = [layer.clone_rows(row_ids_long) for layer in native_layers]
            else:
                self._replace_from(clone_kv_cache_rows(self, row_ids_long))
                return self
            self.batch = int(row_ids_long.numel())
            self._native_cache = None
            self.backend = getattr(self, "backend", "native-paged")
            self._refresh_storage_views()
            return self
        self._replace_from(clone_kv_cache_rows(self, row_ids_long))
        return self


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

    def layer_max_length(self, layer_idx: int) -> int:
        return max(self.lengths[layer_idx], default=0)

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

    def reorder_rows_(self, row_ids: torch.Tensor):
        cloned = clone_kv_cache_rows(self, row_ids)
        self.batch = cloned.batch
        self.n_layers = cloned.n_layers
        self.n_kv_heads = cloned.n_kv_heads
        self.head_dim = cloned.head_dim
        self.pagesize = cloned.pagesize
        self.dtype = cloned.dtype
        self.device = cloned.device
        self.backend = cloned.backend
        self.K = cloned.K
        self.V = cloned.V
        self.lengths = cloned.lengths
        self.capacities = cloned.capacities
        return self


class Int3ContiguousKVCache(RuntimeKVCacheMixin):
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
        backend_name: str = "int3-contiguous",
    ):
        self.batch = int(batch)
        self.n_layers = int(n_layers)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = int(head_dim)
        self.pagesize = max(int(pagesize), 1)
        self.dtype = dtype
        self.device = device
        self.backend = str(backend_name)
        self.K: list[list[torch.Tensor | None]] = [[None for _ in range(self.batch)] for _ in range(self.n_layers)]
        self.V: list[list[torch.Tensor | None]] = [[None for _ in range(self.batch)] for _ in range(self.n_layers)]
        self.K_scale: list[list[torch.Tensor | None]] = [[None for _ in range(self.batch)] for _ in range(self.n_layers)]
        self.V_scale: list[list[torch.Tensor | None]] = [[None for _ in range(self.batch)] for _ in range(self.n_layers)]
        self.lengths = [[0 for _ in range(self.batch)] for _ in range(self.n_layers)]

    def layer_lengths(self, layer_idx: int) -> torch.Tensor:
        return torch.tensor(self.lengths[int(layer_idx)], dtype=torch.long, device=self.device)

    def layer_max_length(self, layer_idx: int) -> int:
        return max(self.lengths[int(layer_idx)], default=0)

    def _pack_chunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_local = x.to(device=self.device, dtype=self.dtype).contiguous()
        packed, scale, dim = quantize_pack_int3_lastdim(x_local)
        if dim != self.head_dim:
            raise ValueError(f"int3 KV chunk head_dim mismatch: got {dim}, expected {self.head_dim}")
        return packed, scale

    def _unpack(self, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return unpack_dequantize_int3_lastdim(packed, scale, self.head_dim, dtype=self.dtype)

    def append(self, layer_idx: int, batch_idx: int, k_chunk: torch.Tensor, v_chunk: torch.Tensor):
        layer_idx = int(layer_idx)
        batch_idx = int(batch_idx)
        k_pack, k_scale = self._pack_chunk(k_chunk)
        v_pack, v_scale = self._pack_chunk(v_chunk)
        if self.K[layer_idx][batch_idx] is None:
            self.K[layer_idx][batch_idx] = k_pack
            self.V[layer_idx][batch_idx] = v_pack
            self.K_scale[layer_idx][batch_idx] = k_scale
            self.V_scale[layer_idx][batch_idx] = v_scale
        else:
            self.K[layer_idx][batch_idx] = torch.cat((self.K[layer_idx][batch_idx], k_pack), dim=1).contiguous()
            self.V[layer_idx][batch_idx] = torch.cat((self.V[layer_idx][batch_idx], v_pack), dim=1).contiguous()
            self.K_scale[layer_idx][batch_idx] = torch.cat((self.K_scale[layer_idx][batch_idx], k_scale), dim=1).contiguous()
            self.V_scale[layer_idx][batch_idx] = torch.cat((self.V_scale[layer_idx][batch_idx], v_scale), dim=1).contiguous()
        self.lengths[layer_idx][batch_idx] += int(k_chunk.shape[1])

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
        ids = (
            torch.arange(self.batch, device=self.device, dtype=torch.long)
            if block_ids is None
            else block_ids.to(device=self.device, dtype=torch.long).contiguous().view(-1)
        )
        assert int(ids.numel()) == int(k.shape[0]), "block_ids must match KV batch size"
        for idx, batch_idx in enumerate(ids.tolist()):
            self.append(int(layer_idx), int(batch_idx), k[idx], v[idx])

    def slice(self, layer_idx: int, batch_idx: int, start: int, end: int):
        layer_idx = int(layer_idx)
        batch_idx = int(batch_idx)
        k = self.K[layer_idx][batch_idx]
        v = self.V[layer_idx][batch_idx]
        ks = self.K_scale[layer_idx][batch_idx]
        vs = self.V_scale[layer_idx][batch_idx]
        if k is None or v is None or ks is None or vs is None:
            return None, None
        s = max(int(start), 0)
        e = min(int(end), int(self.lengths[layer_idx][batch_idx]))
        if s >= e:
            return None, None
        return self._unpack(k[:, s:e, :], ks[:, s:e]), self._unpack(v[:, s:e, :], vs[:, s:e])

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
        out_k = torch.zeros(rows, self.n_kv_heads, width, self.head_dim, dtype=self.dtype, device=self.device)
        out_v = torch.zeros_like(out_k)
        for row, batch_idx in enumerate(ids.tolist()):
            k_row, v_row = self.slice(layer_idx, int(batch_idx), s, e)
            if k_row is None or v_row is None:
                continue
            live = int(k_row.shape[1])
            out_k[row, :, :live, :].copy_(k_row)
            out_v[row, :, :live, :].copy_(v_row)
        return out_k, out_v

    def read(self, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.read_batch(layer_idx, start, end, block_ids=None)

    def evict(self, max_tokens: int, policy: str = "fifo"):
        keep = max(int(max_tokens), 0)
        if policy not in ("fifo", "sliding-window", "lru"):
            return
        for layer_idx in range(self.n_layers):
            for batch_idx in range(self.batch):
                live = int(self.lengths[layer_idx][batch_idx])
                if live <= keep:
                    continue
                if keep <= 0:
                    self.K[layer_idx][batch_idx] = None
                    self.V[layer_idx][batch_idx] = None
                    self.K_scale[layer_idx][batch_idx] = None
                    self.V_scale[layer_idx][batch_idx] = None
                else:
                    self.K[layer_idx][batch_idx] = self.K[layer_idx][batch_idx][:, live - keep : live, :].contiguous()
                    self.V[layer_idx][batch_idx] = self.V[layer_idx][batch_idx][:, live - keep : live, :].contiguous()
                    self.K_scale[layer_idx][batch_idx] = self.K_scale[layer_idx][batch_idx][:, live - keep : live].contiguous()
                    self.V_scale[layer_idx][batch_idx] = self.V_scale[layer_idx][batch_idx][:, live - keep : live].contiguous()
                self.lengths[layer_idx][batch_idx] = keep

    def reorder_rows_(self, row_ids: torch.Tensor):
        cloned = clone_kv_cache_rows(self, row_ids)
        self.batch = cloned.batch
        self.n_layers = cloned.n_layers
        self.n_kv_heads = cloned.n_kv_heads
        self.head_dim = cloned.head_dim
        self.pagesize = cloned.pagesize
        self.dtype = cloned.dtype
        self.device = cloned.device
        self.backend = cloned.backend
        self.K = cloned.K
        self.V = cloned.V
        self.K_scale = cloned.K_scale
        self.V_scale = cloned.V_scale
        self.lengths = cloned.lengths
        return self


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


def clone_kv_cache_rows(cache, row_ids: torch.Tensor):
    from runtime.cache import KVCacheSpec, create_kv_cache

    row_ids_long = row_ids.to(device=cache.device, dtype=torch.long).contiguous().view(-1)
    native_cache = getattr(cache, "_native_cache", None)
    if native_cache is not None and hasattr(native_cache, "clone_rows"):
        return PagedKVCache(
            int(row_ids_long.numel()),
            int(cache.n_layers),
            int(cache.n_kv_heads),
            int(cache.head_dim),
            int(cache.pagesize),
            cache.dtype,
            torch.device(cache.device),
            native_cache_state=native_cache.clone_rows(row_ids_long),
            native_layer_states=None,
            backend_name=getattr(cache, "backend", "native-paged"),
        )
    native_layers = getattr(cache, "_native_layers", None)
    if native_layers is not None and all(hasattr(layer, "clone_rows") for layer in native_layers):
        return PagedKVCache(
            int(row_ids_long.numel()),
            int(cache.n_layers),
            int(cache.n_kv_heads),
            int(cache.head_dim),
            int(cache.pagesize),
            cache.dtype,
            torch.device(cache.device),
            native_cache_state=None,
            native_layer_states=[layer.clone_rows(row_ids_long) for layer in native_layers],
            backend_name=getattr(cache, "backend", "native-paged"),
        )

    spec = KVCacheSpec(
        batch=int(row_ids_long.numel()),
        n_layers=int(cache.n_layers),
        n_kv_heads=int(cache.n_kv_heads),
        head_dim=int(cache.head_dim),
        pagesize=int(cache.pagesize),
        dtype=cache.dtype,
        device=torch.device(cache.device),
        backend=getattr(cache, "backend", "auto"),
    )
    next_cache = create_kv_cache(spec)
    if row_ids_long.numel() == 0:
        return next_cache

    target_ids = torch.arange(int(row_ids_long.numel()), device=spec.device, dtype=torch.long)
    for layer_idx in range(int(cache.n_layers)):
        lengths = cache.layer_lengths(layer_idx).to(device=spec.device, dtype=torch.long).index_select(0, row_ids_long)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len <= 0:
            continue
        if hasattr(cache, "read_batch"):
            k, v = cache.read_batch(layer_idx, 0, max_len, block_ids=row_ids_long)
        else:
            raise ValueError("clone_kv_cache_rows requires a cache implementation with read_batch support")
        for cur_len in torch.unique(lengths, sorted=True).tolist():
            cur_len = int(cur_len)
            if cur_len <= 0:
                continue
            rows = torch.nonzero(lengths == cur_len, as_tuple=False).view(-1)
            next_cache.append_batch(
                layer_idx,
                k.index_select(0, rows)[:, :, :cur_len, :].contiguous(),
                v.index_select(0, rows)[:, :, :cur_len, :].contiguous(),
                block_ids=target_ids.index_select(0, rows),
            )
    return next_cache


def reorder_kv_cache_rows_(cache, row_ids: torch.Tensor):
    if hasattr(cache, "reorder_rows_"):
        out = cache.reorder_rows_(row_ids)
        return cache if out is None else out
    return clone_kv_cache_rows(cache, row_ids)


def concat_kv_caches(caches: list[object | None]):
    from runtime.cache import KVCacheSpec, create_kv_cache

    live = [cache for cache in caches if cache is not None]
    if not live:
        return None

    first = live[0]
    backend = getattr(first, "backend", "auto")
    n_layers = int(getattr(first, "n_layers"))
    n_kv_heads = int(getattr(first, "n_kv_heads"))
    head_dim = int(getattr(first, "head_dim"))
    pagesize = int(getattr(first, "pagesize"))
    dtype = first.dtype
    device = torch.device(first.device)

    total_batch = 0
    for cache in live:
        if (
            int(getattr(cache, "n_layers")) != n_layers
            or int(getattr(cache, "n_kv_heads")) != n_kv_heads
            or int(getattr(cache, "head_dim")) != head_dim
            or int(getattr(cache, "pagesize")) != pagesize
            or cache.dtype != dtype
            or torch.device(cache.device) != device
        ):
            raise ValueError("concat_kv_caches requires all caches to share the same spec")
        total_batch += int(getattr(cache, "batch"))

    next_cache = create_kv_cache(
        KVCacheSpec(
            batch=int(total_batch),
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            pagesize=pagesize,
            dtype=dtype,
            device=device,
            backend=backend,
        )
    )
    if total_batch == 0:
        return next_cache

    row_offset = 0
    for cache in live:
        cache_batch = int(getattr(cache, "batch"))
        if cache_batch <= 0:
            continue
        target_rows = torch.arange(
            row_offset,
            row_offset + cache_batch,
            device=device,
            dtype=torch.long,
        )
        for layer_idx in range(n_layers):
            lengths = cache.layer_lengths(layer_idx).to(device=device, dtype=torch.long)
            max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
            if max_len <= 0:
                continue
            if hasattr(cache, "read_batch"):
                source_rows = torch.arange(cache_batch, device=device, dtype=torch.long)
                k, v = cache.read_batch(layer_idx, 0, max_len, block_ids=source_rows)
            else:
                raise ValueError("concat_kv_caches requires a cache implementation with read_batch support")
            unique_lengths = sorted(
                {
                    int(length)
                    for length in lengths.detach().to(torch.long).cpu().tolist()
                    if int(length) > 0
                }
            )
            for length in unique_lengths:
                rows = torch.nonzero(lengths == int(length), as_tuple=False).view(-1)
                if rows.numel() == 0:
                    continue
                next_cache.append_batch(
                    layer_idx,
                    k.index_select(0, rows).slice(2, 0, int(length)).contiguous(),
                    v.index_select(0, rows).slice(2, 0, int(length)).contiguous(),
                    block_ids=target_rows.index_select(0, rows),
                )
        row_offset += cache_batch
    return next_cache


def split_kv_cache_rows(cache, row_sizes: list[int]):
    if cache is None:
        return [None for _ in row_sizes]
    out = []
    row_offset = 0
    device = torch.device(cache.device)
    for size in row_sizes:
        count = int(size)
        row_ids = torch.arange(row_offset, row_offset + count, device=device, dtype=torch.long)
        out.append(clone_kv_cache_rows(cache, row_ids))
        row_offset += count
    if row_offset != int(getattr(cache, "batch")):
        raise ValueError("split_kv_cache_rows row_sizes must sum to the cache batch size")
    return out


def truncate_kv_cache_prefix(cache, max_tokens: int):
    from runtime.cache import KVCacheSpec, create_kv_cache

    if cache is None:
        return None

    keep = max(int(max_tokens), 0)
    spec = KVCacheSpec(
        batch=int(getattr(cache, "batch")),
        n_layers=int(getattr(cache, "n_layers")),
        n_kv_heads=int(getattr(cache, "n_kv_heads")),
        head_dim=int(getattr(cache, "head_dim")),
        pagesize=int(getattr(cache, "pagesize")),
        dtype=cache.dtype,
        device=torch.device(cache.device),
        backend=getattr(cache, "backend", "auto"),
    )
    next_cache = create_kv_cache(spec)
    if keep <= 0 or int(spec.batch) <= 0:
        return next_cache

    source_rows = torch.arange(int(spec.batch), device=spec.device, dtype=torch.long)
    for layer_idx in range(int(spec.n_layers)):
        lengths = cache.layer_lengths(layer_idx).to(device=spec.device, dtype=torch.long).clamp_(min=0, max=keep)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len <= 0:
            continue
        if hasattr(cache, "read_batch"):
            k, v = cache.read_batch(layer_idx, 0, max_len, block_ids=source_rows)
        else:
            raise ValueError("truncate_kv_cache_prefix requires a cache implementation with read_batch support")
        unique_lengths = sorted(
            {
                int(length)
                for length in lengths.detach().to(torch.long).cpu().tolist()
                if int(length) > 0
            }
        )
        for length in unique_lengths:
            rows = torch.nonzero(lengths == int(length), as_tuple=False).view(-1)
            if rows.numel() == 0:
                continue
            next_cache.append_batch(
                layer_idx,
                k.index_select(0, rows)[:, :, : int(length), :].contiguous(),
                v.index_select(0, rows)[:, :, : int(length), :].contiguous(),
                block_ids=source_rows.index_select(0, rows),
            )
    return next_cache


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


def paged_attention_decode(
    cache: PagedKVCache,
    layer_idx: int,
    q: torch.Tensor,
    k_chunk: torch.Tensor,
    v_chunk: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    scale: float | None = None,
    block_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    cache.append_batch(layer_idx, k_chunk, v_chunk, block_ids=block_ids)
    block_ids_long = cache._normalize_block_ids(block_ids)
    if block_ids_long.numel() == 0:
        return torch.empty_like(q)
    page_count = cache.layer_page_count(layer_idx)
    live_lengths = cache.layer_lengths(layer_idx).index_select(0, block_ids_long)
    block_table = cache.layer_block_table(layer_idx).index_select(0, block_ids_long).contiguous()
    k_pages, v_pages = cache.layer_pages(layer_idx)
    return runtime_paged_attention_decode(
        q,
        k_pages[:page_count],
        v_pages[:page_count],
        block_table,
        live_lengths,
        attn_mask,
        scale=scale,
    )


__all__ = [
    "ContiguousKVCache",
    "Int3ContiguousKVCache",
    "PagedKVCache",
    "clone_kv_cache_rows",
    "concat_kv_caches",
    "reorder_kv_cache_rows_",
    "split_kv_cache_rows",
    "truncate_kv_cache_prefix",
    "init_kv_cache",
    "kv_cache_append",
    "paged_attention_decode",
    "kv_cache_evict",
    "kv_cache_slice",
    "quantize_pack_int3_lastdim",
    "unpack_dequantize_int3_lastdim",
]
