import math
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ArenaStats:
    bytes_budget: int
    bytes_in_use: int
    bytes_free: int
    hits: int
    misses: int
    evictions: int
    highwater_in_use: int


class ArenaScope:
    def __init__(self, arena: "TensorArena"):
        self.arena = arena
        self._owned: List[torch.Tensor] = []

    def alloc(self, *args, **kwargs) -> torch.Tensor:
        t = self.arena.alloc(*args, **kwargs)
        self._owned.append(t)
        return t

    def __enter__(self) -> "ArenaScope":
        return self

    def __exit__(self, *exc):
        for t in reversed(self._owned):
            self.arena.release(t)


class TensorArena:
    """Stream-safe, size-classed tensor recycling arena with soft byte budget.

    Features:
    - Stream-aware safety for CUDA via record_stream + events
    - Geometric size classes to improve reuse vs exact numel bins
    - Optional memory-format bins (contiguous, channels_last 2D/3D)
    - Scopes for RAII-like auto-release
    - Pinned host helpers and async prefetch stubs
    - CUDA Graph handle reservation for stable addresses
    - Telemetry and adaptive budget policy
    """

    def __init__(
        self,
        device: torch.device | str,
        bytes_budget: int,
        *,
        enable_stream_safety: bool = True,
        graph_mode: bool = False,
        growth: float = 1.25,
    ):
        self.device = torch.device(device)
        self.bytes_budget = int(bytes_budget)
        self.bytes_in_use = 0
        self._free_now: Dict[Tuple[torch.dtype, int, str], List[torch.Tensor]] = {}
        # deferred key excludes memfmt; we store memfmt alongside entries
        self._deferred: Dict[
            Tuple[torch.dtype, int], List[Tuple[torch.Tensor, Optional[torch.cuda.Event], str]]
        ] = {}
        self._view_to_base: "weakref.WeakKeyDictionary[torch.Tensor, torch.Tensor]" = (
            weakref.WeakKeyDictionary()
        )

        # Graph handles: id -> (base, dtype, class_size, memfmt)
        self._handles: Dict[int, Tuple[torch.Tensor, torch.dtype, int, str]] = {}
        self._graph_mode = bool(graph_mode)

        # Policy and accounting
        self._growth = float(growth)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._highwater_in_use = 0
        self._free_bytes = 0
        self._handle_bytes = 0

        # Stream safety
        self.enable_stream_safety = bool(enable_stream_safety)

        # Adaptive budgeting
        self._adaptive_enabled = False
        self._adaptive_alpha = 1.0
        self._adaptive_min = 0
        self._adaptive_max = 2**63 - 1

    # -------- Helpers ---------
    def _size_bytes(self, numel: int, dtype: torch.dtype) -> int:
        return int(numel) * torch.tensor([], dtype=dtype).element_size()

    def _class_of(self, numel: int) -> int:
        if numel <= 0:
            return 0
        growth = self._growth
        # geometric growth, at least step of +1 to guarantee progress
        c = 1
        while c < numel:
            c = int(c * growth) + 1
        return c

    def _memfmt_of(self, t: torch.Tensor) -> str:
        if t.is_cuda is False and t.device.type != "cuda":
            return "contig"
        if t.dim() == 5 and t.is_contiguous(memory_format=torch.channels_last_3d):
            return "cl3d"
        if t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last):
            return "cl2d"
        return "contig"

    def _compute_strides(self, shape: Tuple[int, ...], memfmt: str) -> Tuple[int, ...]:
        # contig default strides
        if memfmt == "contig" or len(shape) <= 1:
            if not shape:
                return ()
            strides = [1]
            for s in reversed(shape[1:]):
                strides.append(strides[-1] * s)
            return tuple(reversed(strides))
        if memfmt == "cl2d" and len(shape) == 4:
            n, c, h, w = shape
            c_stride = 1
            w_stride = c
            h_stride = w * w_stride
            n_stride = h * h_stride
            return (n_stride, c_stride, h_stride, w_stride)
        if memfmt == "cl3d" and len(shape) == 5:
            n, c, d, h, w = shape
            c_stride = 1
            w_stride = c
            h_stride = w * w_stride
            d_stride = h * h_stride
            n_stride = d * d_stride
            return (n_stride, c_stride, d_stride, h_stride, w_stride)
        # fallback to contiguous
        return self._compute_strides(shape, "contig")

    def _view_from_base(self, base_1d: torch.Tensor, shape: Tuple[int, ...], memfmt: str) -> torch.Tensor:
        # base_1d is a 1D tensor with storage capacity >= prod(shape)
        numel = int(math.prod(shape) if shape else 1)
        assert base_1d.dim() == 1 and base_1d.numel() >= numel
        strides = self._compute_strides(shape, memfmt)
        view = base_1d.as_strided(size=shape, stride=strides, storage_offset=0)
        return view

    def _key(self, dtype: torch.dtype, cls: int, memfmt: str) -> Tuple[torch.dtype, int, str]:
        return (dtype, int(cls), str(memfmt))

    def _update_highwater(self) -> None:
        if self.bytes_in_use > self._highwater_in_use:
            self._highwater_in_use = self.bytes_in_use

    def _maybe_update_adaptive_budget(self) -> None:
        if not self._adaptive_enabled:
            return
        if self.device.type == "cuda":
            reserved = int(torch.cuda.memory_reserved(self.device))
            target = int(self._adaptive_alpha * reserved)
            target = max(self._adaptive_min, min(self._adaptive_max, target))
            self.bytes_budget = target

    # -------- Public API ---------
    def set_budget(self, bytes_budget: int) -> None:
        self.bytes_budget = int(bytes_budget)

    def set_adaptive_budget(self, alpha: float, min_bytes: int, max_bytes: int) -> None:
        self._adaptive_enabled = True
        self._adaptive_alpha = float(alpha)
        self._adaptive_min = int(min_bytes)
        self._adaptive_max = int(max_bytes)

    def scope(self) -> ArenaScope:
        return ArenaScope(self)

    def alloc(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        *,
        memory_format: str = "contig",
        init: Optional[str] = None,
    ) -> torch.Tensor:
        shape = tuple(int(s) for s in shape)
        numel = int(math.prod(shape) if shape else 1)
        cls = self._class_of(numel)
        class_bytes = self._size_bytes(cls, dtype)
        req_bytes = self._size_bytes(numel, dtype)

        # Opportunistic scavenging before reuse attempt
        self.scavenge()

        key = self._key(dtype, cls, memory_format)
        lst = self._free_now.get(key)
        if lst and len(lst) > 0:
            base = lst.pop()
            self._free_bytes -= class_bytes
            t = self._view_from_base(base, shape, memory_format)
            self._view_to_base[t] = base
            self.bytes_in_use += req_bytes
            self._hits += 1
            self._update_highwater()
            if init == "zero":
                t.zero_()
            elif init == "nan":
                t.fill_(float("nan"))
            return t

        # Miss: allocate new base if budget allows
        self._misses += 1
        self._maybe_update_adaptive_budget()
        if self.bytes_in_use + self._handle_bytes + class_bytes > self.bytes_budget:
            self._evict_until(self.bytes_in_use + self._handle_bytes + class_bytes - self.bytes_budget)

        if self.device.type == "cpu":
            base = torch.empty(cls, dtype=dtype, device=self.device)
        else:
            base = torch.empty(cls, dtype=dtype, device=self.device)
        t = self._view_from_base(base, shape, memory_format)
        self._view_to_base[t] = base
        self.bytes_in_use += req_bytes
        self._update_highwater()
        if init == "zero":
            t.zero_()
        elif init == "nan":
            t.fill_(float("nan"))
        return t

    def alloc_like(
        self,
        other: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
        memory_format: Optional[str] = None,
        init: Optional[str] = None,
    ) -> torch.Tensor:
        use_dtype = dtype if dtype is not None else other.dtype
        use_fmt = memory_format if memory_format is not None else self._memfmt_of(other)
        return self.alloc(tuple(other.shape), dtype=use_dtype, memory_format=use_fmt, init=init)

    def alloc_host(self, shape: Tuple[int, ...], dtype: torch.dtype, pinned: bool = True) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=bool(pinned))

    def prefetch_to(self, t: torch.Tensor, device: torch.device | str, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        """Kick an async copy to device on the given stream and return the destination tensor.
        The returned tensor shares data with the copy destination and can be used after the stream syncs.
        """
        dev = torch.device(device)
        if dev.type != "cuda":
            return t.to(dev)
        s = stream if stream is not None else torch.cuda.current_stream(dev)
        with torch.cuda.stream(s):
            dst = t.to(device=dev, non_blocking=True)
        return dst

    def release(self, t: torch.Tensor) -> None:
        if not isinstance(t, torch.Tensor):
            return
        try:
            req_bytes = self._size_bytes(t.numel(), t.dtype)
        except Exception:
            req_bytes = 0
        self.bytes_in_use = max(0, self.bytes_in_use - req_bytes)

        base = self._view_to_base.pop(t, None)
        if base is None or not isinstance(base, torch.Tensor):
            # Unknown provenance; do not pool to avoid hazards
            return

        cls = int(base.numel())
        class_bytes = self._size_bytes(cls, base.dtype)
        if t.is_cuda and self.enable_stream_safety:
            evt: Optional[torch.cuda.Event] = torch.cuda.Event(enable_timing=False)
            stream = torch.cuda.current_stream(t.device)
            stream.record_event(evt)
            memfmt = self._memfmt_of(t)
            self._deferred.setdefault((base.dtype, cls), []).append((base.detach(), evt, memfmt))
        else:
            memfmt = self._memfmt_of(t)
            self._free_now.setdefault(self._key(base.dtype, cls, memfmt), []).append(base.detach())
            self._free_bytes += class_bytes

    def _evict_until(self, bytes_needed: int) -> None:
        remaining = int(bytes_needed)
        if remaining <= 0:
            return
        # Evict from immediate free bins first
        for key in list(self._free_now.keys()):
            lst = self._free_now[key]
            dtype, cls, _memfmt = key
            class_bytes = self._size_bytes(cls, dtype)
            while lst and remaining > 0:
                _ = lst.pop()
                self._free_bytes = max(0, self._free_bytes - class_bytes)
                remaining -= class_bytes
                self._evictions += 1
            if not lst:
                del self._free_now[key]
            if remaining <= 0:
                return
        # Then drop references to deferred entries (safe; memory freed after events)
        for dkey in list(self._deferred.keys()):
            lst = self._deferred[dkey]
            dtype, cls = dkey
            class_bytes = self._size_bytes(cls, dtype)
            while lst and remaining > 0:
                _ = lst.pop()
                remaining -= class_bytes
                self._evictions += 1
            if not lst:
                del self._deferred[dkey]
            if remaining <= 0:
                return

    def scavenge(self) -> None:
        # Move ready tensors from deferred -> now
        for dkey in list(self._deferred.keys()):
            entries = self._deferred.get(dkey)
            if not entries:
                continue
            ready: List[Tuple[torch.Tensor, str]] = []
            pending: List[Tuple[torch.Tensor, Optional[torch.cuda.Event], str]] = []
            for base, evt, memfmt in entries:
                if evt is None:
                    ready.append((base, memfmt))
                else:
                    if evt.query():
                        ready.append((base, memfmt))
                    else:
                        pending.append((base, evt, memfmt))
            if ready:
                dtype, cls = dkey
                key_add = None  # type: ignore
                class_bytes = self._size_bytes(cls, dtype)
                for base, memfmt in ready:
                    key_add = self._key(dtype, cls, memfmt)
                    self._free_now.setdefault(key_add, []).append(base)
                    self._free_bytes += class_bytes
            if pending:
                self._deferred[dkey] = pending
            else:
                del self._deferred[dkey]

    def shrink_to_fit(self, ratio: float = 0.5) -> None:
        ratio = float(max(0.0, min(1.0, ratio)))
        target = int(self._free_bytes * ratio)
        drop = max(0, self._free_bytes - target)
        if drop > 0:
            self._evict_until(drop)

    # -------- Graph handles ---------
    def reserve_handle(
        self, handle_id: int, numel: int, dtype: torch.dtype, memory_format: str = "contig"
    ) -> None:
        if handle_id in self._handles:
            return
        cls = self._class_of(int(numel))
        class_bytes = self._size_bytes(cls, dtype)
        self._maybe_update_adaptive_budget()
        if self.bytes_in_use + self._handle_bytes + class_bytes > self.bytes_budget:
            self._evict_until(self.bytes_in_use + self._handle_bytes + class_bytes - self.bytes_budget)
        base = torch.empty(cls, dtype=dtype, device=self.device)
        self._handles[handle_id] = (base, dtype, cls, memory_format)
        self._handle_bytes += class_bytes

    def use_handle(
        self, handle_id: int, shape: Tuple[int, ...], dtype: torch.dtype, memory_format: str = "contig"
    ) -> torch.Tensor:
        if handle_id not in self._handles:
            raise KeyError(f"handle {handle_id} not reserved")
        base, hdtype, cls, hfmt = self._handles[handle_id]
        if hdtype != dtype:
            raise AssertionError("dtype mismatch for handle use")
        if hfmt != memory_format:
            raise AssertionError("memory_format mismatch for handle use")
        numel = int(math.prod(shape) if shape else 1)
        if numel > cls:
            raise AssertionError("requested shape exceeds reserved handle capacity")
        view = self._view_from_base(base, tuple(int(s) for s in shape), memory_format)
        return view

    # -------- Telemetry ---------
    def stats(self) -> ArenaStats:
        return ArenaStats(
            bytes_budget=self.bytes_budget,
            bytes_in_use=self.bytes_in_use,
            bytes_free=self._free_bytes,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            highwater_in_use=self._highwater_in_use,
        )

    def debug_dump(self, topk: int = 10) -> str:
        bins = []
        for (dtype, cls, memfmt), lst in self._free_now.items():
            bins.append(((dtype, cls, memfmt), len(lst), self._size_bytes(cls, dtype) * len(lst)))
        bins.sort(key=lambda x: x[2], reverse=True)
        lines = []
        total_free = self._free_bytes
        lines.append(
            f"Arena(device={self.device}, budget={self.bytes_budget}, in_use={self.bytes_in_use}, free={total_free}, hits={self._hits}, misses={self._misses}, evict={self._evictions})"
        )
        for i, (k, cnt, bytes_) in enumerate(bins[: max(0, int(topk))]):
            dtype, cls, memfmt = k
            lines.append(f"  [{i}] {memfmt} {dtype} cls={cls} count={cnt} bytes={bytes_}")
        dump = "\n".join(lines)
        return dump
