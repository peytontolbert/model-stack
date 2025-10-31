from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .batch import Batch


def _discover_shards(root: str | Path) -> List[Path]:
    """Return a sorted list of shard files under root.

    Supported extensions (by priority): .bin (int32), .npy (1D int), .txt (space-separated ints per line).
    """
    p = Path(root)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"shards path not found or not a directory: {root}")
    exts = (".bin", ".npy", ".txt")
    files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix in exts]
    if not files:
        raise FileNotFoundError(f"no shard files with extensions {exts} found under {root}")
    # Sort by extension priority then name for deterministic ordering
    priority = {".bin": 0, ".npy": 1, ".txt": 2}
    files.sort(key=lambda f: (priority.get(f.suffix, 99), f.name))
    return files


class _ShardView:
    """Lightweight view over one shard that yields non-overlapping windows of token ids.

    - For .bin: expects contiguous int32 tokens, uses numpy.memmap for zero-copy slicing
    - For .npy: expects a 1D array of integers
    - For .txt: each line can be a space-separated sequence of ints; lines are concatenated
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.kind = path.suffix
        self._arr: Optional[np.memmap | np.ndarray] = None

    def _load_array(self) -> np.ndarray:
        if self.kind == ".bin":
            mm = np.memmap(self.path, dtype=np.int32, mode="r")
            self._arr = mm
            return mm
        if self.kind == ".npy":
            arr = np.load(self.path, mmap_mode="r")  # type: ignore[no-untyped-call]
            if arr.ndim != 1:
                raise ValueError(f"expected 1D array in {self.path}, got shape {arr.shape}")
            self._arr = arr
            return arr
        # .txt fallback: concatenate ints from all lines
        data: List[int] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data.extend(int(tok) for tok in line.split())
        arr = np.asarray(data, dtype=np.int64)
        self._arr = arr
        return arr

    def windows(self, seq_len: int) -> Iterator[np.ndarray]:
        arr = self._arr if self._arr is not None else self._load_array()
        n = int(arr.shape[0])
        if n < seq_len:
            return iter(())  # empty iterator
        # Non-overlapping contiguous windows of exactly seq_len (+1 when used for loss shift)
        # We will form targets by shifting inside collate; here we provide at least seq_len tokens.
        last = (n // seq_len) * seq_len
        for start in range(0, last, seq_len):
            yield arr[start : start + seq_len]


class ShardedTokenDataset(torch.utils.data.Dataset[torch.Tensor]):
    """Map-style dataset producing fixed-length token windows from multiple shards.

    This dataset iterates shards in order and slices non-overlapping windows of length ``seq_len``.
    It is suitable for simple LM next-token training where targets are computed by shifting.
    """

    def __init__(self, shard_paths: Sequence[Path], seq_len: int, dtype: torch.dtype = torch.long) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.seq_len = int(seq_len)
        self.dtype = dtype
        self._views = [_ShardView(Path(p)) for p in shard_paths]
        # Build index of (shard_idx, window_idx) for __len__/__getitem__ without loading all data
        self._window_counts: List[int] = []
        for v in self._views:
            arr = v._load_array()
            n = int(arr.shape[0])
            count = n // self.seq_len
            self._window_counts.append(count)
        self._prefix: List[int] = [0]
        for c in self._window_counts:
            self._prefix.append(self._prefix[-1] + c)

    def __len__(self) -> int:  # type: ignore[override]
        return self._prefix[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        # binary search over prefix sums
        lo, hi = 0, len(self._prefix) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._prefix[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        shard_idx = lo
        in_shard_idx = idx - self._prefix[shard_idx]
        return shard_idx, in_shard_idx

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx, win_idx = self._locate(idx)
        view = self._views[shard_idx]
        # materialize the specific window
        start = int(win_idx) * self.seq_len
        arr = view._arr if view._arr is not None else view._load_array()
        window = np.asarray(arr[start : start + self.seq_len])
        return torch.as_tensor(window, dtype=self.dtype)


def _default_collate(batch_tokens: List[torch.Tensor], *, device: Optional[torch.device]) -> Batch:
    # Stack into (B, T); attention mask optional (None for dense/casual in model)
    x = torch.stack(batch_tokens, dim=0)
    if device is not None:
        x = x.to(device, non_blocking=True)
    return Batch(input_ids=x, attn_mask=None)


def build_dataloader(
    shards_path: str | Path,
    *,
    batch_size: int,
    seq_len: int,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
    device: Optional[str | torch.device] = None,
    streaming: bool = False,
    distributed: bool = False,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Iterable[Batch]:
    """Build a simple DataLoader over token shards.

    - ``shards_path``: directory containing .bin/.npy/.txt shards of token ids
    - ``seq_len``: length of each training sample (B, T)
    - ``batch_size``: number of samples per batch
    - Returns: an iterable yielding ``Batch`` with fields ``input_ids`` (B, T) and ``attn_mask=None``
    """
    if streaming:
        from .iterable import build_streaming_dataloader
        return build_streaming_dataloader(
            shards_path,
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            shuffle_shards=bool(shuffle),
            repeat=True,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            device=device,
        )

    files = _discover_shards(shards_path)
    ds = ShardedTokenDataset(files, seq_len=int(seq_len))

    # Resolve device
    dev: Optional[torch.device]
    if device is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(device)

    def _collate(items: List[torch.Tensor]) -> Batch:
        return _default_collate(items, device=dev)

    # Use dist.dataloader if requested for consistency with the dist package
    if distributed:
        try:
            from dist.dataloader import build_distributed_dataloader  # type: ignore
            return build_distributed_dataloader(
                ds,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                drop_last=bool(drop_last),
                pin_memory=bool(pin_memory),
                seed=seed,
            )
        except Exception:
            # Fallback to local DistributedSampler path
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(ds, shuffle=bool(shuffle), drop_last=bool(drop_last))
            if seed is not None:
                # type: ignore[attr-defined]
                sampler.seed = int(seed)
            return torch.utils.data.DataLoader(
                ds,
                batch_size=int(batch_size),
                shuffle=False,
                sampler=sampler,
                drop_last=bool(drop_last),
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
                collate_fn=_collate,
            )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=_collate,
    )


__all__ = [
    "Batch",
    "ShardedTokenDataset",
    "build_dataloader",
]


