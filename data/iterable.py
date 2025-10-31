from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence
import itertools
import random

import numpy as np
import torch

from .batch import Batch


def _list_shards(paths: str | Path | Sequence[str | Path]) -> List[Path]:
    if isinstance(paths, (str, Path)):
        base = Path(paths)
        if base.is_dir():
            out = [p for p in sorted(base.iterdir()) if p.is_file() and p.suffix in {".bin", ".npy", ".txt"}]
        else:
            out = [base]
    else:
        out = [Path(p) for p in paths]
    if not out:
        raise FileNotFoundError("no shards provided")
    return out


def _rank_world_from_env() -> tuple[int, int]:
    # Prefer dist.utils if available (initialized or env-backed), fallback to env vars
    try:
        from dist import utils as du  # type: ignore
        return int(du.get_rank(0)), int(du.get_world_size(1))
    except Exception:
        import os
        try:
            r = int(os.getenv("RANK", "0"))
            w = int(os.getenv("WORLD_SIZE", "1"))
        except Exception:
            r, w = 0, 1
        return r, w


class StreamingTokenIterable(torch.utils.data.IterableDataset):
    """Streaming dataset over token shards with optional shard sharding per distributed rank.

    Non-overlapping windows of length ``seq_len`` are produced.
    """

    def __init__(
        self,
        shards: str | Path | Sequence[str | Path],
        *,
        seq_len: int,
        shuffle_shards: bool = True,
        repeat: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._all_shards = _list_shards(shards)
        self.seq_len = int(seq_len)
        self.shuffle = bool(shuffle_shards)
        self.repeat = bool(repeat)
        self.seed = seed

    def _iter_one(self, p: Path) -> Iterator[np.ndarray]:
        if p.suffix == ".bin":
            arr = np.memmap(p, dtype=np.int32, mode="r")
        elif p.suffix == ".npy":
            arr = np.load(p, mmap_mode="r")  # type: ignore
            if arr.ndim != 1:
                return iter(())
        else:  # .txt
            data: List[int] = []
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data.extend(int(tok) for tok in line.split())
            arr = np.asarray(data, dtype=np.int64)
        n = int(arr.shape[0])
        if n < self.seq_len:
            return iter(())
        last = (n // self.seq_len) * self.seq_len
        for s in range(0, last, self.seq_len):
            yield np.asarray(arr[s : s + self.seq_len])

    def _iter_shards_for_rank(self) -> Iterator[Path]:
        all_shards = list(self._all_shards)
        r, w = _rank_world_from_env()
        # partition shards by rank
        my = all_shards[r::max(1, w)]
        if self.shuffle:
            rng = random.Random(self.seed if self.seed is not None else 0)
            rng.shuffle(my)
        if self.repeat:
            return itertools.chain.from_iterable(itertools.repeat(my))  # type: ignore
        return iter(my)

    def __iter__(self) -> Iterator[torch.Tensor]:  # type: ignore[override]
        for shard in self._iter_shards_for_rank():
            for win in self._iter_one(shard):
                yield torch.as_tensor(win, dtype=torch.long)


def build_streaming_dataloader(
    shards: str | Path | Sequence[str | Path],
    *,
    batch_size: int,
    seq_len: int,
    shuffle_shards: bool = True,
    repeat: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    device: Optional[str | torch.device] = None,
) -> torch.utils.data.DataLoader:
    ds = StreamingTokenIterable(shards, seq_len=int(seq_len), shuffle_shards=shuffle_shards, repeat=repeat)
    dev: Optional[torch.device]
    if device is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        dev = torch.device(device)
    def _collate(items: List[torch.Tensor]) -> Batch:
        x = torch.stack(items, dim=0)
        if dev is not None:
            x = x.to(dev, non_blocking=True)
        return Batch(input_ids=x, attn_mask=None)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=True,
        collate_fn=_collate,
    )


__all__ = ["StreamingTokenIterable", "build_streaming_dataloader"]


