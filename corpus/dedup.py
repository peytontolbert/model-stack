from __future__ import annotations

import hashlib
from typing import Iterable, Iterator, Tuple


def _hash_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def dedup_exact(texts: Iterable[str]) -> Iterator[Tuple[str, bool]]:
    """Yield (text, is_unique) using exact SHA1 content hashes.

    For large corpora, consider block-level or MinHash variants; this is a simple baseline.
    """
    seen: set[str] = set()
    for t in texts:
        h = _hash_text(t)
        if h in seen:
            yield t, False
            continue
        seen.add(h)
        yield t, True


__all__ = ["dedup_exact"]


