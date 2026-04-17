from __future__ import annotations

from typing import Any


def getattr_nested(obj: Any, path: str) -> Any:
    cur: Any = obj
    for tok in str(path).split("."):
        if not tok:
            continue
        cur = getattr(cur, tok)
    return cur


__all__ = [
    "getattr_nested",
]
