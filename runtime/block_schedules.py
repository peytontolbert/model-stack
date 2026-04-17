from __future__ import annotations


def drop_path_linear(depth: int, max_drop: float) -> list[float]:
    if depth <= 0:
        return []
    if max_drop <= 0.0:
        return [0.0] * depth
    vals: list[float] = []
    for i in range(depth):
        vals.append(max_drop * float(i) / float(max(depth - 1, 1)))
    return vals


__all__ = [
    "drop_path_linear",
]
