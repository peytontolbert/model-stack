"""
A tiny, self-contained attention demo used for repo-grounding smoke tests.

The functions intentionally include clear docstrings and simple math so
retrieval by the runner can surface these lines in context windows.
"""

from __future__ import annotations

from typing import Tuple
import math


def attention_score(query: float, key: float) -> float:
    """
    Compute an unnormalized attention score between a scalar query and key.

    This mirrors the dot-product component in scaled dot-product attention for
    the 1-D case. In higher dimensions, the score is the dot product of the
    query and key vectors.
    """
    return float(query * key)


def scaled_dot_product_attention(
    query: Tuple[float, ...], key: Tuple[float, ...], value: Tuple[float, ...]
) -> float:
    """
    Minimal scaled dot-product attention for 1-token, 1-head toy inputs.

    Steps:
    1) score = sum_i query[i] * key[i]
    2) scale by sqrt(d) where d is the dimensionality of query/key
    3) softmax over a single score is 1.0 (degenerate 1-token case)
    4) output = weight * sum_i value[i] (here weight==1.0)
    """
    if len(query) != len(key):
        raise ValueError("query and key must have same length")
    d = max(1, len(query))
    score = sum(float(q) * float(k) for q, k in zip(query, key))
    scaled = score / math.sqrt(float(d))
    # With a single score, softmax(scaled) == 1.0; keep explicit for clarity
    weight = 1.0 if math.isfinite(scaled) else 0.0
    return float(weight * sum(float(v) for v in value))


def explain_attention_brief() -> str:
    """
    Return a brief, human-readable explanation of attention suitable for tests.
    """
    return (
        "Attention computes similarity between a query and keys, normalizes the "
        "scores (softmax), and uses them to weight values. The scaled "
        "dot-product form divides by sqrt(d) to keep gradients stable."
    )


