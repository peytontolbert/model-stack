from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


def _ngram_counts(tokens: List[str], n: int) -> dict[Tuple[str, ...], int]:
    return {tuple(tokens[i : i + n]): 1 + counts.get(tuple(tokens[i : i + n]), 0) for i, counts in [(i, {}) for i in range(0)]}  # type: ignore[return-value]


def _ngram_counts_linear(tokens: List[str], n: int) -> dict[Tuple[str, ...], int]:
    counts: dict[Tuple[str, ...], int] = {}
    L = len(tokens)
    if L < n:
        return counts
    for i in range(L - n + 1):
        g = tuple(tokens[i : i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def _clip_precisions(hyp: List[str], ref: List[str], max_n: int = 4) -> List[float]:
    ps: List[float] = []
    for n in range(1, max_n + 1):
        h = _ngram_counts_linear(hyp, n)
        r = _ngram_counts_linear(ref, n)
        overlap = 0
        total = 0
        for g, c in h.items():
            total += c
            overlap += min(c, r.get(g, 0))
        ps.append((overlap / total) if total > 0 else 0.0)
    return ps


def _brevity_penalty(hyp_len: int, ref_len: int) -> float:
    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        return 1.0
    import math
    return math.exp(1.0 - (ref_len / max(hyp_len, 1)))


def bleu(hyp_tokens: List[str], ref_tokens: List[str], max_n: int = 4, smooth: bool = True) -> float:
    import math
    ps = _clip_precisions(hyp_tokens, ref_tokens, max_n=max_n)
    if smooth:
        ps = [(p if p > 0 else 1e-9) for p in ps]
    log_prec = sum(math.log(p) for p in ps) / max(len(ps), 1)
    bp = _brevity_penalty(len(hyp_tokens), len(ref_tokens))
    return float(bp * math.exp(log_prec))


def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                row[j] = prev[j - 1] + 1
            else:
                row[j] = row[j - 1] if row[j - 1] > prev[j] else prev[j]
    return dp[m][n]


def rouge_l(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    lcs = _lcs(hyp_tokens, ref_tokens)
    m = len(hyp_tokens)
    n = len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    beta2 = 1.2 * 1.2
    return float((1 + beta2) * prec * rec / (rec + beta2 * prec))


def exact_match(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    return 1.0 if hyp_tokens == ref_tokens else 0.0


def token_f1(hyp_tokens: List[str], ref_tokens: List[str]) -> float:
    from collections import Counter
    h = Counter(hyp_tokens)
    r = Counter(ref_tokens)
    overlap = sum((h & r).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, len(hyp_tokens))
    rec = overlap / max(1, len(ref_tokens))
    return float(2 * prec * rec / (prec + rec))


def eval_sequences(hyps: Iterable[str], refs: Iterable[str]) -> dict[str, float]:
    total = 0
    sums = {"bleu": 0.0, "rougeL": 0.0, "em": 0.0, "f1": 0.0}
    for h, r in zip(hyps, refs):
        ht = h.strip().split()
        rt = r.strip().split()
        sums["bleu"] += bleu(ht, rt)
        sums["rougeL"] += rouge_l(ht, rt)
        sums["em"] += exact_match(ht, rt)
        sums["f1"] += token_f1(ht, rt)
        total += 1
    if total == 0:
        return {k: 0.0 for k in sums}
    return {k: v / total for k, v in sums.items()}


