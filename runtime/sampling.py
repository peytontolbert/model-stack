from __future__ import annotations

import torch
import torch.nn.functional as F

from runtime.ops import (
    no_repeat_ngram_mask as runtime_no_repeat_ngram_mask,
    presence_frequency_penalty as runtime_presence_frequency_penalty,
    repetition_penalty as runtime_repetition_penalty,
    sample_next_token as runtime_sample_next_token,
    temperature as runtime_temperature,
    topk_mask as runtime_topk_mask,
    topp_mask as runtime_topp_mask,
)


def apply_temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return runtime_temperature(logits, tau)


def apply_repetition_penalty(
    logits: torch.Tensor,
    freq_counts: torch.Tensor,
    presence_counts: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    penalty = alpha * freq_counts + beta * (presence_counts > 0).to(freq_counts.dtype)
    while penalty.ndim < logits.ndim:
        penalty = penalty.unsqueeze(-1)
    return logits - penalty


def apply_transformers_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    return runtime_repetition_penalty(logits, input_ids, penalty)


def apply_min_p_mask(logits: torch.Tensor, p_min: float, dim: int = -1) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=dim)
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    cum = torch.cumsum(sorted_probs, dim=dim)
    keep = cum <= (1.0 - p_min)
    keep[..., 0] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    return mask.scatter(dim, sorted_idx, ~keep)


def apply_typical_mask(logits: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=dim)
    logp = torch.log(probs.clamp_min(1e-45))
    ent = -(probs * logp).sum(dim=dim, keepdim=True)
    dev = (-(logp) - ent).abs()
    sorted_dev, sorted_idx = torch.sort(dev, dim=dim, descending=False)
    cum = torch.cumsum(F.softmax(-sorted_dev / max(tau, 1e-8), dim=dim), dim=dim)
    keep = cum <= 1.0
    keep[..., -1] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    return mask.scatter(dim, sorted_idx, ~keep)


def mixture_of_logits(
    logits_list: list[torch.Tensor],
    weights: list[float],
    temperature: float | None = None,
) -> torch.Tensor:
    assert len(logits_list) == len(weights) and len(weights) > 0
    mix = None
    wsum = sum(weights)
    for logit, w in zip(logits_list, weights):
        cur = logit.float()
        if temperature is not None:
            cur = cur / max(temperature, 1e-8)
        cur = F.log_softmax(cur, dim=-1).exp() * (w / wsum)
        mix = cur if mix is None else (mix + cur)
    return torch.log(mix.clamp_min(1e-45)).to(dtype=logits_list[0].dtype)


def ban_tokens(logits: torch.Tensor, ids: torch.Tensor, dim: int = -1) -> torch.Tensor:
    min_val = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
    return logits.scatter(dim, ids, min_val)


def force_tokens(logits: torch.Tensor, ids: torch.Tensor, dim: int = -1) -> torch.Tensor:
    min_val = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
    forced = torch.full_like(logits, min_val)
    return forced.scatter(dim, ids, logits.gather(dim, ids))


def apply_min_tokens_to_keep_mask(logits: torch.Tensor, k_min: int, dim: int = -1) -> torch.Tensor:
    from tensor.numerics import mask_topk

    return mask_topk(logits, k=k_min, dim=dim)


def apply_topk_mask(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    if dim != -1:
        from tensor.numerics import mask_topk

        return mask_topk(logits, k=k, dim=dim)
    return runtime_topk_mask(logits, k)


def apply_topp_mask(logits: torch.Tensor, p: float, dim: int = -1) -> torch.Tensor:
    if dim != -1:
        from tensor.numerics import mask_topp

        return mask_topp(logits, p=p, dim=dim)
    return runtime_topp_mask(logits, p)


def build_regex_constraint_mask(prefix_ids: torch.Tensor, dfa_state, vocab_size: int) -> torch.Tensor:
    del dfa_state
    return torch.zeros(prefix_ids.shape[0], vocab_size, dtype=torch.bool, device=prefix_ids.device)


def apply_no_repeat_ngram_mask(logits: torch.Tensor, input_ids: torch.Tensor, n: int) -> torch.Tensor:
    return runtime_no_repeat_ngram_mask(input_ids, vocab_size=logits.shape[-1], n=n)


def apply_presence_frequency_penalty(
    logits: torch.Tensor,
    counts: torch.Tensor,
    alpha_presence: float,
    alpha_frequency: float,
) -> torch.Tensor:
    if logits.ndim != counts.ndim:
        penalty = alpha_presence * (counts > 0).to(logits.dtype) + alpha_frequency * counts.to(logits.dtype)
        while penalty.ndim < logits.ndim:
            penalty = penalty.unsqueeze(-1)
        return logits - penalty
    return runtime_presence_frequency_penalty(logits, counts, alpha_presence, alpha_frequency)


def sample_next_token(logits: torch.Tensor, do_sample: bool) -> torch.Tensor:
    return runtime_sample_next_token(logits, do_sample)


def apply_tfs_mask(logits: torch.Tensor, z: float, dim: int = -1) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=dim)
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    diffs = F.pad(sorted_probs[..., :-1] - sorted_probs[..., 1:], (0, 1))
    total = diffs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    cum = torch.cumsum(diffs / total, dim=dim)
    keep = cum <= float(z)
    keep[..., 0] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    return mask.scatter(dim, sorted_idx, ~keep)


def apply_eta_mask(logits: torch.Tensor, eta: float, dim: int = -1) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=dim)
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    cum = torch.cumsum(sorted_probs, dim=dim)
    keep = cum <= (1.0 - float(eta))
    keep[..., 0] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    return mask.scatter(dim, sorted_idx, ~keep)


def mirostat_state(mu: float = 5.0, eta: float = 0.1, version: int = 2):
    return {"mu": float(mu), "eta": float(eta), "version": int(version)}


def mirostat_update(logits: torch.Tensor, token_id: torch.Tensor, state: dict, dim: int = -1):
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=dim)
        picked = probs.gather(dim, token_id.unsqueeze(dim)).squeeze(dim).clamp_min(1e-45)
        surprise = -torch.log(picked)
        err = surprise.mean().item() - float(state.get("mu", 5.0))
        eta = float(state.get("eta", 0.1))
        mu = float(state.get("mu", 5.0)) - eta * err
        state["mu"] = max(0.0, mu)
        scale = float(torch.exp(-surprise.mean()).item())
        return scale, state


def apply_stop_phrases_mask(
    prefix_ids: torch.Tensor,
    stop_phrases: list[list[int]],
    vocab_size: int,
) -> torch.Tensor:
    batch = prefix_ids.shape[0]
    mask = torch.zeros(batch, vocab_size, dtype=torch.bool, device=prefix_ids.device)
    for b in range(batch):
        seq = prefix_ids[b].tolist()
        for sp in stop_phrases:
            if not sp:
                continue
            k = len(sp)
            if k == 1:
                mask[b, sp[0]] = True
                continue
            if len(seq) >= k - 1 and seq[-(k - 1):] == sp[:-1]:
                mask[b, sp[-1]] = True
    return mask


def json_schema_mask(prefix_ids: torch.Tensor, schema, vocab_size: int):
    del schema
    return torch.zeros(prefix_ids.shape[0], vocab_size, dtype=torch.bool, device=prefix_ids.device)


def cfgrammar_mask(prefix_ids: torch.Tensor, grammar, vocab_size: int):
    del grammar
    return torch.zeros(prefix_ids.shape[0], vocab_size, dtype=torch.bool, device=prefix_ids.device)


def sample_gumbel(
    shape: tuple[int, ...],
    eps: float = 1e-20,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U.clamp_min(eps)).clamp_min(eps))


def gumbel_topk(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    g = sample_gumbel(tuple(logits.shape), device=logits.device, dtype=logits.dtype)
    return (logits + g).topk(k, dim=dim).indices


def gumbel_softmax(logits: torch.Tensor, tau: float, hard: bool = False, dim: int = -1) -> torch.Tensor:
    y = torch.nn.functional.softmax(
        (
            (logits + sample_gumbel(tuple(logits.shape), device=logits.device, dtype=logits.dtype))
            / max(tau, 1e-8)
        ).float(),
        dim=dim,
    )
    y = y.to(dtype=logits.dtype)
    if hard:
        idx = y.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(y).scatter(dim, idx, 1.0)
        y = (y_hard - y).detach() + y
    return y


__all__ = [
    "apply_eta_mask",
    "apply_min_p_mask",
    "apply_min_tokens_to_keep_mask",
    "apply_no_repeat_ngram_mask",
    "apply_presence_frequency_penalty",
    "apply_repetition_penalty",
    "apply_stop_phrases_mask",
    "apply_temperature",
    "apply_tfs_mask",
    "apply_topk_mask",
    "apply_topp_mask",
    "apply_transformers_repetition_penalty",
    "apply_typical_mask",
    "ban_tokens",
    "build_regex_constraint_mask",
    "cfgrammar_mask",
    "force_tokens",
    "gumbel_softmax",
    "gumbel_topk",
    "json_schema_mask",
    "mirostat_state",
    "mirostat_update",
    "mixture_of_logits",
    "sample_gumbel",
    "sample_next_token",
]
