import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return logits / max(tau, 1e-8)


def apply_repetition_penalty(logits: torch.Tensor, freq_counts: torch.Tensor, presence_counts: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # subtract frequency alpha*count and presence beta*(count>0)
    penalty = alpha * freq_counts + beta * (presence_counts > 0).to(freq_counts.dtype)
    while penalty.ndim < logits.ndim:
        penalty = penalty.unsqueeze(-1)
    return logits - penalty


def apply_min_p_mask(logits: torch.Tensor, p_min: float, dim: int = -1) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=dim)
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    cum = torch.cumsum(sorted_probs, dim=dim)
    keep = cum <= (1.0 - p_min)
    keep[..., 0] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask = mask.scatter(dim, sorted_idx, ~keep)
    return mask


def apply_typical_mask(logits: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    # typical sampling mask based on deviation from entropy
    probs = F.softmax(logits.float(), dim=dim)
    logp = torch.log(probs.clamp_min(1e-45))
    ent = -(probs * logp).sum(dim=dim, keepdim=True)
    dev = (-(logp) - ent).abs()
    sorted_dev, sorted_idx = torch.sort(dev, dim=dim, descending=False)
    cum = torch.cumsum(F.softmax(-sorted_dev / max(tau, 1e-8), dim=dim), dim=dim)
    keep = cum <= 1.0
    keep[..., -1] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask = mask.scatter(dim, sorted_idx, ~keep)
    return mask


def mixture_of_logits(logits_list: list[torch.Tensor], weights: list[float], temperature: float | None = None) -> torch.Tensor:
    assert len(logits_list) == len(weights) and len(weights) > 0
    mix = None
    wsum = sum(weights)
    for logit, w in zip(logits_list, weights):
        cur = logit.float()
        if temperature is not None:
            cur = cur / max(temperature, 1e-8)
        cur = F.log_softmax(cur, dim=-1).exp() * (w / wsum)
        mix = cur if mix is None else (mix + cur)
    # return logits via log
    return torch.log(mix.clamp_min(1e-45)).to(dtype=logits_list[0].dtype)


def ban_tokens(logits: torch.Tensor, ids: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # ids: (..., K) indices to ban
    min_val = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
    return logits.scatter(dim, ids, min_val)


def force_tokens(logits: torch.Tensor, ids: torch.Tensor, dim: int = -1) -> torch.Tensor:
    min_val = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
    forced = torch.full_like(logits, min_val)
    forced = forced.scatter(dim, ids, logits.gather(dim, ids))
    return forced


def apply_min_tokens_to_keep_mask(logits: torch.Tensor, k_min: int, dim: int = -1) -> torch.Tensor:
    from tensor.numerics import mask_topk
    return mask_topk(logits, k=k_min, dim=dim)


def apply_topk_mask(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    from tensor.numerics import mask_topk
    return mask_topk(logits, k=k, dim=dim)


def apply_topp_mask(logits: torch.Tensor, p: float, dim: int = -1) -> torch.Tensor:
    from tensor.numerics import mask_topp
    return mask_topp(logits, p=p, dim=dim)


def build_regex_constraint_mask(prefix_ids: torch.Tensor, dfa_state, vocab_size: int) -> torch.Tensor:
    # Placeholder: return all False (no constraints). Hook for external DFA integration
    B = prefix_ids.shape[0]
    return torch.zeros(B, vocab_size, dtype=torch.bool, device=prefix_ids.device)


def apply_no_repeat_ngram_mask(logits: torch.Tensor, input_ids: torch.Tensor, n: int) -> torch.Tensor:
    B, V = logits.shape[0], logits.shape[-1]
    mask = torch.zeros(B, V, dtype=torch.bool, device=logits.device)
    if n <= 0 or input_ids.shape[1] < n:
        return mask
    for b in range(B):
        seq = input_ids[b].tolist()
        if len(seq) < n:
            continue
        recent = tuple(seq[-(n - 1):])
        for i in range(len(seq) - n + 1):
            if tuple(seq[i:i + n - 1]) == recent:
                nxt = seq[i + n - 1]
                mask[b, nxt] = True
    return mask


def apply_presence_frequency_penalty(logits: torch.Tensor, counts: torch.Tensor, alpha_presence: float, alpha_frequency: float) -> torch.Tensor:
    penalty = alpha_presence * (counts > 0).to(logits.dtype) + alpha_frequency * counts.to(logits.dtype)
    while penalty.ndim < logits.ndim:
        penalty = penalty.unsqueeze(-1)
    return logits - penalty


# Tail Free Sampling (TFS) and eta sampling
def apply_tfs_mask(logits: torch.Tensor, z: float, dim: int = -1) -> torch.Tensor:
    """Tail Free Sampling mask: True=masked. z in [0,1].

    Uses cumulative normalized first-difference heuristic on sorted probs.
    """
    probs = F.softmax(logits.float(), dim=dim)
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    # Differences between consecutive probabilities (append last diff as last element)
    diffs = F.pad(sorted_probs[..., :-1] - sorted_probs[..., 1:], (0, 1))
    total = diffs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    cum = torch.cumsum(diffs / total, dim=dim)
    keep = cum <= float(z)
    keep[..., 0] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask = mask.scatter(dim, sorted_idx, ~keep)
    return mask


def apply_eta_mask(logits: torch.Tensor, eta: float, dim: int = -1) -> torch.Tensor:
    """Eta sampling mask: keep minimal prefix whose probability mass >= 1 - eta.

    True=masked.
    """
    probs = F.softmax(logits.float(), dim=dim)
    sorted_probs, sorted_idx = torch.sort(probs, dim=dim, descending=True)
    cum = torch.cumsum(sorted_probs, dim=dim)
    keep = cum <= (1.0 - float(eta))
    keep[..., 0] = True
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask = mask.scatter(dim, sorted_idx, ~keep)
    return mask


# Mirostat v1/v2 helpers
def mirostat_state(mu: float = 5.0, eta: float = 0.1, version: int = 2):
    return {"mu": float(mu), "eta": float(eta), "version": int(version)}


def mirostat_update(logits: torch.Tensor, token_id: torch.Tensor, state: dict, dim: int = -1):
    """Update Mirostat state and return temperature scale.

    token_id: (...,) indices of sampled token along dim.
    Returns: scale (float), updated state
    """
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=dim)
        picked = probs.gather(dim, token_id.unsqueeze(dim)).squeeze(dim).clamp_min(1e-45)
        surprise = -torch.log(picked)
        err = surprise.mean().item() - float(state.get("mu", 5.0))
        eta = float(state.get("eta", 0.1))
        mu = float(state.get("mu", 5.0)) - eta * err
        state["mu"] = max(0.0, mu)
        # Map mu to a temperature; simple exponential mapping
        scale = float(torch.exp(-surprise.mean()).item())
        return scale, state


# Stop-phrases / anti-prompt mask
def apply_stop_phrases_mask(prefix_ids: torch.Tensor, stop_phrases: list[list[int]], vocab_size: int) -> torch.Tensor:
    """Return (B,V) boolean mask banning next tokens that would complete a stop phrase.

    Only blocks the immediate next token that would complete any stop phrase given the current prefix.
    """
    B = prefix_ids.shape[0]
    mask = torch.zeros(B, vocab_size, dtype=torch.bool, device=prefix_ids.device)
    for b in range(B):
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


# Constrained decoding placeholders (JSON/CFG)
def json_schema_mask(prefix_ids: torch.Tensor, schema, vocab_size: int):
    # Placeholder: returns all False (no constraint)
    return torch.zeros(prefix_ids.shape[0], vocab_size, dtype=torch.bool, device=prefix_ids.device)


def cfgrammar_mask(prefix_ids: torch.Tensor, grammar, vocab_size: int):
    # Placeholder: returns all False (no constraint)
    return torch.zeros(prefix_ids.shape[0], vocab_size, dtype=torch.bool, device=prefix_ids.device)


# Gumbel utilities (stateless API surface)
def sample_gumbel(shape: tuple[int, ...], eps: float = 1e-20, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U.clamp_min(eps)).clamp_min(eps))


def gumbel_topk(logits: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    g = sample_gumbel(tuple(logits.shape), device=logits.device, dtype=logits.dtype)
    return (logits + g).topk(k, dim=dim).indices


def gumbel_softmax(logits: torch.Tensor, tau: float, hard: bool = False, dim: int = -1) -> torch.Tensor:
    y = torch.nn.functional.softmax(((logits + sample_gumbel(tuple(logits.shape), device=logits.device, dtype=logits.dtype)) / max(tau, 1e-8)).float(), dim=dim)
    y = y.to(dtype=logits.dtype)
    if hard:
        idx = y.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(y).scatter(dim, idx, 1.0)
        y = (y_hard - y).detach() + y
    return y


