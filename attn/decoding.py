from typing import Callable, Optional, List, Tuple
import torch


def beam_search(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    beam_size: int,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    # Simple batched beam search for a single example per batch
    B = input_ids.shape[0]
    beams = input_ids.repeat_interleave(beam_size, dim=0)
    scores = torch.zeros(B * beam_size, device=input_ids.device)
    finished = torch.zeros(B * beam_size, dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        logits = logits_fn(beams)
        logp = torch.log_softmax(logits[:, -1, :], dim=-1)
        next_scores, next_tokens = torch.topk(logp, k=beam_size, dim=-1)
        scores = scores.view(B, beam_size, 1) + next_scores
        scores = scores.view(B, -1)
        best_scores, best_idx = torch.topk(scores, k=beam_size, dim=-1)
        beam_indices = best_idx // beam_size
        token_indices = best_idx % beam_size
        new_beams = []
        new_finished = []
        for b in range(B):
            for i in range(beam_size):
                idx = b * beam_size + beam_indices[b, i]
                tok = next_tokens[idx, token_indices[b, i]]
                seq = torch.cat([beams[idx:idx+1], tok.view(1, 1)], dim=1)
                new_beams.append(seq)
                new_finished.append(tok.item() == eos_id)
        beams = torch.cat(new_beams, dim=0)
        finished = torch.tensor(new_finished, device=beams.device)
        # length penalty
        scores = best_scores / ((beams.shape[1] ** length_penalty))
        if finished.all():
            break
    # pick best beam per batch
    out = []
    for b in range(B):
        s = scores.view(B, beam_size)[b]
        i = int(torch.argmax(s))
        out.append(beams[b * beam_size + i:i * 0 + b * beam_size + i + 1])
    return torch.cat(out, dim=0)


def mirostat_step(mu: float, tau: float, k: int, logits: torch.Tensor) -> Tuple[int, float]:
    # one-step mirostat sampling returning token id and updated mu
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    topk_probs = sorted_probs[:k]
    topk_idx = sorted_idx[:k]
    # approximate entropy
    H = -(topk_probs * torch.log(topk_probs.clamp_min(1e-45))).sum().item()
    if H > tau:
        mu = mu + 0.1 * (H - tau)
    else:
        mu = mu - 0.1 * (tau - H)
    sel = torch.multinomial(topk_probs, num_samples=1).item()
    return int(topk_idx[sel].item()), float(mu)


def apply_regex_constraints(logits: torch.Tensor, state) -> torch.Tensor:
    # Apply constraints by masking disallowed token ids in logits.
    # Supported state formats:
    # - {"mask": BoolTensor[(B, V) or (V,)]}
    # - {"allowed_ids": Tensor/List[int] or List[Tensor/List[int]] per batch}
    # - {"forbidden_ids": Tensor/List[int] or List[Tensor/List[int]] per batch}
    # If no recognized constraints are provided, logits are returned unchanged.
    if state is None or not isinstance(state, dict):
        return logits

    x = logits
    # Work on a view shaped (B, V) for masking; if 3D, we only constrain the last step
    if x.dim() == 3:
        x_to_mask = x[:, -1, :]
    elif x.dim() == 2:
        x_to_mask = x
    else:
        # Unsupported shape; return as-is
        return x

    B, V = x_to_mask.shape
    device = x_to_mask.device
    dtype = x_to_mask.dtype
    min_val = torch.finfo(dtype).min if dtype.is_floating_point else -1e9

    mask: Optional[torch.Tensor] = None

    provided_mask = state.get("mask")
    if provided_mask is not None:
        m = provided_mask.to(device=device, dtype=torch.bool)
        if m.dim() == 1:
            m = m.view(1, -1).expand(B, -1)
        elif m.dim() == 2 and m.shape[0] == 1:
            m = m.expand(B, -1)
        mask = m if mask is None else (mask | m)

    allowed_ids = state.get("allowed_ids")
    if allowed_ids is not None:
        # Start with everything disallowed; unmask allowed ids
        allowed_mask = torch.ones(B, V, dtype=torch.bool, device=device)
        if isinstance(allowed_ids, torch.Tensor):
            if allowed_ids.dim() == 1:
                idx = allowed_ids.to(device=device, dtype=torch.long)
                allowed_mask[:, idx] = False
            elif allowed_ids.dim() == 2:
                for b in range(min(B, allowed_ids.shape[0])):
                    idx = allowed_ids[b].to(device=device, dtype=torch.long)
                    allowed_mask[b, idx] = False
        else:
            # list-like
            if len(allowed_ids) > 0 and isinstance(allowed_ids[0], (list, tuple, torch.Tensor)):
                for b in range(min(B, len(allowed_ids))):
                    ids_b = allowed_ids[b]
                    if isinstance(ids_b, torch.Tensor):
                        idx = ids_b.to(device=device, dtype=torch.long)
                    else:
                        idx = torch.tensor(ids_b, device=device, dtype=torch.long)
                    allowed_mask[b, idx] = False
            else:
                idx = torch.tensor(allowed_ids, device=device, dtype=torch.long)
                allowed_mask[:, idx] = False
        mask = allowed_mask if mask is None else (mask | allowed_mask)

    forbidden_ids = state.get("forbidden_ids")
    if forbidden_ids is not None:
        forbid_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
        if isinstance(forbidden_ids, torch.Tensor):
            if forbidden_ids.dim() == 1:
                idx = forbidden_ids.to(device=device, dtype=torch.long)
                forbid_mask[:, idx] = True
            elif forbidden_ids.dim() == 2:
                for b in range(min(B, forbidden_ids.shape[0])):
                    idx = forbidden_ids[b].to(device=device, dtype=torch.long)
                    forbid_mask[b, idx] = True
        else:
            if len(forbidden_ids) > 0 and isinstance(forbidden_ids[0], (list, tuple, torch.Tensor)):
                for b in range(min(B, len(forbidden_ids))):
                    ids_b = forbidden_ids[b]
                    if isinstance(ids_b, torch.Tensor):
                        idx = ids_b.to(device=device, dtype=torch.long)
                    else:
                        idx = torch.tensor(ids_b, device=device, dtype=torch.long)
                    forbid_mask[b, idx] = True
            else:
                idx = torch.tensor(forbidden_ids, device=device, dtype=torch.long)
                forbid_mask[:, idx] = True
        mask = forbid_mask if mask is None else (mask | forbid_mask)

    if mask is None:
        return x

    masked = x_to_mask.masked_fill(mask, min_val)
    if x.dim() == 3:
        out = x.clone()
        out[:, -1, :] = masked
        return out
    else:
        return masked


