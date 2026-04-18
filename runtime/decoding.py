from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from runtime.native import has_native_op, native_module
from runtime.ops import beam_search_step


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
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device

    beams = (
        input_ids[:, None, :]
        .expand(batch_size, beam_size, prompt_len)
        .reshape(batch_size * beam_size, prompt_len)
    )

    raw_scores = torch.full((batch_size, beam_size), float("-inf"), device=device)
    raw_scores[:, 0] = 0.0
    finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)
    lengths = torch.full((batch_size, beam_size), prompt_len, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        step_out = step_fn(beams)
        logits = logits_fn(step_out)
        beams, raw_scores, finished, lengths, _ = beam_search_step(
            beams,
            logits,
            raw_scores,
            finished,
            lengths,
            beam_size=beam_size,
            eos_id=eos_id,
            pad_id=pad_id,
        )

        if finished.all():
            break

    final_scores = raw_scores / lengths.float().pow(length_penalty)
    best = final_scores.argmax(dim=-1)
    final_beams = beams.view(batch_size, beam_size, -1)
    return final_beams[torch.arange(batch_size, device=device), best]


def incremental_beam_search(
    initial_beams: torch.Tensor,
    initial_logits: torch.Tensor,
    *,
    beam_size: int,
    max_new_tokens: int,
    prompt_length: int,
    eos_id: int,
    pad_id: int,
    advance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if beam_size <= 0:
        raise ValueError("beam_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if initial_beams.dim() != 2:
        raise ValueError("initial_beams must be rank-2 (B*beam, T)")

    if has_native_op("incremental_beam_search"):
        module = native_module()
        if module is not None and hasattr(module, "incremental_beam_search_forward"):
            return module.incremental_beam_search_forward(
                initial_beams,
                initial_logits,
                int(beam_size),
                int(max_new_tokens),
                int(prompt_length),
                int(eos_id),
                int(pad_id),
                advance_fn,
            )

    batch_size = int(initial_beams.shape[0] // beam_size)
    if batch_size * beam_size != int(initial_beams.shape[0]):
        raise ValueError("initial_beams batch does not match beam_size")

    raw_scores = torch.full((batch_size, beam_size), float("-inf"), device=initial_beams.device)
    raw_scores[:, 0] = 0.0
    finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=initial_beams.device)
    lengths = torch.full((batch_size, beam_size), int(prompt_length), dtype=torch.long, device=initial_beams.device)

    beams, raw_scores, finished, lengths, parent_rows = beam_search_step(
        initial_beams,
        initial_logits,
        raw_scores,
        finished,
        lengths,
        beam_size=beam_size,
        eos_id=eos_id,
        pad_id=pad_id,
    )

    for _ in range(1, max_new_tokens):
        if finished.all():
            break
        next_logits = advance_fn(parent_rows, beams)
        if next_logits is None:
            return None
        beams, raw_scores, finished, lengths, parent_rows = beam_search_step(
            beams,
            next_logits,
            raw_scores,
            finished,
            lengths,
            beam_size=beam_size,
            eos_id=eos_id,
            pad_id=pad_id,
        )

    return beams, raw_scores, finished, lengths, parent_rows


def mirostat_step(mu: float, tau: float, k: int, logits: torch.Tensor) -> Tuple[int, float]:
    # One-step mirostat sampling returning token id and updated mu.
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    topk_probs = sorted_probs[:k]
    topk_idx = sorted_idx[:k]
    H = -(topk_probs * torch.log(topk_probs.clamp_min(1e-45))).sum().item()
    if H > tau:
        mu = mu + 0.1 * (H - tau)
    else:
        mu = mu - 0.1 * (tau - H)
    sel = torch.multinomial(topk_probs, num_samples=1).item()
    return int(topk_idx[sel].item()), float(mu)


def apply_regex_constraints(logits: torch.Tensor, state) -> torch.Tensor:
    # Apply constraints by masking disallowed token ids in logits.
    if state is None or not isinstance(state, dict):
        return logits

    x = logits
    if x.dim() == 3:
        x_to_mask = x[:, -1, :]
    elif x.dim() == 2:
        x_to_mask = x
    else:
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
    return masked
