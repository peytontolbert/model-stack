import torch
import torch.nn.functional as F

from tensor.sampling import apply_temperature, apply_topk_mask, apply_topp_mask


@torch.no_grad()
def greedy_generate(model, input_ids: torch.Tensor, *, max_new_tokens: int, eos_id: int | None = None, attn_mask=None) -> torch.Tensor:
    seq = input_ids
    for _ in range(max_new_tokens):
        logits = model(seq, attn_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_token], dim=1)
        if eos_id is not None:
            if (next_token == eos_id).all():
                break
    return seq


@torch.no_grad()
def sample_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    attn_mask=None,
) -> torch.Tensor:
    seq = input_ids
    for _ in range(max_new_tokens):
        logits = model(seq, attn_mask)
        logits = logits[:, -1, :]
        if temperature is not None and temperature != 1.0:
            logits = apply_temperature(logits, temperature)
        mask = None
        if top_k is not None:
            m = apply_topk_mask(logits, top_k)
            mask = m if mask is None else (mask | m)
        if top_p is not None:
            m = apply_topp_mask(logits, top_p)
            mask = m if mask is None else (mask | m)
        if mask is not None:
            min_val = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1e9
            logits = logits.masked_fill(mask, min_val)
        probs = F.softmax(logits.float(), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)
        if eos_id is not None:
            if (next_token == eos_id).all():
                break
    return seq


