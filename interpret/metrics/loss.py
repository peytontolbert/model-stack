from __future__ import annotations

import torch


def token_cross_entropy_map(logits: torch.Tensor, target_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    loss = -logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        loss = loss.masked_fill(attention_mask == 0, 0.0)
    return loss


def token_loss_summary(loss_map: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> dict[str, float]:
    values = loss_map.float()
    if attention_mask is not None:
        values = values[attention_mask != 0]
    if values.numel() == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {"mean": float(values.mean().item()), "max": float(values.max().item()), "min": float(values.min().item())}


def sequence_loss_attribution(logits: torch.Tensor, target_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None, topk: int = 10) -> list[dict[str, float | int]]:
    loss = token_cross_entropy_map(logits, target_ids, attention_mask=attention_mask)
    flat = loss.flatten()
    values, indices = torch.topk(flat, k=min(int(topk), flat.numel()))
    seq_len = int(loss.shape[-1])
    rows = []
    for value, index in zip(values, indices):
        batch = int(index.item() // seq_len)
        pos = int(index.item() % seq_len)
        rows.append({"batch": batch, "position": pos, "target_id": int(target_ids[batch, pos].item()), "loss": float(value.item())})
    return rows
