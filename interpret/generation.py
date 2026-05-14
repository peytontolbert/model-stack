from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.inference_mode()
def generation_logit_trace(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    steps: int = 8,
    temperature: float = 1.0,
    topk: int = 5,
    eos_token_id: Optional[int] = None,
) -> dict[str, object]:
    """Greedy generation trace with per-step entropy, margin, and top-k logits."""

    tokens = input_ids.clone()
    rows: list[dict[str, object]] = []
    for step in range(int(steps)):
        logits = model(tokens)
        next_logits = logits[:, -1, :].float() / max(float(temperature), 1e-8)
        probs = torch.softmax(next_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-45).log()).sum(dim=-1)
        values, indices = torch.topk(next_logits, k=min(int(topk), next_logits.shape[-1]), dim=-1)
        margin = values[:, 0] - values[:, 1] if values.shape[-1] > 1 else torch.zeros_like(values[:, 0])
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        rows.append(
            {
                "step": step,
                "next_token": int(next_token[0, 0].item()),
                "entropy": float(entropy[0].item()),
                "margin": float(margin[0].item()),
                "top_ids": indices[0].detach().cpu().tolist(),
                "top_logits": values[0].detach().cpu().tolist(),
            }
        )
        tokens = torch.cat([tokens, next_token.to(dtype=tokens.dtype)], dim=1)
        if eos_token_id is not None and int(next_token[0, 0].item()) == int(eos_token_id):
            break
    return {"tokens": tokens.detach().cpu(), "steps": rows}


def summarize_generation_trace(trace: dict[str, object]) -> dict[str, object]:
    rows = list(trace.get("steps", []))
    entropies = torch.tensor([float(row["entropy"]) for row in rows], dtype=torch.float32) if rows else torch.empty(0)
    margins = torch.tensor([float(row["margin"]) for row in rows], dtype=torch.float32) if rows else torch.empty(0)
    return {
        "steps": len(rows),
        "generated_tokens": [int(row["next_token"]) for row in rows],
        "entropy_mean": float(entropies.mean().item()) if entropies.numel() else 0.0,
        "margin_mean": float(margins.mean().item()) if margins.numel() else 0.0,
    }
