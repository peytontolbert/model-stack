import torch
from typing import Dict


def dpo_loss(model, batch: Dict[str, torch.Tensor], beta: float = 0.1) -> Dict[str, float]:
    model.train()
    # Expect batch contains paired preferred/rejected inputs
    logits_pref = model(batch["input_ids_pref"])  # [B, T, V]
    logits_rej = model(batch["input_ids_rej"])   # [B, T, V]
    lp_pref = torch.log_softmax(logits_pref, dim=-1).gather(-1, batch["actions_pref"].unsqueeze(-1)).squeeze(-1).sum(-1)
    lp_rej = torch.log_softmax(logits_rej, dim=-1).gather(-1, batch["actions_rej"].unsqueeze(-1)).squeeze(-1).sum(-1)
    # DPO objective (Rafailov et al. 2023): sigmoid(beta * (lp_pref - lp_rej))
    margin = beta * (lp_pref - lp_rej)
    loss = -torch.mean(torch.log(torch.sigmoid(margin)))
    return {"loss": float(loss.item())}


