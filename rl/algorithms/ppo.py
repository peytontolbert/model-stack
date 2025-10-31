import torch
from typing import Dict


def ppo_step(model, optimizer, batch: Dict[str, torch.Tensor], clip_ratio: float = 0.2, kl_target: float = 0.01) -> Dict[str, float]:
    model.train()
    logits = model(batch["input_ids"])  # [B, T, V]
    logprobs = torch.log_softmax(logits, dim=-1)
    act = batch["actions"].unsqueeze(-1)
    new_lp = logprobs.gather(-1, act).squeeze(-1)
    ratio = torch.exp(new_lp - batch["old_logprobs"])  # [B, T]
    adv = batch["advantages"]
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.min(unclipped, clipped))
    # Simple KL proxy using teacher logits if provided
    if "ref_logprobs" in batch:
        kl = torch.mean(batch["ref_logprobs"] - new_lp)
    else:
        kl = torch.mean(torch.zeros_like(new_lp))
    loss = policy_loss + (kl / max(kl_target, 1e-8)) * 0.0  # keep minimal
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()), "kl": float(kl.item())}


