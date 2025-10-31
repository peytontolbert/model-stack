import torch
from typing import Dict, Iterable
from tensor.optim import clip_grad_norm_

from .config import RLConfig
from .algorithms import ppo_step, dpo_loss


class Trainer:
    def __init__(self, model, optimizer, cfg: RLConfig):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def train_steps(self, batches: Iterable[Dict[str, torch.Tensor]], steps: int | None = None) -> Dict[str, float]:
        steps = self.cfg.steps if steps is None else steps
        last_metrics: Dict[str, float] = {}
        it = iter(batches)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            if self.cfg.algo == "ppo":
                last_metrics = ppo_step(
                    self.model,
                    self.optimizer,
                    batch,
                    clip_ratio=self.cfg.clip_ratio,
                    kl_target=self.cfg.kl_target,
                )
            else:
                m = dpo_loss(self.model, batch, beta=self.cfg.beta)
                self.optimizer.zero_grad(set_to_none=True)
                # Recompute forward to get a loss tensor
                logits_pref = self.model(batch["input_ids_pref"])  # [B, T, V]
                logits_rej = self.model(batch["input_ids_rej"])   # [B, T, V]
                lp_pref = torch.log_softmax(logits_pref, dim=-1).gather(-1, batch["actions_pref"].unsqueeze(-1)).squeeze(-1).sum(-1)
                lp_rej = torch.log_softmax(logits_rej, dim=-1).gather(-1, batch["actions_rej"].unsqueeze(-1)).squeeze(-1).sum(-1)
                margin = self.cfg.beta * (lp_pref - lp_rej)
                loss = -torch.mean(torch.log(torch.sigmoid(margin)))
                loss.backward()
                # Optional gradient clipping
                try:
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                except Exception:
                    pass
                self.optimizer.step()
                last_metrics = m
        return last_metrics


