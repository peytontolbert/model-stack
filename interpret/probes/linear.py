from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class LinearProbeConfig:
    task: Literal["classification", "regression"] = "classification"
    l2: float = 0.0
    lr: float = 1e-2
    epochs: int = 50
    batch_size: int = 1024
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


@torch.no_grad()
def evaluate(probe: LinearProbe, x: torch.Tensor, y: torch.Tensor, task: str) -> float:
    probe.eval()
    pred = probe(x)
    if task == "classification":
        acc = (pred.argmax(dim=-1) == y).float().mean().item()
        return acc
    else:
        mse = torch.mean((pred - y) ** 2).item()
        return mse


def fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    *,
    cfg: Optional[LinearProbeConfig] = None,
) -> Tuple[LinearProbe, float]:
    """Train a linear probe on frozen features.

    x_* shapes: [N, D], y_* for classification is [N] with class indices; for regression [N, K]
    Returns (probe, best_metric). For classification, metric is accuracy (higher better). For regression, metric is MSE (lower better).
    """
    cfg = cfg or LinearProbeConfig()
    device = torch.device(cfg.device)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    if x_val is not None:
        x_val = x_val.to(device)
    if y_val is not None:
        y_val = y_val.to(device)

    in_dim = x_train.shape[-1]
    out_dim = int(y_train.max().item()) + 1 if cfg.task == "classification" else y_train.shape[-1]
    probe = LinearProbe(in_dim, out_dim).to(device)

    criterion: nn.Module
    if cfg.task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    loader = _make_loader(x_train, y_train, cfg.batch_size)

    best_score: float
    if cfg.task == "classification":
        best_score = 0.0
    else:
        best_score = float("inf")
    patience_left = cfg.patience

    for _epoch in range(cfg.epochs):
        probe.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if x_val is not None and y_val is not None:
            score = evaluate(probe, x_val, y_val, cfg.task)
            improved = score > best_score if cfg.task == "classification" else score < best_score
            if improved:
                best_score = score
                patience_left = cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

    # Final evaluation if no val set provided
    if x_val is None or y_val is None:
        score = evaluate(probe, x_train, y_train, cfg.task)
        best_score = score

    return probe, best_score


