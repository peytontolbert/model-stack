from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from runtime.ops import activation as runtime_activation
from runtime.ops import linear as runtime_linear


@dataclass
class SAEConfig:
    code_dim: int
    l1: float = 1e-3
    lr: float = 1e-3
    epochs: int = 1000
    batch_size: int = 8192
    patience: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SparseAutoencoder(nn.Module):
    def __init__(self, in_dim: int, code_dim: int, bias: bool = True):
        super().__init__()
        self.encoder = nn.Linear(in_dim, code_dim, bias=bias)
        self.decoder = nn.Linear(code_dim, in_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = runtime_linear(x, self.encoder.weight, self.encoder.bias)
        x_hat = runtime_linear(runtime_activation(z, "relu"), self.decoder.weight, self.decoder.bias)
        return x_hat, z


def _make_loader(x: torch.Tensor, batch_size: int):
    ds = torch.utils.data.TensorDataset(x)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)


@torch.no_grad()
def sae_reconstruction_metrics(sae: SparseAutoencoder, features: torch.Tensor) -> dict[str, float]:
    sae.eval()
    x_hat, z = sae(features)
    return {
        "reconstruction_mse": float(torch.mean((x_hat - features) ** 2).item()),
        "avg_code_l1": float(torch.mean(torch.abs(z)).item()),
        "max_code_activation": float(z.abs().amax().item()) if z.numel() > 0 else 0.0,
    }


def fit_sae(features: torch.Tensor, *, cfg: SAEConfig) -> tuple[SparseAutoencoder, dict]:
    """Train a simple sparse autoencoder on features [N,D].

    Loss = MSE(x_hat, x) + l1 * |z|_1 (with ReLU on z).
    Returns (sae, info) with final metrics.
    """
    device = torch.device(cfg.device)
    x = features.to(device)
    in_dim = x.shape[-1]
    sae = SparseAutoencoder(in_dim, cfg.code_dim).to(device)
    opt = optim.AdamW(sae.parameters(), lr=cfg.lr)

    loader = _make_loader(x, cfg.batch_size)
    best_loss = float("inf")
    best_epoch = -1
    patience_left = cfg.patience
    history: list[float] = []
    for epoch in range(cfg.epochs):
        sae.train()
        running = 0.0
        count = 0
        for (xb,) in loader:
            opt.zero_grad(set_to_none=True)
            x_hat, z = sae(xb)
            rec = torch.mean((x_hat - xb) ** 2)
            sparsity = torch.mean(torch.abs(z))
            loss = rec + cfg.l1 * sparsity
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)
            count += xb.size(0)
        epoch_loss = running / max(count, 1)
        history.append(float(epoch_loss))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    metrics = sae_reconstruction_metrics(sae, x)
    return sae, {
        "loss": float(best_loss),
        "loss_history": history,
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "stopped_early": len(history) < int(cfg.epochs),
        **metrics,
    }


__all__ = [
    "SAEConfig",
    "SparseAutoencoder",
    "fit_sae",
    "sae_reconstruction_metrics",
]
