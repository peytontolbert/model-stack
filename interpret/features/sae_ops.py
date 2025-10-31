from __future__ import annotations

from typing import Iterable, Optional

import torch

from .sae import SparseAutoencoder


@torch.no_grad()
def sae_encode(sae: SparseAutoencoder, x: torch.Tensor) -> torch.Tensor:
    sae.eval()
    _x = x
    if x.ndim == 3:
        B, T, D = x.shape
        _x = x.reshape(B * T, D)
    x_hat, z = sae(_x)
    if x.ndim == 3:
        z = z.view(x.shape[0], x.shape[1], -1)
    return z


@torch.no_grad()
def sae_decode(sae: SparseAutoencoder, z: torch.Tensor) -> torch.Tensor:
    sae.eval()
    _z = z
    if z.ndim == 3:
        B, T, C = z.shape
        _z = z.reshape(B * T, C)
    x_hat = sae.decoder(torch.relu(_z))
    if z.ndim == 3:
        x_hat = x_hat.view(z.shape[0], z.shape[1], -1)
    return x_hat


@torch.no_grad()
def sae_mask_features(sae: SparseAutoencoder, x: torch.Tensor, code_indices: Iterable[int], *, invert: bool = False) -> torch.Tensor:
    """Return reconstruction with selected code indices zeroed (or kept if invert=True)."""
    z = sae_encode(sae, x)
    C = z.shape[-1]
    mask = torch.zeros(C, device=z.device, dtype=z.dtype)
    mask[torch.tensor(list(code_indices), device=z.device, dtype=torch.long)] = 1
    if invert:
        keep = mask.view(*([1] * (z.ndim - 1)), -1)
        z = z * keep
    else:
        drop = (1 - mask).view(*([1] * (z.ndim - 1)), -1)
        z = z * drop
    return sae_decode(sae, z)


@torch.no_grad()
def sae_boost_features(sae: SparseAutoencoder, x: torch.Tensor, code_indices: Iterable[int], *, factor: float = 1.0) -> torch.Tensor:
    z = sae_encode(sae, x)
    idx = torch.tensor(list(code_indices), device=z.device, dtype=torch.long)
    z[..., idx] = z[..., idx] * (1.0 + float(factor))
    return sae_decode(sae, z)


