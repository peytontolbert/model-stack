from __future__ import annotations

import torch


def num_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def num_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_dtype(model: torch.nn.Module, dtype: torch.dtype) -> torch.nn.Module:
    return model.to(dtype=dtype)


def to_device(model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
    return model.to(device=device)


