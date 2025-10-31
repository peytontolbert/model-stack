from typing import Iterable
import torch


def global_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(iter(parameters)).device)
    for p in parameters:
        if p.grad is not None:
            total = total + p.grad.detach().float().pow(2).sum()
    return total.sqrt()


def clip_grad_norm_(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> float:
    norm = global_grad_norm(parameters)
    scale = (max_norm / (norm + eps)).clamp(max=1.0)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)
    return float(norm.item())


def unitwise_l2_norm(t: torch.Tensor) -> torch.Tensor:
    if t.ndim <= 1:
        return t.norm()
    return t.flatten(1).norm(dim=1, keepdim=True).view([t.shape[0]] + [1] * (t.ndim - 1))


def unitwise_clip_(t: torch.Tensor, max_norm: float):
    n = unitwise_l2_norm(t)
    t.mul_((max_norm / (n + 1e-6)).clamp(max=1.0))


def decay_mask_from_module(module: torch.nn.Module, exclude=("bias", "norm")) -> dict[str, bool]:
    mask = {}
    for name, param in module.named_parameters():
        key = True
        if any(x in name.lower() for x in exclude):
            key = False
        mask[name] = key
    return mask


