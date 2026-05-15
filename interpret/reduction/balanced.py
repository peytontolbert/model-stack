from __future__ import annotations

import torch


def empirical_gramian(samples: torch.Tensor, *, center: bool = True, damping: float = 1e-6) -> torch.Tensor:
    x = samples.float().reshape(-1, samples.shape[-1])
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    gram = x.T @ x / max(1, x.shape[0] - 1)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return gram + float(damping) * eye


def empirical_controllability_gramian(activation_deltas: torch.Tensor, *, damping: float = 1e-6) -> torch.Tensor:
    return empirical_gramian(activation_deltas, center=True, damping=damping)


def empirical_observability_gramian(jacobian_rows: torch.Tensor, *, damping: float = 1e-6) -> torch.Tensor:
    j = jacobian_rows.float().reshape(-1, jacobian_rows.shape[-1])
    gram = j.T @ j / max(1, j.shape[0])
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return gram + float(damping) * eye


def balanced_hankel_singular_values(controllability: torch.Tensor, observability: torch.Tensor) -> torch.Tensor:
    if controllability.shape != observability.shape or controllability.ndim != 2:
        raise ValueError("controllability and observability must be square matrices with the same shape")
    c = 0.5 * (controllability.float() + controllability.float().T)
    o = 0.5 * (observability.float() + observability.float().T)
    eig_c, vec_c = torch.linalg.eigh(c)
    sqrt_c = (vec_c * eig_c.clamp_min(0).sqrt().unsqueeze(0)) @ vec_c.T
    core = 0.5 * (sqrt_c @ o @ sqrt_c + (sqrt_c @ o @ sqrt_c).T)
    eig = torch.linalg.eigvalsh(core).clamp_min(0)
    return eig.sqrt().flip(0)


def balanced_projection(
    controllability: torch.Tensor,
    observability: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top balanced directions and Hankel-like singular values.

    Directions are eigenvectors of the symmetrized product proxy
    ``sqrt(C) O sqrt(C)`` mapped back to the original coordinate space.
    """

    c = 0.5 * (controllability.float() + controllability.float().T)
    o = 0.5 * (observability.float() + observability.float().T)
    eig_c, vec_c = torch.linalg.eigh(c)
    sqrt_c = (vec_c * eig_c.clamp_min(0).sqrt().unsqueeze(0)) @ vec_c.T
    inv_sqrt_c = (vec_c * eig_c.clamp_min(1e-12).rsqrt().unsqueeze(0)) @ vec_c.T
    core = 0.5 * (sqrt_c @ o @ sqrt_c + (sqrt_c @ o @ sqrt_c).T)
    eig, vec = torch.linalg.eigh(core)
    order = torch.argsort(eig, descending=True)
    keep = order[: min(int(rank), eig.numel())]
    directions = inv_sqrt_c @ vec[:, keep]
    directions = torch.nn.functional.normalize(directions, dim=0)
    hsv = eig[keep].clamp_min(0).sqrt()
    return directions, hsv


def balanced_energy_retained(hsv: torch.Tensor, rank: int) -> torch.Tensor:
    values = hsv.float().clamp_min(0)
    total = values.sum().clamp_min(1e-12)
    return values[: int(rank)].sum() / total


def project_onto_basis(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x @ basis.to(device=x.device, dtype=x.dtype)


def reconstruct_from_basis(coords: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return coords @ basis.to(device=coords.device, dtype=coords.dtype).T
