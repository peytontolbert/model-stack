import torch


def blocksparse_mask(shape: tuple[int, int], block: tuple[int, int] = (16, 16), density: float = 0.2, device=None) -> torch.Tensor:
    """Return a blocksparse boolean mask of shape (T,S). True=masked (pruned)."""
    T, S = int(shape[0]), int(shape[1])
    Bh, Bw = int(block[0]), int(block[1])
    Th = (T + Bh - 1) // Bh
    Sw = (S + Bw - 1) // Bw
    keep = int(max(1, round((1.0 - float(density)) * Th * Sw)))
    idx = torch.randperm(Th * Sw, device=device)[:keep]
    mask = torch.ones(Th * Sw, dtype=torch.bool, device=device)
    mask[idx] = False
    mask = mask.view(Th, Sw).repeat_interleave(Bh, 0).repeat_interleave(Bw, 1)
    return mask[:T, :S]


def bsr_mm(A_bsr: torch.Tensor, x: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    """Simple block-sparse matmul using dense blocks. A_bsr is dense with zeros on masked blocks."""
    return A_bsr @ x


def sparsify_topk(x: torch.Tensor, k: int, axis: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    vals, idx = torch.topk(x.abs(), k=k, dim=axis)
    return vals, idx


def magnitude_prune(W: torch.Tensor, density: float, block: tuple[int, int] | None = None) -> torch.Tensor:
    if block is None:
        from .regularization import magnitude_mask
        mask = magnitude_mask(W, target_sparsity=float(1.0 - density))
        return W.masked_fill(mask, 0)
    # Block pruning: apply magnitude on block norms
    Bh, Bw = int(block[0]), int(block[1])
    H, S = W.shape
    Th = (H + Bh - 1) // Bh
    Sw = (S + Bw - 1) // Bw
    pad_h = Th * Bh - H
    pad_w = Sw * Bw - S
    Wp = torch.nn.functional.pad(W, (0, pad_w, 0, pad_h))
    Wb = Wp.view(Th, Bh, Sw, Bw).permute(0, 2, 1, 3)
    norms = Wb.reshape(Th, Sw, -1).abs().sum(dim=-1)
    k = max(1, int(round(density * Th * Sw)))
    thresh = norms.view(-1).kthvalue(norms.numel() - k + 1).values
    keep = norms >= thresh
    keep_full = keep.repeat_interleave(Bh, 0).repeat_interleave(Bw, 1)
    Wp = Wp * keep_full.to(Wp.dtype)
    return Wp[:H, :S]


def gather_combine(x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, gather_dim: int = 2) -> torch.Tensor:
    """Gather along gather_dim using indices and combine with weights.

    - x: (..., E, D)
    - indices: (..., k) long
    - weights: (..., k) float (broadcasted across trailing dims)
    Returns: (..., D)
    """
    while indices.ndim < x.ndim - 1:
        indices = indices.unsqueeze(-1)
        weights = weights.unsqueeze(-1)
    expand_shape = list(x.shape)
    expand_shape[gather_dim] = indices.size(-1)
    gathered = x.gather(gather_dim, indices.expand(*expand_shape))
    return (gathered * weights).sum(dim=gather_dim)


