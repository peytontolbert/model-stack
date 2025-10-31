import torch


def svd_lowrank(x: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute truncated SVD: returns (U_r, S_r, V_r).

    Shapes:
      - x: (M, N)
      - U_r: (M, r), S_r: (r,), V_r: (N, r)
    """
    if x.ndim != 2:
        raise ValueError("svd_lowrank expects a 2D matrix")
    r = int(max(0, min(rank, min(x.shape))))
    if r == 0:
        M, N = x.shape
        return x.new_zeros(M, 0), x.new_zeros(0), x.new_zeros(N, 0)
    # Use full SVD for stability; could switch to torch.svd_lowrank if available
    U, S, Vh = torch.linalg.svd(x.float(), full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vh[:r, :].T
    return U_r.to(dtype=x.dtype), S_r.to(dtype=x.dtype), V_r.to(dtype=x.dtype)


def factorized_linear(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return low-rank factors (A, B) such that weight ≈ A @ B.T.

    - weight: (out_features, in_features)
    - A: (out_features, r), B: (in_features, r)
    """
    U, S, V = svd_lowrank(weight, rank)
    # Compose A = U * S and B = V
    A = U * S.unsqueeze(0)
    B = V
    return A.contiguous(), B.contiguous()


def rank_selective_update_(W: torch.Tensor, dW: torch.Tensor, k: int) -> torch.Tensor:
    """In-place update W += (dW)_rank-k using SVD truncation.

    Returns W for chaining.
    """
    if W.shape != dW.shape:
        raise ValueError("W and dW must have the same shape")
    if k <= 0:
        return W
    U, S, V = svd_lowrank(dW, k)
    upd = (U * S.unsqueeze(0)) @ V.T
    W.add_(upd.to(dtype=W.dtype, device=W.device))
    return W


