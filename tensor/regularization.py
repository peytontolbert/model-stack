import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if scale_by_keep and keep_prob > 0.0:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def z_loss_from_logits(logits: torch.Tensor, z_loss_coef: float) -> torch.Tensor:
    # PaLM z-loss: encourages logits to have small magnitude
    z = torch.logsumexp(logits.float(), dim=-1)
    return (z.pow(2) * z_loss_coef).to(dtype=logits.dtype)


def label_smooth(target: torch.Tensor, n_classes: int, eps: float) -> torch.Tensor:
    # target: (B,T) ints -> (B,T,V) float with smoothing
    B, T = target.shape
    out = target.new_full((B, T, n_classes), eps / max(n_classes - 1, 1)).float()
    out.scatter_(-1, target.unsqueeze(-1), 1.0 - eps)
    return out


def grad_noise_std(step: int, eta: float, gamma: float) -> float:
    return float(eta / ((step + 1) ** gamma))


def compute_grad_noise_std(step: int, eta: float, gamma: float) -> float:
    return grad_noise_std(step, eta, gamma)


def build_tokendrop_mask(shape: tuple[int, ...], p: float, device=None) -> torch.Tensor:
    if p <= 0.0:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    return torch.rand(shape, device=device) < p


def build_sequencedrop_mask(B: int, p_seq: float, device=None) -> torch.Tensor:
    if p_seq <= 0.0:
        return torch.zeros(B, dtype=torch.bool, device=device)
    return torch.rand(B, device=device) < p_seq


def schedule_linear(start: float, end: float, step: int, total_steps: int) -> float:
    w = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return float(start + (end - start) * w)


# Structured sparsity helpers
def magnitude_mask(x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """Return boolean mask of positions to prune (True == prune) to reach target sparsity.

    Mask is computed by thresholding absolute values globally.
    """
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError("target_sparsity must be in [0,1)")
    if x.numel() == 0 or target_sparsity == 0.0:
        return torch.zeros_like(x, dtype=torch.bool)
    k = max(int(x.numel() * target_sparsity), 0)
    if k <= 0:
        return torch.zeros_like(x, dtype=torch.bool)
    flat = x.abs().view(-1)
    # kthvalue returns k-th smallest; clamp k to valid range
    k = min(k, flat.numel() - 1)
    thresh = flat.kthvalue(k + 1).values  # 1-indexed style
    return (x.abs() <= thresh)


def prune_topk_(x: torch.Tensor, k: int | None = None, sparsity: float | None = None) -> torch.Tensor:
    """In-place prune. Provide either k (keep top-k by magnitude) or sparsity (fraction to prune).

    Returns the input tensor for chaining.
    """
    if (k is None) == (sparsity is None):
        raise ValueError("Provide exactly one of k or sparsity")
    if k is not None:
        if k <= 0:
            x.zero_()
            return x
        k = int(k)
        flat = x.abs().view(-1)
        if k >= flat.numel():
            return x
        topk_vals = flat.topk(k, largest=True).values
        thresh = topk_vals.min()
        mask = x.abs() < thresh
    else:
        mask = magnitude_mask(x, float(sparsity))
    x.masked_fill_(mask, 0)
    return x


def mixout(param: torch.Tensor, target: torch.Tensor, p: float) -> torch.Tensor:
    """Mixout regularization: randomly replace a fraction p of param with target.

    Returns the mixed tensor (stateless; does not modify inputs).
    """
    if p <= 0.0:
        return param
    mask = torch.rand_like(param) < float(p)
    return torch.where(mask, target.to(dtype=param.dtype, device=param.device), param)


def stochastic_depth_mask(B: int, drop_prob: float, device=None) -> torch.Tensor:
    if drop_prob <= 0.0:
        return torch.ones(B, 1, 1, 1, dtype=torch.bool, device=device)
    keep = torch.rand(B, 1, 1, 1, device=device) >= float(drop_prob)
    return keep

