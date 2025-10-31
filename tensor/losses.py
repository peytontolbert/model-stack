import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    # logits: (B,T,V), targets: (B,T)
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(B, T)
    if attention_mask is not None:
        mask = attention_mask.to(loss.dtype)
        loss = loss * mask
        denom = mask.sum()
        if reduction == "mean":
            return loss.sum() / torch.clamp_min(denom, 1.0)
        if reduction == "sum":
            return loss.sum()
        return loss
    else:
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss


def sequence_nll(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    return masked_cross_entropy(logits, tokens, attention_mask, label_smoothing=0.0, reduction=reduction)


def masked_cross_entropy_ls(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    eps: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    B, T, V = logits.shape
    log_probs = F.log_softmax(logits.float(), dim=-1)
    smoothed = torch.zeros(B, T, V, device=logits.device)
    smoothed.fill_(eps / max(V - 1, 1))
    smoothed.scatter_(-1, targets.unsqueeze(-1), 1.0 - eps)
    loss = -(smoothed * log_probs).sum(dim=-1).to(dtype=logits.dtype)
    if attention_mask is not None:
        loss = loss * attention_mask.to(loss.dtype)
        denom = attention_mask.to(loss.dtype).sum()
        return (loss.sum() / torch.clamp_min(denom, 1.0)) if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return (loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss))


def masked_kl_div(log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=dim)
    if mask is not None:
        kl = kl * mask.to(kl.dtype)
        denom = mask.to(kl.dtype).sum()
        return kl.sum() / torch.clamp_min(denom, 1.0)
    return kl.mean()


def masked_js_div(log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-45))
    js = 0.5 * ((p * (log_p - log_m)).sum(dim=dim) + (q * (log_q - log_m)).sum(dim=dim))
    if mask is not None:
        js = js * mask.to(js.dtype)
        denom = mask.to(js.dtype).sum()
        return js.sum() / torch.clamp_min(denom, 1.0)
    return js.mean()


def sequence_nll_zloss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, z_coef: float) -> torch.Tensor:
    nll = sequence_nll(logits, target, mask, reduction="mean")
    from tensor.regularization import z_loss_from_logits
    z = z_loss_from_logits(logits, z_coef).mean()
    return nll + z


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
    loss = (pred - target).pow(2)
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum()
        return (loss.sum() / torch.clamp_min(denom, 1.0)) if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, delta: float = 1.0, reduction: str = "mean") -> torch.Tensor:
    diff = (pred - target).abs()
    loss = torch.where(diff < delta, 0.5 * diff.pow(2), delta * (diff - 0.5 * delta))
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum()
        return (loss.sum() / torch.clamp_min(denom, 1.0)) if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


def masked_perplexity(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    nll = sequence_nll(logits, target, mask, reduction="mean")
    return torch.exp(nll)


def masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    # multi-class focal loss on hard targets
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    B, T = targets.shape
    pt = p.view(B * T, -1).gather(1, targets.view(-1, 1)).view(B, T)
    mod = (1 - pt).pow(gamma)
    loss = -mod * logp.view(B * T, -1).gather(1, targets.view(-1, 1)).view(B, T)
    if alpha is not None:
        loss = alpha * loss
    loss = loss.to(dtype=logits.dtype)
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum()
        return (loss.sum() / torch.clamp_min(denom, 1.0)) if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


def masked_bce_multilabel(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    # targets/logits shape: (B,T,C) or (N,C)
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    loss = loss.to(dtype=logits.dtype)
    if mask is not None:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum()
        return (loss.sum() / torch.clamp_min(denom, 1.0)) if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


def bce_with_logits_masked(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    loss = loss.to(dtype=logits.dtype)
    if mask is not None:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum()
        return (loss.sum() / torch.clamp_min(denom, 1.0)) if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


def masked_entropy_from_logits(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=dim)
    ent = -(probs * torch.log(probs.clamp_min(1e-45))).sum(dim=dim)
    if mask is not None:
        ent = ent * mask.to(ent.dtype)
        denom = mask.to(ent.dtype).sum()
        return ent.sum() / torch.clamp_min(denom, 1.0)
    return ent.mean()


def masked_ece(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, n_bins: int = 15, dim: int = -1) -> torch.Tensor:
    # Expected Calibration Error
    probs = torch.softmax(logits.float(), dim=dim)
    conf, pred = probs.max(dim=dim)
    correct = (pred == targets).float()
    if mask is not None:
        valid = mask.to(torch.bool)
        conf = conf[valid]
        correct = correct[valid]
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.tensor(0.0, device=logits.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1]) if i < n_bins - 1 else (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            acc = correct[m].mean()
            conf_mean = conf[m].mean()
            ece = ece + (m.float().mean() * (conf_mean - acc).abs())
    return ece.to(dtype=logits.dtype)


def nll_tokenwise(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, reduction: str = "none") -> torch.Tensor:
    """Per-token negative log likelihood from logits and targets.

    Returns shape (B,T) if reduction="none".
    """
    B, T, V = logits.shape
    logp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    nll = -logp.view(B * T, V).gather(1, targets.view(-1, 1)).view(B, T).to(dtype=logits.dtype)
    if mask is not None:
        nll = nll * mask.to(nll.dtype)
    if reduction == "none":
        return nll
    if reduction == "mean":
        if mask is not None:
            denom = mask.to(nll.dtype).sum().clamp_min(1.0)
            return nll.sum() / denom
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def masked_label_margin_loss(logits: torch.Tensor, targets: torch.Tensor, margin: float = 0.0, mask: torch.Tensor | None = None, reduction: str = "mean") -> torch.Tensor:
    """Multi-class margin loss: max(0, margin - (score_true - max_others))."""
    B, T, V = logits.shape
    true = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask_true = torch.ones_like(logits, dtype=torch.bool)
    mask_true.scatter_(-1, targets.unsqueeze(-1), False)
    max_others = logits.masked_fill(~mask_true, float('-inf')).amax(dim=-1)
    loss = torch.relu(margin - (true - max_others))
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        denom = mask.to(loss.dtype).sum().clamp_min(1.0)
        return loss.sum() / denom if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


def masked_log_score(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1, reduction: str = "mean") -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=dim)
    score = -logp.gather(dim, targets.unsqueeze(dim)).squeeze(dim).to(dtype=logits.dtype)
    if mask is not None:
        score = score * mask.to(score.dtype)
        denom = mask.to(score.dtype).sum().clamp_min(1.0)
        return score.sum() / denom if reduction == "mean" else (score.sum() if reduction == "sum" else score)
    return score.mean() if reduction == "mean" else (score.sum() if reduction == "sum" else score)


def masked_spherical_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1, reduction: str = "mean") -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=dim)
    denom = torch.sqrt((probs ** 2).sum(dim=dim).clamp_min(1e-12))
    num = probs.gather(dim, targets.unsqueeze(dim)).squeeze(dim)
    loss = (1.0 - (num / denom)).to(dtype=logits.dtype)
    if mask is not None:
        loss = loss * mask.to(loss.dtype)
        mden = mask.to(loss.dtype).sum().clamp_min(1.0)
        return loss.sum() / mden if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)
    return loss.mean() if reduction == "mean" else (loss.sum() if reduction == "sum" else loss)


