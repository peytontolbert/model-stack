import torch
import torch.nn.functional as F


def masked_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    if mask is not None:
        ok = (~mask) & (target >= 0)
        num = (pred[ok] == target[ok]).sum()
        den = ok.sum().clamp_min(1)
        return (num.float() / den).to(dtype=logits.dtype)
    return ((pred == target).float().mean()).to(dtype=logits.dtype)


def masked_token_f1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # micro-F1 across tokens (binary equality)
    tp = ((pred == target) & (mask is None or (~mask))).sum() if mask is not None else (pred == target).sum()
    fp = ((pred != target) & (mask is None or (~mask))).sum() if mask is not None else (pred != target).sum()
    fn = fp  # symmetric for equality case
    precision = tp.float() / torch.clamp_min((tp + fp).float(), 1.0)
    recall = tp.float() / torch.clamp_min((tp + fn).float(), 1.0)
    f1 = 2 * precision * recall / torch.clamp_min(precision + recall, 1e-8)
    return f1


def masked_topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 1, mask: torch.Tensor | None = None) -> torch.Tensor:
    topk = logits.topk(k, dim=-1).indices
    correct = topk.eq(target.unsqueeze(-1)).any(dim=-1)
    if mask is not None:
        valid = (~mask)
        return (correct & valid).float().sum() / valid.float().sum().clamp_min(1.0)
    return correct.float().mean()


def ece_binning(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, n_bins: int = 15, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=dim)
    conf, pred = probs.max(dim=dim)
    correct = (pred == target).float()
    if mask is not None:
        valid = (~mask).to(torch.bool)
        conf = conf[valid]
        correct = correct[valid]
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.tensor(0.0, device=logits.device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i < n_bins - 1 else (conf > lo) & (conf <= hi)
        if m.any():
            acc = correct[m].mean()
            c = conf[m].mean()
            ece = ece + (m.float().mean() * (c - acc).abs())
    return ece.to(dtype=logits.dtype)


def sequence_logprob(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=dim).gather(dim, target.unsqueeze(dim)).squeeze(dim)
    return logp if mask is None else logp.masked_fill(mask, 0.0)


def brier_score(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=dim)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.shape[-1]).float()
    loss = (probs - one_hot).pow(2).sum(dim=dim)
    if mask is not None:
        loss = loss * (~mask).to(loss.dtype)
        return (loss.sum() / (~mask).to(loss.dtype).sum().clamp_min(1.0)).to(dtype=logits.dtype)
    return loss.mean().to(dtype=logits.dtype)


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=dim)
    ent = -(probs * torch.log(probs.clamp_min(1e-45))).sum(dim=dim)
    if mask is not None:
        ent = ent * (~mask).to(ent.dtype)
        return (ent.sum() / (~mask).to(ent.dtype).sum().clamp_min(1.0)).to(dtype=logits.dtype)
    return ent.mean().to(dtype=logits.dtype)


def auroc_binary(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # logits, targets: (N,) binary; simple threshold sweep AUROC
    probs = torch.sigmoid(logits.float()).view(-1)
    t = targets.view(-1).float()
    if mask is not None:
        m = (~mask).view(-1)
        probs = probs[m]
        t = t[m]
    order = torch.argsort(probs)
    t_sorted = t[order]
    cum_pos = torch.cumsum(t_sorted, dim=0)
    cum_neg = torch.cumsum(1 - t_sorted, dim=0)
    auc = (cum_neg * t_sorted).sum() / (cum_pos[-1].clamp_min(1.0) * cum_neg[-1].clamp_min(1.0))
    return auc.to(dtype=logits.dtype)


# Calibration helpers
def calibrate_temperature_grid(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, temps: list[float] | None = None, dim: int = -1):
    if temps is None:
        temps = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]
    B, T, V = logits.shape
    best_tau = 1.0
    best_nll = None
    for tau in temps:
        scaled = logits.float() / max(tau, 1e-8)
        logp = torch.log_softmax(scaled, dim=dim)
        nll = -logp.view(B * T, V).gather(1, targets.view(-1, 1)).view(B, T)
        if mask is not None:
            nll = nll * (~mask).to(nll.dtype)
            denom = (~mask).to(nll.dtype).sum().clamp_min(1.0)
            nll = nll.sum() / denom
        else:
            nll = nll.mean()
        val = float(nll.item())
        if best_nll is None or val < best_nll:
            best_nll = val
            best_tau = float(tau)
    return best_tau, best_nll


def plascale_logits(logits: torch.Tensor, targets: torch.Tensor, iters: int = 100, lr: float = 0.1):
    # Binary Platt scaling: learn A,B s.t. sigmoid(A*x + B) calibrated
    x = logits.detach().float().view(-1)
    y = targets.detach().float().view(-1)
    A = torch.tensor(1.0, device=logits.device, requires_grad=True)
    B = torch.tensor(0.0, device=logits.device, requires_grad=True)
    opt = torch.optim.SGD([A, B], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        p = torch.sigmoid(A * x + B)
        loss = -(y * torch.log(p.clamp_min(1e-7)) + (1 - y) * torch.log((1 - p).clamp_min(1e-7))).mean()
        loss.backward()
        opt.step()
    return float(A.detach().item()), float(B.detach().item())


def vector_scale_logits(logits: torch.Tensor, scale: torch.Tensor, dim: int = -1) -> torch.Tensor:
    while scale.ndim < logits.ndim:
        scale = scale.unsqueeze(0)
    return logits * scale


def brier_score_masked(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    return brier_score(logits, targets, mask=mask, dim=dim)


def ece_temperature_sweep(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None, n_bins: int = 15, dim: int = -1, temps: list[float] | None = None):
    if temps is None:
        temps = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    best_ece = None
    best_tau = None
    eces = []
    for tau in temps:
        scaled = logits.float() / max(tau, 1e-8)
        ece = ece_binning(scaled, targets, mask=mask, n_bins=n_bins, dim=dim).item()
        eces.append((tau, ece))
        if best_ece is None or ece < best_ece:
            best_ece = ece
            best_tau = tau
    return best_tau if best_tau is not None else 1.0, eces


def sequence_entropy(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=dim)
    ent = -(probs * torch.log(probs.clamp_min(1e-45))).sum(dim=dim)
    if mask is not None:
        ent = ent * (~mask).to(ent.dtype)
        denom = (~mask).to(ent.dtype).sum().clamp_min(1.0)
        return (ent.sum() / denom).to(dtype=logits.dtype)
    return ent.mean().to(dtype=logits.dtype)


# ---- MoE metrics ----
def moe_load_balance_loss(logits: torch.Tensor, num_experts: int, dim: int = -1) -> torch.Tensor:
    """Auxiliary load-balance loss encouraging uniform expert usage.

    Computes MSE between average router probs per expert and uniform 1/E.
    """
    probs = torch.softmax(logits, dim=dim)  # (B,T,E)
    usage = probs.mean(dim=tuple(i for i in range(probs.ndim) if i != dim))  # (E)
    target = torch.full_like(usage, 1.0 / max(1, int(num_experts)))
    return F.mse_loss(usage, target)


def uniq_ngrams(ids: torch.Tensor, n: int = 2, mask: torch.Tensor | None = None) -> torch.Tensor:
    B, T = ids.shape
    out = []
    for b in range(B):
        seq = ids[b]
        if mask is not None:
            m = (~mask[b]).to(torch.bool)
            seq = seq[m]
        s = set()
        arr = seq.tolist()
        for i in range(max(len(arr) - n + 1, 0)):
            s.add(tuple(arr[i:i + n]))
        out.append(len(s))
    return torch.tensor(out, device=ids.device, dtype=ids.dtype)


def masked_span_f1(pred_start: torch.Tensor, pred_end: torch.Tensor, true_start: torch.Tensor, true_end: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    # micro-F1 over spans
    correct = (pred_start == true_start) & (pred_end == true_end)
    if mask is not None:
        valid = (~mask).to(torch.bool)
        correct = correct & valid
        denom = valid.sum().clamp_min(1)
        precision = correct.sum().float() / denom.float()
        recall = precision
    else:
        precision = correct.float().mean()
        recall = precision
    f1 = 2 * precision * recall / torch.clamp_min(precision + recall, 1e-8)
    return f1.to(dtype=pred_start.dtype)


def kl_divergence_many(probs_list: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Average KL divergence to the mean distribution across the list.

    Returns a scalar tensor.
    """
    if not probs_list:
        return torch.tensor(0.0)
    mean_p = torch.stack([p.float() for p in probs_list], dim=0).mean(dim=0)
    mean_p = mean_p / mean_p.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    total = torch.tensor(0.0, device=probs_list[0].device)
    for p in probs_list:
        p = p.float()
        total = total + (p * (torch.log(p.clamp_min(1e-45)) - torch.log(mean_p.clamp_min(1e-45)))).sum(dim=dim).mean()
    return (total / len(probs_list)).to(dtype=probs_list[0].dtype)


def _ngrams(ids: torch.Tensor, n: int) -> set[tuple[int, ...]]:
    arr = ids.tolist()
    if len(arr) < n:
        return set()
    return {tuple(arr[i:i + n]) for i in range(len(arr) - n + 1)}


def distinct_n(ids: torch.Tensor, n: int = 2, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute Distinct-n diversity over batch of sequences.

    ids: (B,T) token ids, mask True for padding positions (optional)
    """
    B, T = ids.shape
    total = 0
    unique: set[tuple[int, ...]] = set()
    for b in range(B):
        seq = ids[b]
        if mask is not None:
            m = (~mask[b]).to(torch.bool)
            seq = seq[m]
        grams = _ngrams(seq, n)
        total += max(len(seq) - n + 1, 0)
        unique |= grams
    if total == 0:
        return torch.tensor(0.0, device=ids.device, dtype=ids.dtype)
    return torch.tensor(len(unique) / max(total, 1), device=ids.device, dtype=ids.dtype)


def self_bleu(ids: torch.Tensor, n: int = 2, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Approximate Self-BLEU-n across batch (higher => less diverse)."""
    B, T = ids.shape
    scores = []
    for i in range(B):
        seq_i = ids[i]
        if mask is not None:
            m = (~mask[i]).to(torch.bool)
            seq_i = seq_i[m]
        ref_ngrams: dict[tuple[int, ...], int] = {}
        ref_total = 0
        for j in range(B):
            if j == i:
                continue
            seq_j = ids[j]
            if mask is not None:
                m = (~mask[j]).to(torch.bool)
                seq_j = seq_j[m]
            grams = list(_ngrams(seq_j, n))
            ref_total += max(len(seq_j) - n + 1, 0)
            for g in grams:
                ref_ngrams[g] = ref_ngrams.get(g, 0) + 1
        cand = list(_ngrams(seq_i, n))
        match = sum(1 for g in cand if ref_ngrams.get(g, 0) > 0)
        denom = max(len(seq_i) - n + 1, 1)
        prec = match / denom
        scores.append(prec)
    if not scores:
        return torch.tensor(0.0, device=ids.device, dtype=ids.dtype)
    return torch.tensor(sum(scores) / len(scores), device=ids.device, dtype=ids.dtype)


