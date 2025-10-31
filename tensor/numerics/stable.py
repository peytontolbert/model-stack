import torch
import torch.nn.functional as F


def safe_softmax_with_logsumexp(logits: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1):
    x = logits.float()
    if mask is not None:
        x = x.masked_fill(mask, torch.finfo(x.dtype).min)
    x_max = x.max(dim=dim, keepdim=True).values
    logZ = x_max + torch.log(torch.exp(x - x_max).sum(dim=dim, keepdim=True))
    probs = torch.exp(x - logZ)
    return probs.to(dtype=logits.dtype), logZ.squeeze(dim).to(dtype=logits.dtype)


def _iter_chunks(x: torch.Tensor, dim: int, chunk: int):
    size = x.size(dim)
    for start in range(0, size, chunk):
        end = min(start + chunk, size)
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(start, end)
        yield start, end, tuple(sl)


def _choose_chunk_size(x: torch.Tensor, dim: int, chunk: int | None) -> int:
    import os
    if chunk is not None and chunk > 0:
        return chunk
    env = int(os.environ.get("TENSOR_CHUNK_SIZE", "0"))
    if env > 0:
        return env
    size = x.size(dim)
    bytes_est = x.numel() * x.element_size()
    if size > 8192 or bytes_est > 64 * 1024 * 1024:
        return 4096
    return size


def chunked_softmax(logits: torch.Tensor, dim: int = -1, chunk: int | None = None) -> torch.Tensor:
    x = logits.float()
    chunk = _choose_chunk_size(x, dim, chunk)
    x_max = x.max(dim=dim, keepdim=True).values
    denom = torch.zeros_like(x_max)
    for _, _, sl in _iter_chunks(x, dim, chunk):
        denom = denom + torch.exp(x[sl] - x_max).sum(dim=dim, keepdim=True)
    out = torch.empty_like(x)
    for _, _, sl in _iter_chunks(x, dim, chunk):
        out[sl] = torch.exp(x[sl] - x_max) / denom
    return out.to(dtype=logits.dtype)


def blockwise_logsumexp(x: torch.Tensor, dim: int = -1, block: int | None = None) -> torch.Tensor:
    x = x.float()
    block = _choose_chunk_size(x, dim, block)
    x_max = x.max(dim=dim, keepdim=True).values
    s = torch.zeros_like(x_max)
    for _, _, sl in _iter_chunks(x, dim, block):
        s = s + torch.exp(x[sl] - x_max).sum(dim=dim, keepdim=True)
    return (x_max + torch.log(s)).squeeze(dim)


def masked_softmax_chunked(logits: torch.Tensor, mask: torch.Tensor | None, dim: int = -1, chunk: int | None = None) -> torch.Tensor:
    x = logits.float()
    chunk = _choose_chunk_size(x, dim, chunk)
    if mask is not None:
        x = x.masked_fill(mask, torch.finfo(x.dtype).min)
    return chunked_softmax(x, dim=dim, chunk=chunk).to(dtype=logits.dtype)



def chunked_norm(x: torch.Tensor, ord: int | float = 2, dim: int = -1, chunk: int | None = None) -> torch.Tensor:
    """Stable norm along a dimension computed in chunks to limit peak memory."""
    xf = x.float()
    c = _choose_chunk_size(xf, dim, chunk)
    if ord == 2 or ord == 2.0:
        acc = torch.zeros_like(xf.select(dim, 0), dtype=torch.float32).unsqueeze(dim)
        for _, _, sl in _iter_chunks(xf, dim, c):
            acc = acc + (xf[sl] * xf[sl]).sum(dim=dim, keepdim=True)
        out = acc.sqrt().squeeze(dim)
    else:
        p = float(ord)
        acc = torch.zeros_like(xf.select(dim, 0), dtype=torch.float32).unsqueeze(dim)
        for _, _, sl in _iter_chunks(xf, dim, c):
            acc = acc + xf[sl].abs().pow(p).sum(dim=dim, keepdim=True)
        out = acc.pow(1.0 / p).squeeze(dim)
    return out.to(dtype=x.dtype)

