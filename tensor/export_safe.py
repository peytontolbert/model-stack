import torch


def gelu_export(x: torch.Tensor, approx: str = "tanh") -> torch.Tensor:
    if approx == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(0.79788456 * (x + 0.044715 * (x ** 3))))
    raise ValueError("Unsupported approx")


def rmsnorm_export(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def gather2d_export(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # x: (B,T,V), index: (B,T) -> (B,T)
    return x.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def scatter_add_export(x: torch.Tensor, index: torch.Tensor, src: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # ONNX-friendly scatter add: result = x with src added at index positions along dim
    out = x.clone()
    return out.scatter_add(dim, index, src)


def gather1d_export(x: torch.Tensor, index: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Generic 1D gather wrapper; works for 1D or ND along given dim
    return x.gather(dim, index)


