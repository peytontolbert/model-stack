import torch
import torch.nn.functional as F
from .numerics import softplus_safe


def gelu(x: torch.Tensor, approx: str = "exact") -> torch.Tensor:
    """GELU with controllable approximation.

    approx: "exact" | "tanh" | "sigmoid"
    """
    if approx in ("exact", "none"):
        return F.gelu(x)
    if approx in ("tanh", "fast"):
        # PyTorch tanh approximate path
        return F.gelu(x, approximate="tanh")
    if approx in ("sigmoid", "quick"):
        return x * torch.sigmoid(1.702 * x)
    raise ValueError(f"Unsupported GELU approx: {approx}")


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def relu2(x: torch.Tensor) -> torch.Tensor:
    y = F.relu(x)
    return y * y


def _apply_act(x: torch.Tensor, act: str, gelu_approx: str | None = None) -> torch.Tensor:
    a = act.lower()
    if a in ("identity", "none"):
        return x
    if a in ("relu",):
        return F.relu(x)
    if a in ("relu2", "squared_relu", "squared-relu"):
        return relu2(x)
    if a in ("silu", "swish"):
        return F.silu(x)
    if a in ("gelu",):
        return gelu(x, approx=gelu_approx or "exact")
    if a in ("tanh_gelu", "tanh-gelu"):
        return gelu(x, approx="tanh")
    if a in ("quick_gelu", "sigmoid_gelu", "quick-gelu"):
        return gelu(x, approx="sigmoid")
    raise ValueError(f"Unsupported activation: {act}")


def with_bias_act(
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    act: str = "identity",
    gelu_approx: str | None = None,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is not None:
        x = x + bias
    x = _apply_act(x, act=act, gelu_approx=gelu_approx)
    if gate is not None:
        x = x * gate
    return x


# Back-compat fused helpers
def bias_gelu(x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return with_bias_act(x, bias=bias, act="gelu")


def bias_silu(x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return with_bias_act(x, bias=bias, act="silu")


def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return gelu(x, approx="tanh")


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return gelu(x, approx="sigmoid")


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(softplus_safe(x))


def tanh_gelu(x: torch.Tensor) -> torch.Tensor:
    return gelu(x, approx="tanh")


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return with_bias_act(x, act="silu", gate=gate)


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return with_bias_act(x, act="gelu", gate=gate)


def reglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return with_bias_act(x, act="relu", gate=gate)


def split_for_glu(y: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    d = y.size(dim)
    h = d // 2
    x = y.narrow(dim, 0, h)
    g = y.narrow(dim, h, d - h)
    return x, g


def glu_chunk_act(
    y: torch.Tensor,
    *,
    act: str = "silu",
    gelu_approx: str | None = None,
    dim: int = -1,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    x, gate = split_for_glu(y, dim=dim)
    return with_bias_act(x, bias=bias, act=act, gelu_approx=gelu_approx, gate=gate)