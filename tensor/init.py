import math
import torch
import torch.nn as nn


def xavier_uniform_linear(module: nn.Linear, gain: float | None = None) -> nn.Linear:
    g = gain if gain is not None else 1.0
    nn.init.xavier_uniform_(module.weight, gain=g)
    if module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(module.bias, -bound, bound)
    return module


def kaiming_uniform_linear(module: nn.Linear, nonlinearity: str = "relu") -> nn.Linear:
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), nonlinearity=nonlinearity)
    if module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(module.bias, -bound, bound)
    return module


def mu_param_linear_init_(lin: nn.Linear, fan_in: int | None = None, mu: float = 0.5):
    # Scales input/output to balance residual growth; simple heuristic
    fi = fan_in if fan_in is not None else lin.in_features
    nn.init.normal_(lin.weight, mean=0.0, std=mu / math.sqrt(fi))
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    return lin


def deepnet_residual_scale(n_layers: int) -> float:
    return 1.0 / math.sqrt(max(n_layers, 1))


def init_swiglu_bias(sizes: tuple[int, int]):
    # return bias tensors for [a|b] projection with zero-mean gate and linear branch
    d_in, d_ff = sizes
    return torch.zeros(2 * d_ff)


def zero_out_proj_bias(linear: nn.Linear):
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def init_rmsnorm_(m, eps: float = 1e-6, gain: float = 1.0):
    if hasattr(m, 'weight'):
        with torch.no_grad():
            m.weight.fill_(gain)
    if hasattr(m, 'eps'):
        m.eps = eps
    return m


def init_qkv_proj_(linear: nn.Linear, d_model: int, n_heads: int, std: float | None = None):
    s = std if std is not None else (0.02 / math.sqrt(d_model))
    nn.init.normal_(linear.weight, mean=0.0, std=s)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def init_out_proj_scaled_(linear: nn.Linear, n_layers: int):
    scale = 1.0 / math.sqrt(max(n_layers, 1))
    nn.init.xavier_uniform_(linear.weight, gain=scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def apply_init_preset(module: nn.Module, preset: str = "llama"):
    # Placeholder for named recipes; no traversal here, just returns preset name for now
    return preset


def mu_param_conv_init_(conv: nn.Conv2d, mu: float = 0.5):
    fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
    nn.init.normal_(conv.weight, mean=0.0, std=mu / math.sqrt(max(fan_in, 1)))
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def xavier_fanfix_linear(lin: nn.Linear, gain: float = 1.0, fix: str = "fan_avg") -> nn.Linear:
    if fix == "fan_in":
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(lin.weight)
        std = gain * math.sqrt(2.0 / max(fan_in + 1e-6, 1.0))
        nn.init.uniform_(lin.weight, -std, std)
    elif fix == "fan_out":
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(lin.weight)
        std = gain * math.sqrt(2.0 / max(fan_out + 1e-6, 1.0))
        nn.init.uniform_(lin.weight, -std, std)
    else:
        nn.init.xavier_uniform_(lin.weight, gain=gain)
    if lin.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(lin.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(lin.bias, -bound, bound)
    return lin


def scaled_silu_init_(lin: nn.Linear, scale: float = 1.0):
    # Heuristic: scale std to account for SiLU variance ~ 0.86
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(lin.weight)
    std = scale * math.sqrt(2.0 / max(fan_in, 1)) * 0.86
    nn.init.normal_(lin.weight, mean=0.0, std=std)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    return lin


# NTK-related rescaling and dynamic fan-in helpers
def ntk_rescale(weight: torch.Tensor, t: int | float) -> torch.Tensor:
    """Rescale weight over time step t to preserve NTK scaling heuristics."""
    s = 1.0 / math.sqrt(max(float(t), 1.0))
    return (weight.float() * s).to(dtype=weight.dtype)


def fan_in_dynamic(weight: torch.Tensor, width: int) -> float:
    """Return recommended std for dynamic fan-in width."""
    fi = max(int(width), 1)
    return float(math.sqrt(2.0 / fi))


def init_schedules(name: str, step: int, total_steps: int, start: float = 1.0, end: float = 1.0) -> float:
    """Time-varying init scaling schedules.

    name in {"linear", "cosine"}. Returns scalar scale factor.
    """
    s = min(max(step / max(total_steps, 1), 0.0), 1.0)
    if name == "cosine":
        import math
        w = (1 - math.cos(math.pi * s)) / 2
    else:
        w = s
    return float(start + (end - start) * w)
