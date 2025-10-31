import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import gelu, silu


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ff_size: int,
        activation: str = "silu",
        dropout_p: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.activation_name = activation
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()

        act = activation.lower()
        if act in ("swiglu", "gated-silu", "geglu", "reglu"):
            self.gated = True
            self.w_in = nn.Linear(hidden_size, 2 * ff_size, bias=bias)
            self.w_out = nn.Linear(ff_size, hidden_size, bias=bias)
        else:
            self.gated = False
            self.w_in = nn.Linear(hidden_size, ff_size, bias=bias)
            self.w_out = nn.Linear(ff_size, hidden_size, bias=bias)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        name = self.activation_name.lower()
        if name in ("gelu",):
            return gelu(x)
        if name in ("silu", "swish"):
            return silu(x)
        if name in ("swiglu", "gated-silu"):
            # handled in forward for gated case
            return x
        if name == "geglu":
            return x  # handled in forward
        if name == "reglu":
            return x  # handled in forward
        return gelu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            x_proj = self.w_in(x)
            a, b = x_proj.chunk(2, dim=-1)
            name = self.activation_name.lower()
            if name in ("swiglu", "gated-silu"):
                x = silu(a) * b
            elif name == "geglu":
                x = F.gelu(a) * b
            elif name == "reglu":
                x = F.relu(a) * b
            else:
                x = silu(a) * b
        else:
            x = self._act(self.w_in(x))
        x = self.dropout(x)
        return self.w_out(x)


