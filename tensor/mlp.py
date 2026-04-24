import torch
import torch.nn as nn

from runtime.ops import gated_activation as runtime_gated_activation
from runtime.ops import linear_module as runtime_linear_module
from runtime.ops import mlp_module as runtime_mlp_module
from .activations import gelu, leaky_relu_0p5_squared, silu


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
        if name in (
            "leaky_relu_0p5_squared",
            "leaky-relu-0p5-squared",
            "leaky_relu_0.5_squared",
            "leaky-relu-0.5-squared",
        ):
            return leaky_relu_0p5_squared(x)
        if name in ("swiglu", "gated-silu"):
            # handled in forward for gated case
            return x
        if name == "geglu":
            return x  # handled in forward
        if name == "reglu":
            return x  # handled in forward
        return gelu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training and not isinstance(self.dropout, nn.Identity)):
            return runtime_mlp_module(
                x,
                self.w_in,
                self.w_out,
                activation=self.activation_name,
                gated=self.gated,
            )
        if self.gated:
            x_proj = runtime_linear_module(x, self.w_in)
            a, b = x_proj.chunk(2, dim=-1)
            name = self.activation_name.lower()
            if name in ("swiglu", "gated-silu"):
                x = runtime_gated_activation(a, b, "silu")
            elif name == "geglu":
                x = runtime_gated_activation(a, b, "gelu")
            elif name == "reglu":
                x = runtime_gated_activation(a, b, "relu")
            elif name in (
                "leaky_relu_0p5_squared",
                "leaky-relu-0p5-squared",
                "leaky_relu_0.5_squared",
                "leaky-relu-0.5-squared",
            ):
                x = runtime_gated_activation(a, b, "leaky_relu_0p5_squared")
            else:
                x = runtime_gated_activation(a, b, "silu")
        else:
            x = self._act(runtime_linear_module(x, self.w_in))
        x = self.dropout(x)
        return runtime_linear_module(x, self.w_out)
