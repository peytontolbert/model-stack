import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int, dropout_p: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p and dropout_p > 0.0 else nn.Identity()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(self.down(x))
        y = self.dropout(y)
        return self.scale * y


class IA3Adapter(nn.Module):
    """Simple multiplicative adapter over last (feature) dim.

    x -> x * diag(s)
    """

    def __init__(self, hidden_size: int, init: float = 1.0):
        super().__init__()
        self.s = nn.Parameter(torch.full((hidden_size,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.s.to(dtype=x.dtype, device=x.device)


def attach_adapters_to_block(block: nn.Module, bottleneck: int | None = None, ia3: bool = False, where: str = "mlp") -> nn.Module:
    """Attach adapters into a Transformer-like block.

    - bottleneck: adds a residual BottleneckAdapter after specified submodule ("attn" or "mlp").
    - ia3: inserts IA3 multiplicative scaling before specified submodule.
    """
    assert where in ("attn", "mlp"), "where must be 'attn' or 'mlp'"
    if bottleneck is not None and bottleneck > 0:
        adapter = BottleneckAdapter(block.cfg.d_model, int(bottleneck))
        # Wrap forward of chosen submodule
        if where == "attn":
            old = block.attn.forward
            def new_attn(*args, **kwargs):
                out = old(*args, **kwargs)
                return out + adapter(args[0] if args else kwargs.get("q"))  # residual on input hidden
            block.attn.forward = new_attn  # type: ignore
        else:
            old_mlp = block.mlp.forward
            def new_mlp(x):
                return old_mlp(x) + adapter(x)
            block.mlp.forward = new_mlp  # type: ignore
    if ia3:
        scaler = IA3Adapter(block.cfg.d_model)
        if where == "attn":
            old = block.attn.forward
            def new_attn(*args, **kwargs):
                q = args[0] if args else kwargs.get("q")
                q = scaler(q)
                if args:
                    args = (q,) + args[1:]
                else:
                    kwargs["q"] = q
                return old(*args, **kwargs)
            block.attn.forward = new_attn  # type: ignore
        else:
            old_mlp = block.mlp.forward
            def new_mlp(x):
                return old_mlp(scaler(x))
            block.mlp.forward = new_mlp  # type: ignore
    return block


