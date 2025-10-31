"""LoRA: Low-Rank Adaptation utilities.

Provides:
- LoRALinear: Drop-in replacement for nn.Linear with LoRA adapters
- inject_lora: Replace modules in a model with LoRA-enabled variants
- merge_lora/unmerge_lora: Fold/unfold LoRA weights into base weights
- get_lora_state_dict/apply_lora_state_dict: Save/restore only LoRA params
- extract_lora_delta: Build delta dict for export compatibility
"""

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor.init import kaiming_uniform_linear


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapters.

    Base weight and bias behave as a standard nn.Linear. Two low-rank matrices
    (A: in_features x r, B: r x out_features) provide a trainable adapter
    whose effective contribution is scaled by alpha/r.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (
            self.lora_alpha / self.lora_rank if self.lora_rank > 0 else 0.0
        )
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if self.lora_rank > 0:
            # A: in_features x r; B: r x out_features
            self.lora_A = nn.Parameter(torch.zeros(in_features, self.lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(self.lora_rank, out_features))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        self._merged = False

    def reset_parameters(self) -> None:
        # Use centralized tensor.init for consistency across the stack
        kaiming_uniform_linear(self)
        if self.lora_rank > 0:
            # Recommended LoRA init: A zeros, B normal small
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B, std=1e-3)

    def _linear(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if self.fan_in_fan_out:
            weight = weight.t()
        return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self._linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and not self._merged:
            # LoRA path
            after_dropout = self.lora_dropout(x)
            # [*, in_features] @ [in_features, r] @ [r, out_features] -> [*, out_features]
            lora_update = after_dropout.matmul(self.lora_A).matmul(self.lora_B)
            result = result + self.lora_scaling * lora_update
        return result

    @torch.no_grad()
    def merge(self) -> None:
        """Fold LoRA weights into the base weight and disable adapters."""
        if self._merged or self.lora_rank == 0:
            self._merged = True
            return
        # Effective delta W = scaling * B^T @ A^T (taking into account linear formulation)
        # Since forward uses x @ A @ B, its equivalent to W + scaling * (B @ A^T)^T
        delta_w = (self.lora_B @ self.lora_A.t()) * self.lora_scaling
        if self.fan_in_fan_out:
            # weight is stored transposed
            self.weight += delta_w.t()
        else:
            self.weight += delta_w
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Unfold LoRA weights from the base weight and re-enable adapters."""
        if not self._merged or self.lora_rank == 0:
            self._merged = False
            return
        delta_w = (self.lora_B @ self.lora_A.t()) * self.lora_scaling
        if self.fan_in_fan_out:
            self.weight -= delta_w.t()
        else:
            self.weight -= delta_w
        self._merged = False


def _should_inject(name: str, include: Optional[Iterable[str]], exclude: Optional[Iterable[str]]) -> bool:
    if include is not None and len(list(include)) > 0:
        if not any(pat in name for pat in include):
            return False
    if exclude is not None and len(list(exclude)) > 0:
        if any(pat in name for pat in exclude):
            return False
    return True


def inject_lora(
    model: nn.Module,
    lora_rank: int,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    fan_in_fan_out_names: Optional[Iterable[str]] = None,
) -> Tuple[nn.Module, Dict[str, LoRALinear]]:
    """Replace nn.Linear modules with LoRALinear in-place.

    Returns the model and a dict of module-name -> LoRALinear for convenience.
    """
    replacements: Dict[str, LoRALinear] = {}
    fifo = set(fan_in_fan_out_names or [])
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _should_inject(name, include, exclude):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            attr_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            l = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                fan_in_fan_out=(name in fifo),
            )
            # copy base weights
            with torch.no_grad():
                l.weight.copy_(module.weight)
                if module.bias is not None and l.bias is not None:
                    l.bias.copy_(module.bias)
            setattr(parent, attr_name, l)
            replacements[name] = l
    return model, replacements


def merge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return only LoRA adapter parameters as a flat state_dict-like mapping."""
    lora_sd: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.lora_rank > 0:
            lora_sd[f"{name}.lora_A"] = module.lora_A.detach().clone()
            lora_sd[f"{name}.lora_B"] = module.lora_B.detach().clone()
    return lora_sd


@torch.no_grad()
def apply_lora_state_dict(model: nn.Module, lora_sd: Dict[str, torch.Tensor]) -> None:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.lora_rank > 0:
            a_key, b_key = f"{name}.lora_A", f"{name}.lora_B"
            if a_key in lora_sd and b_key in lora_sd:
                module.lora_A.copy_(lora_sd[a_key])
                module.lora_B.copy_(lora_sd[b_key])


def extract_lora_delta(base_sd: Dict[str, torch.Tensor], finetuned_sd: Dict[str, torch.Tensor], rank: int) -> Dict[str, torch.Tensor]:
    """Build a delta containing only low-rank updates for Linear weights.

    For each weight W in finetuned that matches base, compute dW = W_ft - W_base and
    compute a thin SVD to approximate dW ≈ U_r @ S_r @ V_r^T; store A = V_r @ S_r^{1/2},
    B = S_r^{1/2} @ U_r^T so that dW ≈ B @ A^T matching our forward A/B shapes.
    """
    device = next(iter(finetuned_sd.values())).device if len(finetuned_sd) > 0 else torch.device("cpu")
    delta: Dict[str, torch.Tensor] = {"__format__": torch.tensor([1], device=device)}
    for k, w_ft in finetuned_sd.items():
        if not (k.endswith(".weight") and k in base_sd and base_sd[k].shape == w_ft.shape):
            continue
        w_base = base_sd[k]
        dW = (w_ft - w_base).to(dtype=torch.float32)
        if dW.ndim != 2:
            continue
        # SVD-based rank-r approximation
        try:
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        except RuntimeError:
            # Fallback to CPU if GPU SVD fails
            U, S, Vh = torch.linalg.svd(dW.cpu(), full_matrices=False)
            U, S, Vh = U.to(dW.device), S.to(dW.device), Vh.to(dW.device)
        r = min(rank, S.shape[0])
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        # Shapes: U_r [o, r], S_r [r], Vh_r [r, i]
        # Choose A: [i, r] and B: [r, o] so that B @ A^T ~ dW
        A = Vh_r.t()  # [i, r]
        B = (S_r[None, :] * U_r.t())  # [r, o]
        delta[f"{k}::lora_A"] = A.contiguous()
        delta[f"{k}::lora_B"] = B.contiguous()
        delta[f"{k}::shape"] = torch.tensor(list(dW.shape), device=device)
    return delta


# Local imports at end to avoid circulars

