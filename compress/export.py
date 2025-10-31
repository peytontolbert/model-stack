"""Export/apply compression deltas (LoRA, pruning masks, quant scales).

The format is a torch.save'd dict with keys:
  - meta: metadata dict
  - lora: state dict of LoRA A/B tensors (optional)
  - pruning_masks: param_name -> mask tensor (optional)
  - quant: module_name -> {type: "int8_pc", inv_scale: Tensor} (optional)
"""

from typing import Dict, Optional

import torch

from .lora import get_lora_state_dict, apply_lora_state_dict
from .quantization import QuantizedLinearInt8


def build_delta(
    model: Optional[torch.nn.Module] = None,
    pruning_masks: Optional[Dict[str, torch.Tensor]] = None,
    extra: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    delta: Dict[str, object] = {"meta": {"format": "delta-v1"}}
    if model is not None:
        # LoRA
        lora_sd = get_lora_state_dict(model)
        if len(lora_sd) > 0:
            delta["lora"] = lora_sd
        # Quant scales
        quant: Dict[str, object] = {}
        for name, m in model.named_modules():
            if isinstance(m, QuantizedLinearInt8):
                quant[name] = {
                    "type": "int8_pc",
                    "inv_scale": m.inv_scale.detach().cpu(),
                }
        if len(quant) > 0:
            delta["quant"] = quant
    if pruning_masks is not None and len(pruning_masks) > 0:
        delta["pruning_masks"] = {k: v.detach().cpu() for k, v in pruning_masks.items()}
    if extra is not None:
        delta["extra"] = extra
    return delta


def export_delta(path: str, delta: Dict[str, object]) -> None:
    torch.save(delta, path)


def load_delta(path: str) -> Dict[str, object]:
    return torch.load(path, map_location="cpu")


@torch.no_grad()
def apply_delta(
    model: torch.nn.Module,
    delta: Dict[str, object],
    apply_lora: bool = True,
    apply_pruning: bool = True,
    apply_quant_scales: bool = True,
) -> None:
    if apply_lora and "lora" in delta:
        apply_lora_state_dict(model, delta["lora"])  # type: ignore[arg-type]
    if apply_pruning and "pruning_masks" in delta:
        masks = delta["pruning_masks"]  # type: ignore[assignment]
        for name, p in model.named_parameters():
            m = masks.get(name)  # type: ignore[attr-defined]
            if m is not None and p.data.shape == m.shape:
                p.data.mul_(m.to(p.data.device))
    if apply_quant_scales and "quant" in delta:
        quant = delta["quant"]  # type: ignore[assignment]
        for name, m in model.named_modules():
            if isinstance(m, QuantizedLinearInt8) and name in quant:
                info = quant[name]
                if info.get("type") == "int8_pc":
                    m.inv_scale.copy_(info["inv_scale"].to(m.inv_scale.device))


__all__ = [
    "build_delta",
    "export_delta",
    "load_delta",
    "apply_delta",
]


