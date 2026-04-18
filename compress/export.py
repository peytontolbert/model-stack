"""Export/apply compression deltas (LoRA, pruning masks, quant state).

The format is a torch.save'd dict with keys:
  - meta: metadata dict
  - lora: state dict of LoRA A/B tensors (optional)
  - pruning_masks: param_name -> mask tensor (optional)
  - quant: module_name -> quantized-module state dict (optional)
"""

from typing import Dict, Optional

import torch

from .lora import get_lora_state_dict, apply_lora_state_dict
from .quantization import QuantizedLinearBitNet, QuantizedLinearFP8, QuantizedLinearInt4, QuantizedLinearInt8


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
        # Quant state
        quant: Dict[str, object] = {}
        for name, m in model.named_modules():
            if isinstance(m, QuantizedLinearInt8):
                quant[name] = {
                    "type": "int8_pc",
                    "qweight": m.qweight.detach().cpu(),
                    "inv_scale": m.inv_scale.detach().cpu(),
                    "pre_scale": m.pre_scale.detach().cpu(),
                    "act_scale": m.act_scale.detach().cpu(),
                    "weight_opt": m.weight_opt,
                    "act_quant_mode": m.act_quant_mode,
                    "act_quant_bits": m.act_quant_bits,
                    "act_quant_method": m.act_quant_method,
                    "act_quant_percentile": m.act_quant_percentile,
                    "spin_enabled": bool(int(m.spin_enabled_flag.item())),
                    "spin_signs": m.spin_signs.detach().cpu(),
                }
            elif isinstance(m, QuantizedLinearBitNet):
                quant[name] = {
                    "type": "bitnet_w2a8",
                    "packed_weight": m.packed_weight.detach().cpu(),
                    "scale_values": m.scale_values.detach().cpu(),
                    "layout_header": m.layout_header.detach().cpu(),
                    "segment_offsets": m.segment_offsets.detach().cpu(),
                    "pre_scale": m.pre_scale.detach().cpu(),
                    "act_scale": m.act_scale.detach().cpu(),
                    "weight_opt": m.weight_opt,
                    "act_quant_mode": m.act_quant_mode,
                    "act_quant_bits": m.act_quant_bits,
                    "act_quant_method": m.act_quant_method,
                    "act_quant_percentile": m.act_quant_percentile,
                    "spin_enabled": bool(int(m.spin_enabled_flag.item())),
                    "spin_signs": m.spin_signs.detach().cpu(),
                }
            elif isinstance(m, QuantizedLinearInt4):
                quant[name] = {
                    "type": "int4_pc_packed",
                    "qweight_packed": m.qweight_packed.detach().cpu(),
                    "inv_scale": m.inv_scale.detach().cpu(),
                    "pre_scale": m.pre_scale.detach().cpu(),
                    "act_scale": m.act_scale.detach().cpu(),
                    "weight_opt": m.weight_opt,
                    "act_quant_mode": m.act_quant_mode,
                    "act_quant_bits": m.act_quant_bits,
                    "act_quant_method": m.act_quant_method,
                    "act_quant_percentile": m.act_quant_percentile,
                    "spin_enabled": bool(int(m.spin_enabled_flag.item())),
                    "spin_signs": m.spin_signs.detach().cpu(),
                }
            elif isinstance(m, QuantizedLinearFP8):
                quant[name] = {
                    "type": "fp8_fake",
                    "weight_fp8": m.weight_fp8.detach().cpu(),
                    "weight_scale": m.weight_scale.detach().cpu(),
                    "amax_observed": m.amax_observed.detach().cpu(),
                    "fp8_dtype": m.fp8_dtype,
                    "pre_scale": m.pre_scale.detach().cpu(),
                    "act_scale": m.act_scale.detach().cpu(),
                    "weight_opt": m.weight_opt,
                    "act_quant_mode": m.act_quant_mode,
                    "act_quant_bits": m.act_quant_bits,
                    "act_quant_method": m.act_quant_method,
                    "act_quant_percentile": m.act_quant_percentile,
                    "spin_enabled": bool(int(m.spin_enabled_flag.item())),
                    "spin_signs": m.spin_signs.detach().cpu(),
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
            if name not in quant:
                continue
            info = quant[name]
            if isinstance(m, QuantizedLinearInt8):
                if info.get("type") == "int8_pc":
                    if "qweight" in info:
                        m.qweight.copy_(info["qweight"].to(m.qweight.device))
                    m.inv_scale.copy_(info["inv_scale"].to(m.inv_scale.device))
                    if "pre_scale" in info:
                        m.pre_scale.copy_(info["pre_scale"].to(m.pre_scale.device))
                    if "act_scale" in info:
                        m.act_scale.copy_(info["act_scale"].to(m.act_scale.device))
                    if "weight_opt" in info:
                        m.weight_opt = str(info["weight_opt"])
                    if "act_quant_mode" in info:
                        m.act_quant_mode = str(info["act_quant_mode"])
                    if "act_quant_bits" in info:
                        m.act_quant_bits = int(info["act_quant_bits"])
                    if "act_quant_method" in info:
                        m.act_quant_method = str(info["act_quant_method"])
                    if "act_quant_percentile" in info:
                        m.act_quant_percentile = float(info["act_quant_percentile"])
                    if "spin_signs" in info:
                        m.spin_signs.copy_(info["spin_signs"].to(m.spin_signs.device))
                    if "spin_enabled" in info:
                        m.spin_enabled_flag.copy_(
                            torch.tensor(1 if info["spin_enabled"] else 0, device=m.spin_enabled_flag.device, dtype=m.spin_enabled_flag.dtype)
                        )
                    m._invalidate_weight_cache()
            elif isinstance(m, QuantizedLinearBitNet):
                if info.get("type") == "bitnet_w2a8":
                    if "packed_weight" in info:
                        m.packed_weight = info["packed_weight"].to(m.packed_weight.device, dtype=m.packed_weight.dtype)
                    if "scale_values" in info:
                        m.scale_values = info["scale_values"].to(m.scale_values.device, dtype=m.scale_values.dtype)
                    if "layout_header" in info:
                        m.layout_header = info["layout_header"].to(m.layout_header.device, dtype=m.layout_header.dtype)
                    if "segment_offsets" in info:
                        m.segment_offsets = info["segment_offsets"].to(m.segment_offsets.device, dtype=m.segment_offsets.dtype)
                    if "pre_scale" in info:
                        m.pre_scale.copy_(info["pre_scale"].to(m.pre_scale.device))
                    if "act_scale" in info:
                        m.act_scale.copy_(info["act_scale"].to(m.act_scale.device))
                    if "weight_opt" in info:
                        m.weight_opt = str(info["weight_opt"])
                    if "act_quant_mode" in info:
                        m.act_quant_mode = str(info["act_quant_mode"])
                    if "act_quant_bits" in info:
                        m.act_quant_bits = int(info["act_quant_bits"])
                    if "act_quant_method" in info:
                        m.act_quant_method = str(info["act_quant_method"])
                    if "act_quant_percentile" in info:
                        m.act_quant_percentile = float(info["act_quant_percentile"])
                    if "spin_signs" in info:
                        m.spin_signs.copy_(info["spin_signs"].to(m.spin_signs.device))
                    if "spin_enabled" in info:
                        m.spin_enabled_flag.copy_(
                            torch.tensor(1 if info["spin_enabled"] else 0, device=m.spin_enabled_flag.device, dtype=m.spin_enabled_flag.dtype)
                        )
                    m._invalidate_weight_cache()
            elif isinstance(m, QuantizedLinearInt4):
                if info.get("type") == "int4_pc_packed":
                    if "qweight_packed" in info:
                        m.qweight_packed.copy_(info["qweight_packed"].to(m.qweight_packed.device))
                    m.inv_scale.copy_(info["inv_scale"].to(m.inv_scale.device))
                    if "pre_scale" in info:
                        m.pre_scale.copy_(info["pre_scale"].to(m.pre_scale.device))
                    if "act_scale" in info:
                        m.act_scale.copy_(info["act_scale"].to(m.act_scale.device))
                    if "weight_opt" in info:
                        m.weight_opt = str(info["weight_opt"])
                    if "act_quant_mode" in info:
                        m.act_quant_mode = str(info["act_quant_mode"])
                    if "act_quant_bits" in info:
                        m.act_quant_bits = int(info["act_quant_bits"])
                    if "act_quant_method" in info:
                        m.act_quant_method = str(info["act_quant_method"])
                    if "act_quant_percentile" in info:
                        m.act_quant_percentile = float(info["act_quant_percentile"])
                    if "spin_signs" in info:
                        m.spin_signs.copy_(info["spin_signs"].to(m.spin_signs.device))
                    if "spin_enabled" in info:
                        m.spin_enabled_flag.copy_(
                            torch.tensor(1 if info["spin_enabled"] else 0, device=m.spin_enabled_flag.device, dtype=m.spin_enabled_flag.dtype)
                        )
                    m._invalidate_weight_cache()
            elif isinstance(m, QuantizedLinearFP8):
                if info.get("type") == "fp8_fake":
                    m.weight_fp8.copy_(info["weight_fp8"].to(m.weight_fp8.device))
                    m.weight_scale.copy_(info["weight_scale"].to(m.weight_scale.device))
                    if "amax_observed" in info:
                        m.amax_observed.copy_(info["amax_observed"].to(m.amax_observed.device))
                    if "fp8_dtype" in info:
                        m.fp8_dtype = str(info["fp8_dtype"])
                    if "pre_scale" in info:
                        m.pre_scale.copy_(info["pre_scale"].to(m.pre_scale.device))
                    if "act_scale" in info:
                        m.act_scale.copy_(info["act_scale"].to(m.act_scale.device))
                    if "weight_opt" in info:
                        m.weight_opt = str(info["weight_opt"])
                    if "act_quant_mode" in info:
                        m.act_quant_mode = str(info["act_quant_mode"])
                    if "act_quant_bits" in info:
                        m.act_quant_bits = int(info["act_quant_bits"])
                    if "act_quant_method" in info:
                        m.act_quant_method = str(info["act_quant_method"])
                    if "act_quant_percentile" in info:
                        m.act_quant_percentile = float(info["act_quant_percentile"])
                    if "spin_signs" in info:
                        m.spin_signs.copy_(info["spin_signs"].to(m.spin_signs.device))
                    if "spin_enabled" in info:
                        m.spin_enabled_flag.copy_(
                            torch.tensor(1 if info["spin_enabled"] else 0, device=m.spin_enabled_flag.device, dtype=m.spin_enabled_flag.dtype)
                        )
                    m._invalidate_weight_cache()


__all__ = [
    "build_delta",
    "export_delta",
    "load_delta",
    "apply_delta",
]
