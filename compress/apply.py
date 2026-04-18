from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch.nn as nn

from .lora import inject_lora
from .quantization import (
    quantize_linear_modules,
    QuantizedLinearBitNet,
    QuantizedLinearFP8,
    QuantizedLinearInt4,
    QuantizedLinearInt8,
)


def apply_compression(
    model: nn.Module,
    *,
    lora: Optional[Dict[str, Any]] = None,
    quant: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply compression transforms to a model based on config dicts.

    lora: {
      "rank": int,
      "alpha": float = 1.0,
      "dropout": float = 0.0,
      "include": Iterable[str] | None,
      "exclude": Iterable[str] | None,
      "fan_in_fan_out_names": Iterable[str] | None,
    }

    quant: {
      "scheme": str = "int8",  # one of: int8, bitnet, int4, fp8
      "include": Iterable[str] | None,
      "exclude": Iterable[str] | None,
      "calibration": str = "absmax",  # QuantizedLinearInt8.from_float argument
      "percentile": float = 0.999,
      "calibration_inputs": Dict[str, torch.Tensor] | None,
      "weight_opt": str = "none",  # one of: none, awq, gptq
      "activation_quant": str = "none",  # one of: none, static_int8, dynamic_int8
      "activation_quant_bits": int = 8,
      "activation_quant_method": str = "absmax",
      "activation_quant_percentile": float = 0.999,
      "spin": bool = False,
      "spin_random": bool = True,
      "spin_seed": int = 0,
    }
    """
    summary: Dict[str, Any] = {"lora": None, "quant": None}

    if lora is not None:
        lrk = int(lora.get("rank", 0))
        if lrk > 0:
            include = lora.get("include")
            exclude = lora.get("exclude")
            # Provide pragmatic defaults if user does not specify
            if include is None and exclude is None:
                include = [
                    ".w_q",
                    ".w_k",
                    ".w_v",
                    ".w_o",
                    ".mlp",
                ]
            model, reps = inject_lora(
                model,
                lora_rank=lrk,
                lora_alpha=float(lora.get("alpha", 1.0)),
                lora_dropout=float(lora.get("dropout", 0.0)),
                include=include,
                exclude=exclude,
                fan_in_fan_out_names=lora.get("fan_in_fan_out_names"),
            )
            summary["lora"] = {"num": len(reps), "modules": list(reps.keys())}
        else:
            summary["lora"] = {"num": 0, "modules": []}

    if quant is not None:
        reps = quantize_linear_modules(
            model,
            include=quant.get("include"),
            exclude=quant.get("exclude"),
            scheme=str(quant.get("scheme", "int8")),
            calibration=str(quant.get("calibration", "absmax")),
            percentile=float(quant.get("percentile", 0.999)),
            calibration_inputs=quant.get("calibration_inputs"),
            weight_opt=str(quant.get("weight_opt", "none")),
            activation_quant=str(quant.get("activation_quant", "none")),
            activation_quant_bits=int(quant.get("activation_quant_bits", 8)),
            activation_quant_method=str(quant.get("activation_quant_method", "absmax")),
            activation_quant_percentile=float(quant.get("activation_quant_percentile", 0.999)),
            spin=bool(quant.get("spin", False)),
            spin_random=bool(quant.get("spin_random", True)),
            spin_seed=int(quant.get("spin_seed", 0)),
        )
        # Quantized wrappers are configured at replacement time; expose the replacement summary here.
        summary["quant"] = {
            "scheme": str(quant.get("scheme", "int8")),
            "weight_opt": str(quant.get("weight_opt", "none")),
            "activation_quant": str(quant.get("activation_quant", "none")),
            "spin": bool(quant.get("spin", False)),
            "num": len(reps),
            "modules": list(reps.keys()),
        }

    return summary


__all__ = [
    "apply_compression",
    "QuantizedLinearInt8",
    "QuantizedLinearBitNet",
    "QuantizedLinearInt4",
    "QuantizedLinearFP8",
]
