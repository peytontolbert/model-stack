from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch.nn as nn

from .lora import inject_lora
from .quantization import quantize_linear_modules, QuantizedLinearInt8


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
      "include": Iterable[str] | None,
      "exclude": Iterable[str] | None,
      "calibration": str = "absmax",  # QuantizedLinearInt8.from_float argument
      "percentile": float = 0.999,
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
        )
        # Apply calibration if specified (QuantizedLinearInt8 supports from_float params on construction only)
        # For now we expose which modules were replaced; detailed calibration control can be added via include/exclude
        summary["quant"] = {"num": len(reps), "modules": list(reps.keys())}

    return summary


__all__ = ["apply_compression", "QuantizedLinearInt8"]


