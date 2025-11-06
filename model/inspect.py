from typing import Dict, Optional, Tuple
import re
import torch
from transformers import AutoModelForCausalLM  # type: ignore


def detect_target_shapes_from_model(model_id: str) -> Optional[Dict[str, Tuple[int, int]]]:
    try:
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(model_id)
        d_model = int(getattr(cfg, "hidden_size", 0) or 0)
        inter = int(getattr(cfg, "intermediate_size", 0) or 0)
        if d_model <= 0:
            return None
        shapes: Dict[str, Tuple[int, int]] = {
            "q_proj": (d_model, d_model),
            "o_proj": (d_model, d_model),
        }
        if inter > 0:
            shapes["up_proj"] = (inter, d_model)
            # Optionally anticipate down_proj if user extends targets later
            shapes["down_proj"] = (d_model, inter)
        return shapes
    except Exception:
        return None


def detect_target_shapes_from_model_full(model_id: str, target_regex: Optional[str] = None) -> Optional[Dict[str, Tuple[int, int]]]:
    """Load model on CPU and enumerate linear layers to infer shapes; filter by regex if provided."""
    try:
        import torch  # local
        from transformers import AutoModelForCausalLM  # type: ignore

        rx = re.compile(str(target_regex)) if target_regex else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model.eval()
        shapes: Dict[str, Tuple[int, int]] = {}
        # Inspect first transformer block if present for canonical names; else full traversal
        root = getattr(model, "model", model)
        layer0 = None
        try:
            layer0 = root.layers[0]
        except Exception:
            layer0 = root
        for name, mod in layer0.named_modules():
            try:
                w = getattr(mod, "weight", None)
                if w is None or getattr(w, "ndim", 0) != 2:
                    continue
                short = name.split(".")[-1]
                if rx is not None:
                    if not rx.search(short):
                        continue
                else:
                    if not short.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")):
                        continue
                d_out, d_in = tuple(w.shape)
                shapes[short] = (int(d_out), int(d_in))
            except Exception:
                continue
        return shapes or None
    except Exception:
        return None


def detect_target_names_from_model_full(model_id: str, target_regex: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Return mapping from short target name (last token) to full module path for export."""
    try:
        import torch  # local
        from transformers import AutoModelForCausalLM  # type: ignore

        rx = re.compile(str(target_regex)) if target_regex else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model.eval()
        names: Dict[str, str] = {}
        root = getattr(model, "model", model)
        layer0 = None
        try:
            layer0 = root.layers[0]
        except Exception:
            layer0 = root
        for name, mod in layer0.named_modules():
            try:
                w = getattr(mod, "weight", None)
                if w is None or getattr(w, "ndim", 0) != 2:
                    continue
                short = name.split(".")[-1]
                if rx is not None:
                    if not rx.search(short) and not rx.search(name):
                        continue
                else:
                    if not short.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")):
                        continue
                names[short] = name
            except Exception:
                continue
        return names or None
    except Exception:
        return None
