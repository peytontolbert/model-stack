from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import re


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
            shapes["down_proj"] = (d_model, inter)
        return shapes
    except Exception:
        return None


def detect_target_shapes_from_model_full(
    model_id: str,
    target_regex: Optional[str] = None,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """Load model on CPU and enumerate linear layers to infer shapes; filter by regex if provided."""
    try:
        import torch
        from transformers import AutoModelForCausalLM  # type: ignore

        rx = re.compile(str(target_regex)) if target_regex else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model.eval()
        shapes: Dict[str, Tuple[int, int]] = {}
        root = getattr(model, "model", model)
        try:
            layer0 = root.layers[0]
        except Exception:
            layer0 = root
        for name, mod in layer0.named_modules():
            try:
                weight = getattr(mod, "weight", None)
                if weight is None or getattr(weight, "ndim", 0) != 2:
                    continue
                short = name.split(".")[-1]
                if rx is not None:
                    if not rx.search(short):
                        continue
                else:
                    if not short.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")):
                        continue
                d_out, d_in = tuple(weight.shape)
                shapes[short] = (int(d_out), int(d_in))
            except Exception:
                continue
        return shapes or None
    except Exception:
        return None


def detect_target_names_from_model_full(
    model_id: str,
    target_regex: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Return mapping from short target name (last token) to full module path for export."""
    try:
        import torch
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
        try:
            layer0 = root.layers[0]
        except Exception:
            layer0 = root
        for name, mod in layer0.named_modules():
            try:
                weight = getattr(mod, "weight", None)
                if weight is None or getattr(weight, "ndim", 0) != 2:
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


def infer_target_shapes(model: Any) -> Dict[str, Tuple[int, int]]:
    try:
        blocks = getattr(model, "blocks")
        if blocks is not None and len(blocks) > 0:
            b0 = blocks[0]
            shapes: Dict[str, Tuple[int, int]] = {}
            shapes["q_proj"] = (int(b0.attn.w_q.weight.shape[0]), int(b0.attn.w_q.weight.shape[1]))
            shapes["k_proj"] = (int(b0.attn.w_k.weight.shape[0]), int(b0.attn.w_k.weight.shape[1]))
            shapes["v_proj"] = (int(b0.attn.w_v.weight.shape[0]), int(b0.attn.w_v.weight.shape[1]))
            shapes["o_proj"] = (int(b0.attn.w_o.weight.shape[0]), int(b0.attn.w_o.weight.shape[1]))
            win_out = int(b0.mlp.w_in.weight.shape[0])
            d_model = int(b0.mlp.w_in.weight.shape[1])
            ff = int(win_out // 2)
            shapes["gate_proj"] = (ff, d_model)
            shapes["up_proj"] = (ff, d_model)
            shapes["down_proj"] = (int(b0.mlp.w_out.weight.shape[0]), int(b0.mlp.w_out.weight.shape[1]))
            return shapes
    except Exception:
        pass
    try:
        from runtime.block_targets import targets_map
        from runtime.block_utils import getattr_nested

        first = getattr(getattr(model, "model", model), "layers")[0]
        shapes: Dict[str, Tuple[int, int]] = {}
        for name, rel in targets_map("hf").items():
            try:
                weight = getattr_nested(first, rel).weight
                shapes[name] = (int(weight.shape[0]), int(weight.shape[1]))
            except Exception:
                continue
        return shapes
    except Exception:
        return {}


def infer_target_shapes_from_config(model_id: str, *, cache_dir: str | None = None) -> Dict[str, tuple[int, int]]:
    """Infer projection matrix shapes from HF config, honoring GQA for K/V."""
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install 'transformers' to run this example") from exc

    cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    d_model = int(getattr(cfg, "hidden_size", 0) or 0)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0)
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    head_dim = int(getattr(cfg, "head_dim", (d_model // n_heads) if (d_model and n_heads) else 0) or 0)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    if d_model <= 0 or n_heads <= 0 or head_dim <= 0:
        raise RuntimeError("Could not infer attention dims from model config")
    kv_out = int(n_kv_heads * head_dim)
    shapes: Dict[str, tuple[int, int]] = {
        "q_proj": (d_model, d_model),
        "k_proj": (kv_out, d_model),
        "v_proj": (kv_out, d_model),
        "o_proj": (d_model, d_model),
    }
    if inter > 0:
        shapes.update({
            "up_proj": (inter, d_model),
            "down_proj": (d_model, inter),
            "gate_proj": (inter, d_model),
        })
    return shapes
