from typing import Dict, Tuple, Any
from transformers import AutoConfig  # type: ignore
from blocks.targets import targets_map
from blocks.utils import getattr_nested

def infer_target_shapes(model: Any) -> Dict[str, Tuple[int, int]]:
    # Local model backend
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
    # HF fallback: inspect first decoder layer
    try:
        first = getattr(getattr(model, "model", model), "layers")[0]
        shapes: Dict[str, Tuple[int, int]] = {}
        for name, rel in targets_map("hf").items():
            try:
                w = getattr_nested(first, rel).weight
                shapes[name] = (int(w.shape[0]), int(w.shape[1]))
            except Exception:
                continue
        return shapes
    except Exception:
        return {}


def infer_target_shapes_from_config(model_id: str, *, cache_dir: str | None = None) -> Dict[str, tuple[int, int]]:
    """Infer projection matrix shapes from HF config, honoring GQA for K/V.

    For LLaMA variants:
      - q_proj: (n_heads*head_dim, d_model) == (d_model, d_model)
      - k_proj/v_proj: (n_kv_heads*head_dim, d_model)
      - o_proj: (d_model, n_heads*head_dim) == (d_model, d_model)
      - up/gate: (intermediate_size, d_model), down: (d_model, intermediate_size)
    """
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception as e:
        raise RuntimeError("Install 'transformers' to run this example") from e

    cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    d_model = int(getattr(cfg, "hidden_size", 0) or 0)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0)
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    head_dim = int(getattr(cfg, "head_dim", (d_model // n_heads) if (d_model and n_heads) else 0) or 0)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    if d_model <= 0 or n_heads <= 0 or head_dim <= 0:
        raise RuntimeError("Could not infer attention dims from model config")
    # Compute KV out dim honoring GQA
    kv_out = int(n_kv_heads * head_dim)
    shapes: Dict[str, tuple[int, int]] = {
        "q_proj": (d_model, d_model),              # (n_heads*Dh, d_model)
        "k_proj": (kv_out, d_model),              # (n_kv_heads*Dh, d_model)
        "v_proj": (kv_out, d_model),              # (n_kv_heads*Dh, d_model)
        "o_proj": (d_model, d_model),             # (d_model, n_heads*Dh)
    }
    if inter > 0:
        shapes.update({
            "up_proj": (inter, d_model),
            "down_proj": (d_model, inter),
            "gate_proj": (inter, d_model),
        })
    return shapes
