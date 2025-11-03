# ActivationTracer hooks, capture pipeline, write JSON (lift from run_repo_adapter.py)

from typing import Dict, Tuple
import torch


def is_block(name: str, _m: torch.nn.Module) -> bool:
    if not name.startswith("model.layers."):
        return False
    rest = name[len("model.layers."):]
    return rest.isdigit() and "." not in rest


def block_out_hook(_key: str, _m: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor | None:
    try:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            v = output[0]
            return v if isinstance(v, torch.Tensor) else None
        v = getattr(output, "hidden_states", None) or getattr(output, "last_hidden_state", None)
        return v if isinstance(v, torch.Tensor) else None
    except Exception:
        return None


def truncate_batch(xx: Dict[str, torch.Tensor], max_tokens: int = 512) -> Dict[str, torch.Tensor]:
    max_t = max(8, int(max_tokens))
    out: Dict[str, torch.Tensor] = {}
    for k, v in xx.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) > max_t:
            out[k] = v[:, -max_t:]
        else:
            out[k] = v
    return out

    
def get_W(m: torch.nn.Module) -> torch.Tensor:
    if hasattr(m, "lm_head") and hasattr(m.lm_head, "weight"):
        return m.lm_head.weight
    if hasattr(m, "get_output_embeddings"):
        we = m.get_output_embeddings()
        if we is not None and hasattr(we, "weight"):
            return we.weight
    raise AttributeError("No output projection found (lm_head or embeddings)")

"""
def _is_block(name: str, _m: torch.nn.Module) -> bool:
    if not name.startswith("model.layers."):
        return False
    rest = name[len("model.layers."):]
    return rest.isdigit() and "." not in rest


def _block_out_hook(_key, _m, _inputs, output):
    try:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            v = output[0]
            return v if isinstance(v, torch.Tensor) else None
        v = getattr(output, "hidden_states", None) or getattr(output, "last_hidden_state", None)
        return v if isinstance(v, torch.Tensor) else None
    except Exception:
        return None


def _truncate_batch(xx):
    max_t = max(8, int(args.interpret_tokens))
    out = {}
    for k, v in xx.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) > max_t:
            out[k] = v[:, -max_t:]
        else:
            out[k] = v
    return out

"""