import torch
from typing import List, Dict, Tuple

def resolve_layer_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    # Heuristic for LLaMA-like transformers
    root = getattr(model, "model", model)
    try:
        return list(getattr(root, "layers"))  # type: ignore[arg-type]
    except Exception:
        # Falcon/MPT variants may use .model.layers
        try:
            return list(getattr(root, "model").layers)  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError("Unable to find transformer layers on model") from e


def find_module_by_relpath(layer_module: torch.nn.Module, relpath: str) -> torch.nn.Module:
    cur = layer_module
    for tok in str(relpath).split("."):
        if not tok:
            continue
        cur = getattr(cur, tok)
    return cur

def apply_weight_deltas(
    model: torch.nn.Module,
    per_layer_deltas: List[Dict[str, torch.Tensor]],
    target_name_map: Dict[str, str],
    *,
    scale: float,
) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    """Add low-rank deltas to Linear weights, returning a list of (param, delta) for cleanup."""
    layers = resolve_layer_modules(model)
    applied: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    num_layers = min(len(layers), len(per_layer_deltas))
    for i in range(num_layers):
        layer = layers[i]
        deltas = per_layer_deltas[i]
        for short, relpath in target_name_map.items():
            if short not in deltas:
                continue
            mod = find_module_by_relpath(layer, relpath)
            w: torch.nn.Parameter = getattr(mod, "weight")  # type: ignore[assignment]
            delta = (scale * deltas[short]).to(w.device, dtype=w.dtype)
            w.data.add_(delta)
            applied.append((w, delta))
    return applied

 


def prepare_head_weight(model: torch.nn.Module, head_use_cpu: bool) -> Tuple[torch.Tensor, torch.device]:
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        W = model.lm_head.weight
    elif hasattr(model, "get_output_embeddings"):
        we = model.get_output_embeddings()
        W = we.weight if (we is not None and hasattr(we, "weight")) else next(model.parameters())
    else:
        W = next(model.parameters())
    if head_use_cpu:
        Wt = W.detach().to(device=torch.device("cpu"), dtype=torch.float32).t().contiguous()
        return Wt, torch.device("cpu")
    dev = next(model.parameters()).device
    Wt = W.detach().to(device=dev).t().contiguous()
    return Wt, dev

    
def local_logits_last(model, input_ids: torch.Tensor) -> torch.Tensor:
    # Use model's own forward to obtain logits directly
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=None)
        if isinstance(out, dict) and ("logits" in out):
            logits_last = out["logits"][:, -1, :]
        elif hasattr(out, "logits"):
            logits_last = out.logits[:, -1, :]
        elif torch.is_tensor(out) and out.dim() == 3:
            logits_last = out[:, -1, :]
        else:
            raise RuntimeError("Unable to extract logits from model output")
    return logits_last.to(torch.float32)
