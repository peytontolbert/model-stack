from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import subprocess

import numpy as np
import torch

# Reuse existing module utilities
from examples.repo_grounded_adapters.modules.peft import (
    detect_target_names_from_model_full as _detect_target_names_from_model_full,
)


@dataclass
class OTFFlags:
    of_sources: str = "question"  # "zoom" | "question"
    zoom_symbol: Optional[str] = None
    zoom_radius: int = 1
    include_text: bool = True
    text_max_bytes: int = 250_000
    max_text_tokens: int = 200_000
    text_weight: float = 0.25


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def find_module_by_relpath(layer_module: torch.nn.Module, relpath: str) -> torch.nn.Module:
    cur = layer_module
    for tok in str(relpath).split("."):
        if not tok:
            continue
        cur = getattr(cur, tok)
    return cur

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


def infer_target_names(model_id: str) -> Dict[str, str]:
    names = _detect_target_names_from_model_full(model_id, target_regex=None) or {}
    # names maps short -> path within the first layer subtree
    # Example: {"q_proj": "self_attn.q_proj", "o_proj": "self_attn.o_proj", ...}
    return names


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


def build_per_layer_deltas(
    adapters: Dict[str, List[Dict[str, Dict[str, np.ndarray]]]],
    target_names: List[str],
    *,
    g_sub: float = 1.0,
    base_adapters: Optional[Dict[str, List[Dict[str, Dict[str, np.ndarray]]]]] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Return per-layer dict of short target name -> delta weight tensor (out,in)."""
    layers_out: List[Dict[str, torch.Tensor]] = []
    num_layers = len(adapters.get("layers", []))
    for i in range(num_layers):
        dest: Dict[str, torch.Tensor] = {}
        cur = adapters["layers"][i]
        base = (
            base_adapters["layers"][i]
            if base_adapters is not None and i < len(base_adapters.get("layers", []))
            else None
        )
        for name in target_names:
            acc: Optional[torch.Tensor] = None
            # Base component
            if base is not None and name in base:
                A = torch.from_numpy(base[name]["A"]).to(torch.float32)
                B = torch.from_numpy(base[name]["B"]).to(torch.float32)
                acc = (A @ B)
            # Subgraph component
            if name in cur:
                A = torch.from_numpy(cur[name]["A"]).to(torch.float32)
                B = torch.from_numpy(cur[name]["B"]).to(torch.float32)
                sub = (A @ B)
                acc = (
                    sub
                    if acc is None
                    else ((1.0 - float(g_sub)) * acc + float(g_sub) * sub)
                )
            if acc is not None:
                dest[name] = acc.contiguous()
        layers_out.append(dest)
    return layers_out



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

def local_logits_last(model, input_ids: torch.Tensor) -> torch.Tensor:
    # Use model's own forward to obtain logits directly
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=None, return_dict=True)
        logits_last = out["logits"][:, -1, :]
    return logits_last.to(torch.float32)

def run_repo_adapter(
    model: str,
    adapters_npz: str,
    repo: str,
    prompt: str,
    *,
    cache_dir: Optional[str] = None,
    device: str = "cpu",
    gpu_ids: Optional[str] = None,
    context_tokens: int = 5000,
    ignore: Optional[List[str]] = None,
    timeout_sec: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Produce an answer with citations via in-process orchestrator; return (exit_code, stdout, stderr)."""
    try:
        from examples.repo_grounded_adapters.modules.runner import generate_answer
        text = generate_answer(
            model_id=model,
            adapters_npz=adapters_npz,
            repo_root=repo,
            prompt=prompt,
            cache_dir=cache_dir,
            context_tokens=int(context_tokens),
            pack_context=True,
            require_citations=True,
            device=device,
            gpu_ids=gpu_ids,
        )
        return 0, text, ""
    except Exception as e:
        return 1, "", str(e)


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