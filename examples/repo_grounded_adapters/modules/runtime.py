from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import subprocess

import numpy as np
import torch

# Reuse existing module utilities
from examples.repo_grounded_adapters.modules.peft import (
)
from model.inspect import detect_target_names_from_model_full

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


def infer_target_names(model_id: str) -> Dict[str, str]:
    names = detect_target_names_from_model_full(model_id, target_regex=None) or {}
    # names maps short -> path within the first layer subtree
    # Example: {"q_proj": "self_attn.q_proj", "o_proj": "self_attn.o_proj", ...}
    return names




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

