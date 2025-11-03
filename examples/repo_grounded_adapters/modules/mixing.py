# delta-cap guard (lift from modular.py)
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from examples.repo_grounded_adapters.code_graph import CodeGraph



def targets_map(backend: str = "local") -> Dict[str, str]:
    if backend == "local":
        return {
            "q_proj": "attn.w_q",
            "k_proj": "attn.w_k",
            "v_proj": "attn.w_v",
            "o_proj": "attn.w_o",
            "up_proj": "mlp.w_in",
            "gate_proj": "mlp.w_in",
            "down_proj": "mlp.w_out",
        }
    return {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
        "gate_proj": "mlp.gate_proj",
    }

def getattr_nested(obj: Any, path: str) -> Any:
    cur: Any = obj
    for tok in str(path).split("."):
        if not tok:
            continue
        cur = getattr(cur, tok)
    return cur


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


def register_hook_mixed_adapters(
    model: Any,
    base_layers: List[Dict[str, Dict[str, np.ndarray]]],
    sub_layers: Optional[List[Dict[str, Dict[str, np.ndarray]]]],
    *,
    alpha_star: float,
    g_sub: float,
    rank: int,
    beta: float,
    target_weights: Optional[Dict[str, float]] = None,
    backend: str = "local",
):
    tmap = targets_map(backend)
    target_weights = target_weights or {}

    def _mat(A: np.ndarray, B: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(A).to(torch.float32) @ torch.from_numpy(B).to(torch.float32)

    per_layer: List[Dict[str, torch.Tensor]] = []
    L = max(len(base_layers or []), len(sub_layers or []))
    for i in range(L):
        cur: Dict[str, torch.Tensor] = {}
        base = base_layers[i] if i < len(base_layers or []) else None
        sub = (sub_layers[i] if (sub_layers is not None and i < len(sub_layers)) else None)
        for name in tmap.keys():
            acc: Optional[torch.Tensor] = None
            if base is not None and name in base:
                try:
                    acc = _mat(base[name]["A"], base[name]["B"]).to(torch.float32)
                except Exception:
                    acc = None
            if sub is not None and name in sub:
                try:
                    sub_m = _mat(sub[name]["A"], sub[name]["B"]).to(torch.float32)
                    acc = sub_m if acc is None else ((1.0 - float(g_sub)) * acc + float(g_sub) * sub_m)
                except Exception:
                    pass
            if acc is not None:
                tw = float(target_weights.get(name, 1.0))
                cur[name] = (tw * acc).contiguous()
        per_layer.append(cur)

    scale = float(alpha_star) / float(max(1, int(rank)))
    # Store CPU-offloaded records of applied deltas to avoid holding large GPU tensors
    # Each record: (param, cpu_delta, mode, extra)
    #   mode == "full"  -> subtract full delta
    #   mode == "slice" -> subtract into row slice [start:end]
    applied: List[Tuple[torch.nn.Parameter, torch.Tensor, str, Tuple[int, int]]] = []
    # layers list
    try:
        layers = list(getattr(getattr(model, "model", model), "layers"))
    except Exception:
        try:
            layers = list(getattr(model, "blocks"))
        except Exception:
            layers = []

    def _cap_delta_if_needed(w: torch.nn.Parameter, d: torch.Tensor) -> torch.Tensor:
        try:
            cap_env = os.environ.get("REPO_ADAPTER_DELTA_CAP", "0.05").strip()
            cap = float(cap_env) if cap_env else 0.0
        except Exception:
            cap = 0.0
        if cap is None or cap <= 0:
            return d
        try:
            wn = float(w.data.norm().item())
            dn = float(d.norm().item())
            if dn > 0 and wn > 0:
                limit = cap * wn
                if dn > limit:
                    s = float(limit / max(dn, 1e-12))
                    return (s * d)
        except Exception:
            return d
        return d

    for i, deltas in enumerate(per_layer):
        if i >= len(layers):
            break
        layer = layers[i]
        for short, rel in tmap.items():
            if short not in deltas:
                continue
            try:
                mod = getattr_nested(layer, rel)
                w: torch.nn.Parameter = getattr(mod, "weight")  # type: ignore[assignment]
                d = (scale * deltas[short]).to(w.device, dtype=w.dtype)
                d = _cap_delta_if_needed(w, d)
                if rel.endswith("mlp.w_in") and short in ("gate_proj", "up_proj"):
                    half = int(w.shape[0] // 2)
                    if short == "gate_proj":
                        part = d[:half, :] if d.shape[0] == w.shape[0] else d
                        part = _cap_delta_if_needed(w, part)
                        w.data[:half, :].add_(part)
                        applied.append((w, part.detach().to("cpu"), "slice", (0, half)))
                    else:
                        part = d[-half:, :] if d.shape[0] == w.shape[0] else d
                        part = _cap_delta_if_needed(w, part)
                        w.data[half:, :].add_(part)
                        applied.append((w, part.detach().to("cpu"), "slice", (half, w.shape[0])))
                else:
                    w.data.add_(d)
                    applied.append((w, d.detach().to("cpu"), "full", (0, 0)))
            except Exception:
                continue

    class _Applied:
        def __init__(self, records: List[Tuple[torch.nn.Parameter, torch.Tensor, str, Tuple[int, int]]]):
            self._records = records
            self._removed = False

        def remove(self) -> None:
            if self._removed:
                return
            for w, d_cpu, mode, extra in self._records:
                try:
                    if mode == "full":
                        d_dev = d_cpu.to(device=w.device, dtype=w.dtype)
                        w.data.sub_(d_dev)
                    elif mode == "slice":
                        start, end = extra
                        d_dev = d_cpu.to(device=w.device, dtype=w.dtype)
                        w.data[start:end, :].sub_(d_dev)
                except Exception:
                    pass
            self._removed = True

    return [_Applied(applied)]
