from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from examples.repo_grounded_adapters.repo_conditioned_adapter import (
    build_repo_embedding,
    build_subgraph_embedding_from_graph,
    generate_lora_from_embedding,
    load_adapters_npz,
    _detect_target_shapes_from_model_full,  # type: ignore[attr-defined]
    _detect_target_names_from_model_full,   # type: ignore[attr-defined]
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


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_layer_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
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


def _infer_target_names(model_id: str) -> Dict[str, str]:
    names = _detect_target_names_from_model_full(model_id, target_regex=None) or {}
    # names maps short -> path within the first layer subtree
    # Example: {"q_proj": "self_attn.q_proj", "o_proj": "self_attn.o_proj", ...}
    return names


def _find_module_by_relpath(layer_module: torch.nn.Module, relpath: str) -> torch.nn.Module:
    cur = layer_module
    for tok in str(relpath).split("."):
        if not tok:
            continue
        cur = getattr(cur, tok)
    return cur


def _apply_weight_deltas(
    model: torch.nn.Module,
    per_layer_deltas: List[Dict[str, torch.Tensor]],
    target_name_map: Dict[str, str],
    *,
    scale: float,
) -> List[Tuple[torch.nn.Parameter, torch.Tensor]]:
    """Add low-rank deltas to Linear weights, returning a list of (param, delta) for cleanup."""
    layers = _resolve_layer_modules(model)
    applied: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    num_layers = min(len(layers), len(per_layer_deltas))
    for i in range(num_layers):
        layer = layers[i]
        deltas = per_layer_deltas[i]
        for short, relpath in target_name_map.items():
            if short not in deltas:
                continue
            mod = _find_module_by_relpath(layer, relpath)
            w: torch.nn.Parameter = getattr(mod, "weight")  # type: ignore[assignment]
            delta = (scale * deltas[short]).to(w.device, dtype=w.dtype)
            w.data.add_(delta)
            applied.append((w, delta))
    return applied


def _build_per_layer_deltas(
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
        base = (base_adapters["layers"][i] if base_adapters is not None and i < len(base_adapters.get("layers", [])) else None)
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
                acc = (sub if acc is None else ((1.0 - float(g_sub)) * acc + float(g_sub) * sub))
            if acc is not None:
                dest[name] = acc.contiguous()
        layers_out.append(dest)
    return layers_out


def main() -> None:
    p = argparse.ArgumentParser()
    # Model & I/O
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapters", default=None, help="Path to base adapters.npz (optional)")
    p.add_argument("--repo", default=os.getcwd())
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default=None, help="Optional path to save response JSON")
    # LoRA/mixing
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--rank", type=int, default=12)
    p.add_argument("--gsub", type=float, default=0.75, help="Weight for on-the-fly subgraph adapters vs base")
    p.add_argument("--mix-beta", type=float, default=0.0, help="Reserved: variance/norm matching strength")
    # OTF selection/embedding
    p.add_argument("--of-sources", default="question", choices=["question", "zoom"])
    p.add_argument("--zoom-symbol", default=None)
    p.add_argument("--zoom-radius", type=int, default=1)
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--text-max-bytes", type=int, default=250_000)
    p.add_argument("--max-text-tokens", type=int, default=200_000)
    p.add_argument("--text-weight", type=float, default=0.25)
    # Generation
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--min-new-tokens", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dev = _get_device()

    # Lazy import transformers to avoid hard dep in non-usage workflows
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError("Install 'transformers' to run this script") from e

    if args.verbose:
        print(json.dumps({"device": str(dev), "model": args.model}, indent=2))

    # Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model)
    if getattr(tok, "pad_token", None) is None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass

    # Prefer bf16 on CUDA for stability; fallback to fp16 otherwise
    torch_dtype = torch.bfloat16 if (dev.type == "cuda") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model.to(dev)
    model.eval()

    # Discover target names and shapes from the HF model id
    target_name_map = _infer_target_names(args.model)
    if not target_name_map:
        # Fallback to canonical LLaMA names if detection failed
        target_name_map = {
            "q_proj": "self_attn.q_proj",
            "k_proj": "self_attn.k_proj",
            "v_proj": "self_attn.v_proj",
            "o_proj": "self_attn.o_proj",
            "up_proj": "mlp.up_proj",
            "down_proj": "mlp.down_proj",
            "gate_proj": "mlp.gate_proj",
        }
    target_shapes = _detect_target_shapes_from_model_full(args.model, target_regex=None) or {}

    # Build on-the-fly adapters from repo evidence
    otf_flags = OTFFlags(
        of_sources=args.of_sources,
        zoom_symbol=args.zoom_symbol,
        zoom_radius=int(args.zoom_radius),
        include_text=bool(args.include_text),
        text_max_bytes=int(args.text_max_bytes),
        max_text_tokens=int(args.max_text_tokens),
        text_weight=float(args.text_weight),
    )

    # Minimal OTF: use repository-level embedding with optional text hashing
    # Advanced zooming via CodeGraph can be added here later.
    emb = build_repo_embedding(
        args.repo,
        dim=1536,
        seed=0,
        include_text=bool(otf_flags.include_text),
        text_max_bytes=int(otf_flags.text_max_bytes),
        max_text_tokens=int(otf_flags.max_text_tokens),
        text_weight=float(otf_flags.text_weight),
        graph_prop_hops=0,
        graph_prop_damp=0.85,
    )

    # Infer model dims from the HF model
    try:
        d_model = int(getattr(model.config, "hidden_size"))
        num_layers = int(len(_resolve_layer_modules(model)))
    except Exception as e:
        raise RuntimeError("Unable to infer d_model/num_layers from model") from e

    targets = list(target_name_map.keys())
    if target_shapes:
        # Restrict targets to those we know shapes for
        targets = [t for t in targets if t in target_shapes]

    sub = generate_lora_from_embedding(
        emb["z"],
        d_model=d_model,
        num_layers=num_layers,
        rank=int(args.rank),
        seed=0,
        targets=targets,
        target_shapes=target_shapes or None,
        layer_gate="zmean",
        target_weights=None,
    )

    base = None
    if args.adapters:
        try:
            base = load_adapters_npz(str(args.adapters))
        except Exception:
            base = None

    # Build combined per-layer deltas
    per_layer = _build_per_layer_deltas(sub, targets, g_sub=float(args.gsub), base_adapters=base)

    # Apply deltas and generate
    scale = float(args.alpha) / float(max(1, int(args.rank)))
    applied = _apply_weight_deltas(model, per_layer, target_name_map, scale=scale)

    try:
        # Simple prompt; instruction-tuned models generally handle plain prompts
        input_ids = tok([args.prompt], return_tensors="pt", padding=True).to(dev)
        gen_kwargs = {
            "max_new_tokens": int(args.max_new_tokens),
            "min_new_tokens": int(max(0, int(args.min_new_tokens))),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
        }
        with torch.no_grad():
            out = model.generate(**input_ids, **gen_kwargs)
        text = tok.decode(out[0], skip_special_tokens=True)
        result = {
            "model": args.model,
            "alpha": float(args.alpha),
            "rank": int(args.rank),
            "gsub": float(args.gsub),
            "response": text,
        }
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
    finally:
        # Cleanup: remove deltas
        for w, delta in applied:
            try:
                w.data.sub_(delta)
            except Exception:
                pass


if __name__ == "__main__":
    main()


