import os
import argparse
from typing import Dict, List, Optional, Tuple, Any
import re
import json

import numpy as np
import torch
from transformers import AutoTokenizer
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None  # type: ignore
from interpret.tracer import ActivationTracer
from interpret.activation_cache import CaptureSpec

from examples.repo_grounded_adapters.repo_conditioned_adapter import (
    load_adapters_npz,
    generate_lora_from_embedding,
    build_subgraph_embedding_from_graph,
)

# Reuse helpers from the on-the-fly runner to avoid duplication
import examples.repo_grounded_adapters.run_llama_with_repo_adapter_on_the_fly as otf  # type: ignore

# --- Compatibility shims for expected helpers in the OTF module --- #
def _shim_parse_target_weights(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec:
        return None
    out: Dict[str, float] = {}
    try:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                try:
                    out[k] = float(v)
                except Exception:
                    continue
            else:
                out[part] = 1.0
        return out or None
    except Exception:
        return None


def _shim_targets_map() -> Dict[str, str]:
    # Canonical LLaMA-style names; model-specific detection may refine later
    return {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
        "gate_proj": "mlp.gate_proj",
    }


def _shim_getattr_nested(obj: Any, path: str) -> Any:
    cur: Any = obj
    for tok in str(path).split("."):
        if not tok:
            continue
        cur = getattr(cur, tok)
    return cur


def _shim_infer_target_shapes(model: Any) -> Dict[str, Tuple[int, int]]:
    shapes: Dict[str, Tuple[int, int]] = {}
    # Local backend: model has .blocks
    try:
        blocks = getattr(model, "blocks")
        if blocks is not None and len(blocks) > 0:
            b0 = blocks[0]
            # attn weights
            shapes["q_proj"] = (int(getattr(b0.attn.w_q.weight, "shape")[0]), int(getattr(b0.attn.w_q.weight, "shape")[1]))
            shapes["k_proj"] = (int(b0.attn.w_k.weight.shape[0]), int(b0.attn.w_k.weight.shape[1]))
            shapes["v_proj"] = (int(b0.attn.w_v.weight.shape[0]), int(b0.attn.w_v.weight.shape[1]))
            shapes["o_proj"] = (int(b0.attn.w_o.weight.shape[0]), int(b0.attn.w_o.weight.shape[1]))
            # mlp: split w_in into gate/up halves; w_out is down_proj
            win_out = int(b0.mlp.w_in.weight.shape[0])
            d_model = int(b0.mlp.w_in.weight.shape[1])
            ff = int(win_out // 2)
            shapes["gate_proj"] = (ff, d_model)
            shapes["up_proj"] = (ff, d_model)
            shapes["down_proj"] = (int(b0.mlp.w_out.weight.shape[0]), int(b0.mlp.w_out.weight.shape[1]))
            return shapes
    except Exception:
        pass
    # HF backend
    try:
        first_layer = getattr(getattr(model, "model", model), "layers")[0]
        for name, rel in _shim_targets_map().items():
            try:
                mod = _shim_getattr_nested(first_layer, rel)
                w = getattr(mod, "weight")
                shapes[name] = (int(w.shape[0]), int(w.shape[1]))
            except Exception:
                continue
        return shapes
    except Exception:
        return {}


_IGN: List[str] = []  # populated from CLI args in main()


def _shim_modules_from_symbols(repo_root: str, seeds: List[str], *, radius: int = 1, top_k: int = 8) -> Tuple[List[str], List[str]]:
    g = CodeGraph.load_or_build(repo_root, ignore=[s for s in (_IGN or []) if s])
    modules_set: Dict[str, bool] = {}
    files_set: Dict[str, bool] = {}
    # Seed by direct symbol matches
    for s in seeds:
        try:
            for sym in g.find_symbol(s):
                modules_set[sym.module] = True
                files_set[os.path.relpath(sym.file, g.root)] = True
        except Exception:
            continue
    # Expand modules by import radius
    if radius > 0 and modules_set:
        cur = list(modules_set.keys())
        seen = set(cur)
        for _ in range(max(0, int(radius))):
            nxt: List[str] = []
            for m in cur:
                for dep in g.module_imports.get(m, []):
                    if dep not in seen:
                        seen.add(dep)
                        nxt.append(dep)
            cur = nxt
        for m in seen:
            modules_set[m] = True
            f = g.file_for_module(m)
            if f:
                files_set[os.path.relpath(f, g.root)] = True
    modules = sorted(modules_set.keys())[: top_k]
    files = sorted(files_set.keys())[: max(top_k, 8)]
    return modules, files


def _shim_question_aware_modules_and_files(repo_root: str, prompt: str, *, top_k: int = 8) -> Tuple[List[str], List[str]]:
    g = CodeGraph.load_or_build(repo_root, ignore=[s for s in (_IGN or []) if s])
    # Simple keyword search over files, rank by hits
    try:
        import re as _re

        toks = [t for t in _re.findall(r"[A-Za-z0-9_]+", (prompt or "").lower()) if len(t) >= 3]
    except Exception:
        toks = []
    score_by_file: Dict[str, int] = {}
    for t in toks[:10]:
        try:
            for (rel, _ln, _txt) in g.search_refs(_re_escape(t)):
                score_by_file[rel] = score_by_file.get(rel, 0) + 1
        except Exception:
            continue
    ranked_files = [fp for fp, _ in sorted(score_by_file.items(), key=lambda x: x[1], reverse=True)]
    files = ranked_files[: max(top_k, 8)]
    modules_set: Dict[str, bool] = {}
    for f in files:
        m = g.module_for_file(f)
        if m:
            modules_set[m] = True
    modules = sorted(modules_set.keys())[: top_k]
    return modules, files


def _re_escape(s: str) -> str:
    try:
        import re as _re

        return _re.escape(s)
    except Exception:
        return s


def _shim_register_hook_mixed_adapters(
    model: Any,
    base_layers: List[Dict[str, Dict[str, np.ndarray]]],
    sub_layers: Optional[List[Dict[str, Dict[str, np.ndarray]]]],
    *,
    alpha_star: float,
    g_sub: float,
    rank: int,
    beta: float,
    target_weights: Optional[Dict[str, float]] = None,
):
    # Compute per-layer weight deltas (out,in) tensors
    # Use the target map provided by the shared OTF helpers so backend-specific
    # name mappings (e.g., local vs HF) are respected.
    targets_map = otf._targets_map()
    target_weights = target_weights or {}
    def _mat(A: np.ndarray, B: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(A).to(torch.float32) @ torch.from_numpy(B).to(torch.float32)

    per_layer: List[Dict[str, torch.Tensor]] = []
    L = max(len(base_layers or []), len(sub_layers or []))
    for i in range(L):
        cur: Dict[str, torch.Tensor] = {}
        base = base_layers[i] if i < len(base_layers or []) else None
        sub = (sub_layers[i] if (sub_layers is not None and i < len(sub_layers)) else None)
        for name in targets_map.keys():
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

    # Apply deltas directly and return removable handles
    scale = float(alpha_star) / float(max(1, int(rank)))
    applied_pairs: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
    # Support HF (.model.layers) and local (.blocks)
    layers = []
    try:
        layers = list(getattr(getattr(model, "model", model), "layers"))
    except Exception:
        try:
            layers = list(getattr(model, "blocks"))
        except Exception:
            layers = []
    # Optional global cap on delta magnitude relative to base weight norm
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
                    scale = float(limit / max(dn, 1e-12))
                    return (scale * d)
        except Exception:
            return d
        return d

    for i, deltas in enumerate(per_layer):
        if i >= len(layers):
            break
        layer = layers[i]
        for short, rel in targets_map.items():
            if short not in deltas:
                continue
            try:
                mod = _shim_getattr_nested(layer, rel)
                w: torch.nn.Parameter = getattr(mod, "weight")  # type: ignore[assignment]
                d = (scale * deltas[short]).to(w.device, dtype=w.dtype)
                d = _cap_delta_if_needed(w, d)
                # Special-case local MLP w_in splitting for gate/up
                if rel.endswith("mlp.w_in") and short in ("gate_proj", "up_proj"):
                    half = int(w.shape[0] // 2)
                    if short == "gate_proj":
                        part = d[:half, :] if d.shape[0] == w.shape[0] else d
                        part = _cap_delta_if_needed(w, part)
                        w.data[:half, :].add_(part)
                    else:
                        part = d[-half:, :] if d.shape[0] == w.shape[0] else d
                        part = _cap_delta_if_needed(w, part)
                        w.data[half:, :].add_(part)
                    applied_pairs.append((w, torch.zeros_like(w)))
                else:
                    w.data.add_(d)
                    applied_pairs.append((w, d))
            except Exception:
                continue

    # Optional debug: print how many parameter deltas were applied when requested
    try:
        if os.environ.get("REPO_ADAPTER_DEBUG", "").strip():
            try:
                print(f"[debug] repo-adapter: applied {len(applied_pairs)} parameter deltas")
            except Exception:
                pass
    except Exception:
        pass

    class _AppliedDeltaHook:
        def __init__(self, pairs: List[Tuple[torch.nn.Parameter, torch.Tensor]]):
            self._pairs = pairs
            self._removed = False

        def remove(self) -> None:
            if self._removed:
                return
            for w, d in self._pairs:
                try:
                    w.data.sub_(d)
                except Exception:
                    pass
            self._removed = True

    return [_AppliedDeltaHook(applied_pairs)]


# Monkey-patch missing helpers if the imported module doesn't provide them
if not hasattr(otf, "_parse_target_weights"):
    setattr(otf, "_parse_target_weights", _shim_parse_target_weights)
if not hasattr(otf, "_targets_map"):
    setattr(otf, "_targets_map", _shim_targets_map)
if not hasattr(otf, "_getattr_nested"):
    setattr(otf, "_getattr_nested", _shim_getattr_nested)
if not hasattr(otf, "_infer_target_shapes"):
    setattr(otf, "_infer_target_shapes", _shim_infer_target_shapes)
if not hasattr(otf, "_modules_from_symbols"):
    setattr(otf, "_modules_from_symbols", _shim_modules_from_symbols)
if not hasattr(otf, "_question_aware_modules_and_files"):
    setattr(otf, "_question_aware_modules_and_files", _shim_question_aware_modules_and_files)
if not hasattr(otf, "_register_hook_mixed_adapters"):
    setattr(otf, "_register_hook_mixed_adapters", _shim_register_hook_mixed_adapters)

try:
    from examples.repo_grounded_adapters.code_graph import CodeGraph
except Exception:
    # Fallback import path
    import importlib.util
    _alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "repo_grounded_adapters", "code_graph.py"))
    _spec = importlib.util.spec_from_file_location("examples.repo_grounded_adapters.code_graph", _alt_path)
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    CodeGraph = getattr(_mod, "CodeGraph")


def _pack_context_heads(repo_root: str, files: List[str], tok, budget_tokens: int) -> str:
    g = CodeGraph.load_or_build(repo_root, ignore=[s for s in (_IGN or []) if s])
    lines_out: List[str] = ["Repository snippets (enhanced):"]
    used = 0
    for rel in files:
        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(g.root, rel))
        try:
            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        head_n = min(len(src_lines), 120)
        block = [f"[ctx] path: {os.path.relpath(abs_fp, g.root)}:1-{head_n}"] + src_lines[:head_n] + [""]
        text = "\n".join(block) + "\n"
        t = len(tok(text).input_ids)
        if used + t > budget_tokens:
            # try smaller
            head_n = min(len(src_lines), 60)
            block = [f"[ctx] path: {os.path.relpath(abs_fp, g.root)}:1-{head_n}"] + src_lines[:head_n] + [""]
            text = "\n".join(block) + "\n"
            t = len(tok(text).input_ids)
            if used + t > budget_tokens:
                continue
        lines_out.extend(block)
        used += t
        if used >= budget_tokens:
            break
    return "\n".join(lines_out) if len(lines_out) > 1 else ""


def _pack_context_windows(repo_root: str, files: List[str], tok, budget_tokens: int) -> str:
    """Naive symbol-window packaging: include windows around first few 'def ' occurrences per file."""
    g = CodeGraph.load_or_build(repo_root, ignore=[s for s in (_IGN or []) if s])
    lines_out: List[str] = ["Repository windows (enhanced):"]
    used = 0
    for rel in files:
        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(g.root, rel))
        try:
            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
        except Exception:
            continue
        # find up to 2 function defs and pack +/- 40 lines
        def_lines: List[int] = []
        for i, line in enumerate(src_lines, start=1):
            s = line.lstrip()
            if s.startswith("def ") or s.startswith("class "):
                def_lines.append(i)
            if len(def_lines) >= 2:
                break
        def_lines = def_lines or [1]
        for ln in def_lines:
            a = max(1, ln - 20)
            b = min(len(src_lines), ln + 40)
            block = [f"[ctx] path: {os.path.relpath(abs_fp, g.root)}:{a}-{b}"] + src_lines[a - 1 : b] + [""]
            text = "\n".join(block) + "\n"
            t = len(tok(text).input_ids)
            if used + t > budget_tokens:
                continue
            lines_out.extend(block)
            used += t
            if used >= budget_tokens:
                break
        if used >= budget_tokens:
            break
    return "\n".join(lines_out) if len(lines_out) > 1 else ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    # Always use local model stack; HF is only used for tokenizer and optional snapshot download
    p.add_argument("--adapters", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--prompt", required=True)

    # Mixing and capacity
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--gsub", type=float, default=0.75)
    p.add_argument("--mix-beta", type=float, default=0.1)
    p.add_argument("--target-weights", default=None)

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device-map", default="none", choices=["auto", "none"])  # no-op for local backend
    p.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids to use (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--max-memory", default=None, help="GPU/CPU memory budget, e.g., '0:22GiB,1:22GiB,cpu:96GiB'")
    p.add_argument("--offload-folder", default=None, help="Directory to offload weights when using device_map=auto")
    # HF-only options removed; kept for CLI compatibility but ignored
    p.add_argument("--attn-impl", default="auto", choices=["auto","flash2","sdpa","eager"], help="(ignored) Attention backend for HF")
    p.add_argument("--load-in-4bit", action="store_true", help="(ignored) Load model in 4-bit quantization")
    p.add_argument("--head-device", default="auto", choices=["auto","cpu","same"], help="Where to run the lm_head projection: auto chooses CPU if GPU free VRAM is low")

    # Selection & context
    p.add_argument("--of-sources", choices=["question", "zoom"], default="question")
    p.add_argument("--zoom-symbol", default=None)
    p.add_argument("--zoom-radius", type=int, default=0)
    p.add_argument("--pack-context", action="store_true")
    p.add_argument("--pack-mode", choices=["heads", "windows"], default="heads")
    p.add_argument("--context-tokens", type=int, default=2000)
    p.add_argument("--require-citations", action="store_true")
    p.add_argument("--citations-per-paragraph", action="store_true")
    p.add_argument("--ignore", action="append", default=None, help="Relative paths to ignore (repeatable), e.g., cloned_repos")
    # Function-first packaging (model-scored)
    p.add_argument("--function-first", action="store_true", help="Use model-scored function windows first, then fill budget")
    p.add_argument("--ff-max-candidates", type=int, default=24, help="Max function windows to score per selection")
    p.add_argument("--ff-window-lines", type=int, default=80, help="Lines per function window (centered on def)")
    p.add_argument("--ff-threshold", type=float, default=0.55, help="Min relevance prob to satisfy definition guarantee")
    p.add_argument("--ff-noise-penalty", type=float, default=0.30, help="Scale of noise prob to subtract from relevance prob")

    # Sampling controls
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--min-new-tokens", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--kv-window", type=int, default=0, help="If >0, prune past_key_values to this window length per step")

    # Telemetry & debug
    p.add_argument("--telemetry-out", default=None, help="Path to write selection and generation metadata JSON")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default=None, help="Cache directory for HF models/tokenizers (defaults to <repo>/checkpoints)")
    p.add_argument("--debug-targets", action="store_true", help="Print discovered target names, shapes, and layer count")
    
    # Interpretability
    p.add_argument("--interpret", action="store_true", help="Capture per-layer outputs and logit-lens top-k (baseline vs adapted)")
    p.add_argument("--interpret-out", default=None, help="Path to write interpret JSON (defaults under artifacts/viz)")
    p.add_argument("--interpret-topk", type=int, default=10)
    p.add_argument("--interpret-tokens", type=int, default=512, help="Max tokens for interpret forward (uses tail of prompt)")
    p.add_argument("--interpret-layer-stride", type=int, default=1, help="Capture every Nth layer to reduce memory")
    p.add_argument("--no-adapters", action="store_true", help="Disable applying adapters; run baseline model only")
    args = p.parse_args()
    # Ensure backend attribute exists (default to local-only backend)
    if not hasattr(args, "backend"):
        setattr(args, "backend", "local")

    # Populate global ignore list for CodeGraph loads in helper fns
    global _IGN
    _IGN = [s for s in (args.ignore or []) if s]

    # Constrain visible GPUs early if requested; reduce fragmentation
    if args.gpu_ids and str(args.gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Parse target weights
    target_weights = otf._parse_target_weights(args.target_weights)  # type: ignore

    # Seeding
    try:
        if args.seed is not None:
            import random
            random.seed(int(args.seed))
            np.random.seed(int(args.seed))
            torch.manual_seed(int(args.seed))
    except Exception:
        pass

    # Resolve cache dir
    cache_dir = args.cache_dir or os.path.join(args.repo, "checkpoints")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

    # Tokenizer (HF ok for text)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=cache_dir)
    torch_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32

    # Ensure local snapshot of the model for weights/config
    def _ensure_snapshot(model_id: str, cache_dir: str) -> str:
        # 1) If user passed a local dir with config.json, use it directly
        if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
            return model_id
        # 2) Try to find an existing snapshot under cache_dir
        try:
            org_name = model_id.strip().split("/")[-2:]
            if len(org_name) == 2:
                org, name = org_name
                # HF cache format: models--ORG--NAME/snapshots/<rev>
                dir1 = os.path.join(cache_dir, f"models--{org}--{name}", "snapshots")
                # Some caches might have slightly different name casing; scan all models--* folders
                candidates: list[str] = []
                if os.path.isdir(dir1):
                    candidates.extend([os.path.join(dir1, d) for d in os.listdir(dir1)])
                else:
                    root = os.path.join(cache_dir)
                    for d in os.listdir(root):
                        if not d.startswith("models--"):
                            continue
                        snap = os.path.join(root, d, "snapshots")
                        if os.path.isdir(snap):
                            for sd in os.listdir(snap):
                                candidates.append(os.path.join(snap, sd))
                # Prefer newest by mtime
                candidates = [p for p in candidates if os.path.isfile(os.path.join(p, "config.json"))]
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    return candidates[0]
        except Exception:
            pass
        # 3) Download if nothing found locally
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required to download the model snapshot")
        return snapshot_download(repo_id=model_id, cache_dir=cache_dir)

    ckpt_dir = _ensure_snapshot(args.model, cache_dir)

    # Build local CausalLM from our stack using config.json dims
    from specs.config import ModelConfig
    from model.factory import build_causal_lm
    from model.hf_llama_loader import load_hf_llama_weights_into_local

    try:
        cfg_obj = json.load(open(os.path.join(ckpt_dir, "config.json"), "r", encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read config.json from {ckpt_dir}") from e
    d_model = int(cfg_obj.get("hidden_size"))
    n_layers = int(cfg_obj.get("num_hidden_layers"))
    n_heads = int(cfg_obj.get("num_attention_heads"))
    d_ff = int(cfg_obj.get("intermediate_size"))
    vocab_size = int(cfg_obj.get("vocab_size"))
    head_dim = int(cfg_obj.get("head_dim", d_model // n_heads))
    n_kv_heads = int(cfg_obj.get("num_key_value_heads", n_heads))
    # Prefer rope_parameters.rope_theta when present (HF convention), else rope_theta
    rope_theta = float(cfg_obj.get("rope_theta", 1e6))
    try:
        rp = cfg_obj.get("rope_parameters", None)
        if isinstance(rp, dict) and rp.get("rope_theta") is not None:
            rope_theta = float(rp.get("rope_theta"))
    except Exception:
        pass
    # RMSNorm epsilon
    rms_eps = float(cfg_obj.get("rms_norm_eps", 1e-6))
    # Parse rope_scaling dict when present
    rs_type = None
    rs_factor = None
    rs_orig = None
    rs_low = None
    rs_high = None
    try:
        rope_scaling = cfg_obj.get("rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rs_type = rope_scaling.get("type")
            if rope_scaling.get("factor") is not None:
                rs_factor = float(rope_scaling.get("factor"))
            if rope_scaling.get("original_max_position_embeddings") is not None:
                rs_orig = int(rope_scaling.get("original_max_position_embeddings"))
            if rope_scaling.get("low_freq_factor") is not None:
                rs_low = float(rope_scaling.get("low_freq_factor"))
            if rope_scaling.get("high_freq_factor") is not None:
                rs_high = float(rope_scaling.get("high_freq_factor"))
    except Exception:
        pass
    mc = ModelConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        head_dim=head_dim,
        rope_theta=rope_theta,
        dtype=("bfloat16" if torch_dtype == torch.bfloat16 else "float32"),
        attn_impl="sdpa",
        rms_norm_eps=rms_eps,
        rope_scaling_type=rs_type,
        rope_scaling_factor=rs_factor,
        rope_scaling_original_max_position_embeddings=rs_orig,
        rope_scaling_low_freq_factor=rs_low,
        rope_scaling_high_freq_factor=rs_high,
    )
    # Respect HF config for tied embeddings/head
    tie_we = bool(cfg_obj.get("tie_word_embeddings", True))
    model = build_causal_lm(mc, block="llama", n_kv_heads=n_kv_heads, tie_weights=tie_we)
    # Pre-cast module dtype so the loader copies weights in the desired dtype (reduces CPU memory)
    try:
        model = model.to(dtype=torch_dtype)
    except Exception:
        pass
    load_hf_llama_weights_into_local(model, ckpt_dir)
    model = model.to(args.device).eval()

    # KV cache toggle (no-op for local unless we add cache implementation flags)

    # Load base adapters
    base = load_adapters_npz(args.adapters)["layers"]

    # Select modules/files
    if args.of_sources == "zoom" and args.zoom_symbol:
        seeds = [s.strip() for s in str(args.zoom_symbol).split(",") if s.strip()]
        modules, files = otf._modules_from_symbols(args.repo, seeds, radius=int(args.zoom_radius), top_k=8)  # type: ignore
        if not modules and not files:
            modules, files = otf._question_aware_modules_and_files(args.repo, args.prompt, top_k=8)  # type: ignore
    else:
        modules, files = otf._question_aware_modules_and_files(args.repo, args.prompt, top_k=8)  # type: ignore

    if args.dry_run:
        print("[dry-run] modules:", modules)
        print("[dry-run] files:", files)
        return

    # Build on-the-fly subgraph adapter
    g = CodeGraph.load_or_build(args.repo, ignore=[s for s in (_IGN or []) if s])
    abs_files = [f if os.path.isabs(f) else os.path.abspath(os.path.join(g.root, f)) for f in files]
    sub_z = build_subgraph_embedding_from_graph(
        g,
        dim=1536,
        seed=int(args.seed) + 1,
        include_modules=modules,
        include_files=abs_files,
        include_text=True,
        text_max_bytes=250000,
        text_weight=0.25,
    )
    # Ensure target map matches backend
    if args.backend == "local":
        local_map = {
            "q_proj": "attn.w_q",
            "k_proj": "attn.w_k",
            "v_proj": "attn.w_v",
            "o_proj": "attn.w_o",
            "up_proj": "mlp.w_in",
            "gate_proj": "mlp.w_in",
            "down_proj": "mlp.w_out",
        }
        setattr(otf, "_targets_map", lambda: local_map)
    target_shapes = otf._infer_target_shapes(model)  # type: ignore
    num_layers = (len(model.blocks) if args.backend == "local" else len(model.model.layers))
    if args.debug_targets:
        try:
            print(json.dumps({
                "layers": int(num_layers),
                "targets": sorted(list(otf._targets_map().keys())),  # type: ignore
                "shapes": {k: list(map(int, v)) for k, v in (target_shapes or {}).items()},
            }, indent=2))
        except Exception:
            pass
    if args.backend == "local":
        d_model = int(getattr(getattr(model, "cfg", None), "d_model", target_shapes.get("q_proj", (0, 0))[0]))
    else:
        d_model = int(getattr(model.config, "hidden_size", target_shapes.get("q_proj", (0, 0))[0]))
    sub = generate_lora_from_embedding(
        sub_z["z"],
        d_model=d_model,
        num_layers=num_layers,
        rank=int(args.rank),
        seed=int(args.seed) + 2,
        targets=list(otf._targets_map().keys()),  # type: ignore
        target_shapes=target_shapes,
    )

    # Defer registering mixed adapters until after optional interpret run
    hooks = []

    # Build final prompt with optional context
    final_prompt = args.prompt
    packed = ""
    if args.pack_context and files:
        if args.pack_mode == "windows":
            if args.function_first:
                # Function-first with model scoring
                def _collect_function_windows(repo_root: str, files_: List[str], lines_each: int) -> List[Tuple[str, int, int, int, List[str]]]:
                    g = CodeGraph.load_or_build(repo_root)
                    out: List[Tuple[str, int, int, int, List[str]]] = []
                    for rel in files_:
                        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(g.root, rel))
                        try:
                            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
                        except Exception:
                            continue
                        # collect def/class anchors
                        anchors: List[int] = []
                        for i, line in enumerate(src_lines, start=1):
                            s = line.lstrip()
                            if s.startswith("def ") or s.startswith("class "):
                                anchors.append(i)
                                if len(anchors) >= max(4, args.ff_max_candidates // max(1, len(files_))):
                                    break
                        if not anchors:
                            continue
                        half = max(10, int(lines_each // 2))
                        for ln in anchors:
                            a = max(1, ln - half)
                            b = min(len(src_lines), ln + half)
                            out.append((rel, a, b, ln, src_lines[a - 1 : b]))
                    return out

                def _model_prob_yes(prompt_q: str, window_txt: str) -> Tuple[float, float]:
                    # Return (relevance_prob, noise_prob) using next-token logits on a binary question
                    def _score_yes_no(q: str) -> float:
                        q_ids = tok(q, return_tensors="pt")
                        # Move to first shard/device
                        dev = next(model.parameters()).device
                        q_ids = {k: v.to(dev) for k, v in q_ids.items()}
                        with torch.no_grad():
                            out_lm = model(**q_ids)
                            logits = out_lm.logits  # [1, T, V]
                            last = logits[:, -1, :]
                            # Token ids for '1' vs '0'
                            t1 = tok("1", add_special_tokens=False).input_ids
                            t0 = tok("0", add_special_tokens=False).input_ids
                            probs = torch.softmax(last, dim=-1)
                            p1 = float(sum(probs[0, i].item() for i in (t1 or [])))
                            p0 = float(sum(probs[0, i].item() for i in (t0 or [])))
                            denom = max(1e-9, p1 + p0)
                            return float(p1 / denom)
                    rel_q = (
                        "Question: "
                        + prompt_q
                        + "\nWindow:\n"
                        + window_txt[:1800]
                        + "\nDoes this window contain the core function or logic to answer the question? Answer 1 or 0."
                    )
                    noise_q = (
                        "Question: "
                        + prompt_q
                        + "\nWindow:\n"
                        + window_txt[:1800]
                        + "\nIs this window likely test/tool/noise unrelated to answering the question? Answer 1 or 0."
                    )
                    return _score_yes_no(rel_q), _score_yes_no(noise_q)

                def _extract_func_name_from_lines(lines_block: List[str], a: int, b: int, anchor_ln: int) -> Optional[str]:
                    try:
                        best_name = None
                        best_dist = 10**9
                        abs_ln = a
                        for ln_text in lines_block:
                            s = ln_text.lstrip()
                            if s.startswith("def ") or s.startswith("class "):
                                m = re.match(r"^(?:def|class)\s+([A-Za-z0-9_]+)", s)
                                if m:
                                    name = m.group(1)
                                    dist = abs(abs_ln - anchor_ln)
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_name = name
                            abs_ln += 1
                        return best_name
                    except Exception:
                        return None

                def _prompt_token_set(prompt_q: str) -> set:
                    try:
                        toks = re.findall(r"[A-Za-z0-9_]+", (prompt_q or "").lower())
                        return set(toks)
                    except Exception:
                        return set()

                def _name_matches_prompt(name: Optional[str], prompt_tokens: set) -> bool:
                    if not name:
                        return False
                    n = name.lower()
                    # quick substring check
                    if n in " ".join(sorted(prompt_tokens)):
                        return True
                    parts = [p for p in n.replace("_", " ").split() if p]
                    if not parts:
                        return False
                    inter = sum(1 for p in parts if p in prompt_tokens)
                    # require at least 2 tokens match for multi-token names, else 1
                    need = 2 if len(parts) >= 2 else 1
                    # also accept if concatenated form appears
                    cat = "".join(parts)
                    return bool(inter >= need or cat in prompt_tokens)

                # 1) collect function windows
                fn_windows = _collect_function_windows(args.repo, files, int(args.ff_window_lines))
                # limit candidates
                fn_windows = fn_windows[: int(args.ff_max_candidates)]
                # 2) score with model
                scored: List[Tuple[float, float, bool, str, int, int, int, List[str], Optional[str]]] = []
                p_tokens = _prompt_token_set(args.prompt)
                for (rel, a, b, anchor_ln, lines_block) in fn_windows:
                    txt_block = "\n".join(lines_block)
                    rel_p, noise_p = _model_prob_yes(args.prompt, txt_block)
                    score = float(max(0.0, min(1.0, rel_p - float(args.ff_noise_penalty) * noise_p)))
                    fname = _extract_func_name_from_lines(lines_block, a, b, anchor_ln)
                    name_match = _name_matches_prompt(fname, p_tokens)
                    # Name-match boost to prefer obviously relevant defs (docs roadmap: name-aware retrieval)
                    boosted = float(min(1.0, score + (0.35 if name_match else 0.0)))
                    scored.append((boosted, score, name_match, rel, a, b, anchor_ln, lines_block, fname))
                scored.sort(key=lambda x: x[0], reverse=True)
                # 3) pack top-scored windows first
                g = CodeGraph.load_or_build(args.repo)
                out_lines: List[str] = ["Repository windows (function-first):"]
                used = 0
                budget = int(args.context_tokens)
                guarantee_ok = False
                # 3a) guarantee: include up to one name-matched window per file first
                included_keys = set()
                seen_file: Dict[str, bool] = {}
                for (boosted, base_score, name_match, rel, a, b, anchor_ln, lines_block, fname) in scored:
                    if not name_match:
                        continue
                    if seen_file.get(rel):
                        continue
                    block = [f"[ctx] path: {rel}:{a}-{b}"] + lines_block + [""]
                    text_block = "\n".join(block) + "\n"
                    t = len(tok(text_block).input_ids)
                    if used + t > budget:
                        continue
                    out_lines.extend(block)
                    used += t
                    included_keys.add((rel, a, b))
                    seen_file[rel] = True
                    if base_score >= float(args.ff_threshold):
                        guarantee_ok = True
                    if used >= int(0.8 * budget):
                        break
                # 3b) fill remaining with highest-scored windows, skipping already included
                for (boosted, base_score, name_match, rel, a, b, anchor_ln, lines_block, fname) in scored:
                    if (rel, a, b) in included_keys:
                        continue
                    block = [f"[ctx] path: {rel}:{a}-{b}"] + lines_block + [""]
                    text_block = "\n".join(block) + "\n"
                    t = len(tok(text_block).input_ids)
                    if used + t > budget:
                        continue
                    out_lines.extend(block)
                    used += t
                    if base_score >= float(args.ff_threshold):
                        guarantee_ok = True
                    if used >= int(0.8 * budget):
                        break
                # 4) fill remainder with auxiliary windows if room remains
                if used < budget:
                    aux = _pack_context_windows(args.repo, files, tok, budget - used)
                    if aux:
                        out_lines.extend(aux.splitlines())
                packed = "\n".join(out_lines) if len(out_lines) > 1 else ""
                if args.verbose:
                    print(f"[debug] function-first guarantee_ok={guarantee_ok} used_tokens={used}")
            else:
                packed = _pack_context_windows(args.repo, files, tok, int(args.context_tokens))
        else:
            packed = _pack_context_heads(args.repo, files, tok, int(args.context_tokens))
        if packed:
            final_prompt = packed + "\n\n" + args.prompt
    if args.require_citations:
        final_prompt = (
            final_prompt
            + "\n\nInstruction: For EVERY claim, append a citation of the form modules/FILE.py:START-END.\n"
              "Use only files shown in [ctx] above. Provide at least 3 citations overall.\n"
              "Example: 'Backpropagation step updates parameters [modules/train.py:210-245]'."
        )

    if args.verbose:
        try:
            prompt_tokens = len(tok(final_prompt, return_tensors="pt").input_ids[0])
        except Exception:
            prompt_tokens = -1
        print(f"[debug] modules={modules} files={files}")
        print(f"[debug] prompt_tokens={prompt_tokens}")

    x = tok(final_prompt, return_tensors="pt")
    if args.device_map == "auto":
        try:
            dev = otf._getattr_nested(model.model.layers[0], "self_attn.q_proj.weight").device  # type: ignore
        except Exception:
            dev = next(model.parameters()).device
        x = {k: v.to(dev) for k, v in x.items()}
    else:
        x = x.to(args.device)

    # Optional: interpretability capture (baseline vs adapted)
    if args.interpret:
        if args.verbose:
            print("[debug] interpret: baseline capturing block outputs...")
        # Minimize peak GPU memory during capture: avoid cloning on device and keep dtype
        spec = CaptureSpec(move_to_cpu=True, dtype=None, detach=True, clone=False)
        tracer_b = ActivationTracer(model, spec=spec)
        # Hook to coerce decoder layer outputs (tuple/ModelOutput) to hidden-state tensor
        def _block_out_hook(_key: str, _module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: Any) -> Optional[torch.Tensor]:
            try:
                if isinstance(output, torch.Tensor):
                    return output
                # HF decoder layers typically return tuple(hidden_states, ...)
                if isinstance(output, (tuple, list)) and output:
                    v = output[0]
                    return v if isinstance(v, torch.Tensor) else None
                # ModelOutput-like
                v = getattr(output, "hidden_states", None)
                if isinstance(v, torch.Tensor):
                    return v
                v = getattr(output, "last_hidden_state", None)
                if isinstance(v, torch.Tensor):
                    return v
            except Exception:
                return None
            return None
        def _is_block(name: str, _m: torch.nn.Module) -> bool:
            if not name.startswith("model.layers."):
                return False
            rest = name[len("model.layers."):]
            return rest.isdigit() and "." not in rest
        tracer_b.add_modules_matching(_is_block, hook=_block_out_hook)
        # Build a truncated input for interpret pass (tail tokens)
        try:
            max_t = max(8, int(args.interpret_tokens))
        except Exception:
            max_t = 512
        def _truncate_batch(xx: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            out = {}
            for k, v in xx.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) > max_t:
                    out[k] = v[:, -max_t:]
                else:
                    out[k] = v
            return out
        x_short = _truncate_batch(x)
        with tracer_b.trace():
            _ = model(**x_short, use_cache=False, output_hidden_states=False, return_dict=True)
            cache_b = dict(tracer_b._cache.items())

        # Apply adapters temporarily
        hooks_tmp = []
        if not bool(args.no_adapters):
            hooks_tmp = otf._register_hook_mixed_adapters(  # type: ignore
                model,
                base,
                sub.get("layers"),
                alpha_star=float(args.alpha),
                g_sub=float(args.gsub),
                rank=int(args.rank),
                beta=float(args.mix_beta),
                target_weights=target_weights,
            )
        if args.verbose:
            print("[debug] interpret: adapted capturing block outputs...")
        tracer_a = ActivationTracer(model, spec=spec)
        tracer_a.add_modules_matching(_is_block, hook=_block_out_hook)
        with tracer_a.trace():
            out_a = model(**x_short, use_cache=False, output_hidden_states=False, return_dict=True)
            cache_a = dict(tracer_a._cache.items())
        # Remove temporary deltas
        for h in hooks_tmp:
            try:
                h.remove()
            except Exception:
                pass

        # Helper: get projection matrix W
        def _get_W(m) -> torch.Tensor:
            if hasattr(m, "lm_head") and hasattr(m.lm_head, "weight"):
                return m.lm_head.weight
            if hasattr(m, "get_output_embeddings"):
                we = m.get_output_embeddings()
                if we is not None and hasattr(we, "weight"):
                    return we.weight
            raise AttributeError("No output projection found (lm_head or embeddings)")

        W = _get_W(model).to(dtype=torch.float32, device="cpu")
        topk = int(args.interpret_topk)
        # Compute per-layer summaries
        layers_sorted = sorted(
            [int(k.split(".")[2]) for k in cache_b.keys() if k.startswith("model.layers.") and k.count(".") == 2]
        )
        stride = max(1, int(args.interpret_layer_stride))
        layers_sorted = [i for i in layers_sorted if (i % stride) == 0]
        lens: Dict[int, Dict[str, List[float]]] = {}
        deltas: Dict[int, float] = {}
        for i in layers_sorted:
            key = f"model.layers.{i}"
            hb = cache_b.get(key)
            ha = cache_a.get(key)
            if hb is None or ha is None:
                continue
            try:
                # Hidden delta mean L2
                d = (ha - hb).float()
                deltas[i] = float(d.pow(2).sum(dim=-1).sqrt().mean().item())
            except Exception:
                deltas[i] = float("nan")
            try:
                # Logit lens on adapted: last token projection
                h_pos = ha[:, -1, :].to(dtype=torch.float32)
                logits = torch.matmul(h_pos, W.t())  # [B,V]
                vals, idx = torch.topk(logits, k=min(topk, logits.shape[-1]), dim=-1)
                lens[i] = {
                    "ids": [int(x) for x in idx[0].tolist()],
                    "scores": [float(x) for x in vals[0].tolist()],
                }
            except Exception:
                continue

        # Delta logits L2 on last token
        try:
            lb = (torch.matmul(cache_b[f"model.layers.{layers_sorted[-1]}"][:, -1, :].to(torch.float32), W.t()))
            la = (torch.matmul(cache_a[f"model.layers.{layers_sorted[-1]}"][:, -1, :].to(torch.float32), W.t()))
            delta_logits_l2 = float((la - lb).norm(p=2, dim=-1).mean().item())
        except Exception:
            delta_logits_l2 = None

        # Write JSON
        interpret_obj = {
            "alpha": float(args.alpha),
            "rank": int(args.rank),
            "gsub": float(args.gsub),
            "layers": layers_sorted,
            "hidden_delta_mean_l2": {int(k): float(v) for k, v in deltas.items()},
            "logit_lens": {int(k): v for k, v in lens.items()},
            "delta_logits_l2": delta_logits_l2,
        }
        out_path = (
            args.interpret_out
            or (args.telemetry_out and os.path.join(os.path.dirname(os.path.abspath(args.telemetry_out)), "interpret.json"))
            or os.path.join(os.path.dirname(os.path.abspath(args.adapters)), "..", "viz", "interpret.json")
        )
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(interpret_obj, fh, indent=2)
            if args.verbose:
                print(f"[debug] interpret written: {out_path}")
        except Exception as e:
            if args.verbose:
                print(f"[warn] failed to write interpret: {e}")

    # Now register mixed hooks for generation
    hooks = []
    if not bool(args.no_adapters):
        hooks = otf._register_hook_mixed_adapters(  # type: ignore
            model,
            base,
            sub.get("layers"),
            alpha_star=float(args.alpha),
            g_sub=float(args.gsub),
            rank=int(args.rank),
            beta=float(args.mix_beta),
            target_weights=target_weights,
        )

    # For local backend, prefer built-in generate to avoid HF-only cache paths
    try:
        args.kv_window = 0
        args.head_device = "same"
    except Exception:
        pass
    # Generation: optional KV-pruned manual loop (disabled for local backend)
    if False and (int(args.kv_window) > 0 or args.head_device != "same"):
        if args.verbose:
            print(f"[debug] generating with kv_window={int(args.kv_window)}...")
        max_new = int(args.max_new_tokens)
        do_sample = bool(args.do_sample)
        temperature = float(args.temperature)
        top_p = float(args.top_p)
        rep_pen = float(args.repetition_penalty)

        # Decide head device and prepare projection weight once
        head_mode = str(args.head_device)
        head_use_cpu = False
        # Inspect current lm_head device if available
        try:
            lm_head = getattr(model, "lm_head", None)
            head_param = next(lm_head.parameters()).device if (lm_head is not None) else next(model.parameters()).device
        except Exception:
            head_param = next(model.parameters()).device
        if head_mode == "cpu":
            head_use_cpu = True
        elif head_mode == "auto":
            try:
                if head_param.type == "cuda":
                    free, total = torch.cuda.mem_get_info(head_param)
                    # Require a modest safety margin for logits matmul
                    head_use_cpu = bool(free < (256 * 1024 * 1024))
                else:
                    head_use_cpu = True
            except Exception:
                head_use_cpu = True
        else:
            head_use_cpu = False

        # Prepare W^T on target device
        def _prepare_head_weight() -> Tuple[torch.Tensor, torch.device]:
            we = model.get_output_embeddings()
            W = we.weight
            if head_use_cpu:
                Wt = W.detach().to(device=torch.device("cpu"), dtype=torch.float32).t().contiguous()
                return Wt, torch.device("cpu")
            # use same device as hidden state; keep dtype
            dev = next(model.parameters()).device
            Wt = W.detach().to(device=dev).t().contiguous()
            return Wt, dev

        Wt, head_dev = _prepare_head_weight()

        def _top_p_sample(logits: torch.Tensor, top_p_val: float, temp: float) -> torch.Tensor:
            if temp and temp > 0:
                logits = logits / float(temp)
            probs = torch.softmax(logits, dim=-1)
            if 0.0 < top_p_val < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > top_p_val
                # keep at least one
                mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            return torch.multinomial(probs, num_samples=1)

        # Prepare initial inputs
        input_ids = x["input_ids"] if isinstance(x, dict) else x.input_ids
        attention_mask = x.get("attention_mask") if isinstance(x, dict) else getattr(x, "attention_mask", None)
        seq = input_ids
        # Prepare EOS set (can be int or list)
        eos_ids = getattr(model.config, "eos_token_id", None)
        if isinstance(eos_ids, (list, tuple)):
            eos_set = {int(x) for x in eos_ids if x is not None}
        elif eos_ids is not None:
            eos_set = {int(eos_ids)}
        else:
            eos_set = set()

        for _ in range(max_new):
            with torch.no_grad():
                # Evaluate on the last kv_window tokens to bound compute/memory
                win = int(args.kv_window)
                ctx = seq[:, -win:] if win > 0 else seq[:, -1:]
                # Run core model to get hidden states (bypass HF head)
                core = model.model(input_ids=ctx, attention_mask=None, use_cache=False, return_dict=True)
                h_last = core.last_hidden_state[:, -1, :]
                if head_use_cpu and h_last.device.type != "cpu":
                    h_proj = h_last.to(device=head_dev, dtype=Wt.dtype)
                else:
                    # ensure on same device/dtype as Wt
                    h_proj = h_last.to(device=head_dev, dtype=Wt.dtype)
                logits = torch.matmul(h_proj, Wt)
            # repetition penalty (simple)
            if rep_pen != 1.0:
                gather_ids = seq
                logits = logits.scatter_add(1, gather_ids, torch.full_like(gather_ids, -abs(rep_pen - 1.0), dtype=logits.dtype))
            if do_sample:
                next_id = _top_p_sample(logits, top_p, temperature)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
            # No explicit HF cache; sliding window enforced via ctx slicing above
            # Early stop
            if eos_set:
                try:
                    eos_t = torch.tensor(sorted(list(eos_set)), device=next_id.device, dtype=next_id.dtype)
                    # next_id: (B,1); check if each batch element is in eos set
                    if torch.isin(next_id.view(-1), eos_t).all():
                        break
                except Exception:
                    # Fallback: Python-side check
                    if all(int(x) in eos_set for x in next_id.view(-1).tolist()):
                        break
        out = seq
    else:
        gen_kwargs = {
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
        }
        if args.do_sample:
            gen_kwargs.update({
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
                "min_new_tokens": int(max(0, int(args.min_new_tokens))),
            })
        if args.verbose:
            print("[debug] generating...")
        try:
            out = model.generate(**x, **gen_kwargs)
        except RuntimeError as e:
            msg = str(e)
            if ("CUBLAS_STATUS_ALLOC_FAILED" in msg) or ("CUDA out of memory" in msg):
                if args.verbose:
                    print("[warn] CUDA allocation failed; retrying on CPU...")
                try:
                    # Best-effort to free some GPU memory
                    import torch as _torch  # local alias
                    if _torch.cuda.is_available():
                        try:
                            _torch.cuda.empty_cache()
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    model = model.to(device="cpu", dtype=torch.float32)
                except Exception:
                    model = model.to(device="cpu")
                try:
                    x = {k: v.to("cpu") for k, v in x.items()} if isinstance(x, dict) else x.to("cpu")
                except Exception:
                    pass
                out = model.generate(**x, **gen_kwargs)
            else:
                raise
    if args.verbose:
        print("[debug] decode...")
    # Handle local generate return structure
    try:
        out_ids = out[0] if hasattr(out, "__getitem__") else out
        if isinstance(out, dict) and "sequences" in out:
            out_ids = out["sequences"]
        # Decode only the newly generated tokens (exclude prompt/context)
        try:
            inp_len = int(x["input_ids"].shape[1]) if isinstance(x, dict) else int(x.input_ids.shape[1])
        except Exception:
            inp_len = 0
        seq_ids = out_ids[0]
        new_ids = seq_ids[inp_len:] if (hasattr(seq_ids, "shape") and seq_ids.shape[0] > inp_len) else seq_ids
        text = tok.decode(new_ids, skip_special_tokens=True)
    except Exception:
        # Fallback: best-effort decode
        try:
            seq = out.get("sequences") if isinstance(out, dict) else out
            # Best-effort prompt-trim
            inp_len = int(x["input_ids"].shape[1]) if isinstance(x, dict) else int(getattr(x, "input_ids", [0]).shape[1])
            seq_ids = seq[0]
            new_ids = seq_ids[inp_len:] if (hasattr(seq_ids, "shape") and seq_ids.shape[0] > inp_len) else seq_ids
            text = tok.decode(new_ids, skip_special_tokens=True)
        except Exception:
            text = "[error] failed to decode output"

    # Citation checks with one retry strategy
    def _has_citations(s: str, per_para: bool) -> bool:
        rx = re.compile(r"(?:path:\s*)?[A-Za-z0-9_./-]+?\.\w+:\d+(?:-\d+)?")
        if not rx.search(s):
            return False
        if per_para:
            paras = [p.strip() for p in s.split("\n\n") if p.strip()]
            return all(rx.search(p) for p in paras)
        return True

    if args.require_citations and not _has_citations(text, bool(args.citations_per_paragraph)):
        if args.verbose:
            print("[debug] retry without sampling, beam search + stronger scaffold...")
        retry_prompt = (
            final_prompt
            + "\n\nRewrite the answer. For EACH paragraph, end with a citation like [modules/train.py:123-160].\n"
              "Use at least 3 citations overall and only files shown in [ctx]."
        )
        x2 = tok(retry_prompt, return_tensors="pt")
        if args.device_map == "auto":
            try:
                dev = otf._getattr_nested(model.model.layers[0], "self_attn.q_proj.weight").device  # type: ignore
            except Exception:
                dev = next(model.parameters()).device
            x2 = {k: v.to(dev) for k, v in x2.items()}
        else:
            x2 = x2.to(args.device)
        retry_kwargs = {
            "max_new_tokens": int(max(int(args.max_new_tokens), 256)),
            "do_sample": False,
        }
        out2 = model.generate(**x2, **retry_kwargs)
        text2 = tok.decode(out2[0], skip_special_tokens=True)
        if _has_citations(text2, bool(args.citations_per_paragraph)):
            text = text2
        else:
            # Last resort: append a references block from packed files so the user sees anchors
            refs = []
            for rel in files[:4]:
                path = rel if rel.startswith("modules/") else f"modules/{os.path.basename(rel)}"
                refs.append(f"- {path}:1-120")
            if refs:
                text = text2 + "\n\nReferences (from context):\n" + "\n".join(refs)
            else:
                text = text2 or "INSUFFICIENT_EVIDENCE: No path:line citations found per policy."

    # Telemetry
    if args.telemetry_out:
        meta = {
            "modules": modules,
            "files": files,
            "alpha": float(args.alpha),
            "rank": int(args.rank),
            "gsub": float(args.gsub),
            "pack_mode": str(args.pack_mode),
            "context_tokens": int(args.context_tokens),
            "do_sample": bool(args.do_sample),
            "max_new_tokens": int(args.max_new_tokens),
            "require_citations": bool(args.require_citations),
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.telemetry_out)), exist_ok=True)
            with open(args.telemetry_out, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        except Exception:
            pass

    print(text)

    # Cleanup hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass


if __name__ == "__main__":
    main()


