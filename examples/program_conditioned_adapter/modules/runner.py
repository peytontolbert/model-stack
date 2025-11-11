from __future__ import annotations

import os
import json
import time
from typing import Optional, Tuple, List, Dict, Any, Set

import numpy as np
import torch
from data.tokenizer import LocalLlamaTokenizer

from examples.program_conditioned_adapter.modules.adapter import (
    load_adapters_npz,
    generate_lora_from_embedding,
)
from model.llama_bootstrap import build_local_llama_from_snapshot
from model.hf_snapshot import ensure_snapshot
from blocks.targets import targets_map
from blocks.inspect import infer_target_shapes
from model.inspect import detect_target_names_from_model_full
from examples.program_conditioned_adapter.modules.mixing import (
    register_hook_mixed_adapters,
)
from examples.program_conditioned_adapter.modules.caches import load_manifest, load_windows_index, pick_files_from_windows
from examples.program_conditioned_adapter.modules.citations import CitationManager, CitationPolicy
from examples.program_conditioned_adapter.modules.interpret import (
    is_block,
    block_out_hook,
    truncate_batch,
    get_W,
)
from examples.program_conditioned_adapter.modules.program_state import (
    load_program_state,
    save_program_state,
    join_program_states,
    new_state_from_run,
    changed_bits,
)

def _normalize_citations_regex():
    try:
        import re
        return re.compile(r"(?:path:\s*)?([A-Za-z0-9_./\\\-]+?\.\w+):(\d+)(?:-(\d+))?")
    except Exception:
        return None

def normalize_citations(text: str) -> List[tuple[str, int, int]]:
    rx = _normalize_citations_regex()
    if rx is None:
        return []
    out: List[tuple[str, int, int]] = []
    try:
        import os as _os
        for m in rx.finditer(text or ""):
            p = m.group(1)
            try:
                a = int(m.group(2) or "0")
            except Exception:
                a = 0
            try:
                b = int(m.group(3) or a)
            except Exception:
                b = a
            out.append((p.replace("\\\\", "/"), min(a, b), max(a, b)))
        return out
    except Exception:
        return []

def has_citations(s: str, per_para: bool) -> bool:
    try:
        import re as _re
        rx = _re.compile(r"(?:path:\s*)?[A-Za-z0-9_./\\-]+?\.\w+:\d+(?:-\d+)?")
        if not rx.search(s or ""):
            return False
        if per_para:
            paras = [p.strip() for p in (s or "").split("\n\n") if p.strip()]
            return all(rx.search(p) for p in paras)
        return True
    except Exception:
        return False

def generate_answer(
    model_id: str,
    adapters_npz: str,
    prompt: str,
    *,
    program_root: Optional[str] = None,
    delta_cap: float = 0.05,
    cache_dir: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    device_map: str = "none",
    gpu_ids: Optional[str] = None,
    # selection
    of_sources: str = "question",
    zoom_symbol: Optional[str] = None,
    zoom_radius: int = 0,
    ignore: Optional[List[str]] = None,
    # context
    pack_context: bool = False,
    pack_mode: str = "heads",
    context_tokens: int = 2000,
    function_first: bool = False,
    ff_max_candidates: int = 24,
    ff_window_lines: int = 80,
    ff_threshold: float = 0.55,
    ff_noise_penalty: float = 0.30,
    # capacity/mixing
    alpha: float = 16.0,
    rank: int = 12,
    gsub: float = 0.75,
    beta: float = 0.1,
    entropy_aware: bool = False,
    rank_min: int = 8,
    rank_max: int = 32,
    gsub_min: float = 0.6,
    gsub_max: float = 0.9,
    entropy_weights: str = "program=0.4,subgraph=0.4,question=0.2",
    target_weights: Optional[str] = None,
    layer_schedule: bool = False,
    q_aware_weights: bool = False,
    mixture_m: int = 0,
    adapters_bank: Optional[str] = None,
    per_target_rank_schedule: bool = False,
    rank_budget: int = 0,
    ablate_attn: bool = False,
    ablate_mlp: bool = False,
    alpha_warmup: bool = False,
    adapter_aware_decoding: bool = False,
    layer_rank_tiers: bool = False,
    # cones/rounding
    # cones/rounding
    cone_rank: int = 2,
    cone_weight: float = 0.5,
    round_lora: bool = False,
    round_threshold: float = 0.5,
    # citations
    require_citations: bool = False,
    citations_per_paragraph: bool = False,
    # retrieval/rerank
    rerank: bool = True,
    self_queries_path: Optional[str] = None,
    # generation
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    min_new_tokens: int = 64,
    max_new_tokens: int = 256,
    kv_window: int = 0,
    head_device: str = "same",  # "cpu" | "auto" | "same"
    # misc
    seed: int = 0,
    verbose: bool = False,
    # provenance footer
    commit_footer: bool = False,
    # monotone selection for non-structured path
    monotone_selection: bool = False,
) -> str:
    # Program root (path or URI) for reading artifacts/windows
    root = str(program_root) if (program_root is not None and str(program_root).strip()) else ""
    if gpu_ids and str(gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Seed
    try:
        import random
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
    except Exception:
        pass

    # Prefer explicit cache_dir; else env; else project root (/..../checkpoints)
    if not cache_dir:
        env_cache = os.environ.get("TRANSFORMER_CACHE_DIR") or os.environ.get("HF_HOME")
        if env_cache:
            cache_dir = env_cache
        else:
            mod_dir = os.path.dirname(__file__)
            proj_root = os.path.abspath(os.path.join(mod_dir, "..", "..", ".."))
            cache_dir = os.path.join(proj_root, "checkpoints")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

    # Local tokenizer (no transformers)
    snap_dir = ensure_snapshot(model_id, cache_dir)
    tok_local = LocalLlamaTokenizer(snap_dir)
    class _TokAdapter:
        def __init__(self, base):
            self._b = base
        class _Ret:
            def __init__(self, ids):
                self.input_ids = ids
        def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
            ids = self._b.encode(text)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return _TokAdapter._Ret(ids)
    tok = _TokAdapter(tok_local)
    def _tok_encode(text: str) -> List[int]:
        return tok_local.encode(text)
    def _tok_len(text: str) -> int:
        try:
            return len(tok_local.encode(text))
        except Exception:
            return 0
    def _tok_decode(ids: torch.Tensor | List[int]) -> str:
        if isinstance(ids, torch.Tensor):
            try:
                ids_l = ids.to("cpu").tolist()
            except Exception:
                ids_l = []
        else:
            ids_l = list(ids)
        try:
            return tok_local.decode(ids_l, skip_special_tokens=True)
        except Exception:
            return tok_local.decode(ids_l)
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    model, _cfg = build_local_llama_from_snapshot(snap_dir, device, torch_dtype, device_map=device_map, gpu_ids=gpu_ids)

    # Load base adapters
    base_layers = load_adapters_npz(adapters_npz)["layers"]

    # Selection via caches (program-agnostic): prefer windows with prompt overlap
    files: List[str] = []
    try:
        adapters_dir = os.path.dirname(adapters_npz)
        windows = load_windows_index(adapters_dir)
        files = pick_files_from_windows(root, windows, prompt, k=max(8, int(ff_max_candidates)))
    except Exception:
        files = []
    modules: List[str] = []

    # Subgraph embedding
    # Program-agnostic: skip subgraph embedding if no backend embedder; use base only
    sub = {"layers": []}

    # Shapes and targets
    tmap = targets_map("local")
    t_shapes = infer_target_shapes(model)
    num_layers = len(getattr(model, "blocks", []))
    d_model_local = int(t_shapes.get("q_proj", (0, 0))[0]) or int(getattr(getattr(model, "cfg", None), "d_model", 0) or 0)

    # Entropy-aware capacity
    scaled_rank = int(rank)
    scaled_gsub = float(gsub)
    entropy_diag: Optional[Dict[str, float]] = None

    # No subgraph z available -> keep sub empty (base-only mixing)
    sub = {"layers": []}

    # Function-first cones: build and merge
    cr = max(0, int(cone_rank))
    # Skip function-first cones in program-agnostic core (requires backend embedder)

    # Optional mixture bank: mix top-m module adapters from bank by concatenation (Σ π_i Δθ_i)
    if adapters_bank and int(mixture_m) > 0:
        try:
            import glob
            bank_root = os.path.abspath(os.path.expanduser(os.path.expandvars(adapters_bank)))
            sel_mods = [m for m in (modules or []) if m]
            picked = 0
            for mod in sel_mods:
                if picked >= int(mixture_m):
                    break
                mod_dir1 = os.path.join(bank_root, "sub_adapters", mod.replace("/", "_"))
                mod_dir2 = os.path.join(bank_root, mod.replace("/", "_"))
                cand = None
                for d in (mod_dir1, mod_dir2):
                    fp = os.path.join(d, "adapters.npz")
                    if os.path.isfile(fp):
                        cand = fp
                        break
                if not cand:
                    continue
                try:
                    bank_ad = load_adapters_npz(cand)
                except Exception:
                    continue
                # uniform π for now
                w = 1.0 / float(min(len(sel_mods), int(mixture_m)))
                sw = float(max(0.0, min(1.0, w))) ** 0.5
                # per-layer concat
                merged_layers = []
                for i in range(num_layers):
                    baseL = sub["layers"][i]
                    bL = bank_ad["layers"][i] if i < len(bank_ad.get("layers", [])) else {}
                    dst: Dict[str, Dict[str, np.ndarray]] = {}
                    for name in tmap.keys():
                        if (name in baseL) and (name in bL):
                            A1 = baseL[name]["A"]; B1 = baseL[name]["B"]
                            A2 = (sw * bL[name]["A"]).astype(np.float32); B2 = (sw * bL[name]["B"]).astype(np.float32)
                            A = np.concatenate([A1, A2], axis=1)
                            B = np.concatenate([B1, B2], axis=0)
                            dst[name] = {"A": A, "B": B, "gate": baseL[name].get("gate", np.array([0.0], dtype=np.float32))}
                        elif name in baseL:
                            dst[name] = baseL[name]
                        elif name in bL:
                            A = (sw * bL[name]["A"]).astype(np.float32)
                            dst[name] = {"A": A, "B": bL[name]["B"], "gate": bL[name].get("gate", np.array([0.0], dtype=np.float32))}
                    merged_layers.append(dst)
                sub = {"layers": merged_layers}
                # try infer rank increment from a present target
                try:
                    for i in range(num_layers):
                        any_name = next((n for n in tmap.keys() if n in bank_ad["layers"][i]), None)
                        if any_name:
                            inc = int(bank_ad["layers"][i][any_name]["B"].shape[0])
                            scaled_rank = int(scaled_rank + inc)
                            break
                except Exception:
                    pass
                picked += 1
        except Exception:
            pass

    # Optional rounding of LoRA factors
    if bool(round_lora):
        try:
            thr = float(max(0.0, round_threshold))
            for i in range(len(sub.get("layers", []))):
                for name, tensors in sub["layers"][i].items():
                    for key in ("A", "B"):
                        arr = tensors.get(key)
                        if not isinstance(arr, np.ndarray):
                            continue
                        q = float(np.median(np.abs(arr))) if arr.size > 0 else 0.0
                        if q <= 0:
                            continue
                        out = np.where(np.abs(arr) < (thr * q), 0.0, np.sign(arr) * q).astype(np.float32)
                        tensors[key] = out
        except Exception:
            pass

    # Final prompt with optional context
    final_prompt = prompt
    if bool(pack_context) and files:
        adapters_dir = os.path.dirname(adapters_npz)
        mf = load_manifest(adapters_dir)
        # Simple windows-based packing: use windows_index ranges, read from program root
        try:
            windows = load_windows_index(adapters_dir)
        except Exception:
            windows = []
        lines_out: List[str] = ["Program windows:"]
        used = 0
        budget = int(context_tokens)
        # Choose up to one top window per selected file
        for rel in files:
            rel_n = rel.replace("\\", "/")
            cand = [w for w in windows if (w.get("path") or "").replace("\\", "/") == rel_n]
            if not cand:
                continue
            w = cand[0]
            a = int(w.get("start_line", 1))
            b = int(w.get("end_line", max(a, a + 60)))
            abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(root, rel))
            try:
                src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
            except Exception:
                continue
            a = max(1, min(a, len(src_lines)))
            b = max(a, min(b, len(src_lines)))
            block = [f"[ctx] path: {rel}:{a}-{b}"] + src_lines[a - 1 : b] + [""]
            text_block = "\n".join(block) + "\n"
            t = _tok_len(text_block)
            if used + t > budget:
                continue
            lines_out.extend(block)
            used += t
            if used >= budget:
                break
        packed = "\n".join(lines_out) if len(lines_out) > 1 else ""
        if packed:
            final_prompt = packed + "\n\n" + final_prompt
    if require_citations:
        # Tailor the example path to the first selected file for better compliance
        try:
            example_rel = files[0] if files else None
            example_path = example_rel if example_rel else "file.py"
        except Exception:
            example_path = "file.py"
        final_prompt = (
            final_prompt
            + f"\n\nInstruction: For EVERY claim, append a citation of the form {example_path}:START-END.\n"
              "Use only files shown in [ctx] above. Provide at least 3 citations overall.\n"
        )

    # Adapter-aware decoding: pointer-first nudge when citations are required
    if bool(adapter_aware_decoding) and bool(require_citations):
        try:
            example_rel = files[0] if files else None
            example_path = example_rel if example_rel else "file.py"
        except Exception:
            example_path = "file.py"
        final_prompt = (
            f"[pointer-first] Start with a citation like [{example_path}:A-B], then explain.\n\n"
            + final_prompt
        )
    ids = torch.tensor([_tok_encode(final_prompt)], dtype=torch.long, device=device)
    x = {"input_ids": ids}

    # Apply mixed adapters
    hooks = []
    def _parse_target_weights(spec: Optional[str]) -> Optional[Dict[str, float]]:
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
    tw = _parse_target_weights(target_weights) or {}
    # Optional question-aware reweighting
    if bool(q_aware_weights):
        try:
            ql = str(prompt).lower()
            mul: Dict[str, float] = {}
            if any(k in ql for k in ["signature", "param", "argument", "type", "prototype"]):
                mul.update({"o_proj": 1.10, "v_proj": 1.08})
            if any(k in ql for k in ["why", "fail", "error", "behavior", "incorrect", "bug"]):
                mul.update({"up_proj": 1.06, "down_proj": 1.05, "gate_proj": 1.04})
            if any(k in ql for k in ["where", "defined", "definition", "locate", "find"]):
                mul.update({"q_proj": 1.03})
            for k, m in mul.items():
                tw[k] = float(tw.get(k, 1.0)) * float(m)
        except Exception:
            pass
    # Optional ablations
    if bool(ablate_attn):
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tw[k] = 0.0
    if bool(ablate_mlp):
        for k in ("up_proj", "down_proj", "gate_proj"):
            tw[k] = 0.0
    # Optional per-layer schedule (gentle rise toward top third)
    layer_multipliers: Optional[List[float]] = None
    if bool(layer_schedule):
        try:
            L_layers = len(base_layers or [])
            if L_layers > 0:
                layer_multipliers = []
                for i in range(L_layers):
                    frac = float(i) / float(max(1, L_layers - 1))
                    if frac < (1.0 / 3.0):
                        lm = 0.95 + 0.15 * (frac / (1.0 / 3.0))
                    elif frac < (2.0 / 3.0):
                        lm = 1.05 + 0.05 * ((frac - (1.0 / 3.0)) / (1.0 / 3.0))
                    else:
                        lm = 1.10 + 0.05 * ((frac - (2.0 / 3.0)) / (1.0 / 3.0))
                    layer_multipliers.append(float(lm))
        except Exception:
            layer_multipliers = None
    # Optional per-target rank trimming (global per target; optional budget)
    per_target_keep: Optional[Dict[str, int]] = None
    if bool(per_target_rank_schedule):
        try:
            # Heuristic fractions by target group
            base_frac: Dict[str, float] = {
                "o_proj": 1.00, "up_proj": 1.00, "down_proj": 0.90, "gate_proj": 0.80,
                "q_proj": 0.70, "k_proj": 0.65, "v_proj": 0.60,
            }
            per_target_keep = {}
            for t, frac in base_frac.items():
                keep = int(max(1, min(int(scaled_rank), round(int(scaled_rank) * float(frac)))))
                per_target_keep[t] = keep
            # Apply global per-layer rank budget if requested
            try:
                budget = int(max(0, int(rank_budget)))
            except Exception:
                budget = 0
            if budget > 0 and per_target_keep:
                total = int(sum(int(v) for v in per_target_keep.values()))
                if total > budget:
                    scale = float(budget) / float(max(1, total))
                    for t in list(per_target_keep.keys()):
                        per_target_keep[t] = int(max(1, round(int(per_target_keep[t]) * scale)))
        except Exception:
            per_target_keep = None
    alpha_used = float(alpha * (0.5 if bool(alpha_warmup) else 1.0))
    # Optional layer-tiered per-target keeps
    per_target_keep_layers: Optional[List[Dict[str, int]]] = None
    if bool(layer_rank_tiers):
        try:
            L_layers = len(base_layers or [])
            if L_layers > 0:
                per_target_keep_layers = []
                def tier_for(frac: float) -> str:
                    if frac < (1.0/3.0): return "low"
                    if frac < (2.0/3.0): return "mid"
                    return "top"
                for i in range(L_layers):
                    frac = float(i) / float(max(1, L_layers - 1))
                    tier = tier_for(frac)
                    # desired keeps by group
                    if tier == "low":
                        vals = {"q_proj": 2, "k_proj": 2, "v_proj": 8, "o_proj": 8, "up_proj": 12, "down_proj": 12}
                        vals["gate_proj"] = min(8, int(0.5 * vals["up_proj"]))  # 6
                    elif tier == "mid":
                        vals = {"q_proj": 3, "k_proj": 3, "v_proj": 12, "o_proj": 12, "up_proj": 16, "down_proj": 16}
                        vals["gate_proj"] = min(8, int(0.5 * vals["up_proj"]))  # 8
                    else:
                        vals = {"q_proj": 4, "k_proj": 4, "v_proj": 16, "o_proj": 16, "up_proj": 24, "down_proj": 24}
                        vals["gate_proj"] = min(8, int(0.5 * vals["up_proj"]))  # 8
                    # cap by scaled_rank and >=1
                    for k in list(vals.keys()):
                        vals[k] = int(max(1, min(int(scaled_rank), int(vals[k]))))
                    per_target_keep_layers.append(vals)
        except Exception:
            per_target_keep_layers = None
    hooks = register_hook_mixed_adapters(
        model,
        base_layers,
        sub.get("layers"),
        alpha_star=float(alpha_used),
        g_sub=float(scaled_gsub),
        rank=int(scaled_rank),
        beta=float(beta),
        target_weights=tw,
        backend="local",
        layer_multipliers=layer_multipliers,
        per_target_keep=per_target_keep,
        per_target_keep_layers=per_target_keep_layers,
        delta_cap=float(max(0.0, delta_cap)),
    )

    # Generation (always use model.generate for stability)
    if verbose:
        print("[debug] generating...")
    try:
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            gen_kwargs.update({
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "min_new_tokens": int(max(0, int(min_new_tokens))),
            })
        if bool(adapter_aware_decoding) and bool(require_citations):
            try:
                # gentle nudge: slightly higher top_p, slightly lower repetition penalty
                gen_kwargs["top_p"] = float(min(0.99, float(gen_kwargs.get("top_p", top_p)) + 0.03))
                gen_kwargs["repetition_penalty"] = float(max(1.0, float(gen_kwargs.get("repetition_penalty", repetition_penalty)) - 0.05))
            except Exception:
                pass
        out = model.generate(**x, **gen_kwargs)
    except RuntimeError as e:
        msg = str(e)
        if ("CUBLAS_STATUS_ALLOC_FAILED" in msg) or ("CUDA out of memory" in msg):
            if verbose:
                print("[warn] CUDA allocation failed; retrying on CPU...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
            out = model.generate(**x, max_new_tokens=int(max_new_tokens), do_sample=False)
        else:
            raise

    # Decode (sequences-first, no logits decoding; ensure int dtype)
    try:
        seq = out.get("sequences") if isinstance(out, dict) else out
        if isinstance(seq, (list, tuple)):
            seq0 = seq[0]
        else:
            try:
                seq0 = seq[0]
            except Exception:
                seq0 = seq
        try:
            inp_len = int(x["input_ids"].shape[1])
        except Exception:
            inp_len = 0
        # Validate dtype is integer-like; otherwise fail decoding safely
        try:
            dt = getattr(seq0, "dtype", None)
            if dt is not None and not (str(dt).startswith("torch.int") or str(dt).startswith("torch.long")):
                raise ValueError("non-integer sequence output")
        except Exception:
            pass
        new_ids = seq0[inp_len:] if (hasattr(seq0, "shape") and getattr(seq0, "shape", [0])[0] > inp_len) else seq0
        text = _tok_decode(new_ids)
    except Exception:
        text = "[error] failed to decode output"

    # Citations enforcement with one retry
    if require_citations:
        if verbose:
            print("[debug] retry without sampling, stronger scaffold...")
        # Match example path to selected file (same as primary scaffold)
        try:
            example_rel = files[0] if files else None
            example_path = example_rel if example_rel else "file.py"
        except Exception:
            example_path = "file.py"
        retry_prompt = (
            final_prompt
            + f"\n\nRewrite the answer. For EACH paragraph, end with a citation like [{example_path}:123-160].\n"
              "Use at least 3 citations overall and only files shown in [ctx]."
        )
        # Stage-2 fallback: re-register hooks with stronger top-layer emphasis and slightly higher g_sub
        try:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            # reuse tw; build boosted layer multipliers even if layer_schedule was off
            L_layers = len(base_layers or [])
            layer_multipliers2: Optional[List[float]] = None
            if L_layers > 0:
                layer_multipliers2 = []
                for i in range(L_layers):
                    frac = float(i) / float(max(1, L_layers - 1))
                    # Emphasize top third more aggressively
                    if frac < (1.0 / 3.0):
                        lm = 0.95
                    elif frac < (2.0 / 3.0):
                        lm = 1.05
                    else:
                        lm = 1.20
                    layer_multipliers2.append(float(lm))
            boosted_gsub = float(min(0.95, float(scaled_gsub) * 1.10))
            # Re-register with same per-target keep and boosted schedule
            hooks = register_hook_mixed_adapters(
                model,
                base_layers,
                sub.get("layers"),
                alpha_star=float(alpha),
                g_sub=float(boosted_gsub),
                rank=int(scaled_rank),
                beta=float(beta),
                target_weights=tw,
                backend="local",
                layer_multipliers=layer_multipliers2,
                per_target_keep=per_target_keep,
                delta_cap=float(max(0.0, delta_cap)),
            )
        except Exception:
            pass
        x2_ids = torch.tensor([_tok_encode(retry_prompt)], dtype=torch.long, device=device)
        x2 = {"input_ids": x2_ids}
        retry_kwargs = {
            "max_new_tokens": int(max(int(max_new_tokens), 256)),
            "do_sample": False,
        }
        out2 = model.generate(**x2, **retry_kwargs)
        # Robust decode across dict/tensor/list outputs; only accept int token ids
        try:
            if isinstance(out2, dict):
                seq = out2.get("sequences", None)
                if seq is None:
                    raise ValueError("missing sequences in generate output")
            else:
                seq = out2
            if isinstance(seq, (list, tuple)):
                seq0 = seq[0]
            else:
                try:
                    seq0 = seq[0]
                except Exception:
                    seq0 = seq
            # Ensure integer dtype
            dt = getattr(seq0, "dtype", None)
            if dt is not None and not (str(dt).startswith("torch.int") or str(dt).startswith("torch.long")):
                raise ValueError("non-integer sequence output")
            try:
                inp_len2 = int(x2["input_ids"].shape[1])
            except Exception:
                inp_len2 = 0
            new_ids2 = seq0[inp_len2:] if (hasattr(seq0, "shape") and getattr(seq0, "shape", [0])[0] > inp_len2) else seq0
            text2 = _tok_decode(new_ids2)
        except Exception:
            # Last resort: stringify
            text2 = str(out2)
        # Accept retry result (program-agnostic runner does not attempt advanced repair)
        text = text2

    # Repair citations to anchored spans when present
    # Program-agnostic: skip repair; leave retry result as-is

    # Append commit footer for provenance if requested
    if bool(commit_footer):
        try:
            # Try to read manifest next to adapters_npz
            mn = None
            try:
                base_dir = os.path.dirname(adapters_npz)
                mf = os.path.join(base_dir, "manifest.json")
                if os.path.exists(mf):
                    obj = json.loads(open(mf, "r", encoding="utf-8").read())
                    mn = str(obj.get("commit") or obj.get("git", {}).get("commit") or "")
            except Exception:
                mn = None
            if not mn:
                try:
                    mn = os.popen(f"git -C {root} rev-parse HEAD").read().strip()
                except Exception:
                    mn = None
            if mn:
                short = mn[:12]
                text = text.rstrip() + f"\n\nAnswer valid for commit {short}."
        except Exception:
            pass

    # Clean up hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    return text



def generate_answer_structured(
    model_id: str,
    adapters_npz: str,
    prompt: str,
    *,
    program_root: Optional[str] = None,
    delta_cap: float = 0.05,
    # Same knobs as generate_answer with a few extras
    cache_dir: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    device_map: str = "none",
    gpu_ids: Optional[str] = None,
    of_sources: str = "question",
    zoom_symbol: Optional[str] = None,
    zoom_radius: int = 0,
    ignore: Optional[List[str]] = None,
    pack_context: bool = True,
    pack_mode: str = "windows",
    context_tokens: int = 3000,
    function_first: bool = True,
    ff_max_candidates: int = 24,
    ff_window_lines: int = 80,
    ff_threshold: float = 0.55,
    ff_noise_penalty: float = 0.30,
    alpha: float = 16.0,
    rank: int = 12,
    gsub: float = 0.75,
    beta: float = 0.1,
    entropy_aware: bool = False,
    rank_min: int = 8,
    rank_max: int = 32,
    gsub_min: float = 0.6,
    gsub_max: float = 0.9,
    entropy_weights: str = "program=0.4,subgraph=0.4,question=0.2",
    target_weights: Optional[str] = None,
    layer_schedule: bool = False,
    q_aware_weights: bool = False,
    mixture_m: int = 0,
    adapters_bank: Optional[str] = None,
    # Extra mixing/decoding knobs (opt-in)
    per_target_rank_schedule: bool = False,
    rank_budget: int = 0,
    ablate_attn: bool = False,
    ablate_mlp: bool = False,
    alpha_warmup: bool = False,
    adapter_aware_decoding: bool = False,
    layer_rank_tiers: bool = False,
    cone_rank: int = 2,
    cone_weight: float = 0.5,
    round_lora: bool = False,
    round_threshold: float = 0.5,
    require_citations: bool = True,
    citations_per_paragraph: bool = False,
    rerank: bool = True,
    self_queries_path: Optional[str] = None,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    min_new_tokens: int = 64,
    max_new_tokens: int = 256,
    kv_window: int = 0,
    head_device: str = "same",
    seed: int = 0,
    verbose: bool = False,
    commit_footer: bool = False,
    # New dcpo/lfp controls
    lfp_iters: int = 1,
    budget_H: float = 0.0,
    monotone_selection: bool = True,
    program_state_path: Optional[str] = None,
    # Powerdomain strength
    samples: int = 1,
    # Delta join semantics
    cone_join: str = "concat",  # "concat" | "weighted"
    # Telemetry
    telemetry_out: Optional[str] = None,
    telemetry_verify_tests: bool = False,
) -> Dict[str, Any]:
    """Generate a structured answer with citations and program state updates.

    Returns a dict: {text, citations, must, may, selection, lfp_passes, converged, provenance, confidence}.
    """
    # Program root (path or URI) for reading artifacts/windows/state
    root = str(program_root) if (program_root is not None and str(program_root).strip()) else ""
    t_start = time.time()
    if gpu_ids and str(gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Seed
    try:
        import random
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
    except Exception:
        pass

    # Cache
    if not cache_dir:
        env_cache = os.environ.get("TRANSFORMER_CACHE_DIR") or os.environ.get("HF_HOME")
        cache_dir = env_cache or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")), "checkpoints")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

    # Local tokenizer (no transformers) and adapter for context/scoring utilities
    snap_dir = ensure_snapshot(model_id, cache_dir)
    tok_local = LocalLlamaTokenizer(snap_dir)
    class _TokAdapter:
        def __init__(self, base):
            self._b = base
        class _Ret:
            def __init__(self, ids):
                self.input_ids = ids
        def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
            ids = self._b.encode(text)
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}
            return _TokAdapter._Ret(ids)
    tok = _TokAdapter(tok_local)
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model, _cfg = build_local_llama_from_snapshot(snap_dir, device, torch_dtype, device_map=device_map, gpu_ids=gpu_ids)

    base_layers = load_adapters_npz(adapters_npz)["layers"]
    # Program-agnostic structured path: skip graph embedder; retain main flow
    _ps_path = program_state_path if (program_state_path is not None and str(program_state_path).strip()) else None
    state_prev = load_program_state(root, path=_ps_path)

    must_set: Set[Tuple[str, int, int]] = set()
    union_set: Set[Tuple[str, int, int]] = set()
    selected_mods_acc: Set[str] = set(state_prev.candidates_modules) if monotone_selection else set()
    selected_files_acc: Set[str] = set(state_prev.candidates_files) if monotone_selection else set()
    lfp_passes = 0
    converged = False

    def _one_pass(mods_hint: List[str], files_hint: List[str]) -> Tuple[str, List[str], List[str], Dict[str, Any], Set[Tuple[str,int,int]], Set[Tuple[str,int,int]], Dict[str, Any]]:
        nonlocal model
        # Initial selection (question or zoom), union with hints for monotonicity
        ignore_list = [s for s in (ignore or []) if s]
        adapters_dir = os.path.dirname(adapters_npz)
        windows = load_windows_index(adapters_dir)
        files = pick_files_from_windows(root, windows, prompt, k=max(8, int(ff_max_candidates)))
        # initialize modules list for return and monotone selection merging
        modules: List[str] = []
        if monotone_selection:
            modules = sorted(list(set(mods_hint) | set(modules)))
            files = sorted(list(set(files_hint) | set(files)))
        # Subgraph emb
        # No subgraph embedding in program-agnostic structured path
        # Shapes/targets
        tmap = targets_map("local")
        t_shapes = infer_target_shapes(model)
        num_layers = len(getattr(model, "blocks", []))
        d_model_local = int(t_shapes.get("q_proj", (0, 0))[0]) or int(getattr(getattr(model, "cfg", None), "d_model", 0) or 0)
        # Entropy-aware
        scaled_rank = int(rank)
        scaled_gsub = float(gsub)
            # Build sub adapters
        sub = {"layers": []}
        # Optional cones
        cr = max(0, int(cone_rank))
        if bool(function_first) and cr > 0:
            try:
                pass
            except Exception:
                pass
        # Optional mixture bank: mix top-m module adapters from bank by concatenation (Σ π_i Δθ_i)
        if adapters_bank and int(mixture_m) > 0:
            try:
                import glob
                bank_root = os.path.abspath(os.path.expanduser(os.path.expandvars(adapters_bank)))
                sel_mods = [m for m in (modules or []) if m]
                picked = 0
                for mod in sel_mods:
                    if picked >= int(mixture_m):
                        break
                    mod_dir1 = os.path.join(bank_root, "sub_adapters", mod.replace("/", "_"))
                    mod_dir2 = os.path.join(bank_root, mod.replace("/", "_"))
                    cand = None
                    for d in (mod_dir1, mod_dir2):
                        fp = os.path.join(d, "adapters.npz")
                        if os.path.isfile(fp):
                            cand = fp
                            break
                    if not cand:
                        continue
                    try:
                        bank_ad = load_adapters_npz(cand)
                    except Exception:
                        continue
                    # uniform π for now
                    w = 1.0 / float(min(len(sel_mods), int(mixture_m)))
                    sw = float(max(0.0, min(1.0, w))) ** 0.5
                    # per-layer concat
                    merged_layers = []
                    for i in range(num_layers):
                        baseL = sub["layers"][i]
                        bL = bank_ad["layers"][i] if i < len(bank_ad.get("layers", [])) else {}
                        dst: Dict[str, Dict[str, np.ndarray]] = {}
                        for name in tmap.keys():
                            if (name in baseL) and (name in bL):
                                A1 = baseL[name]["A"]; B1 = baseL[name]["B"]
                                A2 = (sw * bL[name]["A"]).astype(np.float32); B2 = (sw * bL[name]["B"]).astype(np.float32)
                                A = np.concatenate([A1, A2], axis=1)
                                B = np.concatenate([B1, B2], axis=0)
                                dst[name] = {"A": A, "B": B, "gate": baseL[name].get("gate", np.array([0.0], dtype=np.float32))}
                            elif name in baseL:
                                dst[name] = baseL[name]
                            elif name in bL:
                                A = (sw * bL[name]["A"]).astype(np.float32)
                                dst[name] = {"A": A, "B": bL[name]["B"], "gate": bL[name].get("gate", np.array([0.0], dtype=np.float32))}
                        merged_layers.append(dst)
                    sub = {"layers": merged_layers}
                    # try infer rank increment from a present target
                    try:
                        for i in range(num_layers):
                            any_name = next((n for n in tmap.keys() if n in bank_ad["layers"][i]), None)
                            if any_name:
                                inc = int(bank_ad["layers"][i][any_name]["B"].shape[0])
                                scaled_rank = int(scaled_rank + inc)
                                break
                    except Exception:
                        pass
                    picked += 1
            except Exception:
                pass
        # Rounding
        if bool(round_lora):
            try:
                thr = float(max(0.0, round_threshold))
                for i in range(len(sub.get("layers", []))):
                    for name, tensors in sub["layers"][i].items():
                        for key in ("A", "B"):
                            arr = tensors.get(key)
                            if not isinstance(arr, np.ndarray):
                                continue
                            q = float(np.median(np.abs(arr))) if arr.size > 0 else 0.0
                            if q <= 0:
                                continue
                            out = np.where(np.abs(arr) < (thr * q), 0.0, np.sign(arr) * q).astype(np.float32)
                            tensors[key] = out
            except Exception:
                pass

        # Delta norm diagnostics (approximate AB norms per target averaged across layers)
        delta_norms: Dict[str, float] = {}
        try:
            sums: Dict[str, float] = {}
            counts: Dict[str, int] = {}
            for i in range(len(sub.get("layers", []))):
                for name, tensors in sub["layers"][i].items():
                    try:
                        A = tensors.get("A"); B = tensors.get("B")
                        if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
                            AB = (A @ B)
                            n = float(np.linalg.norm(AB)) if AB.size > 0 else 0.0
                            sums[name] = float(sums.get(name, 0.0) + n)
                            counts[name] = int(counts.get(name, 0) + 1)
                    except Exception:
                        continue
            for k, v in sums.items():
                c = max(1, int(counts.get(k, 1)))
                delta_norms[k] = float(v / float(c))
        except Exception:
            delta_norms = {}

        # Prompt + context
        final_prompt = prompt
        files_for_ctx = list(files)
        if bool(pack_context) and files_for_ctx:
            packed = ""
            ident_header = ""
            if pack_mode == "windows":
                if bool(function_first):
                    # Simplified: pack top window per file like non-structured path
                    adapters_dir = os.path.dirname(adapters_npz)
                    try:
                        windows = load_windows_index(adapters_dir)
                    except Exception:
                        windows = []
                    out_lines: List[str] = ["Program windows (function-first):"]
                    used = 0
                    budget = int(context_tokens)
                    for rel in files_for_ctx:
                        cand = [w for w in windows if (w.get("path") or "").replace("\\", "/") == rel.replace("\\", "/")]
                        if not cand:
                            continue
                        w = cand[0]
                        a = int(w.get("start_line", 1)); b = int(w.get("end_line", a + 60))
                        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(root, rel))
                        try:
                            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
                        except Exception:
                            continue
                        a = max(1, min(a, len(src_lines))); b = max(a, min(b, len(src_lines)))
                        block = [f"[ctx] path: {rel}:{a}-{b}"] + src_lines[a - 1 : b] + [""]
                        text_block = "\n".join(block) + "\n"
                        t = len(tok(text_block).input_ids)
                        if used + t > budget:
                            continue
                        out_lines.extend(block)
                        used += t
                        if used >= budget:
                            break
                    packed = "\n".join(out_lines) if len(out_lines) > 1 else ""
                else:
                    # Fallback: identical to function-first simplified path
                    adapters_dir = os.path.dirname(adapters_npz)
                    try:
                        windows = load_windows_index(adapters_dir)
                    except Exception:
                        windows = []
                    out_lines: List[str] = ["Program windows:"]
                    used = 0
                    budget = int(context_tokens)
                    for rel in files_for_ctx:
                        cand = [w for w in windows if (w.get("path") or "").replace("\\", "/") == rel.replace("\\", "/")]
                        if not cand:
                            continue
                        w = cand[0]
                        a = int(w.get("start_line", 1)); b = int(w.get("end_line", a + 60))
                        abs_fp = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(root, rel))
                        try:
                            src_lines = open(abs_fp, "r", encoding="utf-8", errors="ignore").read().splitlines()
                        except Exception:
                            continue
                        a = max(1, min(a, len(src_lines))); b = max(a, min(b, len(src_lines)))
                        block = [f"[ctx] path: {rel}:{a}-{b}"] + src_lines[a - 1 : b] + [""]
                        text_block = "\n".join(block) + "\n"
                        t = len(tok(text_block).input_ids)
                        if used + t > budget:
                            continue
                        out_lines.extend(block)
                        used += t
                        if used >= budget:
                            break
                    packed = "\n".join(out_lines) if len(out_lines) > 1 else ""
            else:
                packed = ""  # heads mode omitted in program-agnostic core
            if packed:
                final_prompt = (ident_header + packed + "\n\n" + final_prompt) if ident_header else (packed + "\n\n" + final_prompt)
        if require_citations:
            try:
                example_rel = files_for_ctx[0] if files_for_ctx else None
                example_path = example_rel if example_rel else "file.py"
            except Exception:
                example_path = "file.py"
            final_prompt = (
                final_prompt
                + f"\n\nInstruction: For EVERY claim, append a citation of the form {example_path}:START-END.\n"
                  "Use only files shown in [ctx] above. Provide at least 3 citations overall.\n"
            )
            # Adapter-aware decoding prompt nudge (pointer-first) for consistency
            if bool(adapter_aware_decoding):
                try:
                    example_rel = files_for_ctx[0] if files_for_ctx else None
                    example_path = example_rel if example_rel else "file.py"
                except Exception:
                    example_path = "file.py"
                final_prompt = (
                    f"[pointer-first] Start with a citation like [{example_path}:A-B], then explain.\n\n"
                    + final_prompt
                )

        # Apply mixed adapters
        x_ids = torch.tensor([[i for i in tok_local.encode(final_prompt)]], dtype=torch.long, device=device)
        x = {"input_ids": x_ids}
        def _parse_target_weights(spec: Optional[str]) -> Optional[Dict[str, float]]:
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
                        k = k.strip(); out[k] = float(v)
                    else:
                        out[part] = 1.0
                return out or None
            except Exception:
                return None
        tw = _parse_target_weights(target_weights) or {}
        # Optional question-aware reweighting
        if bool(q_aware_weights):
            try:
                ql = str(prompt).lower()
                mul: Dict[str, float] = {}
                if any(k in ql for k in ["signature", "param", "argument", "type", "prototype"]):
                    mul.update({"o_proj": 1.10, "v_proj": 1.08})
                if any(k in ql for k in ["why", "fail", "error", "behavior", "incorrect", "bug"]):
                    mul.update({"up_proj": 1.06, "down_proj": 1.05, "gate_proj": 1.04})
                if any(k in ql for k in ["where", "defined", "definition", "locate", "find"]):
                    mul.update({"q_proj": 1.03})
                for k, m in mul.items():
                    tw[k] = float(tw.get(k, 1.0)) * float(m)
            except Exception:
                pass
        # Optional per-layer schedule
        layer_multipliers: Optional[List[float]] = None
        if bool(layer_schedule):
            try:
                L_layers = len(base_layers or [])
                if L_layers > 0:
                    layer_multipliers = []
                    for i in range(L_layers):
                        frac = float(i) / float(max(1, L_layers - 1))
                        if frac < (1.0 / 3.0):
                            lm = 0.95 + 0.15 * (frac / (1.0 / 3.0))
                        elif frac < (2.0 / 3.0):
                            lm = 1.05 + 0.05 * ((frac - (1.0 / 3.0)) / (1.0 / 3.0))
                        else:
                            lm = 1.10 + 0.05 * ((frac - (2.0 / 3.0)) / (1.0 / 3.0))
                        layer_multipliers.append(float(lm))
            except Exception:
                layer_multipliers = None
        # Apply ablations
        if bool(ablate_attn):
            for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tw[k] = 0.0
        if bool(ablate_mlp):
            for k in ("up_proj", "down_proj", "gate_proj"):
                tw[k] = 0.0
        # Optional per-target rank trimming
        per_target_keep: Optional[Dict[str, int]] = None
        if bool(per_target_rank_schedule):
            try:
                base_frac: Dict[str, float] = {
                    "o_proj": 1.00, "up_proj": 1.00, "down_proj": 0.90, "gate_proj": 0.80,
                    "q_proj": 0.70, "k_proj": 0.65, "v_proj": 0.60,
                }
                per_target_keep = {}
                for t, frac in base_frac.items():
                    keep = int(max(1, min(int(scaled_rank), round(int(scaled_rank) * float(frac)))))
                    per_target_keep[t] = keep
                budget = int(max(0, int(rank_budget)))
                if budget > 0 and per_target_keep:
                    total = int(sum(int(v) for v in per_target_keep.values()))
                    if total > budget:
                        scale = float(budget) / float(max(1, total))
                        for t in list(per_target_keep.keys()):
                            per_target_keep[t] = int(max(1, round(int(per_target_keep[t]) * scale)))
            except Exception:
                per_target_keep = None
        # Optional per-layer rank tiers
        per_target_keep_layers: Optional[List[Dict[str, int]]] = None
        if bool(layer_rank_tiers):
            try:
                L_layers = len(base_layers or [])
                if L_layers > 0:
                    per_target_keep_layers = []
                    def tier_for(frac: float) -> str:
                        if frac < (1.0/3.0): return "low"
                        if frac < (2.0/3.0): return "mid"
                        return "top"
                    for i in range(L_layers):
                        frac = float(i) / float(max(1, L_layers - 1))
                        tier = tier_for(frac)
                        if tier == "low":
                            vals = {"q_proj": 2, "k_proj": 2, "v_proj": 8, "o_proj": 8, "up_proj": 12, "down_proj": 12}
                            vals["gate_proj"] = min(8, int(0.5 * vals["up_proj"]))
                        elif tier == "mid":
                            vals = {"q_proj": 3, "k_proj": 3, "v_proj": 12, "o_proj": 12, "up_proj": 16, "down_proj": 16}
                            vals["gate_proj"] = min(8, int(0.5 * vals["up_proj"]))
                        else:
                            vals = {"q_proj": 4, "k_proj": 4, "v_proj": 16, "o_proj": 16, "up_proj": 24, "down_proj": 24}
                            vals["gate_proj"] = min(8, int(0.5 * vals["up_proj"]))
                        for k in list(vals.keys()):
                            vals[k] = int(max(1, min(int(scaled_rank), int(vals[k]))))
                        per_target_keep_layers.append(vals)
            except Exception:
                per_target_keep_layers = None
        # Alpha warmup on first structured pass
        alpha_used = float(alpha * (0.5 if (bool(alpha_warmup) and int(lfp_passes) == 0) else 1.0))
        hooks = register_hook_mixed_adapters(
            model,
            base_layers,
            sub.get("layers"),
            alpha_star=float(alpha_used),
            g_sub=float(scaled_gsub),
            rank=int(scaled_rank),
            beta=float(beta),
            target_weights=tw,
            backend="local",
            layer_multipliers=layer_multipliers,
            per_target_keep=per_target_keep,
            per_target_keep_layers=per_target_keep_layers,
            delta_cap=float(max(0.0, delta_cap)),
        )

        # Generate (possibly multiple samples)
        union_cites: Set[Tuple[str,int,int]] = set()
        inter_cites: Optional[Set[Tuple[str,int,int]]] = None
        text_first = ""
        gen_err = None
        oom_retry = False
        for si in range(max(1, int(samples))):
            try:
                gen_kwargs = {"max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample)}
                if do_sample:
                    gen_kwargs.update({
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "repetition_penalty": float(repetition_penalty),
                        "min_new_tokens": int(max(0, int(min_new_tokens))),
                    })
                if bool(adapter_aware_decoding) and bool(require_citations):
                    try:
                        gen_kwargs["top_p"] = float(min(0.99, float(gen_kwargs.get("top_p", top_p)) + 0.03))
                        gen_kwargs["repetition_penalty"] = float(max(1.0, float(gen_kwargs.get("repetition_penalty", repetition_penalty)) - 0.05))
                    except Exception:
                        pass
                out = model.generate(**x, **gen_kwargs)
            except RuntimeError as e:
                msg = str(e)
                if ("CUBLAS_STATUS_ALLOC_FAILED" in msg) or ("CUDA out of memory" in msg):
                    oom_retry = True
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
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
                    out = model.generate(**x, max_new_tokens=int(max_new_tokens), do_sample=False)
                else:
                    gen_err = msg
                    break
            try:
                seq = out.get("sequences") if isinstance(out, dict) else out
                seq0 = seq[0] if hasattr(seq, "__getitem__") else seq
                inp_len = int(x["input_ids"].shape[1]) if isinstance(x, dict) else int(x.input_ids.shape[1])
                new_ids = seq0[inp_len:] if (hasattr(seq0, "shape") and getattr(seq0, "shape", [0])[0] > inp_len) else seq0
                try:
                    text = tok_local.decode(new_ids.tolist(), skip_special_tokens=True)  # type: ignore[arg-type]
                except Exception:
                    text = tok_local.decode(new_ids.tolist())  # type: ignore[arg-type]
            except Exception:
                text = "[error] failed to decode output"

            # Retry for citations per policy
            missing_citations = False
            if require_citations:
                missing_citations = True
                try:
                    example_rel = files_for_ctx[0] if files_for_ctx else None
                    example_path = example_rel if example_rel else "file.py"
                except Exception:
                    example_path = "file.py"
                retry_prompt = (
                    final_prompt
                    + f"\n\nRewrite the answer. For EACH paragraph, end with a citation like [{example_path}:123-160].\n"
                      "Use at least 3 citations overall and only files shown in [ctx]."
                )
                x2_ids = torch.tensor([[i for i in tok_local.encode(retry_prompt)]], dtype=torch.long, device=device)
                x2 = {"input_ids": x2_ids}
                out2 = model.generate(**x2, max_new_tokens=int(max(int(max_new_tokens), 256)), do_sample=False)
                try:
                    seq = out2.get("sequences") if isinstance(out2, dict) else out2
                    seq0 = seq[0] if hasattr(seq, "__getitem__") else seq
                    try:
                        text2 = tok_local.decode(seq0.tolist(), skip_special_tokens=True)  # type: ignore[arg-type]
                    except Exception:
                        text2 = tok_local.decode(seq0.tolist())  # type: ignore[arg-type]
                except Exception:
                    text2 = str(out2)
                text = text2
                missing_citations = False
            # If still missing citations, append smart references from anchored spans
            if require_citations and missing_citations:
                pass

            # Repair citations to anchored spans when present (before capturing first text)
            # Skip repair in program-agnostic path

            if not text_first:
                text_first = text
            cset = set(normalize_citations(text))
            union_cites.update(cset)
            if inter_cites is None:
                inter_cites = set(cset)
            else:
                inter_cites &= set(cset)

        # Clean hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        diag = {"rank": scaled_rank, "gsub": scaled_gsub, "targets": list(tmap.keys()), "delta_norms": (delta_norms or None)}
        beh = {"oom_retry": bool(oom_retry), "gen_error": gen_err is not None}
        return text_first, modules, files, diag, (union_cites or set()), (inter_cites or set()), beh

        # Clean hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        diag = {"rank": scaled_rank, "gsub": scaled_gsub, "targets": list(tmap.keys())}
        return text, modules, files, diag

    # LFP loop
    prev_checksum = state_prev.checksum()
    text_last = ""
    retries_used = 0
    for it in range(max(1, int(lfp_iters))):
        mods_hint = sorted(list(selected_mods_acc))
        files_hint = sorted(list(selected_files_acc))
        text, sel_mods, sel_files, diag, cites_u, cites_i, beh_diag = _one_pass(mods_hint, files_hint)
        lfp_passes += 1
        try:
            if bool(beh_diag.get("gen_error")):
                retries_used += 1
        except Exception:
            pass
        # Facts from citations (union/intersection across samples per pass)
        union_set.update(cites_u)
        if it == 0:
            must_set = set(cites_i)
        else:
            must_set = must_set & set(cites_i)
        # Accumulate selection (monotone)
        selected_mods_acc.update(sel_mods)
        selected_files_acc.update(sel_files)
        # Update RepoState (anytime vector = subgraph z)
        z_all = None
        st_new = new_state_from_run(
            root,
            modules=sorted(list(selected_mods_acc)),
            files=sorted(list(selected_files_acc)),
            citations=list(union_set),
            z_vec=(z_all if isinstance(z_all, np.ndarray) else None),
            beh_event={"type": "gen", "ok": bool(text and len(text) > 0), "ts": time.time(), **beh_diag},
            H_increment=float(max(0.0, budget_H)),
        )
        state_cur = join_program_states(state_prev, st_new)
        # Convergence: no new bits added
        if not changed_bits(state_prev, state_cur):
            converged = True
            text_last = text
            state_prev = state_cur
            break
        # else continue
        text_last = text
        state_prev = state_cur

    # Save state
    try:
        save_program_state(state_prev, path=_ps_path)
    except Exception:
        pass

    # Provenance and confidence
    try:
        commit_sha = os.popen(f"git -C {root} rev-parse HEAD").read().strip()
    except Exception:
        commit_sha = None
    para = [p for p in (text_last or "").split("\n\n") if p.strip()]
    cite_ok = sum(1 for p in para if has_citations(p, False))
    confidence = float(cite_ok) / float(max(1, len(para)))

    must = sorted(list(must_set))
    may = sorted(list(union_set - must_set))

    elapsed = None
    try:
        elapsed = float(max(0.0, time.time() - t_start))
    except Exception:
        elapsed = None

    result = {
        "text": text_last,
        "citations": sorted(list(union_set)),
        "must": must,
        "may": may,
        "selection": {
            "modules": sorted(list(selected_mods_acc)),
            "files": sorted(list(selected_files_acc)),
        },
        "lfp_passes": int(lfp_passes),
        "converged": bool(converged),
        "provenance": {"commit": commit_sha},
        "confidence": float(confidence),
        "metrics": {
            "citations_total": int(len(union_set)),
            "citations_must": int(len(must)),
            "modules_selected": int(len(selected_mods_acc)),
            "files_selected": int(len(selected_files_acc)),
            "elapsed_sec": elapsed,
            "retries": int(retries_used),
            "delta_norms": (diag.get("delta_norms") if isinstance(diag, dict) else None),
        },
    }
    # Optional signature edit distance (rough, per-citation snippet vs answer)
    try:
        def _lev_norm(a: str, b: str) -> float:
            a = a.strip()[:256]; b = b.strip()[:1024]
            if not a or not b:
                return 1.0
            la, lb = len(a), len(b)
            prev = list(range(lb + 1))
            cur = [0] * (lb + 1)
            for i in range(1, la + 1):
                cur[0] = i
                ai = a[i - 1]
                for j in range(1, lb + 1):
                    cost = 0 if ai == b[j - 1] else 1
                    cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
                prev, cur = cur, prev
            dist = float(prev[lb])
            norm = dist / float(max(1, max(la, lb)))
            return max(0.0, min(1.0, norm))
        ed_vals: List[float] = []
        ans_txt = result["text"] or ""
        for (rel, a_ln, b_ln) in result.get("citations", [])[:6]:
            try:
                pth = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(root, rel))
                with open(pth, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = fh.read().splitlines()
                a0 = max(1, int(a_ln)); b0 = min(len(lines), int(b_ln))
                snippet = "\n".join(lines[a0 - 1 : b0])
                ed_vals.append(_lev_norm(snippet, ans_txt))
            except Exception:
                continue
        if ed_vals:
            result["metrics"]["signature_edit_mean"] = float(sum(ed_vals) / float(len(ed_vals)))
            result["metrics"]["signature_edit_min"] = float(min(ed_vals))
    except Exception:
        pass
    # Optional tests verification (best-effort, core is program-agnostic)
    if bool(telemetry_verify_tests):
        # Left as a no-op in core. Example backends can perform verification and
        # populate these metrics in their own wrappers.
        result.setdefault("metrics", {})["tests_checked"] = 0
        result.setdefault("metrics", {})["tests_passed"] = 0

    # Telemetry sidecar
    if telemetry_out:
        try:
            os.makedirs(os.path.dirname(telemetry_out), exist_ok=True)
        except Exception:
            pass
        try:
            open(telemetry_out, "w", encoding="utf-8").write(json.dumps(result, indent=2))
        except Exception:
            pass

    return result



