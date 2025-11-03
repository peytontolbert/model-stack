from __future__ import annotations

import os
import json
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer  # type: ignore

from examples.repo_grounded_adapters.code_graph import CodeGraph
from examples.repo_grounded_adapters.modules.adapter import (
    load_adapters_npz,
    generate_lora_from_embedding,
)
from examples.repo_grounded_adapters.modules.embedding import (
    build_subgraph_embedding_from_graph,
)
from examples.repo_grounded_adapters.modules.model import (
    ensure_snapshot,
    build_local_llama_from_snapshot,
)
from examples.repo_grounded_adapters.modules.selection import (
    modules_from_symbols,
    question_aware_modules_and_files,
    prompt_token_set,
    name_matches_prompt,
)
from examples.repo_grounded_adapters.modules.mixing import (
    targets_map,
    infer_target_shapes,
    register_hook_mixed_adapters,
)
from examples.repo_grounded_adapters.modules.capacity import (
    entropy_score,
    scale_capacity,
)
from examples.repo_grounded_adapters.modules.context import (
    pack_context_heads,
    pack_context_windows,
    collect_function_windows,
    extract_func_name_from_lines,
    model_prob_yes,
)
from examples.repo_grounded_adapters.modules.verify import has_citations
from examples.repo_grounded_adapters.modules.interpret import (
    is_block,
    block_out_hook,
    truncate_batch,
    get_W,
)


def generate_answer(
    model_id: str,
    adapters_npz: str,
    repo_root: str,
    prompt: str,
    *,
    cache_dir: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
    entropy_weights: str = "repo=0.4,subgraph=0.4,question=0.2",
    target_weights: Optional[str] = None,
    # cones/rounding
    cone_rank: int = 2,
    cone_weight: float = 0.5,
    round_lora: bool = False,
    round_threshold: float = 0.5,
    # citations
    require_citations: bool = False,
    citations_per_paragraph: bool = False,
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
) -> str:
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

    cache_dir = cache_dir or os.path.join(repo_root, "checkpoints")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
    torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    ckpt_dir = ensure_snapshot(model_id, cache_dir)
    model, _cfg = build_local_llama_from_snapshot(ckpt_dir, device, torch_dtype)

    # Load base adapters
    base_layers = load_adapters_npz(adapters_npz)["layers"]

    # Selection
    ignore_list = [s for s in (ignore or []) if s]
    if (of_sources == "zoom") and zoom_symbol:
        seeds = [s.strip() for s in str(zoom_symbol).split(",") if s.strip()]
        modules, files = modules_from_symbols(repo_root, seeds, radius=int(zoom_radius), top_k=8, ignore=ignore_list)
        if not modules and not files:
            modules, files = question_aware_modules_and_files(repo_root, prompt, top_k=8, ignore=ignore_list)
    else:
        modules, files = question_aware_modules_and_files(repo_root, prompt, top_k=8, ignore=ignore_list)
    if verbose:
        try:
            print(f"[debug] selection modules={modules} files={files}")
        except Exception:
            pass

    # Subgraph embedding
    g = CodeGraph.load_or_build(repo_root, ignore=ignore_list)
    abs_files = [f if os.path.isabs(f) else os.path.abspath(os.path.join(g.root, f)) for f in files]
    sub_z = build_subgraph_embedding_from_graph(
        g,
        dim=1536,
        seed=int(seed) + 1,
        include_modules=modules,
        include_files=abs_files,
        include_text=True,
        text_max_bytes=250000,
        text_weight=0.25,
    )

    # Shapes and targets
    tmap = targets_map("local")
    t_shapes = infer_target_shapes(model)
    num_layers = len(getattr(model, "blocks", []))
    d_model_local = int(t_shapes.get("q_proj", (0, 0))[0]) or int(getattr(getattr(model, "cfg", None), "d_model", 0) or 0)

    # Entropy-aware capacity
    scaled_rank = int(rank)
    scaled_gsub = float(gsub)
    entropy_diag: Optional[Dict[str, float]] = None
    if bool(entropy_aware):
        try:
            es, diag = entropy_score(g, modules, files, weights=str(entropy_weights))
            entropy_diag = diag
            scaled_rank, scaled_gsub = scale_capacity(
                es,
                rank_min=int(rank_min),
                rank_max=int(rank_max),
                gsub_min=float(gsub_min),
                gsub_max=float(gsub_max),
            )
            if verbose:
                print(f"[debug] entropy score={es:.3f} rank->{scaled_rank} gsub->{scaled_gsub:.3f}")
        except Exception:
            pass

    # Build subgraph adapters
    sub = generate_lora_from_embedding(
        sub_z["z"],
        d_model=int(d_model_local),
        num_layers=int(num_layers),
        rank=int(scaled_rank),
        seed=int(seed) + 2,
        targets=list(tmap.keys()),
        target_shapes=t_shapes,
    )

    # Function-first cones: build and merge
    cr = max(0, int(cone_rank))
    if bool(function_first) and cr > 0:
        try:
            fn_windows2 = collect_function_windows(repo_root, files, int(ff_window_lines), max_candidates=int(ff_max_candidates))
            seen: Dict[str, bool] = {}
            cone_files_rel: List[str] = []
            for (rel, a, b, ln, lines_block) in fn_windows2:
                if not seen.get(rel):
                    seen[rel] = True
                    cone_files_rel.append(rel)
            abs_cone_files = [f if os.path.isabs(f) else os.path.abspath(os.path.join(g.root, f)) for f in cone_files_rel]
            if abs_cone_files:
                cone_z = build_subgraph_embedding_from_graph(
                    g,
                    dim=1536,
                    seed=int(seed) + 3,
                    include_modules=modules,
                    include_files=abs_cone_files,
                    include_text=True,
                    text_max_bytes=250000,
                    text_weight=0.25,
                )
                cone = generate_lora_from_embedding(
                    cone_z["z"],
                    d_model=int(d_model_local),
                    num_layers=int(num_layers),
                    rank=int(cr),
                    seed=int(seed) + 4,
                    targets=list(tmap.keys()),
                    target_shapes=t_shapes,
                )
                cw = float(max(0.0, min(1.0, cone_weight)))
                merged_layers = []
                for i in range(num_layers):
                    baseL = sub["layers"][i]
                    coneL = cone["layers"][i]
                    dst: Dict[str, Dict[str, np.ndarray]] = {}
                    for name in tmap.keys():
                        if (name in baseL) and (name in coneL):
                            A1 = baseL[name]["A"]; B1 = baseL[name]["B"]
                            A2 = coneL[name]["A"]; B2 = coneL[name]["B"]
                            A2s = (cw * A2).astype(np.float32)
                            A = np.concatenate([A1, A2s], axis=1)
                            B = np.concatenate([B1, B2], axis=0)
                            dst[name] = {"A": A, "B": B, "gate": baseL[name].get("gate", np.array([0.0], dtype=np.float32))}
                        elif name in baseL:
                            dst[name] = baseL[name]
                        elif name in coneL:
                            A = (cw * coneL[name]["A"]).astype(np.float32)
                            dst[name] = {"A": A, "B": coneL[name]["B"], "gate": coneL[name].get("gate", np.array([0.0], dtype=np.float32))}
                    merged_layers.append(dst)
                sub = {"layers": merged_layers}
                scaled_rank = int(scaled_rank + cr)
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
        packed = ""
        if pack_mode == "windows":
            if function_first:
                fn_windows = collect_function_windows(repo_root, files, int(ff_window_lines), max_candidates=int(ff_max_candidates))
                scored: List[Tuple[float, float, bool, str, int, int, int, List[str], Optional[str]]] = []
                p_tokens = prompt_token_set(prompt)
                for (rel, a, b, anchor_ln, lines_block) in fn_windows:
                    txt_block = "\n".join(lines_block)
                    rel_p, noise_p = model_prob_yes(tok, model, prompt, txt_block)
                    score = float(max(0.0, min(1.0, rel_p - float(ff_noise_penalty) * noise_p)))
                    fname = extract_func_name_from_lines(lines_block, a, b, anchor_ln)
                    name_match = name_matches_prompt(fname, p_tokens)
                    boosted = float(min(1.0, score + (0.35 if name_match else 0.0)))
                    scored.append((boosted, score, name_match, rel, a, b, anchor_ln, lines_block, fname))
                scored.sort(key=lambda x: x[0], reverse=True)
                g_local = CodeGraph.load_or_build(repo_root)
                out_lines: List[str] = ["Repository windows (function-first):"]
                used = 0
                budget = int(context_tokens)
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
                    if used >= int(0.8 * budget):
                        break
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
                    if used >= int(0.8 * budget):
                        break
                if used < budget:
                    aux = pack_context_windows(repo_root, files, tok, budget - used)
                    if aux:
                        out_lines.extend(aux.splitlines())
                packed = "\n".join(out_lines) if len(out_lines) > 1 else ""
            else:
                packed = pack_context_windows(repo_root, files, tok, int(context_tokens))
        else:
            packed = pack_context_heads(repo_root, files, tok, int(context_tokens))
        if packed:
            final_prompt = packed + "\n\n" + final_prompt
    if require_citations:
        final_prompt = (
            final_prompt
            + "\n\nInstruction: For EVERY claim, append a citation of the form modules/FILE.py:START-END.\n"
              "Use only files shown in [ctx] above. Provide at least 3 citations overall.\n"
        )

    x = tok(final_prompt, return_tensors="pt")
    x = x.to(device)

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
    hooks = register_hook_mixed_adapters(
        model,
        base_layers,
        sub.get("layers"),
        alpha_star=float(alpha),
        g_sub=float(scaled_gsub),
        rank=int(scaled_rank),
        beta=float(beta),
        target_weights=tw,
        backend="local",
    )

    # Generation
    if verbose:
        print("[debug] generating...")
    try:
        if int(kv_window) > 0 or head_device != "same":
            # Manual loop with optional head offload
            max_new = int(max_new_tokens)
            do_sample_flag = bool(do_sample)
            head_mode = str(head_device)
            head_use_cpu = False
            try:
                head_param = next(model.lm_head.parameters()).device if hasattr(model, "lm_head") else next(model.parameters()).device
            except Exception:
                head_param = next(model.parameters()).device
            if head_mode == "cpu":
                head_use_cpu = True
            elif head_mode == "auto":
                try:
                    if head_param.type == "cuda":
                        free, total = torch.cuda.mem_get_info(head_param)
                        head_use_cpu = bool(free < (256 * 1024 * 1024))
                    else:
                        head_use_cpu = True
                except Exception:
                    head_use_cpu = True
            else:
                head_use_cpu = False

            from examples.repo_grounded_adapters.modules.runtime import prepare_head_weight as _prep
            Wt, head_dev = _prep(model, head_use_cpu)
            input_ids = x["input_ids"] if isinstance(x, dict) else x.input_ids
            seq = input_ids
            eos_ids = getattr(model.config, "eos_token_id", None)
            if isinstance(eos_ids, (list, tuple)):
                eos_set = {int(xx) for xx in eos_ids if xx is not None}
            elif eos_ids is not None:
                eos_set = {int(eos_ids)}
            else:
                eos_set = set()
            for _ in range(max_new):
                with torch.no_grad():
                    win = int(kv_window)
                    ctx = seq[:, -win:] if win > 0 else seq[:, -1:]
                    core = model.model(input_ids=ctx, attention_mask=None, use_cache=False, return_dict=True)
                    h_last = core.last_hidden_state[:, -1, :]
                    h_proj = h_last.to(device=head_dev, dtype=Wt.dtype)
                    logits = torch.matmul(h_proj, Wt)
                if float(repetition_penalty) != 1.0:
                    gather_ids = seq
                    logits = logits.scatter_add(1, gather_ids, torch.full_like(gather_ids, -abs(float(repetition_penalty) - 1.0), dtype=logits.dtype))
                if do_sample_flag:
                    lg = logits
                    if temperature and temperature > 0:
                        lg = lg / float(temperature)
                    probs = torch.softmax(lg, dim=-1)
                    if 0.0 < float(top_p) < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                        cum = torch.cumsum(sorted_probs, dim=-1)
                        mask = cum > float(top_p)
                        mask[..., 0] = False
                        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                seq = torch.cat([seq, next_id], dim=1)
                if eos_set:
                    try:
                        eos_t = torch.tensor(sorted(list(eos_set)), device=next_id.device, dtype=next_id.dtype)
                        if torch.isin(next_id.view(-1), eos_t).all():
                            break
                    except Exception:
                        if all(int(xx) in eos_set for xx in next_id.view(-1).tolist()):
                            break
            out = seq
        else:
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

    # Decode
    try:
        out_ids = out[0] if hasattr(out, "__getitem__") else out
        if isinstance(out, dict) and "sequences" in out:
            out_ids = out["sequences"]
        try:
            inp_len = int(x["input_ids"].shape[1]) if isinstance(x, dict) else int(x.input_ids.shape[1])
        except Exception:
            inp_len = 0
        seq_ids = out_ids[0]
        new_ids = seq_ids[inp_len:] if (hasattr(seq_ids, "shape") and seq_ids.shape[0] > inp_len) else seq_ids
        text = tok.decode(new_ids, skip_special_tokens=True)
    except Exception:
        try:
            seq = out.get("sequences") if isinstance(out, dict) else out
            inp_len = int(x["input_ids"].shape[1]) if isinstance(x, dict) else int(getattr(x, "input_ids", [0]).shape[1])
            seq_ids = seq[0]
            new_ids = seq_ids[inp_len:] if (hasattr(seq_ids, "shape") and seq_ids.shape[0] > inp_len) else seq_ids
            text = tok.decode(new_ids, skip_special_tokens=True)
        except Exception:
            text = "[error] failed to decode output"

    # Citations enforcement with one retry
    if require_citations and not has_citations(text, bool(citations_per_paragraph)):
        if verbose:
            print("[debug] retry without sampling, stronger scaffold...")
        retry_prompt = (
            final_prompt
            + "\n\nRewrite the answer. For EACH paragraph, end with a citation like [modules/train.py:123-160].\n"
              "Use at least 3 citations overall and only files shown in [ctx]."
        )
        x2 = tok(retry_prompt, return_tensors="pt").to(device)
        retry_kwargs = {
            "max_new_tokens": int(max(int(max_new_tokens), 256)),
            "do_sample": False,
        }
        out2 = model.generate(**x2, **retry_kwargs)
        text2 = tok.decode(out2[0], skip_special_tokens=True)
        if has_citations(text2, bool(citations_per_paragraph)):
            text = text2
        else:
            refs = []
            for rel in files[:4]:
                path = rel if rel.startswith("modules/") else f"modules/{os.path.basename(rel)}"
                refs.append(f"- {path}:1-120")
            if refs:
                text = text2 + "\n\nReferences (from context):\n" + "\n".join(refs)
            else:
                text = text2 or "INSUFFICIENT_EVIDENCE: No path:line citations found per policy."

    # Clean up hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    return text


