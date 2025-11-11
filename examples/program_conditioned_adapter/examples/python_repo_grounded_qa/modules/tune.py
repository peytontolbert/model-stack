from __future__ import annotations

import os
import json
import time
import hashlib
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Set

from ..code_graph import CodeGraph
from .prompts import build_prompts_for_module
from .verify import verify_with_tests, extract_citations
from examples.program_conditioned_adapter.modules.embedding import build_subgraph_embedding_from_graph
from examples.program_conditioned_adapter.modules.adapter import (
    save_npz,
    generate_lora_from_embedding,
)
from model.inspect import detect_target_shapes_from_model_full, detect_target_shapes_from_model


def _run_repo_adapter_cli(
    model: str,
    adapters_dir: str,
    repo: str,
    prompt: str,
    *,
    cache_dir: Optional[str] = None,
    device: str = "cpu",
    gpu_ids: Optional[str] = None,
    context_tokens: int = 5000,
    timeout_sec: Optional[int] = None,
) -> Tuple[int, str, str]:
    """Minimal wrapper around the generic run CLI to obtain an answer text."""
    cmd = [
        sys.executable,
        "-m",
        "examples.program_conditioned_adapter.run",
        "--model", model,
        "--sources", repo,
        "--adapters-dir", adapters_dir,
        "--prompt", prompt,
        "--context-tokens", str(int(context_tokens)),
    ]
    if cache_dir:
        cmd += ["--cache-dir", str(cache_dir)]
    if device and device != "cpu":
        cmd += ["--device-map", "auto"]
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        return int(proc.returncode), proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as te:
        return 124, "", str(te)
    except Exception as e:
        return 1, "", str(e)


def distill_repo(
    repo: str,
    model: str,
    adapters: str,
    out_dir: str,
    *,
    ignore: Optional[List[str]] = None,
    max_prompts: int = 3,
    cache_dir: Optional[str] = None,
    device: str = "cpu",
    gpu_ids: Optional[str] = None,
    context_tokens: int = 5000,
    timeout_sec: Optional[int] = None,
    resume: bool = False,
    log_every: int = 25,
    citations_per_paragraph: bool = False,
) -> Tuple[Set[str], str]:
    out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(out_dir)))
    os.makedirs(out_dir, exist_ok=True)
    g = CodeGraph.load_or_build(repo, ignore=[s for s in (ignore or []) if s])

    buffer_path = os.path.join(out_dir, "distill.jsonl")
    verified_modules: Dict[str, bool] = {}
    processed: Dict[Tuple[str, str], bool] = {}
    if resume and os.path.isfile(buffer_path):
        try:
            with open(buffer_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        k = (str(row.get("module", "")), str(row.get("prompt", "")))
                        processed[k] = True
                    except Exception:
                        continue
        except Exception:
            processed = {}

    mode = ("a" if resume and os.path.isfile(buffer_path) else "w")
    t0 = time.time()
    logs_dir = os.path.join(out_dir, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass

    env = os.environ.copy()
    with open(buffer_path, mode, encoding="utf-8") as fh:
        for idx, module in enumerate(sorted(g.modules.keys()), start=1):
            mi = g.modules[module]
            if mi.is_test:
                continue
            prompts = build_prompts_for_module(g, module, max_q=int(max_prompts))
            module_verified = False
            for pidx, q in enumerate(prompts, start=1):
                if resume and ((module, q) in processed):
                    continue
                try:
                    print(json.dumps({
                        "event": "start_prompt",
                        "module": module,
                        "prompt_index": pidx,
                        "context_tokens": int(context_tokens),
                    }), flush=True)
                except Exception:
                    pass
                rc, ans_text, err_text = _run_repo_adapter_cli(
                    model,
                    adapters,
                    repo,
                    q,
                    cache_dir=cache_dir,
                    device=device,
                    gpu_ids=gpu_ids,
                    context_tokens=int(context_tokens),
                    timeout_sec=(int(timeout_sec) if (timeout_sec and int(timeout_sec) > 0) else None),
                )
                ok = (rc == 0)
                if ok:
                    cites = extract_citations(ans_text)
                    ok = bool(cites)
                    if ok:
                        ok = verify_with_tests(g, module, repo_root=repo, env=env)
                try:
                    h = hashlib.sha1((module + "\n" + q).encode("utf-8", errors="ignore")).hexdigest()[:12]
                    log_fp = os.path.join(logs_dir, f"{module.replace('/', '_')}.{pidx}.{h}.log")
                    with open(log_fp, "w", encoding="utf-8") as lf:
                        lf.write(ans_text)
                        lf.write("\n\n[stderr]\n")
                        lf.write(err_text or "")
                except Exception:
                    pass
                fh.write(json.dumps({
                    "module": module,
                    "prompt": q,
                    "answer": ans_text,
                    "verified": bool(ok),
                    "citations": extract_citations(ans_text),
                    "ignore": [s for s in (ignore or []) if s],
                    "device": device,
                    "context_tokens": int(context_tokens),
                    "rc": int(rc),
                }) + "\n")
                module_verified = module_verified or bool(ok)
            if module_verified:
                verified_modules[module] = True
            if (idx % max(1, int(log_every))) == 0:
                try:
                    elapsed = time.time() - t0
                except Exception:
                    elapsed = -1.0
                print(json.dumps({"progress": {"done": idx, "total": len(g.modules), "elapsed_sec": elapsed}}))

    return set(verified_modules.keys()), buffer_path


def export_tuned_adapters(
    repo: str,
    model: str,
    verified_modules: List[str],
    out_dir: str,
    *,
    per_module_adapters: bool = False,
    include_deps: bool = False,
    max_deps: int = 4,
    rank: int = 8,
    cache_dir: Optional[str] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    g = CodeGraph.load_or_build(repo)
    emb = build_subgraph_embedding_from_graph(
        g,
        dim=1536,
        seed=0,
        include_modules=sorted(list(verified_modules)),
        include_files=None,
        include_text=True,
        text_max_bytes=250000,
        max_text_tokens=0,
        text_weight=0.25,
        graph_prop_hops=0,
        graph_prop_damp=0.85,
    )
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    adapters = {"layers": [], "rank": 8, "d_model": 0, "targets": targets, "gates": []}
    manifest = {
        "repo": os.path.abspath(repo),
        "verified_modules": sorted(list(verified_modules)),
        "buffer": os.path.join(out_dir, "distill.jsonl"),
        "schema_version": 1,
        "note": "Embedding from verified modules; adapter layers empty for downstream mixing.",
    }
    save_npz(out_dir, embedding=emb, adapters=adapters, manifest=manifest)

    if not per_module_adapters:
        return

    # Try to infer model dims and target shapes for per-module export
    num_layers: int = 0
    d_model: int = 0
    target_shapes: Optional[Dict[str, Tuple[int, int]]] = None
    try:
        target_shapes = detect_target_shapes_from_model_full(model, target_regex=None)
    except Exception:
        target_shapes = None
    if not target_shapes:
        try:
            target_shapes = detect_target_shapes_from_model(model)
        except Exception:
            target_shapes = None
    try:
        from transformers import AutoConfig  # type: ignore
        cfg = AutoConfig.from_pretrained(model, cache_dir=cache_dir)
        num_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        d_model = int(getattr(cfg, "hidden_size", 0) or 0)
    except Exception:
        num_layers, d_model = 0, 0

    sub_root = os.path.join(out_dir, "sub_adapters")
    os.makedirs(sub_root, exist_ok=True)
    for module in sorted(list(verified_modules)):
        mods = [module]
        if include_deps:
            deps = list(g.module_imports.get(module, []) or [])
            if int(max_deps) > 0:
                deps = deps[: int(max_deps)]
            mods.extend([m for m in deps if m])
        mods_unique = sorted({m for m in mods})
        emb_m = build_subgraph_embedding_from_graph(
            g,
            dim=1536,
            seed=0,
            include_modules=mods_unique,
            include_files=None,
            include_text=True,
            text_max_bytes=250000,
            max_text_tokens=0,
            text_weight=0.25,
            graph_prop_hops=0,
            graph_prop_damp=0.85,
        )
        if (num_layers <= 0) or (d_model <= 0):
            sub_dir = os.path.join(sub_root, module.replace("/", "_"))
            os.makedirs(sub_dir, exist_ok=True)
            save_npz(
                sub_dir,
                embedding=emb_m,
                adapters={"layers": [], "rank": rank, "d_model": 0, "targets": [], "gates": []},
                manifest={
                    "module": module,
                    "include_modules": mods_unique,
                    "note": "Embedding-only; shapes not inferred",
                },
            )
            continue
        try:
            adapters_m = generate_lora_from_embedding(
                emb_m["z"],
                d_model=int(d_model),
                num_layers=int(num_layers),
                rank=int(rank),
                seed=0,
                targets=list((target_shapes or {}).keys()) if target_shapes else None,
                target_shapes=target_shapes,
                layer_gate="zmean",
                target_weights=None,
            )
            sub_dir = os.path.join(sub_root, module.replace("/", "_"))
            os.makedirs(sub_dir, exist_ok=True)
            save_npz(
                sub_dir,
                embedding=emb_m,
                adapters=adapters_m,
                manifest={
                    "module": module,
                    "include_modules": mods_unique,
                    "rank": int(rank),
                    "d_model": int(d_model),
                    "layers": int(num_layers),
                },
            )
        except Exception:
            continue


