import os
import argparse
import json
from typing import Optional, Dict, Tuple, List

import numpy as np

from examples.repo_grounded_adapters.code_graph import CodeGraph
from examples.repo_grounded_adapters.modules.embedding import build_repo_embedding, build_subgraph_embedding_from_graph
from examples.repo_grounded_adapters.modules.adapter import generate_lora_from_embedding, save_npz, detect_target_shapes_from_model, detect_target_shapes_from_model_full
from examples.repo_grounded_adapters.modules.capacity import entropy_score
from examples.repo_grounded_adapters.modules.embedding import auto_model_dims

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapters-dir", required=True)
    # Embedding
    p.add_argument("--embed-dim", type=int, default=1536)
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--text-max-bytes", type=int, default=0)
    p.add_argument("--max-text-tokens", type=int, default=0)
    p.add_argument("--text-weight", type=float, default=0.25)
    p.add_argument("--graph-prop-hops", type=int, default=0)
    p.add_argument("--graph-prop-damp", type=float, default=0.85)
    p.add_argument("--ignore", action="append", default=None)
    # Base adapter
    p.add_argument("--base-rank", type=int, default=8)
    p.add_argument("--target-weights", default=None)
    p.add_argument("--knowledge-preset", action="store_true")
    # Priors & rounding
    p.add_argument("--kbann-priors", action="store_true")
    p.add_argument("--round-lora", action="store_true")
    p.add_argument("--round-threshold", type=float, default=0.5)
    # Per-module export
    p.add_argument("--per-module", action="store_true")
    p.add_argument("--include-deps", action="store_true")
    p.add_argument("--max-deps", type=int, default=4)
    p.add_argument("--sub-rank", type=int, default=8)
    # HF cache/model
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--probe-full", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    # Optional self-tune/distillation
    p.add_argument("--self-tune-out", default=None, help="If set, run distillation over modules and export tuned adapters into this dir")
    p.add_argument("--context-tokens", type=int, default=5000)
    p.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    p.add_argument("--gpu-ids", default=None)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--citations-per-paragraph", action="store_true")
    p.add_argument("--per-module-adapters", action="store_true")
    p.add_argument("--distill-include-deps", action="store_true")
    p.add_argument("--distill-max-deps", type=int, default=4)
    p.add_argument("--distill-rank", type=int, default=8)
    args = p.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.adapters_dir)))
    os.makedirs(out_dir, exist_ok=True)

    # Build CodeGraph and embedding (full repo)
    g = CodeGraph.load_or_build(args.repo, ignore=[s for s in (args.ignore or []) if s])
    emb = build_repo_embedding(
        args.repo,
        dim=int(args.embed_dim),
        seed=int(args.seed),
        include_text=bool(args.include_text),
        text_max_bytes=int(args.text_max_bytes),
        max_text_tokens=int(args.max_text_tokens),
        text_weight=float(args.text_weight),
        graph_prop_hops=int(args.graph_prop_hops),
        graph_prop_damp=float(args.graph_prop_damp),
        ignore=[s for s in (args.ignore or []) if s],
    )

    # Shapes & dims
    cache_dir = args.cache_dir or os.path.join(args.repo, "checkpoints")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass
    target_shapes: Optional[Dict[str, Tuple[int, int]]] = None
    if args.probe_full:
        target_shapes = detect_target_shapes_from_model_full(args.model, target_regex=None)
    if not target_shapes:
        target_shapes = detect_target_shapes_from_model(args.model)
    num_layers, d_model = auto_model_dims(args.model, cache_dir)

    # Target weights: parse or preset
    def _parse_tw(spec: Optional[str]) -> Optional[Dict[str, float]]:
        if not spec:
            return None
        out: Dict[str, float] = {}
        try:
            parts = [p.strip() for p in str(spec).split(",") if p.strip()]
            for p in parts:
                if "=" not in p:
                    continue
                n, v = p.split("=", 1)
                out[n.strip()] = float(v)
            return out or None
        except Exception:
            return None
    tw = _parse_tw(args.target_weights)
    if (tw is None) and bool(args.knowledge_preset):
        tw = {"q_proj": 0.95, "k_proj": 0.95, "v_proj": 0.95, "o_proj": 1.10, "up_proj": 1.10, "down_proj": 1.05}

    # Optional KBANN priors (domain-derived boosts)
    if bool(args.kbann_priors):
        try:
            es, diag = entropy_score(g, list(g.modules.keys()), [], weights="repo=0.8,subgraph=0.2,question=0.0")
            scomp = float(diag.get("repo_component", 0.5))
            kb_tw = {"o_proj": 1.0 + 0.15 * scomp, "up_proj": 1.0 + 0.15 * scomp, "down_proj": 1.0 + 0.10 * scomp, "v_proj": 0.95}
            if tw is None:
                tw = kb_tw
            else:
                for k, v in kb_tw.items():
                    tw[k] = float(tw.get(k, 1.0)) * float(v)
        except Exception:
            pass

    # Generate base adapters
    adapters = generate_lora_from_embedding(
        emb["z"],
        d_model=(d_model or 0),
        num_layers=(num_layers or 0),
        rank=int(args.base_rank),
        seed=int(args.seed),
        targets=(list(target_shapes.keys()) if target_shapes else None),
        target_shapes=target_shapes,
        target_weights=tw,
    )

    # Optional rounding
    if bool(args.round_lora):
        try:
            thr = float(max(0.0, args.round_threshold))
            for i in range(len(adapters.get("layers", []))):
                for name, tensors in adapters["layers"][i].items():
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

    manifest = {
        "repo": os.path.abspath(args.repo),
        "model": args.model,
        "embed_dim": int(args.embed_dim),
        "layers": int(num_layers),
        "d_model": int(d_model),
        "rank": int(args.base_rank),
        "include_text": bool(args.include_text),
        "graph_prop_hops": int(args.graph_prop_hops),
        "graph_prop_damp": float(args.graph_prop_damp),
        "target_weights": tw,
    }
    save_npz(out_dir, embedding=emb, adapters=adapters, manifest=manifest)

    if args.verbose:
        print(json.dumps({"status": "ok", "adapters": os.path.join(out_dir, "adapters.npz"), "embedding": os.path.join(out_dir, "embedding.npz")}, indent=2))

    # Optional per-module export
    if bool(args.per_module):
        sub_root = os.path.join(out_dir, "sub_adapters")
        os.makedirs(sub_root, exist_ok=True)
        # Shapes again for export
        shapes = target_shapes
        if not shapes:
            shapes = detect_target_shapes_from_model(args.model)
        for module in sorted(g.modules.keys()):
            if g.modules[module].is_test:
                continue
            mods = [module]
            if args.include_deps:
                deps = list(g.module_imports.get(module, []) or [])
                if int(args.max_deps) > 0:
                    deps = deps[: int(args.max_deps)]
                mods.extend([m for m in deps if m])
            mods = sorted({m for m in mods})
            emb_m = build_subgraph_embedding_from_graph(
                g,
                dim=int(args.embed_dim),
                seed=int(args.seed),
                include_modules=mods,
                include_files=None,
                include_text=bool(args.include_text),
                text_max_bytes=int(args.text_max_bytes),
                max_text_tokens=int(args.max_text_tokens),
                text_weight=float(args.text_weight),
                graph_prop_hops=int(args.graph_prop_hops),
                graph_prop_damp=float(args.graph_prop_damp),
            )
            try:
                adapters_m = generate_lora_from_embedding(
                    emb_m["z"],
                    d_model=int(d_model),
                    num_layers=int(num_layers),
                    rank=int(args.sub_rank),
                    seed=int(args.seed),
                    targets=(list(shapes.keys()) if shapes else None),
                    target_shapes=shapes,
                    target_weights=None,
                )
                sub_dir = os.path.join(sub_root, module.replace("/", "_"))
                os.makedirs(sub_dir, exist_ok=True)
                save_npz(
                    sub_dir,
                    embedding=emb_m,
                    adapters=adapters_m,
                    manifest={"module": module, "include_modules": mods, "rank": int(args.sub_rank), "d_model": int(d_model), "layers": int(num_layers)},
                )
            except Exception:
                continue

    # Optional: self-tune and export tuned adapters
    if args.self_tune_out:
        from examples.repo_grounded_adapters.modules.tune import distill_repo, export_tuned_adapters
        tuned_out = os.path.abspath(os.path.expanduser(os.path.expandvars(args.self_tune_out)))
        os.makedirs(tuned_out, exist_ok=True)
        verified, buffer_path = distill_repo(
            repo=args.repo,
            model=args.model,
            adapters=os.path.join(out_dir, "adapters.npz"),
            out_dir=tuned_out,
            ignore=[s for s in (args.ignore or []) if s],
            max_prompts=3,
            cache_dir=args.cache_dir,
            device=args.device,
            gpu_ids=args.gpu_ids,
            context_tokens=int(args.context_tokens),
            timeout_sec=(int(args.timeout_sec) if int(args.timeout_sec) > 0 else None),
            resume=bool(args.resume),
            log_every=int(args.log_every),
            citations_per_paragraph=bool(args.citations_per_paragraph),
        )
        export_tuned_adapters(
            repo=args.repo,
            model=args.model,
            verified_modules=sorted(list(verified)),
            out_dir=tuned_out,
            per_module_adapters=bool(args.per_module_adapters),
            include_deps=bool(args.distill_include_deps),
            max_deps=int(args.distill_max_deps),
            rank=int(args.distill_rank),
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()


