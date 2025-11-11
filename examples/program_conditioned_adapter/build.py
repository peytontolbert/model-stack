import os
import argparse
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, List
import sys
import random
import platform
import importlib

import numpy as np

from examples.program_conditioned_adapter.modules.embedding import (  # type: ignore
    build_program_embedding,
    build_subgraph_embedding_from_program,
)
from examples.program_conditioned_adapter.modules.adapter import (  # type: ignore
    generate_lora_from_embedding,
    generate_lora_from_embedding_torch,
    save_npz,
)
from model.hf_snapshot import ensure_snapshot
from model.inspect import detect_target_shapes_from_model
from examples.program_conditioned_adapter.modules.capacity import entropy_score  # type: ignore
#
# NOTE: Build is program-agnostic; no backend-specific imports here.
from examples.program_conditioned_adapter.modules.program_graph import ProgramGraph  # type: ignore

def _load_symbol(path: Optional[str]):
    if not path:
        return None
    mod, _, attr = path.partition(":")
    if not mod or not attr:
        raise ValueError(f"Invalid symbol path '{path}', expected 'module:attr'")
    m = importlib.import_module(mod)
    return getattr(m, attr)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", required=False, help="Program sources root (path or URI)")
    p.add_argument("--program", required=False, help="Alias for --sources (program root path or URI)")
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
    # Adapter mapping is always enhanced; no CLI toggle
    p.add_argument("--code-recall-preset", action="store_true", help="Opt-in preset tuned for code recall (o,v,up,down,gate emphasized; light q,k)")
    # Priors & rounding
    p.add_argument("--kbann-priors", action="store_true")
    p.add_argument("--kbann-strong", action="store_true", help="Stronger inhibitory priors: boost o/up/down; damp v,k (slightly q)")
    p.add_argument("--round-lora", action="store_true")
    p.add_argument("--round-threshold", type=float, default=0.5)
    p.add_argument("--round-mode", choices=["none", "hard", "soft"], default=None, help="Rounding/sparsification mode for adapter matrices")
    p.add_argument("--round-soft-kp", type=float, default=10.0, help="Soft mode: percent of entries to keep (top-|.|) per axis")
    p.add_argument("--round-axis", choices=["row", "col", "global"], default="row", help="Soft mode axis for top-k sparsification")
    p.add_argument("--zero-b", action="store_true", help="After generation, set all B matrices to zero (official LoRA init)")
    p.add_argument("--learn-bias", action="store_true", help="Export zero bias vectors so downstream can fine-tune bias only")
    # Contracts (optional snapshot into manifest and embedding contracts channel)
    p.add_argument("--contracts-require-citations", action="store_true")
    p.add_argument("--contracts-retrieval-policy", default=None)
    p.add_argument("--contracts-retrieval-temp", type=float, default=None)
    p.add_argument("--contracts-weight", type=float, default=0.10)
    # Per-module export (built by default; use --no-per-module to skip)
    p.add_argument("--per-module", action="store_true", help="DEPRECATED: no effect; per-module bank is built by default. Use --no-per-module to disable.")
    p.add_argument("--no-per-module", action="store_true", help="Disable building per-module sub_adapters bank (built by default)")
    p.add_argument("--include-deps", action="store_true")
    p.add_argument("--max-deps", type=int, default=4)
    p.add_argument("--sub-rank", type=int, default=8)
    p.add_argument("--files-only", action="store_true", help="Export a files-only sub-adapter using an explicit file allowlist")
    p.add_argument(
        "--files-allowlist",
        action="append",
        default=None,
        help="Files-only mode: relative file paths to include (repeatable)",
    )
    # HF cache/model
    p.add_argument("--cache-dir", default="/data/transformer_10/checkpoints")
    p.add_argument("--probe-full", action="store_true")
    p.add_argument("--gen-backend", choices=["numpy", "torch"], default="numpy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fallback-topology", action="store_true", help="If embedding is empty, fall back to topology-only component z_top")
    # ProgramGraph/Embedder plugins (backend-agnostic)
    p.add_argument("--pg-backend", default=None, help="Dotted path to ProgramGraph factory, e.g. 'examples.program_conditioned_adapter.examples.python_repo_grounded_qa.python_repo_graph:PythonRepoGraph'")
    p.add_argument("--embedder-fn", default=None, help="Dotted path to embedder function '(pg, sources_root, **opts)->{z,...}'. Defaults to PCA program embedder.")
    # Capacity scheduling
    p.add_argument("--auto-rank", action="store_true", help="Auto-schedule rank based on program complexity")
    p.add_argument("--rank-min", type=int, default=None)
    p.add_argument("--rank-max", type=int, default=None)
    p.add_argument("--mdl-budget-params", type=int, default=0, help="Optional MDL-style global parameter budget across all layers/targets (approximate)")
    # Program seeding
    p.add_argument("--init-program-state", action="store_true", help="Initialize or update baseline ProgramState (.program_state.json) after build")
    p.add_argument("--program-state-path", default=None, help="Optional explicit path to write ProgramState JSON; defaults to <sources_root>/.program_state.json")
    # (self-tune is no longer part of build; use modules/tune.py externally if desired)
    args = p.parse_args()

    # Warn on deprecated/ignored flags for per-module export
    try:
        if bool(args.per_module) and bool(args.no_per_module):
            print("[pca.build] Warning: Both --per-module and --no-per-module were provided; proceeding with --no-per-module (disables per-module bank).", file=sys.stderr)
        elif bool(args.per_module):
            print("[pca.build] Warning: --per-module is deprecated and has no effect; per-module bank is built by default. Use --no-per-module to disable.", file=sys.stderr)
    except Exception:
        pass
    # Determinism: set seeds and hash seed (best-effort)
    try:
        os.environ["PYTHONHASHSEED"] = str(int(args.seed))
    except Exception:
        pass
    try:
        random.seed(int(args.seed))
    except Exception:
        pass
    try:
        np.random.seed(int(args.seed))
    except Exception:
        pass
    torch_seeded = False
    try:
        import torch as _torch  # type: ignore
        _torch.manual_seed(int(args.seed))
        torch_seeded = True
    except Exception:
        torch_seeded = False

    out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.adapters_dir)))
    os.makedirs(out_dir, exist_ok=True)

    # Normalize sources root (prefer --program, else --sources)
    src_arg = args.program if getattr(args, "program", None) else args.sources
    if not src_arg:
        raise SystemExit("--sources or --program is required")
    sources_root = os.path.abspath(os.path.expanduser(os.path.expandvars(src_arg)))

    # Construct ProgramGraph via plugin (if provided)
    pg: Optional[ProgramGraph] = None
    try:
        pg_ctor = _load_symbol(args.pg_backend)
        if pg_ctor:
            pg = pg_ctor(sources_root, ignore=[s for s in (args.ignore or []) if s])
    except Exception:
        pg = None

    # Build embedding (program-agnostic) via plugin or default PCA program embedder
    embedder = _load_symbol(args.embedder_fn) if args.embedder_fn else None
    if embedder is not None:
        try:
            # Prefer new program-graph signature
            emb = embedder(
                pg,
                sources_root=sources_root,
                dim=int(args.embed_dim),
                seed=int(args.seed),
                include_text=bool(args.include_text),
                text_max_bytes=int(args.text_max_bytes),
                max_text_tokens=int(args.max_text_tokens),
                text_weight=float(args.text_weight),
                calls_weight=0.25,
                types_weight=0.20,
                tests_weight=0.15,
                graph_prop_hops=int(args.graph_prop_hops),
                graph_prop_damp=float(args.graph_prop_damp),
                ignore=[s for s in (args.ignore or []) if s],
            )
        except Exception:
            # Backward-compat: allow sources-root signature
            emb = embedder(
                sources_root,
                dim=int(args.embed_dim),
                seed=int(args.seed),
                include_text=bool(args.include_text),
                text_max_bytes=int(args.text_max_bytes),
                max_text_tokens=int(args.max_text_tokens),
                text_weight=float(args.text_weight),
                calls_weight=0.25,
                types_weight=0.20,
                tests_weight=0.15,
                graph_prop_hops=int(args.graph_prop_hops),
                graph_prop_damp=float(args.graph_prop_damp),
                ignore=[s for s in (args.ignore or []) if s],
            )
    else:
        emb = build_program_embedding(
            pg if pg is not None else None,  # type: ignore[arg-type]
            sources_root=sources_root,
            dim=int(args.embed_dim),
            seed=int(args.seed),
            include_text=bool(args.include_text),
            text_max_bytes=int(args.text_max_bytes),
            max_text_tokens=int(args.max_text_tokens),
            text_weight=float(args.text_weight),
            calls_weight=0.25,
            types_weight=0.20,
            tests_weight=0.15,
            contracts_kv={
                "require_citations": (True if bool(args.contracts_require_citations) else False),
                "retrieval_policy": (str(args.contracts_retrieval_policy) if args.contracts_retrieval_policy else ""),
                "retrieval_temp": (str(args.contracts_retrieval_temp) if args.contracts_retrieval_temp is not None else ""),
            },
            contracts_weight=float(max(0.0, args.contracts_weight)),
            graph_prop_hops=int(args.graph_prop_hops),
            graph_prop_damp=float(args.graph_prop_damp),
            ignore=[s for s in (args.ignore or []) if s],
        )

    # Crash early on empty embeddings; optionally fall back to topology-only
    try:
        z = emb.get("z")
        z_norm = float(np.linalg.norm(z)) if isinstance(z, np.ndarray) else 0.0
        fams = ["z_sym", "z_doc", "z_mod", "z_top", "z_text"]
        fam_norms = {k: (float(np.linalg.norm(emb[k])) if isinstance(emb.get(k), np.ndarray) else 0.0) for k in fams if k in emb}
        embed_empty = (z_norm == 0.0)
        fallback_used = False
        if embed_empty:
            if bool(args.fallback_topology) and (fam_norms.get("z_top", 0.0) > 0.0):
                emb["z"] = emb["z_top"].astype(np.float32)
                fallback_used = True
            else:
                detail = {
                    "z_norm": z_norm,
                    "family_norms": fam_norms,
                    "hint": "Relax --ignore, enable --include-text, increase --text-max-bytes/--max-text-tokens, or pass --fallback-topology",
                }
                raise RuntimeError(f"Empty program embedding (||z||=0). Details: {json.dumps(detail)}")
    except Exception as _e:
        if not isinstance(_e, RuntimeError):
            raise

    # Shapes & dims
    # Prefer explicit cache_dir; else env; else project root (/..../checkpoints)
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
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
    # Infer shapes/dims from local snapshot config.json (no transformers)
    snap_dir = ensure_snapshot(args.model, cache_dir)
    cfg_path = os.path.join(snap_dir, "config.json")
    cfg_obj = json.load(open(cfg_path, "r", encoding="utf-8"))
    d_model = int(cfg_obj.get("hidden_size", 4096))
    num_layers = int(cfg_obj.get("num_hidden_layers", 32))
    n_heads = int(cfg_obj.get("num_attention_heads", 32))
    n_kv_heads = int(cfg_obj.get("num_key_value_heads", n_heads))
    d_ff = int(cfg_obj.get("intermediate_size", 11008))
    head_dim = int(cfg_obj.get("head_dim", d_model // max(1, n_heads)))
    target_shapes: Optional[Dict[str, Tuple[int, int]]] = {
        "q_proj": (n_heads * head_dim, d_model),
        "k_proj": (n_kv_heads * head_dim, d_model),
        "v_proj": (n_kv_heads * head_dim, d_model),
        "o_proj": (d_model, n_heads * head_dim),
        "up_proj": (d_ff, d_model),
        "gate_proj": (d_ff, d_model),
        "down_proj": (d_model, d_ff),
    }

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
    if tw is None:
        if bool(args.code_recall_preset):
            tw = {"o_proj": 1.15, "v_proj": 1.10, "up_proj": 1.10, "down_proj": 1.05, "gate_proj": 1.00, "q_proj": 0.95, "k_proj": 0.90}
        elif bool(args.knowledge_preset):
            tw = {"q_proj": 0.95, "k_proj": 0.95, "v_proj": 0.95, "o_proj": 1.10, "up_proj": 1.10, "down_proj": 1.05}

    # Optional KBANN priors (domain-derived boosts) - program-agnostic
    kbann_mode = None
    if bool(args.kbann_priors):
        try:
            # Estimate structural complexity and density from ProgramGraph (if available)
            ents_cnt = 0
            edges_cnt = 0
            try:
                if 'pg' in locals() and (pg is not None):
                    try:
                        ents_cnt = sum(1 for _ in pg.entities())
                    except Exception:
                        ents_cnt = 0
                    try:
                        edges_cnt = sum(1 for _ in pg.edges())  # type: ignore[attr-defined]
                    except Exception:
                        edges_cnt = 0
            except Exception:
                ents_cnt = 0
                edges_cnt = 0
            scomp = float(max(0.0, min(1.0, ents_cnt / 10000.0)))  # normalize entity count
            density = float(edges_cnt) / float(max(1, ents_cnt))
            dens_term = 0.05 * float(min(1.0, density / 8.0))
            if bool(args.kbann_strong):
                kbann_mode = "strong"
                kb_tw = {
                    "o_proj": 1.0 + 0.22 * scomp + 1.2 * dens_term,
                    "up_proj": 1.0 + 0.22 * scomp + 1.2 * dens_term,
                    "down_proj": 1.0 + 0.15 * scomp + dens_term,
                    "v_proj": 0.85,   # stronger inhibitory prior
                    "k_proj": 0.95,
                    "q_proj": 0.98,
                }
            else:
                kbann_mode = "standard"
                # Prior emphasis: o/up/down (composition/usage), mild downweight v, neutral k/q
                kb_tw = {
                    "o_proj": 1.0 + 0.15 * scomp + dens_term,
                    "up_proj": 1.0 + 0.15 * scomp + dens_term,
                    "down_proj": 1.0 + 0.10 * scomp + (dens_term * 0.8),
                    "v_proj": 0.95,
                    "k_proj": 1.00,
                    "q_proj": 1.00,
                }
            if tw is None:
                tw = kb_tw
            else:
                for k, v in kb_tw.items():
                    tw[k] = float(tw.get(k, 1.0)) * float(v)
        except Exception:
            pass

    # Targets & shape safety
    NON_SQUARE_TARGETS = {"k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"}
    if target_shapes:
        selected_targets: List[str] = list(target_shapes.keys())
        # shape sanity
        for t, shp in target_shapes.items():
            if not isinstance(shp, tuple) or len(shp) != 2:
                raise RuntimeError(f"Invalid target shape for {t}: {shp}")
            a, b = int(shp[0]), int(shp[1])
            if a <= 0 or b <= 0:
                raise RuntimeError(f"Non-positive dims for {t}: {shp}")
    else:
        # Without shapes, only allow square-safe targets
        selected_targets = ["q_proj", "o_proj"]
        # If user expects non-square behavior (e.g., via presets), hard-fail
        if any(t in NON_SQUARE_TARGETS for t in selected_targets):
            raise RuntimeError("Non-square targets requested without inferred shapes. Use --probe-full or restrict targets to q_proj,o_proj.")

    # Per-layer/dims consistency
    if int(num_layers) <= 0 or int(d_model) <= 0:
        raise RuntimeError("Could not infer model dims (layers/d_model). Pass a valid --model or use --probe-full.")

    # Final normalization/clipping of target weights (single place)
    if tw:
        try:
            vals = [float(v) for v in tw.values() if v is not None]
            if vals:
                mean_v = float(sum(vals) / float(len(vals)))
                if mean_v > 0:
                    for k in list(tw.keys()):
                        tw[k] = float(tw[k]) / mean_v
                # clip to [0.7, 1.3]
                for k in list(tw.keys()):
                    v = float(tw[k])
                    if v < 0.7:
                        tw[k] = 0.7
                    elif v > 1.3:
                        tw[k] = 1.3
        except Exception:
            pass

    # Auto rank scheduling
    rank_min = int(args.rank_min) if args.rank_min is not None else int(args.base_rank)
    rank_max = int(args.rank_max) if args.rank_max is not None else int(max(rank_min, args.base_rank))
    rank_global = int(args.base_rank)
    scomp_ar = 0.5
    density_ar = 0.0
    if bool(args.auto_rank):
        try:
            ents_cnt_ar = 0
            edges_cnt_ar = 0
            if 'pg' in locals() and (pg is not None):
                try:
                    ents_cnt_ar = sum(1 for _ in pg.entities())
                except Exception:
                    ents_cnt_ar = 0
                try:
                    edges_cnt_ar = sum(1 for _ in pg.edges())  # type: ignore[attr-defined]
                except Exception:
                    edges_cnt_ar = 0
            density_ar = float(edges_cnt_ar) / float(max(1, ents_cnt_ar))
            scomp_ar = float(max(0.0, min(1.0, ents_cnt_ar / 10000.0)))
            comp = max(0.0, min(1.0, 0.5 * scomp_ar + 0.5 * min(1.0, density_ar / 8.0)))
            rank_global = int(max(1, round(rank_min + comp * float(max(0, rank_max - rank_min)))))
        except Exception:
            rank_global = int(args.base_rank)

    # Generate base adapters (backend selectable)
    backend_used = str(args.gen_backend)
    try:
        if backend_used == "torch":
            adapters = generate_lora_from_embedding_torch(
                emb["z"],
                d_model=int(d_model),
                num_layers=int(num_layers),
                rank=int(rank_global),
                seed=int(args.seed),
                targets=selected_targets,
                target_shapes=target_shapes,
                target_weights=tw,
            )
            if bool(args.learn_bias):
                # Add zero bias vectors for downstream fine-tuning
                try:
                    for i in range(len(adapters.get("layers", []))):
                        for name, tensors in adapters["layers"][i].items():
                            if isinstance(tensors.get("A"), np.ndarray):
                                d_out = int(tensors["A"].shape[0])
                                tensors["bias"] = np.zeros((d_out,), dtype=np.float32)
                except Exception:
                    pass
        else:
            adapters = generate_lora_from_embedding(
                emb["z"],
                d_model=int(d_model),
                num_layers=int(num_layers),
                rank=int(rank_global),
                seed=int(args.seed),
                targets=selected_targets,
                target_shapes=target_shapes,
                target_weights=tw,
                learn_bias=bool(args.learn_bias),
            )
    except Exception:
        # Fallback to numpy backend on any error
        backend_used = "numpy"
        adapters = generate_lora_from_embedding(
            emb["z"],
            d_model=int(d_model),
            num_layers=int(num_layers),
            rank=int(rank_global),
            seed=int(args.seed),
            targets=selected_targets,
            target_shapes=target_shapes,
            target_weights=tw,
            learn_bias=bool(args.learn_bias),
        )

    # Optional capacity schedule per target (effective rank via zeroing)
    per_target_keep: Dict[str, int] = {}
    if bool(args.auto_rank) and rank_global > 0:
        base_frac: Dict[str, float] = {
            "o_proj": 1.00, "up_proj": 1.00, "down_proj": 0.90, "gate_proj": 0.80,
            "q_proj": 0.70, "k_proj": 0.65, "v_proj": 0.60,
        }
        comp_adj = max(0.85, min(1.15, 1.0 + 0.15 * (scomp_ar - 0.5)))
        for t in selected_targets:
            frac = base_frac.get(t, 0.8) * comp_adj
            keep = int(max(1, min(rank_global, round(rank_global * frac))))
            per_target_keep[t] = keep
        # MDL-style global budget: cap total params across layers/targets
        try:
            budget = int(max(0, int(args.mdl_budget_params)))
        except Exception:
            budget = 0
        if budget > 0:
            # estimate params per target per layer: keep * (d_out + d_in)
            def _shape_for(t: str) -> Tuple[int, int]:
                if target_shapes and t in target_shapes:
                    a, b = target_shapes[t]
                    return int(a), int(b)
                # fallback square
                return int(d_model), int(d_model)
            total = 0
            per_t_cost: Dict[str, int] = {}
            for t, k in per_target_keep.items():
                a, b = _shape_for(t)
                cost = int(max(1, k)) * int(max(1, a + b))
                per_t_cost[t] = cost
                total += cost
            total *= int(num_layers)
            if total > budget:
                scale = float(budget) / float(total)
                # rescale keeps proportionally, ensure at least 1
                for t in list(per_target_keep.keys()):
                    k = int(per_target_keep[t])
                    k2 = int(max(1, round(float(k) * scale)))
                    per_target_keep[t] = min(int(rank_global), k2)
        try:
            for i in range(len(adapters.get("layers", []))):
                for name, tensors in adapters["layers"][i].items():
                    keep = int(per_target_keep.get(name, rank_global))
                    A = tensors.get("A"); B = tensors.get("B")
                    if isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[1] >= rank_global:
                        if keep < rank_global:
                            A[:, keep:] = 0.0
                            tensors["A"] = A
                    if isinstance(B, np.ndarray) and B.ndim == 2 and B.shape[0] >= rank_global:
                        if keep < rank_global:
                            B[keep:, :] = 0.0
                            tensors["B"] = B
        except Exception:
            per_target_keep = {}

    # Optional zero-B initialization after generation
    if bool(args.zero_b):
        try:
            for i in range(len(adapters.get("layers", []))):
                for name, tensors in adapters["layers"][i].items():
                    if isinstance(tensors.get("B"), np.ndarray):
                        tensors["B"] = np.zeros_like(tensors["B"], dtype=np.float32)
        except Exception:
            pass

    # Optional rounding/sparsification
    try:
        mode = (args.round_mode or ("hard" if bool(args.round_lora) else "none"))
        if mode == "hard":
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
        elif mode == "soft":
            kp = float(max(0.0, args.round_soft_kp))
            axis = str(args.round_axis)
            keep_frac = max(0.0, min(100.0, kp)) / 100.0
            def _sparsify_topk(a: np.ndarray) -> np.ndarray:
                if a.size == 0 or keep_frac <= 0.0:
                    return np.zeros_like(a)
                if keep_frac >= 1.0:
                    return a
                if axis == "global":
                    k = int(np.ceil(keep_frac * a.size))
                    if k <= 0:
                        return np.zeros_like(a)
                    flat_idx = np.argpartition(np.abs(a).ravel(), -k)[-k:]
                    mask = np.zeros(a.size, dtype=bool)
                    mask[flat_idx] = True
                    return (a.ravel() * mask).reshape(a.shape)
                elif axis == "row":
                    rows, cols = a.shape
                    k = int(np.ceil(keep_frac * cols))
                    if k <= 0:
                        return np.zeros_like(a)
                    out = np.zeros_like(a)
                    for r in range(rows):
                        idx = np.argpartition(np.abs(a[r]), -k)[-k:]
                        out[r, idx] = a[r, idx]
                    return out
                else:  # col
                    rows, cols = a.shape
                    k = int(np.ceil(keep_frac * rows))
                    if k <= 0:
                        return np.zeros_like(a)
                    out = np.zeros_like(a)
                    for c in range(cols):
                        idx = np.argpartition(np.abs(a[:, c]), -k)[-k:]
                        out[idx, c] = a[idx, c]
                    return out
            for i in range(len(adapters.get("layers", []))):
                for name, tensors in adapters["layers"][i].items():
                    for key in ("A", "B"):
                        arr = tensors.get(key)
                        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                            continue
                        tensors[key] = _sparsify_topk(arr).astype(np.float32)
        # else: none -> no rounding
    except Exception:
        pass

    # Git metadata (best-effort)
    commit_sha: Optional[str] = None
    tree_sha: Optional[str] = None
    try:
        commit_sha = subprocess.check_output(["git", "-C", sources_root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit_sha = None
    try:
        tree_sha = subprocess.check_output(["git", "-C", sources_root, "rev-parse", "HEAD^{tree}"], text=True).strip()
    except Exception:
        tree_sha = None
    dirty_state = None
    try:
        # exit code 0 = clean; non-zero = dirty
        rc = subprocess.run(["git", "-C", sources_root, "diff-index", "--quiet", "HEAD", "--"], capture_output=True).returncode
        dirty_state = (rc != 0)
    except Exception:
        dirty_state = None

    # Sources path and count (ProgramGraph artifacts if available)
    sources_path = os.path.join(out_dir, "sources.jsonl")
    if pg is not None:
        try:
            sources_count = len(list(pg.artifacts("source")))
        except Exception:
            sources_count = 0
    else:
        sources_count = 0

    # Selection summary (base build = whole program)
    selection_summary = {"modules": None, "files": None}

    # Targets summary
    try:
        targets_list = list(adapters.get("targets", [])) if isinstance(adapters.get("targets"), list) else (list(target_shapes.keys()) if target_shapes else None)
    except Exception:
        targets_list = None

    manifest = {
        "sources_root": sources_root,
        "commit": commit_sha,
        "tree": tree_sha,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema_version": 1,
        "tool_version": "program_conditioned_adapter.build/1.0",
        "argv": list(sys.argv),
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "os": os.name,
        },
        "program_id": (getattr(pg, "program_id", None) if pg is not None else os.path.basename(sources_root)),
        "model": args.model,
        "embed_dim": int(args.embed_dim),
        "layers": int(num_layers),
        "d_model": int(d_model),
        "rank": int(rank_global),
        "targets": targets_list,
        "target_shapes": target_shapes,
        "include_text": bool(args.include_text),
        "graph_prop_hops": int(args.graph_prop_hops),
        "graph_prop_damp": float(args.graph_prop_damp),
        "target_weights": tw,
        "kbann_mode": kbann_mode,
        "contracts": {
            "require_citations": bool(args.contracts_require_citations),
            "retrieval_policy": (str(args.contracts_retrieval_policy) if args.contracts_retrieval_policy else None),
            "retrieval_temp": (float(args.contracts_retrieval_temp) if args.contracts_retrieval_temp is not None else None),
            "weight": float(max(0.0, args.contracts_weight)),
        },
        "round": {
            "mode": (args.round_mode or ("hard" if bool(args.round_lora) else "none")),
            "threshold": float(args.round_threshold),
            "soft_k_percent": float(args.round_soft_kp),
            "axis": str(args.round_axis),
        },
        "init": {
            "zero_b": bool(args.zero_b),
            "learn_bias": bool(args.learn_bias),
        },
        "selection": selection_summary,
        "seeds": {
            "python": int(args.seed),
            "numpy": int(args.seed),
            "torch": (int(args.seed) if torch_seeded else None),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        # code_graph metrics removed from core manifest; program examples may enrich
        "sources_file": sources_path,
        "sources_count": int(sources_count),
        "embedding_diagnostics": {
            "fallback_topology_used": bool('fallback_used' in locals() and fallback_used),
            "family_norms": (fam_norms if 'fam_norms' in locals() else None),
            "family_sparsity": (emb.get("sparsity") if isinstance(emb, dict) else None),
        },
        "git": {
            "commit": commit_sha,
            "tree": tree_sha,
            "dirty": dirty_state,
        },
        "auto_rank": {
            "enabled": bool(args.auto_rank),
            "rank_min": int(rank_min),
            "rank_max": int(rank_max),
            "rank_used": int(rank_global),
            "complexity": {"scomp": float(scomp_ar), "density": float(density_ar), "score": float(max(0.0, min(1.0, 0.5 * scomp_ar + 0.5 * min(1.0, density_ar / 8.0))))},
            "per_target_keep": per_target_keep or None,
            "mdl_budget_params": (int(args.mdl_budget_params) if args.mdl_budget_params else 0),
        },
        "caches": {
            "symbol_index": None,
            "windows_index": None,
            "facts": None,
            "rerank_features": None,
            "self_queries": None,
        },
    }
    # Record dcpo capability and program state path (if requested)
    if bool(getattr(args, "init_program_state", False)):
        try:
            manifest["dcpo"] = {"enabled": True}
            psp = getattr(args, "program_state_path", None)
            if psp:
                manifest["dcpo"]["program_state_path"] = os.path.abspath(psp)
        except Exception:
            pass
    save_npz(out_dir, embedding=emb, adapters=adapters, manifest=manifest)

    # Emit caches next to adapters: symbol_index.jsonl, windows_index.jsonl, facts.jsonl, rerank_features.npz, self_queries.jsonl
    # symbol_index.jsonl
    try:
        if pg is not None:
            sym_path = os.path.join(out_dir, "symbol_index.jsonl")
            with open(sym_path, "w", encoding="utf-8") as sf:
                count = 0
                for e in pg.entities():
                    if e.kind not in ("function", "class", "module"):
                        continue
                    rec = {"uri": e.uri, "id": e.id, "name": e.name, "kind": e.kind, "owner": e.owner}
                    sf.write(json.dumps(rec) + "\n")
                    count += 1
                    if count >= 50000:
                        break
            try:
                mf = os.path.join(out_dir, "manifest.json")
                obj = json.loads(open(mf, "r", encoding="utf-8").read())
                with open(sym_path, "rb") as rf:
                    sym_sha = hashlib.sha256(rf.read()).hexdigest()
                obj.setdefault("caches", {})
                obj["caches"]["symbol_index"] = {"path": sym_path, "sha256": sym_sha}
                open(mf, "w", encoding="utf-8").write(json.dumps(obj, indent=2))
            except Exception:
                pass
    except Exception:
        pass

    # windows_index.jsonl (function-first spans)
    try:
        if pg is not None:
            win_path = os.path.join(out_dir, "windows_index.jsonl")
            with open(win_path, "w", encoding="utf-8") as wf:
                for e in pg.entities():
                    if e.kind not in ("function", "class"):
                        continue
                    try:
                        ra = pg.resolve(e.uri)
                    except Exception:
                        continue
                    rec = {
                        "symbol": e.id,
                        "uri": e.uri,
                        "path": ra.artifact_uri.split("/artifact/", 1)[-1],
                        "start_line": int(ra.span.start_line),
                        "end_line": int(ra.span.end_line),
                        "span_sha256": ra.hash,
                    }
                    wf.write(json.dumps(rec) + "\n")
            try:
                mf = os.path.join(out_dir, "manifest.json")
                obj = json.loads(open(mf, "r", encoding="utf-8").read())
                with open(win_path, "rb") as rf:
                    win_sha = hashlib.sha256(rf.read()).hexdigest()
                obj.setdefault("caches", {})
                obj["caches"]["windows_index"] = {"path": win_path, "sha256": win_sha}
                open(mf, "w", encoding="utf-8").write(json.dumps(obj, indent=2))
            except Exception:
                pass
    except Exception:
        pass

    # facts.jsonl (entity definitions anchored to artifacts)
    try:
        if pg is not None:
            facts_path = os.path.join(out_dir, "facts.jsonl")
            pid = getattr(pg, "program_id", os.path.basename(sources_root))
            with open(facts_path, "w", encoding="utf-8") as ff:
                for e in pg.entities():
                    try:
                        ra = pg.resolve(e.uri)
                    except Exception:
                        continue
                    fact = {
                        "uri": e.uri,
                        "program_id": pid,
                        "entity": {"id": e.id, "kind": e.kind, "name": e.name, "owner": e.owner},
                        "artifact_uri": ra.artifact_uri,
                        "span": {"start": {"line": ra.span.start_line}, "end": {"line": ra.span.end_line}},
                        "artifact_hash": ra.hash,
                    }
                    ff.write(json.dumps(fact) + "\n")
            try:
                mf = os.path.join(out_dir, "manifest.json")
                obj = json.loads(open(mf, "r", encoding="utf-8").read())
                with open(facts_path, "rb") as rf:
                    facts_sha = hashlib.sha256(rf.read()).hexdigest()
                obj.setdefault("caches", {})
                obj["caches"]["facts"] = {"path": facts_path, "sha256": facts_sha}
                open(mf, "w", encoding="utf-8").write(json.dumps(obj, indent=2))
            except Exception:
                pass
    except Exception:
        pass

    # rerank_features.npz (light placeholder features)
    try:
        rr_path = os.path.join(out_dir, "rerank_features.npz")
        feats = {
            "bm25": np.zeros((1,), dtype=np.float32),
            "graph": np.zeros((1,), dtype=np.float32),
        }
        np.savez_compressed(rr_path, **feats)
        try:
            mf = os.path.join(out_dir, "manifest.json")
            obj = json.loads(open(mf, "r", encoding="utf-8").read())
            with open(rr_path, "rb") as rf:
                rr_sha = hashlib.sha256(rf.read()).hexdigest()
            obj.setdefault("caches", {})
            obj["caches"]["rerank_features"] = {"path": rr_path, "sha256": rr_sha}
            open(mf, "w", encoding="utf-8").write(json.dumps(obj, indent=2))
        except Exception:
            pass
    except Exception:
        pass

    # self_queries.jsonl (empty scaffold by default)
    try:
        sq_path = os.path.join(out_dir, "self_queries.jsonl")
        if not os.path.exists(sq_path):
            open(sq_path, "w", encoding="utf-8").write("")
        try:
            mf = os.path.join(out_dir, "manifest.json")
            obj = json.loads(open(mf, "r", encoding="utf-8").read())
            with open(sq_path, "rb") as rf:
                sq_sha = hashlib.sha256(rf.read()).hexdigest()
            obj.setdefault("caches", {})
            obj["caches"]["self_queries"] = {"path": sq_path, "sha256": sq_sha}
            open(mf, "w", encoding="utf-8").write(json.dumps(obj, indent=2))
        except Exception:
            pass
    except Exception:
        pass
    # Also export a JSONL of program sources (path, sha256, bytes)
    try:
        if pg is not None:
            with open(sources_path, "w", encoding="utf-8") as sf:
                for art in pg.artifacts("source"):
                    try:
                        rel = art.uri.split("/artifact/", 1)[-1]
                    except Exception:
                        rel = None
                    try:
                        abs_fp = os.path.abspath(os.path.join(sources_root, rel)) if rel else None
                        if abs_fp and os.path.isfile(abs_fp):
                            with open(abs_fp, "rb") as rf:
                                raw = rf.read()
                            h = hashlib.sha256(raw).hexdigest()
                            size = len(raw)
                        else:
                            h = ""
                            size = 0
                    except Exception:
                        h = ""
                        size = 0
                    rec = {"path": (rel.replace("\\", "/") if rel else None), "sha256": (getattr(art, "hash", "") or h), "bytes": int(size)}
                    sf.write(json.dumps(rec) + "\n")
    except Exception:
        pass

    # Export a lightweight identifiers table
    try:
        ids_path = os.path.join(out_dir, "identifiers.jsonl")
        with open(ids_path, "w", encoding="utf-8") as idf:
            count = 0
            # Use ProgramGraph entities for portability
            pg_ids = pg
            if pg_ids is not None:
                for e in pg_ids.entities():
                    try:
                        rec = {
                            "symbol": str(e.id),
                            "name": str(e.name),
                            "module": str(e.owner or ""),
                            "file": None,
                            "signature": None,
                            "kind": e.kind,
                        }
                        idf.write(json.dumps(rec) + "\n")
                        count += 1
                        if count >= 20000:
                            break
                    except Exception:
                        continue
        # Patch manifest with identifiers path
        try:
            mf = os.path.join(out_dir, "manifest.json")
            obj = json.loads(open(mf, "r", encoding="utf-8").read())
            obj["identifiers_file"] = ids_path
            open(mf, "w", encoding="utf-8").write(json.dumps(obj, indent=2))
        except Exception:
            pass
    except Exception:
        pass

    # Initialize or update ProgramState if requested
    try:
        if bool(getattr(args, "init_program_state", False)):
            # Prefer explicit path; else default to <sources_root>/.program_state.json
            psp = getattr(args, "program_state_path", None)
            if psp:
                ps_path = os.path.abspath(psp)
            else:
                ps_path = os.path.abspath(os.path.join(sources_root, ".program_state.json"))
            # Load latest manifest (with caches patched)
            try:
                mf_path = os.path.join(out_dir, "manifest.json")
                mf_obj = json.loads(open(mf_path, "r", encoding="utf-8").read())
            except Exception:
                mf_obj = manifest
            # Minimal ProgramState snapshot
            prog_state = {
                "schema_version": 1,
                "program_id": mf_obj.get("program_id"),
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "sources_root": sources_root,
                "adapters_dir": out_dir,
                "manifest_path": os.path.join(out_dir, "manifest.json"),
                "caches": mf_obj.get("caches"),
                "embedding_diagnostics": mf_obj.get("embedding_diagnostics"),
                "git": mf_obj.get("git"),
                "selection": mf_obj.get("selection"),
            }
            os.makedirs(os.path.dirname(ps_path), exist_ok=True)
            with open(ps_path, "w", encoding="utf-8") as pf:
                pf.write(json.dumps(prog_state, indent=2))
    except Exception:
        pass

    if args.verbose:
        print(json.dumps({"status": "ok", "adapters": os.path.join(out_dir, "adapters.npz"), "embedding": os.path.join(out_dir, "embedding.npz")}, indent=2))

    # Per-module export (default-on unless explicitly disabled)
    if not bool(args.no_per_module):
        sub_root = os.path.join(out_dir, "sub_adapters")
        os.makedirs(sub_root, exist_ok=True)
        # Shapes again for export
        shapes = target_shapes
        if not shapes:
            shapes = detect_target_shapes_from_model(args.model)

        # Files-only sub-adapter if requested
        try:
            if bool(args.files_only) and (args.files_allowlist is not None):
                # Normalize file paths to absolute
                inc_files_abs: List[str] = []
                for f in [s for s in (args.files_allowlist or []) if s]:
                    pth = f
                    if not os.path.isabs(pth):
                        pth = os.path.join(sources_root, pth)
                    inc_files_abs.append(os.path.abspath(pth))
                # Subgraph embedding restricted to allowlisted files via ProgramGraph
                emb_f = build_subgraph_embedding_from_program(
                    pg if pg is not None else None,  # type: ignore[arg-type]
                    sources_root=sources_root,
                    dim=int(args.embed_dim),
                    seed=int(args.seed),
                    include_owners=None,
                    include_artifact_paths=inc_files_abs,
                    include_text=bool(args.include_text),
                    text_max_bytes=int(args.text_max_bytes),
                    max_text_tokens=int(args.max_text_tokens),
                    text_weight=float(args.text_weight),
                    calls_weight=0.25,
                    types_weight=0.20,
                    tests_weight=0.15,
                    graph_prop_hops=int(args.graph_prop_hops),
                    graph_prop_damp=float(args.graph_prop_damp),
                )
                # Generate
                if backend_used == "torch":
                    adapters_f = generate_lora_from_embedding_torch(
                        emb_f["z"],
                        d_model=int(d_model),
                        num_layers=int(num_layers),
                        rank=int(args.sub_rank),
                        seed=int(args.seed),
                        targets=(list(shapes.keys()) if shapes else None),
                        target_shapes=shapes,
                        target_weights=None,
                    )
                    if bool(args.learn_bias):
                        for i in range(len(adapters_f.get("layers", []))):
                            for name, tensors in adapters_f["layers"][i].items():
                                if isinstance(tensors.get("A"), np.ndarray):
                                    d_out = int(tensors["A"].shape[0])
                                    tensors["bias"] = np.zeros((d_out,), dtype=np.float32)
                else:
                    adapters_f = generate_lora_from_embedding(
                        emb_f["z"],
                        d_model=int(d_model),
                        num_layers=int(num_layers),
                        rank=int(args.sub_rank),
                        seed=int(args.seed),
                        targets=(list(shapes.keys()) if shapes else None),
                        target_shapes=shapes,
                        target_weights=None,
                        learn_bias=bool(args.learn_bias),
                    )
                sub_dir_f = os.path.join(sub_root, "files_only")
                os.makedirs(sub_dir_f, exist_ok=True)
                # Normalize allowlist to sources-root-relative for manifest
                inc_files_rel = []
                for pth in inc_files_abs:
                    try:
                        inc_files_rel.append(os.path.relpath(pth, sources_root))
                    except Exception:
                        inc_files_rel.append(pth)
                save_npz(
                    sub_dir_f,
                    embedding=emb_f,
                    adapters=adapters_f,
                    manifest={
                        "mode": "files_only",
                        "include_files": inc_files_rel,
                        "rank": int(args.sub_rank),
                        "d_model": int(d_model),
                        "layers": int(num_layers),
                    },
                )
        except Exception:
            pass
        # Per-module export: iterate modules via ProgramGraph entities owners
        modules_all = []
        if pg is not None:
            try:
                modules_all = sorted({e.name for e in pg.entities() if e.kind == "module"})
            except Exception:
                modules_all = []
        for module in modules_all:
            # Skip obvious tests by name
            if module.endswith(".tests") or module.endswith("tests"):
                continue
            mods = [module]
            # Dependency inclusion not available via ProgramGraph yet; keep single module
            mods = sorted({m for m in mods})
            emb_m = build_subgraph_embedding_from_program(
                pg if pg is not None else None,  # type: ignore[arg-type]
                sources_root=sources_root,
                dim=int(args.embed_dim),
                seed=int(args.seed),
                include_owners=mods,
                include_artifact_paths=None,
                include_text=bool(args.include_text),
                text_max_bytes=int(args.text_max_bytes),
                max_text_tokens=int(args.max_text_tokens),
                text_weight=float(args.text_weight),
                calls_weight=0.25,
                types_weight=0.20,
                tests_weight=0.15,
                graph_prop_hops=int(args.graph_prop_hops),
                graph_prop_damp=float(args.graph_prop_damp),
            )
            try:
                if backend_used == "torch":
                    adapters_m = generate_lora_from_embedding_torch(
                        emb_m["z"],
                        d_model=int(d_model),
                        num_layers=int(num_layers),
                        rank=int(args.sub_rank),
                        seed=int(args.seed),
                        targets=(list(shapes.keys()) if shapes else None),
                        target_shapes=shapes,
                        target_weights=None,
                    )
                    if bool(args.learn_bias):
                        for i in range(len(adapters_m.get("layers", []))):
                            for name, tensors in adapters_m["layers"][i].items():
                                if isinstance(tensors.get("A"), np.ndarray):
                                    d_out = int(tensors["A"].shape[0])
                                    tensors["bias"] = np.zeros((d_out,), dtype=np.float32)
                else:
                    adapters_m = generate_lora_from_embedding(
                        emb_m["z"],
                        d_model=int(d_model),
                        num_layers=int(num_layers),
                        rank=int(args.sub_rank),
                        seed=int(args.seed),
                        targets=(list(shapes.keys()) if shapes else None),
                        target_shapes=shapes,
                        target_weights=None,
                        learn_bias=bool(args.learn_bias),
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


if __name__ == "__main__":
    main()


