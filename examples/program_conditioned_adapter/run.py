from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import subprocess
from typing import Dict, Any
import json
import os as _os
import importlib

import numpy as np
from model.hf_snapshot import ensure_snapshot
from examples.program_conditioned_adapter.modules.retrieval_policy import RetrievalPolicy  # type: ignore
from examples.program_conditioned_adapter.modules.runner_core import select_region, prepare_citations  # type: ignore

def _root() -> Path:
    return Path(__file__).resolve().parents[2]



def main() -> None:
    p = argparse.ArgumentParser()
    # Model & sources
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--sources", default=str(_root()), help="Program sources root (path or URI)")
    p.add_argument("--program", default=None, help="Alias for --sources (program root path or URI)")
    p.add_argument("--prompt", default=(
        "Explain how generation works for this program. Cite path:line for each claim."
    ))

    # PCA program options (program-agnostic; backend not enforced here)
    p.add_argument("--backend", default=None, help="Program backend id (informational; use --pg-backend to enable ProgramGraph features)")
    p.add_argument("--retrieval-policy", dest="retrieval_policy", default=None, help='Retrieval mix like "sim:0.6,struct:0.4"')
    p.add_argument("--retrieval-temp", dest="retrieval_temp", type=float, default=None)
    p.add_argument("--use-cache", action="store_true", help="Prefer caches when available (symbol/windows/facts)")
    p.add_argument("--citations-enforce", action="store_true", help="Require citations policy in PCA layer")
    p.add_argument("--citations-repair", action="store_true")
    p.add_argument("--pg-backend", default=None, help="Dotted path to ProgramGraph factory, e.g. 'examples.program_conditioned_adapter.examples.python_repo_grounded_qa.python_repo_graph:PythonRepoGraph'")
    p.add_argument("--program-state", default=None, help="Optional path to .program_state.json (defaults to <adapters_dir>/.program_state.json if present)")

    # Base adapters generation
    p.add_argument("--adapters-dir", default=None, help="Directory to save/find base adapters (defaults under examples/.../artifacts/base_adapters)")
    p.add_argument("--base-rank", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=1536)
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--text-max-bytes", type=int, default=0)
    p.add_argument("--max-text-tokens", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    # Runner mixing and capacity
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--rank", type=int, default=12)
    p.add_argument("--gsub", type=float, default=0.75)
    p.add_argument("--mix-beta", type=float, default=0.1)
    p.add_argument("--target-weights", default=None, help="CSV like q_proj=1,o_proj=1.1,up_proj=1.1,down_proj=1.05")
    p.add_argument("--knowledge-preset", action="store_true", help="Use a preset of target weights tuned for knowledge recall (boost o/up/down; modest q/k/v)")
    p.add_argument("--code-recall-preset", action="store_true", help="Opt-in preset tuned for code recall (o,v,up,down,gate emphasized; light q,k)")
    p.add_argument("--delta-cap", type=float, default=0.05, help="AB-norm clipping cap relative to base weight norm per layer (0 disables)")
    # Entropy-aware capacity (forward to enhanced runner)
    p.add_argument("--entropy-aware", action="store_true")
    p.add_argument("--rank-min", type=int, default=8)
    p.add_argument("--rank-max", type=int, default=32)
    p.add_argument("--gsub-min", type=float, default=0.6)
    p.add_argument("--gsub-max", type=float, default=0.9)
    p.add_argument("--entropy-weights", default="program=0.4,subgraph=0.4,question=0.2")

    # Selection & context packing
    p.add_argument("--of-sources", choices=["question", "zoom"], default="question")
    p.add_argument("--zoom-symbol", default=None)
    p.add_argument("--zoom-radius", type=int, default=0)
    p.add_argument("--pack-context", action="store_true")
    p.add_argument("--pack-mode", choices=["heads", "windows"], default="heads")
    p.add_argument("--context-tokens", type=int, default=3000)
    p.add_argument("--require-citations", action="store_true")
    p.add_argument("--citations-per-paragraph", action="store_true")
    p.add_argument("--function-first", action="store_true")
    p.add_argument("--ff-max-candidates", type=int, default=24)
    p.add_argument("--ff-window-lines", type=int, default=80)
    p.add_argument("--ff-threshold", type=float, default=0.55)
    p.add_argument("--ff-noise-penalty", type=float, default=0.30)
    # Layer schedule and q-aware weights (opt-in)
    p.add_argument("--layer-schedule", action="store_true", help="Enable a light per-layer multiplier rising toward top third (additive; default off)")
    p.add_argument("--q-aware-weights", action="store_true", help="Heuristic reweighting of targets by question intent (additive; default off)")
    p.add_argument("--per-target-rank-schedule", action="store_true", help="Trim per-target effective rank at run-time (additive; default off)")
    p.add_argument("--rank-budget", type=int, default=0, help="Optional per-layer rank budget across targets; rescale keeps to meet budget (0 disables)")
    p.add_argument("--ablate-attn", action="store_true", help="Ablate attention targets (q/k/v/o) by zeroing their weights")
    p.add_argument("--ablate-mlp", action="store_true", help="Ablate MLP targets (up/down/gate) by zeroing their weights")
    p.add_argument("--layer-rank-tiers", action="store_true", help="Use top/mid/low thirds per-layer rank keeps by target group (opt-in)")
    # Mixture bank (opt-in)
    p.add_argument("--mixture-m", type=int, default=0, help="Top-m subgraph adapters from bank to mix (0 disables)")
    p.add_argument("--adapters-bank", default=None, help="Path to a bank of sub_adapters (from build --per-module)")
    # Adapter mapping is always enhanced; no CLI toggle
    # Alpha warmup and decoding hooks (opt-in)
    p.add_argument("--alpha-warmup", action="store_true", help="Use a lighter alpha on first attempt (or first structured pass), then full alpha on retry/subsequent passes")
    p.add_argument("--adapter-aware-decoding", action="store_true", help="Slightly relax sampling and prompt pointer-first when citations are required")
    # Telemetry verification (opt-in)
    p.add_argument("--telemetry-tests", action="store_true", help="Attempt simple verify_with_tests() on a few selected modules and record results (structured path only)")
    # Reranking
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--self-queries", default=None, help="Path to self_queries.jsonl for retrieval boosts")

    # Sampling controls
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--min-new-tokens", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=512)
    # Local generation/perf controls
    p.add_argument("--kv-window", type=int, default=0, help="Optional sliding KV window length; 0 disables")
    p.add_argument("--head-device", choices=["same", "cpu", "auto"], default="same", help="Place lm_head on cpu/auto to save VRAM")

    # Device & telemetry
    p.add_argument("--device-map", default="auto", choices=["auto", "none"])
    p.add_argument("--gpu-ids", default=None)
    p.add_argument("--max-memory", default=None)
    p.add_argument("--telemetry-out", default=None)
    # DCPO/Structured controls
    p.add_argument("--structured", action="store_true")
    p.add_argument("--lfp-iters", type=int, default=1)
    p.add_argument("--budget-H", type=float, default=0.0)
    p.add_argument("--monotone-selection", action="store_true")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--cone-join", choices=["concat", "weighted"], default="concat")
    p.add_argument("--cache-dir", default="/data/transformer_10/checkpoints", help="Cache directory for HF models/tokenizers")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-adapters", action="store_true", help="Disable applying adapters in the enhanced runner")
    p.add_argument("--commit-footer", action="store_true", help="Append 'answer valid for commit X' footer")
    args = p.parse_args()

    # Normalize program root (prefer --sources; allow --program alias)
    src_arg = args.sources
    if getattr(args, "program", None):
        src_arg = args.program if args.program else src_arg
    program_root = Path(src_arg)
    # Example directory for artifacts colocated with this run.py
    example_dir = Path(__file__).resolve().parent
    artifacts = example_dir / "artifacts"
    base_dir = Path(args.adapters_dir) if args.adapters_dir else (artifacts / "base_adapters")
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    adapters_npz = base_dir / "adapters.npz"
    manifest_path = base_dir / "manifest.json"
    # Resolve program state path
    ps_path = None
    if args.program_state:
        ps_path = Path(args.program_state)
    else:
        default_ps = base_dir / ".program_state.json"
        if default_ps.exists():
            ps_path = default_ps

    # Generate base adapters if not present by delegating to build.py (program-agnostic)
    if not adapters_npz.exists():
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "build.py"),
            "--sources", str(program_root),
            "--model", str(args.model),
            "--adapters-dir", str(base_dir),
            "--embed-dim", str(int(args.embed_dim)),
            "--seed", str(int(args.seed)),
        ]
        if bool(args.include_text):
            cmd.append("--include-text")
        if int(args.text_max_bytes) > 0:
            cmd += ["--text-max-bytes", str(int(args.text_max_bytes))]
        if int(args.max_text_tokens) > 0:
            cmd += ["--max-text-tokens", str(int(args.max_text_tokens))]
        if bool(args.knowledge_preset):
            cmd.append("--knowledge-preset")
        if bool(args.code_recall_preset):
            cmd.append("--code-recall-preset")
        subprocess.check_call(cmd)

    # Run directly via orchestrator
    from examples.program_conditioned_adapter.modules.runner import generate_answer
    from examples.program_conditioned_adapter.modules.runner import generate_answer_structured

    if args.gpu_ids and str(args.gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ProgramGraph + RetrievalPolicy selection (seed zoom) if requested
    zoom_seed = None
    try:
        if args.pg_backend:
            # Load PG factory dynamically only when explicitly provided
            def _load_symbol(path: str):
                mod, _, attr = path.partition(":")
                m = importlib.import_module(mod)
                return getattr(m, attr)
            pg_ctor = _load_symbol(args.pg_backend)
            pg = pg_ctor(str(program_root), ignore=None)
            pol = RetrievalPolicy.from_spec(args.retrieval_policy, temp=args.retrieval_temp)
            region_ids = select_region(args.prompt, pg, pol, top_k=8)
            ents_by_id = {e.id: e for e in pg.entities()}
            names = []
            for eid in region_ids[:4]:
                e = ents_by_id.get(eid)
                if e and e.name:
                    names.append(e.name)
            if names:
                zoom_seed = ",".join(names)
    except Exception:
        zoom_seed = None

    if bool(args.structured):
        res = generate_answer_structured(
            model_id=args.model,
            adapters_npz=str(adapters_npz),
            program_root=str(program_root),
            delta_cap=float(max(0.0, args.delta_cap)),
            prompt=args.prompt,
            cache_dir=str(cache_dir),
            device_map=str(args.device_map),
            alpha=float(args.alpha),
            rank=int(args.rank),
            gsub=float(args.gsub),
            beta=float(args.mix_beta),
            of_sources=("zoom" if (zoom_seed and not args.zoom_symbol) else args.of_sources),
            zoom_symbol=(args.zoom_symbol or (zoom_seed or None)),
            zoom_radius=int(args.zoom_radius),
            pack_context=bool(args.pack_context),
            pack_mode=args.pack_mode,
            context_tokens=int(args.context_tokens),
            require_citations=bool(args.require_citations),
            citations_per_paragraph=bool(args.citations_per_paragraph),
            function_first=bool(args.function_first),
            ff_max_candidates=int(args.ff_max_candidates),
            ff_window_lines=int(args.ff_window_lines),
            ff_threshold=float(args.ff_threshold),
            ff_noise_penalty=float(args.ff_noise_penalty),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            min_new_tokens=int(args.min_new_tokens),
            max_new_tokens=int(args.max_new_tokens),
            kv_window=int(args.kv_window),
            head_device=str(args.head_device),
            seed=int(args.seed),
            entropy_aware=bool(args.entropy_aware),
            rank_min=int(args.rank_min),
            rank_max=int(args.rank_max),
            gsub_min=float(args.gsub_min),
            gsub_max=float(args.gsub_max),
            entropy_weights=str(args.entropy_weights),
            target_weights=(str(args.target_weights) if args.target_weights else (
                "o_proj=1.15,v_proj=1.10,up_proj=1.10,down_proj=1.05,gate_proj=1.00,q_proj=0.95,k_proj=0.90" if args.code_recall_preset else (
                    "q_proj=0.95,k_proj=0.95,v_proj=0.95,o_proj=1.10,up_proj=1.10,down_proj=1.05" if args.knowledge_preset else None
                )
            )),
            rerank=bool(args.rerank),
            self_queries_path=(str(args.self_queries) if args.self_queries else None),
            commit_footer=bool(args.commit_footer),
            verbose=bool(args.verbose),
            lfp_iters=int(args.lfp_iters),
            budget_H=float(args.budget_H),
            monotone_selection=bool(args.monotone_selection),
            program_state_path=(str(ps_path) if ps_path else None),
            samples=int(args.samples),
            cone_join=str(args.cone_join),
            telemetry_out=(str(args.telemetry_out) if args.telemetry_out else None),
            layer_schedule=bool(args.layer_schedule),
            q_aware_weights=bool(args.q_aware_weights),
            mixture_m=int(args.mixture_m),
            adapters_bank=(str(args.adapters_bank) if args.adapters_bank else None),
            # Forward additional knobs to structured path
            per_target_rank_schedule=bool(args.per_target_rank_schedule),
            rank_budget=int(args.rank_budget),
            ablate_attn=bool(args.ablate_attn),
            ablate_mlp=bool(args.ablate_mlp),
            alpha_warmup=bool(args.alpha_warmup),
            adapter_aware_decoding=bool(args.adapter_aware_decoding),
            layer_rank_tiers=bool(args.layer_rank_tiers),
            telemetry_verify_tests=bool(args.telemetry_tests),
        )
        # PCA evidence/provenance stamping when enabled
        try:
            if (args.citations_enforce or args.citations_per_paragraph or args.citations_repair) and args.pg_backend:
                # Load manifest (for provenance)
                mf_obj: Dict[str, Any] = {}
                try:
                    if os.path.exists(manifest_path):
                        mf_obj = json.loads(open(manifest_path, "r", encoding="utf-8").read())
                except Exception:
                    mf_obj = {}
                # Minimal unit
                unit = {"text": res.get("text", ""), "evidence": []}
                citations_policy = {
                    "enforce": bool(args.citations_enforce),
                    "per_paragraph": bool(args.citations_per_paragraph),
                    "repair": bool(args.citations_repair),
                }
                # Build ProgramGraph for stamping when provided
                def _load_symbol2(path: str):
                    mod, _, attr = path.partition(":")
                    m = importlib.import_module(mod)
                    return getattr(m, attr)
                pg_ctor2 = _load_symbol2(args.pg_backend)
                pg2 = pg_ctor2(str(program_root), ignore=None)
                pol2 = RetrievalPolicy.from_spec(args.retrieval_policy, temp=args.retrieval_temp)
                region_ids2 = select_region(args.prompt, pg2, pol2, top_k=8)
                stamped = prepare_citations([unit], region_ids2, pg2, citations_policy=citations_policy, manifest=mf_obj)
                if stamped:
                    res["evidence"] = stamped[0].get("evidence", [])
                    res["provenance"] = stamped[0].get("provenance", res.get("provenance"))
        except Exception:
            pass
        print(res.get("text", ""))
        if args.verbose:
            try:
                import json as _json
                print(_json.dumps({
                    "must": len(res.get("must", [])),
                    "may": len(res.get("may", [])),
                    "lfp_passes": res.get("lfp_passes"),
                    "converged": res.get("converged"),
                    "confidence": res.get("confidence"),
                }, indent=2))
            except Exception:
                pass
    else:
        text = generate_answer(
            model_id=args.model,
            adapters_npz=str(adapters_npz),
            program_root=str(program_root),
            delta_cap=float(max(0.0, args.delta_cap)),
            prompt=args.prompt,
            cache_dir=str(cache_dir),
            device_map=str(args.device_map),
            alpha=float(args.alpha),
            rank=int(args.rank),
            gsub=float(args.gsub),
            beta=float(args.mix_beta),
            of_sources=("zoom" if (zoom_seed and not args.zoom_symbol) else args.of_sources),
            zoom_symbol=(args.zoom_symbol or (zoom_seed or None)),
            zoom_radius=int(args.zoom_radius),
            pack_context=bool(args.pack_context),
            pack_mode=args.pack_mode,
            context_tokens=int(args.context_tokens),
            require_citations=bool(args.require_citations),
            citations_per_paragraph=bool(args.citations_per_paragraph),
            function_first=bool(args.function_first),
            ff_max_candidates=int(args.ff_max_candidates),
            ff_window_lines=int(args.ff_window_lines),
            ff_threshold=float(args.ff_threshold),
            ff_noise_penalty=float(args.ff_noise_penalty),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            min_new_tokens=int(args.min_new_tokens),
            max_new_tokens=int(args.max_new_tokens),
            kv_window=int(args.kv_window),
            head_device=str(args.head_device),
            seed=int(args.seed),
            entropy_aware=bool(args.entropy_aware),
            rank_min=int(args.rank_min),
            rank_max=int(args.rank_max),
            gsub_min=float(args.gsub_min),
            gsub_max=float(args.gsub_max),
            entropy_weights=str(args.entropy_weights),
            target_weights=(str(args.target_weights) if args.target_weights else (
                "o_proj=1.15,v_proj=1.10,up_proj=1.10,down_proj=1.05,gate_proj=1.00,q_proj=0.95,k_proj=0.90" if args.code_recall_preset else (
                    "q_proj=0.95,k_proj=0.95,v_proj=0.95,o_proj=1.10,up_proj=1.10,down_proj=1.05" if args.knowledge_preset else None
                )
            )),
            rerank=bool(args.rerank),
            self_queries_path=(str(args.self_queries) if args.self_queries else None),
            commit_footer=bool(args.commit_footer),
            verbose=bool(args.verbose),
            monotone_selection=bool(args.monotone_selection),
            layer_schedule=bool(args.layer_schedule),
            q_aware_weights=bool(args.q_aware_weights),
            mixture_m=int(args.mixture_m),
            adapters_bank=(str(args.adapters_bank) if args.adapters_bank else None),
            per_target_rank_schedule=bool(args.per_target_rank_schedule),
            rank_budget=int(args.rank_budget),
            ablate_attn=bool(args.ablate_attn),
            ablate_mlp=bool(args.ablate_mlp),
            # map mode hook reserved for structured path
            alpha_warmup=bool(args.alpha_warmup),
            adapter_aware_decoding=bool(args.adapter_aware_decoding),
            layer_rank_tiers=bool(args.layer_rank_tiers),
        )
        print(text)


if __name__ == "__main__":
    main()


