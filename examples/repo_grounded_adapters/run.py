from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import subprocess
from typing import Dict
import json
import os as _os

import numpy as np

from examples.repo_grounded_adapters.modules.embedding import build_repo_embedding
from examples.repo_grounded_adapters.modules.adapter import generate_lora_from_embedding, save_npz
from blocks.inspect import infer_target_shapes_from_config as _infer_shapes_unused  # legacy; avoid transformers
from model.hf_snapshot import ensure_snapshot

def _root() -> Path:
    return Path(__file__).resolve().parents[2]



def main() -> None:
    p = argparse.ArgumentParser()
    # Model & repo
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--repo", default=str(_root()))
    p.add_argument("--prompt", default=(
        "Explain how generation works in this repo. Cite path:line for each claim."
    ))

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
    # Entropy-aware capacity (forward to enhanced runner)
    p.add_argument("--entropy-aware", action="store_true")
    p.add_argument("--rank-min", type=int, default=8)
    p.add_argument("--rank-max", type=int, default=32)
    p.add_argument("--gsub-min", type=float, default=0.6)
    p.add_argument("--gsub-max", type=float, default=0.9)
    p.add_argument("--entropy-weights", default="repo=0.4,subgraph=0.4,question=0.2")

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

    root = Path(args.repo)
    ex_dir = root / "examples" / "repo_grounded_adapters"
    artifacts = ex_dir / "artifacts"
    base_dir = Path(args.adapters_dir) if args.adapters_dir else (artifacts / "base_adapters")
    base_dir.mkdir(parents=True, exist_ok=True)
    proj_root = _root()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    adapters_npz = base_dir / "adapters.npz"

    # Generate base adapters for the HF model dims if not present
    if not adapters_npz.exists():
        emb = build_repo_embedding(
            str(root),
            dim=int(args.embed_dim),
            seed=int(args.seed),
            include_text=bool(args.include_text),
            text_max_bytes=int(args.text_max_bytes),
            max_text_tokens=int(args.max_text_tokens),
        )
        # Infer shapes and dims from local snapshot config.json (no transformers)
        snap_dir = ensure_snapshot(args.model, str(cache_dir))
        cfg_path = _os.path.join(snap_dir, "config.json")
        cfg_obj = json.load(open(cfg_path, "r", encoding="utf-8"))
        d_model = int(cfg_obj.get("hidden_size", 4096))
        num_layers = int(cfg_obj.get("num_hidden_layers", 32))
        n_heads = int(cfg_obj.get("num_attention_heads", 32))
        n_kv_heads = int(cfg_obj.get("num_key_value_heads", n_heads))
        d_ff = int(cfg_obj.get("intermediate_size", 11008))
        head_dim = int(cfg_obj.get("head_dim", d_model // max(1, n_heads)))
        shapes: Dict[str, tuple[int, int]] = {
            "q_proj": (n_heads * head_dim, d_model),
            "k_proj": (n_kv_heads * head_dim, d_model),
            "v_proj": (n_kv_heads * head_dim, d_model),
            "o_proj": (d_model, n_heads * head_dim),
            "up_proj": (d_ff, d_model),
            "gate_proj": (d_ff, d_model),
            "down_proj": (d_model, d_ff),
        }
        # Parse or synthesize target weights (knowledge preset if requested and none provided)
        def _parse_tw(spec: str | None) -> dict[str, float] | None:
            if not spec:
                return None
            out: dict[str, float] = {}
            try:
                parts = [p.strip() for p in str(spec).split(",") if p.strip()]
                for p in parts:
                    if "=" not in p:
                        out[p] = 1.0
                        continue
                    k, v = p.split("=", 1)
                    out[k.strip()] = float(v)
            except Exception:
                return None
            return out or None
        tw_spec = (args.target_weights if args.target_weights else ("q_proj=0.95,k_proj=0.95,v_proj=0.95,o_proj=1.10,up_proj=1.10,down_proj=1.05" if args.knowledge_preset else None))
        tw = _parse_tw(tw_spec)
        adapters = generate_lora_from_embedding(
            emb["z"],
            d_model=d_model,
            num_layers=num_layers,
            rank=int(args.base_rank),
            seed=int(args.seed),
            targets=list(shapes.keys()),
        target_shapes=shapes,
        layer_gate="zmean",
        target_weights=tw,
        )
        manifest = {
            "model": args.model,
            "d_model": d_model,
            "layers": num_layers,
            "rank": int(args.base_rank),
            "embed_dim": int(args.embed_dim),
        }
        save_npz(str(base_dir), embedding=emb, adapters=adapters, manifest=manifest)

    # Run directly via orchestrator
    from examples.repo_grounded_adapters.modules.runner import generate_answer
    from examples.repo_grounded_adapters.modules.runner import generate_answer_structured

    if args.gpu_ids and str(args.gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if bool(args.structured):
        res = generate_answer_structured(
            model_id=args.model,
            adapters_npz=str(adapters_npz),
            repo_root=str(root),
            prompt=args.prompt,
            cache_dir=str(cache_dir),
            device_map=str(args.device_map),
            alpha=float(args.alpha),
            rank=int(args.rank),
            gsub=float(args.gsub),
            beta=float(args.mix_beta),
            of_sources=args.of_sources,
            zoom_symbol=args.zoom_symbol,
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
            target_weights=(str(args.target_weights) if args.target_weights else ("q_proj=0.95,k_proj=0.95,v_proj=0.95,o_proj=1.10,up_proj=1.10,down_proj=1.05" if args.knowledge_preset else None)),
            rerank=bool(args.rerank),
            self_queries_path=(str(args.self_queries) if args.self_queries else None),
            commit_footer=bool(args.commit_footer),
            verbose=bool(args.verbose),
            lfp_iters=int(args.lfp_iters),
            budget_H=float(args.budget_H),
            monotone_selection=bool(args.monotone_selection),
            repo_state_path=None,
            samples=int(args.samples),
            cone_join=str(args.cone_join),
            telemetry_out=(str(args.telemetry_out) if args.telemetry_out else None),
        )
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
            repo_root=str(root),
            prompt=args.prompt,
            cache_dir=str(cache_dir),
            device_map=str(args.device_map),
            alpha=float(args.alpha),
            rank=int(args.rank),
            gsub=float(args.gsub),
            beta=float(args.mix_beta),
            of_sources=args.of_sources,
            zoom_symbol=args.zoom_symbol,
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
            target_weights=(str(args.target_weights) if args.target_weights else ("q_proj=0.95,k_proj=0.95,v_proj=0.95,o_proj=1.10,up_proj=1.10,down_proj=1.05" if args.knowledge_preset else None)),
            rerank=bool(args.rerank),
            self_queries_path=(str(args.self_queries) if args.self_queries else None),
            commit_footer=bool(args.commit_footer),
            verbose=bool(args.verbose),
            monotone_selection=bool(args.monotone_selection),
        )
        print(text)


if __name__ == "__main__":
    main()


