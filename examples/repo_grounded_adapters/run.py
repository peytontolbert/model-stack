from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import subprocess
from typing import Dict

import numpy as np

from examples.repo_grounded_adapters.modules.embedding import build_repo_embedding
from examples.repo_grounded_adapters.modules.adapter import generate_lora_from_embedding, save_npz


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _infer_target_shapes_from_config(model_id: str, *, cache_dir: str | None = None) -> Dict[str, tuple[int, int]]:
    """Infer projection matrix shapes from HF config, honoring GQA for K/V.

    For LLaMA variants:
      - q_proj: (n_heads*head_dim, d_model) == (d_model, d_model)
      - k_proj/v_proj: (n_kv_heads*head_dim, d_model)
      - o_proj: (d_model, n_heads*head_dim) == (d_model, d_model)
      - up/gate: (intermediate_size, d_model), down: (d_model, intermediate_size)
    """
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception as e:
        raise RuntimeError("Install 'transformers' to run this example") from e

    cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    d_model = int(getattr(cfg, "hidden_size", 0) or 0)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0)
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    head_dim = int(getattr(cfg, "head_dim", (d_model // n_heads) if (d_model and n_heads) else 0) or 0)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
    if d_model <= 0 or n_heads <= 0 or head_dim <= 0:
        raise RuntimeError("Could not infer attention dims from model config")
    # Compute KV out dim honoring GQA
    kv_out = int(n_kv_heads * head_dim)
    shapes: Dict[str, tuple[int, int]] = {
        "q_proj": (d_model, d_model),              # (n_heads*Dh, d_model)
        "k_proj": (kv_out, d_model),              # (n_kv_heads*Dh, d_model)
        "v_proj": (kv_out, d_model),              # (n_kv_heads*Dh, d_model)
        "o_proj": (d_model, d_model),             # (d_model, n_heads*Dh)
    }
    if inter > 0:
        shapes.update({
            "up_proj": (inter, d_model),
            "down_proj": (d_model, inter),
            "gate_proj": (inter, d_model),
        })
    return shapes


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
    p.add_argument("--cache-dir", default=None, help="Cache directory for HF models/tokenizers (defaults to <repo>/checkpoints)")
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
    cache_dir = Path(args.cache_dir) if args.cache_dir else (proj_root / "checkpoints")
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
        # Infer shapes from HF model config
        shapes = _infer_target_shapes_from_config(args.model, cache_dir=str(cache_dir))
        # Infer dims from config
        try:
            from transformers import AutoConfig  # type: ignore
            cfg = AutoConfig.from_pretrained(args.model, cache_dir=str(cache_dir))
            num_layers = int(getattr(cfg, "num_hidden_layers", 32) or 32)
            d_model = int(getattr(cfg, "hidden_size", 4096) or 4096)
        except Exception:
            num_layers, d_model = 32, 4096
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

    if args.gpu_ids and str(args.gpu_ids).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    )
    print(text)


if __name__ == "__main__":
    main()


