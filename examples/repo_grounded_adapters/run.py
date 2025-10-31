from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import subprocess
from typing import Dict

import numpy as np

from examples.repo_grounded_adapters.repo_conditioned_adapter import (
    build_repo_embedding,
    generate_lora_from_embedding,
    save_npz,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _infer_target_shapes_from_config(model_id: str, *, cache_dir: str | None = None) -> Dict[str, tuple[int, int]]:
    try:
        from transformers import AutoConfig  # type: ignore
    except Exception as e:
        raise RuntimeError("Install 'transformers' to run this example") from e

    cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    d_model = int(getattr(cfg, "hidden_size", 0) or 0)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0)
    if d_model <= 0:
        raise RuntimeError("Could not infer hidden_size from model config")
    shapes: Dict[str, tuple[int, int]] = {
        "q_proj": (d_model, d_model),
        "k_proj": (d_model, d_model),
        "v_proj": (d_model, d_model),
        "o_proj": (d_model, d_model),
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

    # Sampling controls
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--min-new-tokens", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=512)

    # Device & telemetry
    p.add_argument("--device-map", default="auto", choices=["auto", "none"])
    p.add_argument("--gpu-ids", default=None)
    p.add_argument("--max-memory", default=None)
    p.add_argument("--telemetry-out", default=None)
    p.add_argument("--cache-dir", default=None, help="Cache directory for HF models/tokenizers (defaults to <repo>/checkpoints)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    root = Path(args.repo)
    ex_dir = root / "examples" / "repo_grounded_adapters"
    artifacts = ex_dir / "artifacts"
    base_dir = Path(args.adapters_dir) if args.adapters_dir else (artifacts / "base_adapters")
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (root / "checkpoints")
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
        adapters = generate_lora_from_embedding(
            emb["z"],
            d_model=d_model,
            num_layers=num_layers,
            rank=int(args.base_rank),
            seed=int(args.seed),
            targets=list(shapes.keys()),
        target_shapes=shapes,
        layer_gate="zmean",
        target_weights=None,
        )
        manifest = {
            "model": args.model,
            "d_model": d_model,
            "layers": num_layers,
            "rank": int(args.base_rank),
            "embed_dim": int(args.embed_dim),
        }
        save_npz(str(base_dir), embedding=emb, adapters=adapters, manifest=manifest)

    # Build command for the enhanced HF runner (pass-through of options)
    cmd = [
        sys.executable,
        "-m",
        "examples.repo_grounded_adapters.run_llama_with_repo_adapter_enhanced",
        "--model", args.model,
        "--adapters", str(adapters_npz),
        "--repo", str(root),
        "--prompt", args.prompt,
        "--cache-dir", str(cache_dir),
        "--alpha", str(args.alpha),
        "--rank", str(args.rank),
        "--gsub", str(args.gsub),
        "--mix-beta", str(args.mix_beta),
        "--of-sources", args.of_sources,
        "--zoom-symbol", str(args.zoom_symbol) if args.zoom_symbol is not None else "",
        "--zoom-radius", str(int(args.zoom_radius)),
        "--pack-mode", args.pack_mode,
        "--context-tokens", str(int(args.context_tokens)),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--repetition-penalty", str(args.repetition_penalty),
        "--min-new-tokens", str(int(args.min_new_tokens)),
        "--max-new-tokens", str(int(args.max_new_tokens)),
        "--device-map", args.device_map,
        "--seed", str(int(args.seed)),
    ]
    if args.pack_context:
        cmd.append("--pack-context")
    if args.require_citations:
        cmd.append("--require-citations")
    if args.citations_per_paragraph:
        cmd.append("--citations-per-paragraph")
    if args.function_first:
        cmd.extend([
            "--function-first",
            "--ff-max-candidates", str(int(args.ff_max_candidates)),
            "--ff-window-lines", str(int(args.ff_window_lines)),
            "--ff-threshold", str(float(args.ff_threshold)),
            "--ff-noise-penalty", str(float(args.ff_noise_penalty)),
        ])
    if args.do_sample:
        cmd.append("--do-sample")
    if args.target_weights:
        cmd.extend(["--target-weights", str(args.target_weights)])
    if args.telemetry_out:
        cmd.extend(["--telemetry-out", str(args.telemetry_out)])
    if args.verbose:
        cmd.append("--verbose")

    # Device/memory environment
    env = os.environ.copy()
    if args.gpu_ids and str(args.gpu_ids).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids).strip()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.max_memory and str(args.max_memory).strip():
        cmd.extend(["--max-memory", str(args.max_memory)])

    # Clean empty zoom-symbol arg if not provided
    try:
        if "--zoom-symbol" in cmd:
            zi = cmd.index("--zoom-symbol")
            if cmd[zi + 1] == "":
                del cmd[zi:zi + 2]
    except Exception:
        pass

    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()


