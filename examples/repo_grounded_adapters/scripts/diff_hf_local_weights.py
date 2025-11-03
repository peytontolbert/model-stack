import os
import json
import argparse
from typing import Dict, List, Tuple

import torch


def _ensure_snapshot(model_id: str, cache_dir: str) -> str:
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    org_name = model_id.strip().split("/")[-2:]
    if len(org_name) == 2:
        org, name = org_name
        dir1 = os.path.join(cache_dir, f"models--{org}--{name}", "snapshots")
        cands = []
        if os.path.isdir(dir1):
            cands.extend([os.path.join(dir1, d) for d in os.listdir(dir1)])
        cands = [p for p in cands if os.path.isfile(os.path.join(p, "model.safetensors.index.json"))]
        if cands:
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return cands[0]
    from huggingface_hub import snapshot_download  # type: ignore
    return snapshot_download(repo_id=model_id, cache_dir=cache_dir)


def _index_shards(model_dir: str) -> Dict[str, List[str]]:
    index_fp = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_fp):
        raise FileNotFoundError(f"Missing index file: {index_fp}")
    with open(index_fp, "r", encoding="utf-8") as fh:
        index_obj = json.load(fh)
    weight_map: Dict[str, str] = index_obj.get("weight_map", {})
    by_file: Dict[str, List[str]] = {}
    for k, fn in weight_map.items():
        by_file.setdefault(fn, []).append(k)
    return by_file


def _read_hf_tensor(model_dir: str, key: str) -> torch.Tensor:
    from safetensors import safe_open  # type: ignore
    by_file = _index_shards(model_dir)
    for shard_rel, keys in by_file.items():
        if key not in keys:
            continue
        fp = os.path.join(model_dir, shard_rel)
        with safe_open(fp, framework="pt", device="cpu") as f:  # type: ignore
            return f.get_tensor(key)  # type: ignore[attr-defined]
    raise KeyError(f"tensor {key} not found in shards")


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a32 = a.detach().to(torch.float32, copy=False)
    b32 = b.detach().to(torch.float32, copy=False)
    diff = (a32 - b32).view(-1)
    l2 = float(diff.norm().item())
    max_abs = float(diff.abs().max().item())
    denom = float(max(1e-12, a32.norm().item()))
    rel = float(l2 / denom)
    cos = float((torch.nn.functional.cosine_similarity(a32.view(1, -1), b32.view(1, -1))).item())
    return {"l2": l2, "max_abs": max_abs, "rel_l2": rel, "cos": cos}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--cache-dir", default="/data/transformer_10/checkpoints")
    p.add_argument("--layers", default="0", help="Comma-separated layer indices to compare")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    ckpt = _ensure_snapshot(args.model, args.cache_dir)

    # Build local model and load HF weights via our loader
    from specs.config import ModelConfig
    from model.factory import build_causal_lm
    from model.hf_llama_loader import load_hf_llama_weights_into_local

    cfg = json.load(open(os.path.join(ckpt, "config.json"), "r", encoding="utf-8"))
    use_cuda = torch.cuda.is_available()
    mc = ModelConfig(
        d_model=int(cfg.get("hidden_size")),
        n_heads=int(cfg.get("num_attention_heads")),
        n_layers=int(cfg.get("num_hidden_layers")),
        d_ff=int(cfg.get("intermediate_size")),
        vocab_size=int(cfg.get("vocab_size")),
        head_dim=int(cfg.get("head_dim", int(cfg.get("hidden_size")) // int(cfg.get("num_attention_heads")))),
        rope_theta=float(cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 1e6))),
        dtype=("bfloat16" if use_cuda else "float32"),
        attn_impl="sdpa",
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        rope_scaling_type=(cfg.get("rope_scaling", {}) or {}).get("type"),
        rope_scaling_factor=(cfg.get("rope_scaling", {}) or {}).get("factor"),
        rope_scaling_original_max_position_embeddings=(cfg.get("rope_scaling", {}) or {}).get("original_max_position_embeddings"),
        rope_scaling_low_freq_factor=(cfg.get("rope_scaling", {}) or {}).get("low_freq_factor"),
        rope_scaling_high_freq_factor=(cfg.get("rope_scaling", {}) or {}).get("high_freq_factor"),
    )
    n_kv_heads = int(cfg.get("num_key_value_heads", mc.n_heads))
    tie_we = bool(cfg.get("tie_word_embeddings", True))
    local = build_causal_lm(mc, block="llama", n_kv_heads=n_kv_heads, tie_weights=tie_we)
    load_hf_llama_weights_into_local(local, ckpt)
    local = (local.to(device="cuda", dtype=torch.bfloat16) if use_cuda else local.to(device="cpu", dtype=torch.float32)).eval()

    report: Dict[str, Dict[str, float]] = {}

    # Embedding and lm_head
    try:
        hf_embed = _read_hf_tensor(ckpt, "model.embed_tokens.weight").to("cpu")
        report["embed.weight"] = _metrics(local.embed.weight.data.detach().to("cpu"), hf_embed)
    except Exception:
        report["embed.weight"] = {"error": 1.0}  # type: ignore
    try:
        hf_lm = _read_hf_tensor(ckpt, "lm_head.weight").to("cpu")
        report["lm_head.weight"] = _metrics(local.lm_head.weight.data.detach().to("cpu"), hf_lm)
    except Exception:
        report["lm_head.weight"] = {"error": 1.0}  # type: ignore

    layers = [int(x.strip()) for x in str(args.layers).split(",") if x.strip()]
    for li in layers:
        key = f"layers.{li}"
        try:
            blk = local.blocks[li]
        except Exception:
            report[key] = {"error": 1.0}  # type: ignore
            continue

        # Attention projections
        for short, hf_name, local_w in (
            ("q_proj", f"model.layers.{li}.self_attn.q_proj.weight", blk.attn.w_q.weight),
            ("k_proj", f"model.layers.{li}.self_attn.k_proj.weight", blk.attn.w_k.weight),
            ("v_proj", f"model.layers.{li}.self_attn.v_proj.weight", blk.attn.w_v.weight),
            ("o_proj", f"model.layers.{li}.self_attn.o_proj.weight", blk.attn.w_o.weight),
        ):
            try:
                hf_t = _read_hf_tensor(ckpt, hf_name).to("cpu")
                report[f"{key}.{short}"] = _metrics(local_w.data.detach().to("cpu"), hf_t)
            except Exception:
                report[f"{key}.{short}"] = {"error": 1.0}  # type: ignore

        # Layer norms
        for short, hf_name, local_w in (
            ("n1", f"model.layers.{li}.input_layernorm.weight", blk.n1.weight),
            ("n2", f"model.layers.{li}.post_attention_layernorm.weight", blk.n2.weight),
        ):
            try:
                hf_t = _read_hf_tensor(ckpt, hf_name).to("cpu")
                report[f"{key}.{short}"] = _metrics(local_w.data.detach().to("cpu"), hf_t)
            except Exception:
                report[f"{key}.{short}"] = {"error": 1.0}  # type: ignore

        # MLP down (w_out)
        try:
            hf_down = _read_hf_tensor(ckpt, f"model.layers.{li}.mlp.down_proj.weight").to("cpu")
            report[f"{key}.mlp.down"] = _metrics(blk.mlp.w_out.weight.data.detach().to("cpu"), hf_down)
        except Exception:
            report[f"{key}.mlp.down"] = {"error": 1.0}  # type: ignore

        # MLP fused w_in vs concat(gate, up)
        try:
            hf_gate = _read_hf_tensor(ckpt, f"model.layers.{li}.mlp.gate_proj.weight").to("cpu")
            hf_up = _read_hf_tensor(ckpt, f"model.layers.{li}.mlp.up_proj.weight").to("cpu")
            fused = torch.cat([hf_gate, hf_up], dim=0).contiguous()
            report[f"{key}.mlp.w_in"] = _metrics(blk.mlp.w_in.weight.data.detach().to("cpu"), fused)
        except Exception:
            report[f"{key}.mlp.w_in"] = {"error": 1.0}  # type: ignore

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


