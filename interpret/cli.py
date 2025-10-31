from __future__ import annotations

import argparse
import importlib
from typing import Callable, Tuple

import torch

from .logit_lens import logit_lens
from .causal.patching import causal_trace_restore_fraction
from .attn.weights import attention_weights_for_layer, attention_entropy_for_layer
from .attn.rollout import attention_rollout
from .neuron.mlp_lens import mlp_lens
from .causal.head_patching import causal_trace_heads_restore_table
from .logit_diff import logit_diff_lens
from .attn.saliency import head_grad_saliencies
from .causal.steer import steer_residual
from .attribution.occlusion import token_occlusion_importance
from .search.greedy import greedy_head_recovery
from .importance.module_scan import module_importance_scan
from .analysis.residual import residual_norms
from .analysis.flops import estimate_layer_costs
from .analysis.mask_effects import logit_change_with_mask


def load_model(factory_path: str):
    """Load a model from a callable path module:function -> model.

    The callable should return a torch.nn.Module on invocation with no args,
    or a tuple (model, tokenizer). This CLI only uses the model.
    """
    module_name, func_name = factory_path.split(":")
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    out = fn()
    if isinstance(out, tuple):
        model = out[0]
    else:
        model = out
    return model


def parse_tokens(s: str) -> torch.Tensor:
    ids = [int(x) for x in s.split(",") if x.strip()]
    return torch.tensor([ids], dtype=torch.long)


def cmd_logit_lens(args):
    model = load_model(args.model)
    input_ids = parse_tokens(args.tokens).to(next(model.parameters()).device)
    out = logit_lens(model, input_ids, topk=args.topk)
    for layer, (idx, val) in sorted(out.items()):
        print(f"layer {layer}: top{args.topk} ids={idx.tolist()} scores={val.tolist()}")


def cmd_causal_trace(args):
    model = load_model(args.model)
    clean = parse_tokens(args.clean).to(next(model.parameters()).device)
    corrupted = parse_tokens(args.corrupted).to(next(model.parameters()).device)
    patch_points = [p.strip() for p in args.points.split(",") if p.strip()]
    frac = causal_trace_restore_fraction(model, clean_input_ids=clean, corrupted_input_ids=corrupted, patch_points=patch_points)
    topv, topi = torch.topk(frac, k=min(args.topk, frac.shape[0]))
    print(f"recovery top{args.topk}: ids={topi.tolist()} frac={topv.tolist()}")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m interpret.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("logit-lens", help="Run logit lens on a token sequence")
    p1.add_argument("--model", required=True, help="Module:function that returns a model")
    p1.add_argument("--tokens", required=True, help="Comma-separated token ids, e.g. '1,2,3'")
    p1.add_argument("--topk", type=int, default=10)
    p1.set_defaults(func=cmd_logit_lens)

    p2 = sub.add_parser("causal-trace", help="Causal tracing via activation patching")
    p2.add_argument("--model", required=True, help="Module:function that returns a model")
    p2.add_argument("--clean", required=True, help="Comma-separated clean token ids")
    p2.add_argument("--corrupted", required=True, help="Comma-separated corrupted token ids")
    p2.add_argument("--points", required=True, help="Comma-separated module names to patch (from model.named_modules)")
    p2.add_argument("--topk", type=int, default=10)
    p2.set_defaults(func=cmd_causal_trace)

    p3 = sub.add_parser("attn-weights", help="Dump attention weights for a layer")
    p3.add_argument("--model", required=True)
    p3.add_argument("--tokens", required=True)
    p3.add_argument("--layer", type=int, required=True)
    p3.set_defaults(func=lambda args: (
        (lambda _m, _ids: print(attention_weights_for_layer(_m, _ids, args.layer).cpu().tolist()))(
            load_model(args.model), parse_tokens(args.tokens)
        )
    ))

    p4 = sub.add_parser("attn-entropy", help="Compute attention entropy for a layer")
    p4.add_argument("--model", required=True)
    p4.add_argument("--tokens", required=True)
    p4.add_argument("--layer", type=int, required=True)
    def _run_entropy(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        ent = attention_entropy_for_layer(m, ids, a.layer)
        print(ent.cpu().tolist())
    p4.set_defaults(func=_run_entropy)

    p5 = sub.add_parser("attn-rollout", help="Compute attention rollout across layers")
    p5.add_argument("--model", required=True)
    p5.add_argument("--tokens", required=True)
    def _run_rollout(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        R = attention_rollout(m, ids)
        print(R[0].cpu().tolist())
    p5.set_defaults(func=_run_rollout)

    p6 = sub.add_parser("mlp-lens", help="Run MLP lens top-k per layer")
    p6.add_argument("--model", required=True)
    p6.add_argument("--tokens", required=True)
    p6.add_argument("--topk", type=int, default=10)
    def _run_mlp_lens(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        out = mlp_lens(m, ids, topk=a.topk)
        for layer, (idx, val) in sorted(out.items()):
            print(f"layer {layer}: top{a.topk} ids={idx.tolist()} scores={val.tolist()}")
    p6.set_defaults(func=_run_mlp_lens)

    p7 = sub.add_parser("head-trace", help="Per-head causal tracing table (LxH)")
    p7.add_argument("--model", required=True)
    p7.add_argument("--clean", required=True)
    p7.add_argument("--corrupted", required=True)
    def _run_head_trace(a):
        m = load_model(a.model)
        clean = parse_tokens(a.clean).to(next(m.parameters()).device)
        corr = parse_tokens(a.corrupted).to(next(m.parameters()).device)
        T = causal_trace_heads_restore_table(m, clean_input_ids=clean, corrupted_input_ids=corr)
        print(T.cpu().tolist())
    p7.set_defaults(func=_run_head_trace)

    p8 = sub.add_parser("logit-diff-lens", help="Per-layer contributions to a logit difference")
    p8.add_argument("--model", required=True)
    p8.add_argument("--tokens", required=True)
    p8.add_argument("--target", type=int, required=True)
    p8.add_argument("--baseline", type=int, required=True)
    def _run_ld(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        res = logit_diff_lens(m, ids, a.target, a.baseline)
        print({int(k): float(v) for k, v in res.items()})
    p8.set_defaults(func=_run_ld)

    p9 = sub.add_parser("head-saliency", help="Grad×output per-head saliency table (LxH)")
    p9.add_argument("--model", required=True)
    p9.add_argument("--tokens", required=True)
    def _run_head_sal(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        S = head_grad_saliencies(m, ids)
        print(S.cpu().tolist())
    p9.set_defaults(func=_run_head_sal)

    p10 = sub.add_parser("occlude", help="Token occlusion attribution by zeroing embeddings")
    p10.add_argument("--model", required=True)
    p10.add_argument("--tokens", required=True)
    p10.add_argument("--mode", default="logit", choices=["logit","prob","nll"])
    def _run_occ(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        scores = token_occlusion_importance(m, ids, mode=a.mode)
        print(scores.tolist())
    p10.set_defaults(func=_run_occ)

    p11 = sub.add_parser("head-greedy", help="Greedy head selection for recovery")
    p11.add_argument("--model", required=True)
    p11.add_argument("--clean", required=True)
    p11.add_argument("--corrupted", required=True)
    p11.add_argument("--k", type=int, default=5)
    def _run_greedy(a):
        m = load_model(a.model)
        clean = parse_tokens(a.clean).to(next(m.parameters()).device)
        corr = parse_tokens(a.corrupted).to(next(m.parameters()).device)
        res = greedy_head_recovery(m, clean_input_ids=clean, corrupted_input_ids=corr, k=a.k)
        sel = [(int(li), int(h)) for (li, h) in res["selected"]]
        print({"selected": sel, "curve": [float(x) for x in res["curve"]]})
    p11.set_defaults(func=_run_greedy)

    p12 = sub.add_parser("scan-modules", help="Rank modules by output ablation importance")
    p12.add_argument("--model", required=True)
    p12.add_argument("--tokens", required=True)
    p12.add_argument("--mode", default="logit", choices=["logit","prob","nll"])
    def _run_scan(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        res = module_importance_scan(m, ids, mode=a.mode)
        print(res)
    p12.set_defaults(func=_run_scan)

    p13 = sub.add_parser("residual-stats", help="Residual L2 norms pre/post per layer")
    p13.add_argument("--model", required=True)
    p13.add_argument("--tokens", required=True)
    def _run_residual(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        out = residual_norms(m, ids)
        print({k: v.tolist() if v.ndim == 1 else "tensor" for k, v in out.items()})
    p13.set_defaults(func=_run_residual)

    p14 = sub.add_parser("flops", help="Estimate per-layer FLOPs and activation bytes")
    p14.add_argument("--model", required=True)
    p14.add_argument("--seq", type=int, required=True)
    p14.add_argument("--batch", type=int, default=1)
    p14.add_argument("--dtype", default="bf16")
    def _run_flops(a):
        m = load_model(a.model)
        info = estimate_layer_costs(m, seq_len=a.seq, batch_size=a.batch, dtype=a.dtype)
        print({"total_flops": int(info["total_flops"]), "bytes_per_token": int(info["bytes_per_token"])})
    p14.set_defaults(func=_run_flops)

    p15 = sub.add_parser("mask-effect", help="Delta target logit under alternative attention mask")
    p15.add_argument("--model", required=True)
    p15.add_argument("--tokens", required=True)
    p15.add_argument("--type", default="sliding", choices=["sliding","block","dilated"])
    p15.add_argument("--window", type=int, default=128)
    p15.add_argument("--block", type=int, default=128)
    p15.add_argument("--dilation", type=int, default=2)
    def _run_mask(a):
        m = load_model(a.model)
        ids = parse_tokens(a.tokens).to(next(m.parameters()).device)
        delta = logit_change_with_mask(m, ids, attn_mask_type=a.type, window=a.window, block=a.block, dilation=a.dilation)
        print(float(delta))
    p15.set_defaults(func=_run_mask)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()


