from __future__ import annotations

import argparse
import importlib
from typing import Callable, Tuple

import torch

from .attribution.direct import component_logit_attribution
from .logit_lens import logit_lens
from .causal.patching import causal_trace_restore_fraction
from .attn.weights import attention_weights_for_layer, attention_entropy_for_layer
from .attn.rollout import attention_rollout
from .neuron.mlp_lens import mlp_lens
from .causal.head_patching import causal_trace_heads_restore_table
from .causal.sweeps import block_output_patch_sweep, head_patch_sweep, path_patch_sweep
from .logit_diff import logit_diff_lens
from .attn.saliency import head_grad_saliencies
from .causal.steer import steer_residual
from .attribution.occlusion import token_occlusion_importance
from .search.greedy import greedy_head_recovery
from .importance.module_scan import module_importance_scan
from .analysis.residual import residual_norms
from .analysis.flops import estimate_layer_costs
from .analysis.mask_effects import logit_change_with_mask
from .reporting import summarize_patch_sweep, summarize_path_patch_sweep
from .model_adapter import ModelInputs
from .analysis.similarity import compare_model_representations
from .causal.attribution_patching import module_attribution_patching, summarize_attribution_patching
from .diffusion import DiffusionTracer, diffusion_attention_phase_summary, patch_denoiser_latents, prompt_token_occlusion_importance, summarize_diffusion_steps, summarize_prompt_token_attribution
from .metrics.faithfulness import faithfulness_summary, token_deletion_insertion_curves
from .search.circuit import greedy_module_circuit, summarize_module_circuit
from .safety.triggers import token_trigger_append_scan


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


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def maybe_parse_tokens(s: str | None, device: torch.device) -> torch.Tensor | None:
    if s is None:
        return None
    return parse_tokens(s).to(device)


def parse_floats(s: str) -> torch.Tensor:
    values = [float(x) for x in s.split(",") if x.strip()]
    return torch.tensor([values], dtype=torch.float32)


def add_forward_input_args(parser: argparse.ArgumentParser, *, include_stack: bool = False, include_kind: bool = False) -> None:
    parser.add_argument("--tokens", help="Comma-separated token ids for causal/encoder models")
    parser.add_argument("--enc-tokens", help="Comma-separated encoder token ids for seq2seq models")
    parser.add_argument("--dec-tokens", help="Comma-separated decoder token ids for seq2seq models")
    if include_stack:
        parser.add_argument("--stack", choices=["causal", "encoder", "decoder"])
    if include_kind:
        parser.add_argument("--kind", default="self", choices=["self", "cross"])


def add_pair_input_args(parser: argparse.ArgumentParser, *, include_stack: bool = False, include_kind: bool = False) -> None:
    parser.add_argument("--clean", help="Comma-separated clean token ids for causal/encoder models")
    parser.add_argument("--corrupted", help="Comma-separated corrupted token ids for causal/encoder models")
    parser.add_argument("--enc-clean", help="Comma-separated clean encoder token ids for seq2seq models")
    parser.add_argument("--dec-clean", help="Comma-separated clean decoder token ids for seq2seq models")
    parser.add_argument("--enc-corrupted", help="Comma-separated corrupted encoder token ids for seq2seq models")
    parser.add_argument("--dec-corrupted", help="Comma-separated corrupted decoder token ids for seq2seq models")
    if include_stack:
        parser.add_argument("--stack", choices=["causal", "encoder", "decoder"])
    if include_kind:
        parser.add_argument("--kind", default="self", choices=["self", "cross"])


def load_forward_kwargs(args, model):
    device = model_device(model)
    tokens = maybe_parse_tokens(getattr(args, "tokens", None), device)
    enc_tokens = maybe_parse_tokens(getattr(args, "enc_tokens", None), device)
    dec_tokens = maybe_parse_tokens(getattr(args, "dec_tokens", None), device)
    if tokens is None and (enc_tokens is None or dec_tokens is None):
        raise ValueError("Provide --tokens or the pair --enc-tokens/--dec-tokens")
    kwargs = {
        "input_ids": tokens,
        "enc_input_ids": enc_tokens,
        "dec_input_ids": dec_tokens,
    }
    if hasattr(args, "stack"):
        kwargs["stack"] = getattr(args, "stack", None)
    if hasattr(args, "kind"):
        kwargs["kind"] = getattr(args, "kind", "self")
    return kwargs


def load_pair_kwargs(args, model):
    device = model_device(model)
    clean = maybe_parse_tokens(getattr(args, "clean", None), device)
    corrupted = maybe_parse_tokens(getattr(args, "corrupted", None), device)
    enc_clean = maybe_parse_tokens(getattr(args, "enc_clean", None), device)
    dec_clean = maybe_parse_tokens(getattr(args, "dec_clean", None), device)
    enc_corrupted = maybe_parse_tokens(getattr(args, "enc_corrupted", None), device)
    dec_corrupted = maybe_parse_tokens(getattr(args, "dec_corrupted", None), device)
    kwargs = {}
    if clean is not None or corrupted is not None:
        if clean is None or corrupted is None:
            raise ValueError("Provide both --clean and --corrupted")
        kwargs["clean_input_ids"] = clean
        kwargs["corrupted_input_ids"] = corrupted
    else:
        if None in (enc_clean, dec_clean, enc_corrupted, dec_corrupted):
            raise ValueError("Provide either --clean/--corrupted or the full seq2seq set --enc-clean/--dec-clean/--enc-corrupted/--dec-corrupted")
        kwargs["clean_inputs"] = ModelInputs.encoder_decoder(enc_clean, dec_clean)
        kwargs["corrupted_inputs"] = ModelInputs.encoder_decoder(enc_corrupted, dec_corrupted)
    if hasattr(args, "stack"):
        kwargs["stack"] = getattr(args, "stack", None)
    if hasattr(args, "kind"):
        kwargs["kind"] = getattr(args, "kind", "self")
    return kwargs


def cmd_logit_lens(args):
    model = load_model(args.model)
    out = logit_lens(model, topk=args.topk, **load_forward_kwargs(args, model))
    for layer, (idx, val) in sorted(out.items()):
        print(f"layer {layer}: top{args.topk} ids={idx.tolist()} scores={val.tolist()}")


def cmd_causal_trace(args):
    model = load_model(args.model)
    patch_points = [p.strip() for p in args.points.split(",") if p.strip()]
    frac = causal_trace_restore_fraction(model, patch_points=patch_points, **load_pair_kwargs(args, model))
    topv, topi = torch.topk(frac, k=min(args.topk, frac.shape[0]))
    print(f"recovery top{args.topk}: ids={topi.tolist()} frac={topv.tolist()}")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m interpret.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("logit-lens", help="Run logit lens on a token sequence")
    p1.add_argument("--model", required=True, help="Module:function that returns a model")
    add_forward_input_args(p1, include_stack=True, include_kind=True)
    p1.add_argument("--topk", type=int, default=10)
    p1.set_defaults(func=cmd_logit_lens)

    p2 = sub.add_parser("causal-trace", help="Causal tracing via activation patching")
    p2.add_argument("--model", required=True, help="Module:function that returns a model")
    add_pair_input_args(p2, include_stack=True, include_kind=True)
    p2.add_argument("--points", required=True, help="Comma-separated module names to patch (from model.named_modules)")
    p2.add_argument("--topk", type=int, default=10)
    p2.set_defaults(func=cmd_causal_trace)

    p3 = sub.add_parser("attn-weights", help="Dump attention weights for a layer")
    p3.add_argument("--model", required=True)
    add_forward_input_args(p3, include_stack=True, include_kind=True)
    p3.add_argument("--layer", type=int, required=True)
    def _run_attn_weights(a):
        m = load_model(a.model)
        kwargs = load_forward_kwargs(a, m)
        print(attention_weights_for_layer(m, layer_index=a.layer, **kwargs).cpu().tolist())
    p3.set_defaults(func=_run_attn_weights)

    p4 = sub.add_parser("attn-entropy", help="Compute attention entropy for a layer")
    p4.add_argument("--model", required=True)
    add_forward_input_args(p4, include_stack=True, include_kind=True)
    p4.add_argument("--layer", type=int, required=True)
    def _run_entropy(a):
        m = load_model(a.model)
        ent = attention_entropy_for_layer(m, layer_index=a.layer, **load_forward_kwargs(a, m))
        print(ent.cpu().tolist())
    p4.set_defaults(func=_run_entropy)

    p5 = sub.add_parser("attn-rollout", help="Compute attention rollout across layers")
    p5.add_argument("--model", required=True)
    add_forward_input_args(p5, include_stack=True, include_kind=True)
    def _run_rollout(a):
        m = load_model(a.model)
        kwargs = load_forward_kwargs(a, m)
        R = attention_rollout(m, **kwargs)
        print(R[0].cpu().tolist())
    p5.set_defaults(func=_run_rollout)

    p6 = sub.add_parser("mlp-lens", help="Run MLP lens top-k per layer")
    p6.add_argument("--model", required=True)
    add_forward_input_args(p6, include_stack=True, include_kind=True)
    p6.add_argument("--topk", type=int, default=10)
    def _run_mlp_lens(a):
        m = load_model(a.model)
        out = mlp_lens(m, topk=a.topk, **load_forward_kwargs(a, m))
        for layer, (idx, val) in sorted(out.items()):
            print(f"layer {layer}: top{a.topk} ids={idx.tolist()} scores={val.tolist()}")
    p6.set_defaults(func=_run_mlp_lens)

    p7 = sub.add_parser("head-trace", help="Per-head causal tracing table (LxH)")
    p7.add_argument("--model", required=True)
    add_pair_input_args(p7, include_stack=True, include_kind=True)
    def _run_head_trace(a):
        m = load_model(a.model)
        T = causal_trace_heads_restore_table(m, **load_pair_kwargs(a, m))
        print(T.cpu().tolist())
    p7.set_defaults(func=_run_head_trace)

    p8 = sub.add_parser("logit-diff-lens", help="Per-layer contributions to a logit difference")
    p8.add_argument("--model", required=True)
    add_forward_input_args(p8, include_stack=True, include_kind=True)
    p8.add_argument("--target", type=int, required=True)
    p8.add_argument("--baseline", type=int, required=True)
    def _run_ld(a):
        m = load_model(a.model)
        res = logit_diff_lens(m, target_token_id=a.target, baseline_token_id=a.baseline, **load_forward_kwargs(a, m))
        print({int(k): float(v) for k, v in res.items()})
    p8.set_defaults(func=_run_ld)

    p9 = sub.add_parser("head-saliency", help="Grad×output per-head saliency table (LxH)")
    p9.add_argument("--model", required=True)
    add_forward_input_args(p9, include_stack=True, include_kind=True)
    def _run_head_sal(a):
        m = load_model(a.model)
        S = head_grad_saliencies(m, **load_forward_kwargs(a, m))
        print(S.cpu().tolist())
    p9.set_defaults(func=_run_head_sal)

    p10 = sub.add_parser("occlude", help="Token occlusion attribution by zeroing embeddings")
    p10.add_argument("--model", required=True)
    add_forward_input_args(p10, include_stack=True)
    p10.add_argument("--mode", default="logit", choices=["logit","prob","nll"])
    def _run_occ(a):
        m = load_model(a.model)
        scores = token_occlusion_importance(m, mode=a.mode, **load_forward_kwargs(a, m))
        print(scores.tolist())
    p10.set_defaults(func=_run_occ)

    p11 = sub.add_parser("head-greedy", help="Greedy head selection for recovery")
    p11.add_argument("--model", required=True)
    add_pair_input_args(p11, include_stack=True, include_kind=True)
    p11.add_argument("--k", type=int, default=5)
    def _run_greedy(a):
        m = load_model(a.model)
        res = greedy_head_recovery(m, k=a.k, **load_pair_kwargs(a, m))
        sel = [(int(li), int(h)) for (li, h) in res["selected"]]
        print({"selected": sel, "curve": [float(x) for x in res["curve"]]})
    p11.set_defaults(func=_run_greedy)

    p12 = sub.add_parser("scan-modules", help="Rank modules by output ablation importance")
    p12.add_argument("--model", required=True)
    add_forward_input_args(p12, include_stack=True)
    p12.add_argument("--mode", default="logit", choices=["logit","prob","nll"])
    def _run_scan(a):
        m = load_model(a.model)
        res = module_importance_scan(m, mode=a.mode, **load_forward_kwargs(a, m))
        print(res)
    p12.set_defaults(func=_run_scan)

    p13 = sub.add_parser("residual-stats", help="Residual L2 norms pre/post per layer")
    p13.add_argument("--model", required=True)
    add_forward_input_args(p13, include_stack=True)
    def _run_residual(a):
        m = load_model(a.model)
        out = residual_norms(m, **load_forward_kwargs(a, m))
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
    add_forward_input_args(p15, include_stack=True)
    p15.add_argument("--type", default="sliding", choices=["sliding","block","dilated"])
    p15.add_argument("--window", type=int, default=128)
    p15.add_argument("--block", type=int, default=128)
    p15.add_argument("--dilation", type=int, default=2)
    def _run_mask(a):
        m = load_model(a.model)
        delta = logit_change_with_mask(m, attn_mask_type=a.type, window=a.window, block=a.block, dilation=a.dilation, **load_forward_kwargs(a, m))
        print(float(delta))
    p15.set_defaults(func=_run_mask)

    p16 = sub.add_parser("component-attribution", help="Direct score attribution over residual components")
    p16.add_argument("--model", required=True)
    add_forward_input_args(p16, include_stack=True)
    def _run_component_attr(a):
        m = load_model(a.model)
        out = component_logit_attribution(m, **load_forward_kwargs(a, m))
        print({k: float(v) for k, v in out.items()})
    p16.set_defaults(func=_run_component_attr)

    p17 = sub.add_parser("head-sweep", help="Patch one head/position at a time and report recovery")
    p17.add_argument("--model", required=True)
    add_pair_input_args(p17, include_stack=True, include_kind=True)
    p17.add_argument("--topk", type=int, default=10)
    def _run_head_sweep(a):
        m = load_model(a.model)
        out = head_patch_sweep(m, **load_pair_kwargs(a, m))
        print(summarize_patch_sweep(out["names"], out["scores"], topk=a.topk, unit_label="head_index", time_label="source_index"))
    p17.set_defaults(func=_run_head_sweep)

    p18 = sub.add_parser("block-sweep", help="Patch one block/time position at a time and summarize recovery")
    p18.add_argument("--model", required=True)
    add_pair_input_args(p18, include_stack=True)
    p18.add_argument("--topk", type=int, default=10)
    def _run_block_sweep(a):
        m = load_model(a.model)
        out = block_output_patch_sweep(m, **load_pair_kwargs(a, m))
        print(summarize_patch_sweep(out["names"], out["scores"], topk=a.topk, time_label="token_index"))
    p18.set_defaults(func=_run_block_sweep)

    p19 = sub.add_parser("path-sweep", help="Evaluate path patching across source/receiver module grids")
    p19.add_argument("--model", required=True)
    add_pair_input_args(p19)
    p19.add_argument("--sources", required=True, help="Comma-separated source module names")
    p19.add_argument("--receivers", required=True, help="Comma-separated receiver module names")
    p19.add_argument("--topk", type=int, default=10)
    def _run_path_sweep(a):
        m = load_model(a.model)
        kwargs = load_pair_kwargs(a, m)
        out = path_patch_sweep(
            m,
            source_modules=[x.strip() for x in a.sources.split(",") if x.strip()],
            receiver_modules=[x.strip() for x in a.receivers.split(",") if x.strip()],
            **kwargs,
        )
        print(summarize_path_patch_sweep(out, topk=a.topk))
    p19.set_defaults(func=_run_path_sweep)

    p20 = sub.add_parser("diffusion-trace", help="Trace denoiser timesteps, latents, and cross-attention for a text-to-image pipeline")
    p20.add_argument("--model", required=True, help="Module:function that returns a callable diffusion pipeline")
    p20.add_argument("--prompt", required=True)
    p20.add_argument("--steps", type=int, default=20)
    def _run_diffusion_trace(a):
        pipe = load_model(a.model)
        tracer = DiffusionTracer(pipe)
        _output, cache, records = tracer.trace_generation(a.prompt, num_inference_steps=a.steps)
        keys = list(cache.keys())
        print({"trace": summarize_diffusion_steps(records), "keys": keys})
    p20.set_defaults(func=_run_diffusion_trace)

    p21 = sub.add_parser("diffusion-occlude", help="Rank prompt tokens by text-to-image output change under token removal")
    p21.add_argument("--model", required=True, help="Module:function that returns a callable diffusion pipeline")
    p21.add_argument("--prompt", required=True)
    p21.add_argument("--steps", type=int, default=20)
    p21.add_argument("--topk", type=int, default=10)
    def _run_diffusion_occlude(a):
        pipe = load_model(a.model)
        rows = prompt_token_occlusion_importance(
            pipe,
            a.prompt,
            generation_kwargs={"num_inference_steps": a.steps},
        )
        print(summarize_prompt_token_attribution(rows, topk=a.topk))
    p21.set_defaults(func=_run_diffusion_occlude)

    p22 = sub.add_parser("diffusion-patch-latents", help="Patch clean denoiser latents into a corrupted text-to-image run")
    p22.add_argument("--model", required=True, help="Module:function that returns a callable diffusion pipeline")
    p22.add_argument("--clean-prompt", required=True)
    p22.add_argument("--corrupted-prompt", required=True)
    p22.add_argument("--steps", type=int, default=20)
    p22.add_argument("--patch-steps", default="", help="Comma-separated denoising step indices; empty patches all available steps")
    def _run_diffusion_patch_latents(a):
        pipe = load_model(a.model)
        patch_steps = [int(x) for x in a.patch_steps.split(",") if x.strip()] or None
        result = patch_denoiser_latents(
            pipe,
            clean_prompt=a.clean_prompt,
            corrupted_prompt=a.corrupted_prompt,
            patch_steps=patch_steps,
            num_inference_steps=a.steps,
        )
        print(
            {
                "clean_corrupted_distance": result.clean_corrupted_distance,
                "clean_patched_distance": result.clean_patched_distance,
                "recovery_fraction": result.recovery_fraction,
                "patched_keys": list(result.patched_keys),
            }
        )
    p22.set_defaults(func=_run_diffusion_patch_latents)

    p23 = sub.add_parser("faithfulness", help="Deletion/insertion faithfulness curves for token scores")
    p23.add_argument("--model", required=True)
    p23.add_argument("--tokens", required=True, help="Comma-separated token ids")
    p23.add_argument("--scores", required=True, help="Comma-separated token explanation scores aligned to --tokens")
    p23.add_argument("--baseline-token", type=int, default=0)
    def _run_faithfulness(a):
        m = load_model(a.model)
        device = model_device(m)
        tokens = parse_tokens(a.tokens).to(device)
        scores = parse_floats(a.scores).to(device)
        curves = token_deletion_insertion_curves(m, tokens, scores, baseline_token_id=a.baseline_token)
        summary = faithfulness_summary(curves)
        print({**summary, "order": curves["order"].tolist()})
    p23.set_defaults(func=_run_faithfulness)

    p24 = sub.add_parser("module-circuit", help="Greedily select modules that recover clean behavior under activation patching")
    p24.add_argument("--model", required=True)
    add_pair_input_args(p24)
    p24.add_argument("--candidates", required=True, help="Comma-separated module names")
    p24.add_argument("--k", type=int, default=5)
    p24.add_argument("--topk", type=int, default=10)
    def _run_module_circuit(a):
        m = load_model(a.model)
        result = greedy_module_circuit(
            m,
            candidate_modules=[x.strip() for x in a.candidates.split(",") if x.strip()],
            k=a.k,
            **load_pair_kwargs(a, m),
        )
        print({"selected": result["selected"], "curve": result["curve"], "top": summarize_module_circuit(result, topk=a.topk)})
    p24.set_defaults(func=_run_module_circuit)

    p25 = sub.add_parser("attribution-patch", help="Rank modules with first-order attribution patching")
    p25.add_argument("--model", required=True)
    add_pair_input_args(p25)
    p25.add_argument("--candidates", required=True, help="Comma-separated module names")
    p25.add_argument("--topk", type=int, default=10)
    def _run_attribution_patch(a):
        m = load_model(a.model)
        result = module_attribution_patching(
            m,
            candidate_modules=[x.strip() for x in a.candidates.split(",") if x.strip()],
            **load_pair_kwargs(a, m),
        )
        print(summarize_attribution_patching(result, topk=a.topk))
    p25.set_defaults(func=_run_attribution_patch)

    p26 = sub.add_parser("trigger-append", help="Scan appended trigger token ids by target score delta")
    p26.add_argument("--model", required=True)
    p26.add_argument("--tokens", required=True)
    p26.add_argument("--triggers", required=True, help="Comma-separated candidate trigger token ids")
    def _run_trigger_append(a):
        m = load_model(a.model)
        device = model_device(m)
        rows = token_trigger_append_scan(
            m,
            parse_tokens(a.tokens).to(device),
            [int(x) for x in a.triggers.split(",") if x.strip()],
        )
        print(rows)
    p26.set_defaults(func=_run_trigger_append)

    p27 = sub.add_parser("compare-representations", help="Compare module representations between two model factories")
    p27.add_argument("--model-a", required=True)
    p27.add_argument("--model-b", required=True)
    p27.add_argument("--tokens", required=True)
    p27.add_argument("--modules", required=True, help="Comma-separated module names to compare")
    def _run_compare_representations(a):
        ma = load_model(a.model_a)
        mb = load_model(a.model_b)
        device = model_device(ma)
        out = compare_model_representations(
            ma,
            mb,
            input_ids=parse_tokens(a.tokens).to(device),
            module_names=[x.strip() for x in a.modules.split(",") if x.strip()],
        )
        print(out)
    p27.set_defaults(func=_run_compare_representations)

    p28 = sub.add_parser("diffusion-phase", help="Summarize cross-attention entropy/change over diffusion denoising steps")
    p28.add_argument("--model", required=True, help="Module:function that returns a callable diffusion pipeline")
    p28.add_argument("--prompt", required=True)
    p28.add_argument("--steps", type=int, default=20)
    def _run_diffusion_phase(a):
        pipe = load_model(a.model)
        tracer = DiffusionTracer(pipe)
        _output, cache, _records = tracer.trace_generation(a.prompt, num_inference_steps=a.steps)
        print(diffusion_attention_phase_summary(cache))
    p28.set_defaults(func=_run_diffusion_phase)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
