from __future__ import annotations

import argparse
import importlib
from typing import Optional

import torch

from specs.config import ModelConfig
from model.factory import build_model
from model.checkpoint import load_config, load_pretrained
from data.loader import build_dataloader
from .loop import evaluate_lm_next_token
from .bench import benchmark_forward, benchmark_generate
from .calibration import evaluate_ece
from .report import write_json, append_csv, viz_log_scalar
from .seq import eval_sequences
from .latency import latency_forward, latency_generate
from .memory import report_memory
from .suite import run_basic_suite


def _load_model_from_factory(spec: str) -> torch.nn.Module:
    mod, fn = spec.split(":", 1)
    m = importlib.import_module(mod)
    fn = getattr(m, fn)
    out = fn()
    return out[0] if isinstance(out, tuple) else out


def cmd_ppl(args: argparse.Namespace) -> None:
    if args.model_dir is not None:
        cfg: ModelConfig = load_config(args.model_dir)
        model = build_model(cfg)
        model = load_pretrained(model, args.model_dir)
    else:
        model = _load_model_from_factory(args.model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    loader = build_dataloader(
        args.shards,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        num_workers=int(args.num_workers),
        drop_last=True,
        pin_memory=True,
        device=device,
        streaming=args.streaming,
        distributed=False,
        seed=args.seed,
        shuffle=False,
    )

    res = evaluate_lm_next_token(model, loader, device=device, max_batches=args.max_batches, report_accuracy=not args.no_acc)
    payload = {"nll": res.nll, "ppl": res.ppl, "acc": res.acc, "tokens": res.num_tokens}
    if args.outdir:
        write_json(args.outdir, "ppl", payload)
        append_csv(args.outdir, "ppl", {"nll": res.nll, "ppl": res.ppl, "acc": res.acc or 0.0, "tokens": res.num_tokens})
        viz_log_scalar(args.viz_log_dir, 0, "eval.ppl", float(res.ppl))
        if res.acc is not None:
            viz_log_scalar(args.viz_log_dir, 0, "eval.acc", float(res.acc))
    print(payload)


def cmd_bench_fwd(args: argparse.Namespace) -> None:
    model = _load_or_build(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    res = benchmark_forward(model, batch_size=args.batch_size, seq_len=args.seq_len, vocab_size=args.vocab_size, warmup_steps=args.warmup, steps=args.steps, device=device)
    payload = {"tokens_per_sec": res.tokens_per_sec, "latency_ms": res.latency_ms, "total_tokens": res.total_tokens, "total_time_s": res.total_time_s}
    if args.outdir:
        write_json(args.outdir, "bench_forward", payload)
        append_csv(args.outdir, "bench_forward", payload)
        viz_log_scalar(args.viz_log_dir, 0, "bench.forward.tokens_per_sec", float(res.tokens_per_sec))
    print(payload)


def cmd_bench_gen(args: argparse.Namespace) -> None:
    model = _load_or_build(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    # Optional KV cache factory via serve.runtime if desired
    kvf = None
    try:
        from serve.runtime import ModelRuntime  # type: ignore
        rt = ModelRuntime(model=model, cfg=ModelConfig(d_model=1, n_heads=1, n_layers=1, d_ff=1, vocab_size=max(args.vocab_size or 2, 2)), device=device, dtype=torch.bfloat16, kv_pagesize=512)  # type: ignore[arg-type]
        kvf = rt.allocate_cache
    except Exception:
        kvf = None
    res = benchmark_generate(model, batch_size=args.batch_size, seq_len=args.seq_len, max_new_tokens=args.max_new_tokens, vocab_size=args.vocab_size, warmup_steps=args.warmup, repeats=args.repeats, device=device, kv_cache_factory=kvf)
    payload = {"tokens_per_sec": res.tokens_per_sec, "latency_ms": res.latency_ms, "total_tokens": res.total_tokens, "total_time_s": res.total_time_s}
    if args.outdir:
        write_json(args.outdir, "bench_generate", payload)
        append_csv(args.outdir, "bench_generate", payload)
        viz_log_scalar(args.viz_log_dir, 0, "bench.generate.tokens_per_sec", float(res.tokens_per_sec))
    print(payload)


def cmd_ece(args: argparse.Namespace) -> None:
    model = _load_or_build(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    loader = build_dataloader(args.shards, batch_size=args.batch_size, seq_len=args.seq_len, num_workers=args.num_workers, drop_last=True, pin_memory=True, device=device, streaming=args.streaming, distributed=False, seed=args.seed, shuffle=False)
    res = evaluate_ece(model, loader, device=device, n_bins=args.n_bins, max_batches=args.max_batches)
    payload = {"ece": res.ece, "tokens": res.num_tokens}
    if args.outdir:
        write_json(args.outdir, "ece", payload)
        append_csv(args.outdir, "ece", payload)
        viz_log_scalar(args.viz_log_dir, 0, "eval.ece", float(res.ece))
    print(payload)


def cmd_seq(args: argparse.Namespace) -> None:
    hyps = open(args.hyp, "r", encoding="utf-8").read().splitlines()
    refs = open(args.ref, "r", encoding="utf-8").read().splitlines()
    res = eval_sequences(hyps, refs)
    if args.outdir:
        write_json(args.outdir, "seq_metrics", res)
        append_csv(args.outdir, "seq_metrics", res)
        viz_log_scalar(args.viz_log_dir, 0, "eval.bleu", float(res.get("bleu", 0.0)))
        viz_log_scalar(args.viz_log_dir, 0, "eval.rougeL", float(res.get("rougeL", 0.0)))
    print(res)


def cmd_latency(args: argparse.Namespace) -> None:
    model = _load_or_build(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    if args.mode == "forward":
        dist = latency_forward(model, repeats=args.repeats, batch_size=args.batch_size, seq_len=args.seq_len, vocab_size=args.vocab_size, device=device)
    else:
        dist = latency_generate(model, repeats=args.repeats, batch_size=args.batch_size, seq_len=args.seq_len, max_new_tokens=args.max_new_tokens, vocab_size=args.vocab_size, device=device)
    payload = {"p50_ms": dist.p50_ms, "p95_ms": dist.p95_ms, "p99_ms": dist.p99_ms, "samples": dist.samples}
    if args.outdir:
        write_json(args.outdir, "latency", payload)
        append_csv(args.outdir, "latency", payload)
    print(payload)


def cmd_mem(args: argparse.Namespace) -> None:
    model = _load_or_build(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    # Best-effort: simple forward for peak
    def _fwd():
        x = torch.randint(0, getattr(model, "vocab_size", 32000), (args.batch_size, args.seq_len), dtype=torch.long, device=device)
        return model(x, None, None)
    rep = report_memory(model, run_forward=_fwd if args.peak else None)
    payload = {"params_mb": rep.params_mb, "buffers_mb": rep.buffers_mb, "peak_cuda_mb": rep.peak_cuda_mb}
    if args.outdir:
        write_json(args.outdir, "memory", payload)
        append_csv(args.outdir, "memory", payload)
    print(payload)


def cmd_suite(args: argparse.Namespace) -> None:
    model = _load_or_build(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    loader = build_dataloader(args.shards, batch_size=args.batch_size, seq_len=args.seq_len, num_workers=args.num_workers, drop_last=True, pin_memory=True, device=device, streaming=args.streaming, distributed=False, seed=args.seed, shuffle=False)
    res = run_basic_suite(model, loader, device=device)
    d = res.to_dict()
    if args.outdir:
        write_json(args.outdir, "suite", d)
    print(d)


def _load_or_build(args: argparse.Namespace) -> torch.nn.Module:
    if getattr(args, "model_dir", None):
        cfg: ModelConfig = load_config(args.model_dir)
        model = build_model(cfg)
        model = load_pretrained(model, args.model_dir)
    else:
        model = _load_model_from_factory(args.model)
    return model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.cli", description="Evaluation utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("ppl", help="Compute perplexity on token shards")
    p1.add_argument("--model-dir", type=str, default=None, help="Directory with config/weights")
    p1.add_argument("--model", type=str, default=None, help="Module:function returning a model (alternative to --model-dir)")
    p1.add_argument("--shards", type=str, required=True, help="Directory of token shards")
    p1.add_argument("--batch-size", type=int, default=8)
    p1.add_argument("--seq-len", type=int, default=512)
    p1.add_argument("--num-workers", type=int, default=0)
    p1.add_argument("--device", type=str, default=None)
    p1.add_argument("--max-batches", type=int, default=None)
    p1.add_argument("--seed", type=int, default=1337)
    p1.add_argument("--streaming", action="store_true")
    p1.add_argument("--no-acc", action="store_true")
    p1.add_argument("--outdir", type=str, default=None)
    p1.add_argument("--viz-log-dir", type=str, default=None)
    p1.set_defaults(func=cmd_ppl)

    p2 = sub.add_parser("bench-forward", help="Throughput for forward pass (tokens/sec)")
    p2.add_argument("--model-dir", type=str, default=None)
    p2.add_argument("--model", type=str, default=None)
    p2.add_argument("--batch-size", type=int, default=8)
    p2.add_argument("--seq-len", type=int, default=512)
    p2.add_argument("--vocab-size", type=int, default=None)
    p2.add_argument("--warmup", type=int, default=5)
    p2.add_argument("--steps", type=int, default=20)
    p2.add_argument("--device", type=str, default=None)
    p2.add_argument("--outdir", type=str, default=None)
    p2.add_argument("--viz-log-dir", type=str, default=None)
    p2.set_defaults(func=cmd_bench_fwd)

    p3 = sub.add_parser("bench-generate", help="Decode throughput (new tokens/sec)")
    p3.add_argument("--model-dir", type=str, default=None)
    p3.add_argument("--model", type=str, default=None)
    p3.add_argument("--batch-size", type=int, default=1)
    p3.add_argument("--seq-len", type=int, default=128)
    p3.add_argument("--max-new-tokens", type=int, default=128)
    p3.add_argument("--vocab-size", type=int, default=None)
    p3.add_argument("--warmup", type=int, default=2)
    p3.add_argument("--repeats", type=int, default=5)
    p3.add_argument("--device", type=str, default=None)
    p3.add_argument("--outdir", type=str, default=None)
    p3.add_argument("--viz-log-dir", type=str, default=None)
    p3.set_defaults(func=cmd_bench_gen)

    p4 = sub.add_parser("ece", help="Expected Calibration Error on token shards")
    p4.add_argument("--model-dir", type=str, default=None)
    p4.add_argument("--model", type=str, default=None)
    p4.add_argument("--shards", type=str, required=True)
    p4.add_argument("--batch-size", type=int, default=8)
    p4.add_argument("--seq-len", type=int, default=512)
    p4.add_argument("--num-workers", type=int, default=0)
    p4.add_argument("--device", type=str, default=None)
    p4.add_argument("--max-batches", type=int, default=None)
    p4.add_argument("--seed", type=int, default=1337)
    p4.add_argument("--streaming", action="store_true")
    p4.add_argument("--n-bins", type=int, default=15)
    p4.add_argument("--outdir", type=str, default=None)
    p4.add_argument("--viz-log-dir", type=str, default=None)
    p4.set_defaults(func=cmd_ece)

    p5 = sub.add_parser("seq", help="Sequence metrics from hyp/ref text files")
    p5.add_argument("--hyp", type=str, required=True, help="Path to candidate (one per line)")
    p5.add_argument("--ref", type=str, required=True, help="Path to reference (one per line)")
    p5.add_argument("--outdir", type=str, default=None)
    p5.add_argument("--viz-log-dir", type=str, default=None)
    p5.set_defaults(func=cmd_seq)

    p6 = sub.add_parser("latency", help="Latency percentiles for forward/generate")
    p6.add_argument("--model-dir", type=str, default=None)
    p6.add_argument("--model", type=str, default=None)
    p6.add_argument("--mode", type=str, default="forward", choices=["forward","generate"])
    p6.add_argument("--repeats", type=int, default=100)
    p6.add_argument("--batch-size", type=int, default=1)
    p6.add_argument("--seq-len", type=int, default=128)
    p6.add_argument("--max-new-tokens", type=int, default=1)
    p6.add_argument("--vocab-size", type=int, default=None)
    p6.add_argument("--device", type=str, default=None)
    p6.add_argument("--outdir", type=str, default=None)
    p6.set_defaults(func=cmd_latency)

    p7 = sub.add_parser("mem", help="Model memory footprint and optional peak CUDA")
    p7.add_argument("--model-dir", type=str, default=None)
    p7.add_argument("--model", type=str, default=None)
    p7.add_argument("--batch-size", type=int, default=1)
    p7.add_argument("--seq-len", type=int, default=128)
    p7.add_argument("--device", type=str, default=None)
    p7.add_argument("--peak", action="store_true")
    p7.add_argument("--outdir", type=str, default=None)
    p7.set_defaults(func=cmd_mem)

    p8 = sub.add_parser("suite", help="Run a basic evaluation suite (ppl, benches, ece)")
    p8.add_argument("--model-dir", type=str, default=None)
    p8.add_argument("--model", type=str, default=None)
    p8.add_argument("--shards", type=str, required=True)
    p8.add_argument("--batch-size", type=int, default=8)
    p8.add_argument("--seq-len", type=int, default=512)
    p8.add_argument("--num-workers", type=int, default=0)
    p8.add_argument("--device", type=str, default=None)
    p8.add_argument("--seed", type=int, default=1337)
    p8.add_argument("--streaming", action="store_true")
    p8.add_argument("--outdir", type=str, default=None)
    p8.set_defaults(func=cmd_suite)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()


