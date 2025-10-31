from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional

import torch

from specs.config import ModelConfig
from model.factory import build_model
from viz.session import VizSession
from viz.render import render_index
from specs.export import ExportConfig
from export.exporter import export as export_model


def _random_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
    return type("Batch", (), {"input_ids": input_ids, "attn_mask": None})


def cmd_e2e(args: argparse.Namespace) -> None:
    # Build config and model
    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        attn_impl=args.attn_impl,
        dtype=args.dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)

    # Viz
    viz = VizSession(type("Cfg", (), {"log_dir": args.log_dir}))
    viz.set_step(0)

    # Optional interpret probes (best-effort)
    if args.interpret_probes:
        try:
            from interpret.tracer import ActivationTracer  # type: ignore
            tracer = ActivationTracer(model)
            tracer.add_block_outputs()
        except Exception:
            tracer = None
    else:
        tracer = None

    # Training loop (toy random data)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model.train()
    for step in range(int(args.steps)):
        batch = _random_batch(args.batch_size, args.seq_len, args.vocab_size, device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and args.dtype != "float32"):
            if tracer is None:
                logits = model(batch.input_ids, None, None)
            else:
                with tracer.trace():
                    logits = model(batch.input_ids, None, None)
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                batch.input_ids[..., 1:].contiguous().view(-1),
            )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        viz.log_scalar(step, "train.loss", float(loss))
        viz.set_step(step + 1)

    # If we captured activations, log a quick histogram for the first block output
    if args.interpret_probes and tracer is not None:
        # Best-effort: try a known key like "blocks.0"
        try:
            # Run one forward to populate cache
            with tracer.trace() as cache:
                _ = model(_random_batch(1, args.seq_len, args.vocab_size, device).input_ids, None, None)
                x0 = cache.get("blocks.0")
                if x0 is not None:
                    viz.log_histogram(None, "blocks.0.out", x0)
        except Exception:
            pass

    # Render dashboard
    render_index(args.log_dir, title=args.title or "Training")

    # Optional export
    if args.export_target is not None:
        exp_cfg = ExportConfig(
            target=args.export_target,
            opset=args.export_opset,
            quantize=args.export_quantize,
            dynamic_axes=args.export_dynamic,
            outdir=args.export_outdir,
        )
        out = export_model(model, exp_cfg, vocab_size=cfg.vocab_size, d_model=cfg.d_model)
        print(str(out))


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="tt", description="Transformer toolkit CLI")
    sub = p.add_subparsers(dest="cmd")

    p_e2e = sub.add_parser("e2e", help="Build model, run toy training, optional export")
    p_e2e.add_argument("--d-model", type=int, default=256)
    p_e2e.add_argument("--n-heads", type=int, default=8)
    p_e2e.add_argument("--n-layers", type=int, default=4)
    p_e2e.add_argument("--d-ff", type=int, default=1024)
    p_e2e.add_argument("--vocab-size", type=int, default=32000)
    p_e2e.add_argument("--attn-impl", type=str, default="eager", choices=["eager","flash","triton"])
    p_e2e.add_argument("--dtype", type=str, default="bfloat16", choices=["float16","bfloat16","float32"])
    p_e2e.add_argument("--batch-size", type=int, default=8)
    p_e2e.add_argument("--seq-len", type=int, default=256)
    p_e2e.add_argument("--steps", type=int, default=50)
    p_e2e.add_argument("--lr", type=float, default=3e-4)
    p_e2e.add_argument("--log-dir", type=str, default=".viz")
    p_e2e.add_argument("--title", type=str, default=None)
    p_e2e.add_argument("--interpret-probes", action="store_true")
    # Export options
    p_e2e.add_argument("--export-target", type=str, default=None, choices=["torchscript","onnx","tensorrt"])
    p_e2e.add_argument("--export-opset", type=int, default=19)
    p_e2e.add_argument("--export-quantize", type=str, default=None, choices=["int8","fp8"])
    p_e2e.add_argument("--export-dynamic", action="store_true", default=True)
    p_e2e.add_argument("--export-outdir", type=str, default="artifacts/")

    args = p.parse_args(argv)
    if args.cmd == "e2e":
        cmd_e2e(args)
        return
    p.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()


