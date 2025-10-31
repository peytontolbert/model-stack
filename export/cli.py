from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from specs.export import ExportConfig
from .exporter import export_from_dir, export_model


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="export", description="Model export utilities")
    p.add_argument("--model-dir", type=str, help="Path to model directory with config and weights")
    p.add_argument("--target", type=str, default="onnx", choices=["onnx", "torchscript", "tensorrt"])
    p.add_argument("--opset", type=int, default=19)
    p.add_argument("--quantize", type=str, default=None, choices=["int8", "fp8"])  # weight-only int8
    p.add_argument("--dynamic-axes", action="store_true", default=True)
    p.add_argument("--outdir", type=str, default="artifacts/")
    p.add_argument("--trt-workspace-mb", type=int, default=4096)

    args = p.parse_args(argv)

    cfg = ExportConfig(
        target=args.target,
        opset=args.opset,
        quantize=args.quantize,
        dynamic_axes=args.dynamic_axes,
        outdir=args.outdir,
        trt_max_workspace_mb=args.trt_workspace_mb,
    )

    out = export_from_dir(args.model_dir, cfg)
    print(str(out))


if __name__ == "__main__":  # pragma: no cover
    main()


