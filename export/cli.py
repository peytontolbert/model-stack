from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from specs.export import ExportConfig
from .exporter import export_from_dir, export_model


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="export", description="Model export utilities")
    p.add_argument("--model-dir", type=str, help="Path to model directory with config and weights")
    p.add_argument("--target", type=str, default="onnx", choices=["onnx", "torchscript", "tensorrt", "browser-bitnet"])
    p.add_argument("--opset", type=int, default=19)
    p.add_argument("--quantize", type=str, default=None, choices=["int8", "int4", "nf4", "fp8", "bitnet"])
    p.add_argument("--quant-spin", action="store_true", default=False)
    p.add_argument("--quant-spin-seed", type=int, default=0)
    p.add_argument("--quant-weight-opt", type=str, default="none", choices=["none", "awq", "gptq"])
    p.add_argument(
        "--quant-activation-quant",
        type=str,
        default=None,
        choices=["static_int8", "dynamic_int8"],
        help="Optional activation fake-quant mode for quantized export.",
    )
    p.add_argument("--quant-activation-quant-bits", type=int, default=8)
    p.add_argument(
        "--quant-activation-quant-method",
        type=str,
        default="absmax",
        choices=["absmax", "percentile", "mse"],
    )
    p.add_argument("--quant-activation-quant-percentile", type=float, default=0.999)
    p.add_argument(
        "--quant-calibration-inputs",
        type=str,
        default=None,
        help="Path to torch-saved calibration input map used for AWQ/GPTQ and static activation quantization.",
    )
    p.add_argument("--dynamic-axes", action="store_true", default=True)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--outdir", type=str, default="artifacts/")
    p.add_argument("--trt-workspace-mb", type=int, default=4096)

    args = p.parse_args(argv)

    cfg = ExportConfig(
        target=args.target,
        opset=args.opset,
        quantize=args.quantize,
        quant_spin=args.quant_spin,
        quant_spin_seed=args.quant_spin_seed,
        quant_weight_opt=args.quant_weight_opt,
        quant_activation_quant=args.quant_activation_quant,
        quant_activation_quant_bits=args.quant_activation_quant_bits,
        quant_activation_quant_method=args.quant_activation_quant_method,
        quant_activation_quant_percentile=args.quant_activation_quant_percentile,
        quant_calibration_inputs_path=args.quant_calibration_inputs,
        dynamic_axes=args.dynamic_axes,
        max_seq_len=args.max_seq_len,
        outdir=args.outdir,
        trt_max_workspace_mb=args.trt_workspace_mb,
    )

    out = export_from_dir(args.model_dir, cfg)
    print(str(out))


if __name__ == "__main__":  # pragma: no cover
    main()
