#!/usr/bin/env python3
"""Materialize a CPU-resident Wan Animate BF16 checkpoint into INT8 blocks.

This runs once on host RAM.  It deliberately does not call CUDA or FSDP: the
result is intended for the block-offload runtime, where only one transformed
block group is placed on an RTX 3090 at a time. Wan Animate defaults to
weight-only INT8 because dynamic activation quantization can soften diffusion
outputs across denoising steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", choices=("absmax", "percentile", "mse"), default="absmax")
    parser.add_argument("--percentile", type=float, default=0.999)
    parser.add_argument("--min-weight-elements", type=int, default=16_384)
    parser.add_argument("--activation-quant", choices=("dynamic_int8", "none"), default="none")
    parser.add_argument("--storage-format", choices=("safetensors", "pt"), default="safetensors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wan_root.is_dir():
        raise FileNotFoundError(args.wan_root)
    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(args.checkpoint_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {args.output_dir}")

    sys.path.insert(0, str(args.wan_root))
    sys.path.insert(0, "/data/transformer_10")
    from wan.animate import load_wan_animate_transformer_cpu
    from runtime.wan_animate_int8 import (
        WanInt8Config,
        convert_wan_linears_to_int8,
        save_int8_block_state,
        write_int8_block_manifest,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = WanInt8Config(
        calibration=args.calibration,
        percentile=args.percentile,
        activation_quant=args.activation_quant,
        min_weight_elements=args.min_weight_elements,
    )
    print("loading BF16 Wan Animate transformer on CPU", flush=True)
    model = load_wan_animate_transformer_cpu(str(args.checkpoint_dir), torch.bfloat16)
    model.eval().requires_grad_(False)
    print("converting eligible projection matrices to per-channel INT8", flush=True)
    inventory = convert_wan_linears_to_int8(model, config)
    print(f"converted={len(inventory.quantized)} skipped={len(inventory.skipped)}", flush=True)
    for index in range(len(model.blocks)):
        destination = save_int8_block_state(model, args.output_dir, block_index=index, storage_format=args.storage_format)
        print(f"wrote {destination}", flush=True)
    manifest = write_int8_block_manifest(
        args.output_dir,
        source_checkpoint=args.checkpoint_dir,
        config=config,
        inventory=inventory,
        num_blocks=len(model.blocks),
        storage_formats=(args.storage_format,),
    )
    # Non-block modules remain BF16 and are small relative to the 40 blocks.
    non_block_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if not key.startswith("blocks.")
    }
    from runtime.wan_animate_int8 import _save_tensor_state
    non_block_suffix = ".safetensors" if args.storage_format == "safetensors" else ".pt"
    non_block_path = _save_tensor_state(non_block_state, args.output_dir / f"non_blocks{non_block_suffix}")
    print(f"wrote {non_block_path}", flush=True)
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
