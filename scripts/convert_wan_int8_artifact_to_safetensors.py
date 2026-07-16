#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def _convert_file(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"exists {destination}", flush=True)
        return
    state = torch.load(source, map_location="cpu", weights_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file({key: tensor.detach().cpu().contiguous() for key, tensor in state.items()}, str(temporary))
    os.replace(temporary, destination)
    print(f"converted {source} -> {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Wan INT8 block artifact from torch .pt files to safetensors.")
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = args.artifact_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    num_blocks = int(payload["num_blocks"])

    _convert_file(args.artifact_dir / "non_blocks.pt", args.artifact_dir / "non_blocks.safetensors", overwrite=args.overwrite)
    for index in range(num_blocks):
        _convert_file(
            args.artifact_dir / "blocks" / f"block_{index:02d}.pt",
            args.artifact_dir / "blocks" / f"block_{index:02d}.safetensors",
            overwrite=args.overwrite,
        )

    payload["storage_formats"] = sorted(set([*payload.get("storage_formats", []), "pt", "safetensors"]))
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"updated {manifest}", flush=True)


if __name__ == "__main__":
    main()
