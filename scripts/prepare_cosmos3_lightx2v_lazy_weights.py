#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def _target_file(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    if len(parts) > 2 and parts[0] == "layers" and parts[1].isdigit():
        return f"block_{parts[1]}.safetensors"
    return "non_block.safetensors"


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a Cosmos3 Diffusers transformer snapshot into LightX2V lazy-load block files.")
    parser.add_argument("transformer_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    index_path = args.transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    if not weight_map:
        raise SystemExit(f"No weight_map found in {index_path}")

    output_to_keys: dict[str, list[str]] = defaultdict(list)
    key_to_source: dict[str, str] = {}
    for key, source in weight_map.items():
        target = _target_file(key)
        output_to_keys[target].append(key)
        key_to_source[key] = source

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for output_name in tqdm(sorted(output_to_keys, key=lambda n: (n != "non_block.safetensors", n)), desc="Writing Cosmos3 lazy blocks"):
        output_path = args.output_dir / output_name
        if output_path.exists() and not args.force:
            for key in output_to_keys[output_name]:
                index["weight_map"][key] = output_name
            index["metadata"]["total_size"] += output_path.stat().st_size
            continue

        keys_by_source: dict[str, list[str]] = defaultdict(list)
        for key in output_to_keys[output_name]:
            keys_by_source[key_to_source[key]].append(key)

        tensors = {}
        for source_name, keys in keys_by_source.items():
            source_path = args.transformer_dir / source_name
            with safe_open(source_path, framework="pt", device="cpu") as source:
                for key in keys:
                    tensors[key] = source.get_tensor(key)
        save_file(tensors, output_path)
        for key in tensors:
            index["weight_map"][key] = output_name
        index["metadata"]["total_size"] += os.path.getsize(output_path)
        del tensors

    (args.output_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "files": len(output_to_keys), "tensors": len(weight_map), "total_size": index["metadata"]["total_size"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
