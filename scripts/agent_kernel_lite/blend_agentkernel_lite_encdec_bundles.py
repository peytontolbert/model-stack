#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"expected state dict at {path}, got {type(state)!r}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend two AgentKernel Lite encoder-decoder bundles.")
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument("--correction-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--minimal", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    base = Path(args.base_bundle).expanduser().resolve()
    correction = Path(args.correction_bundle).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    alpha = float(args.alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    shutil.rmtree(output, ignore_errors=True)
    if int(args.minimal):
        (output / "model").mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / "model" / "config.json", output / "model" / "config.json")
        shutil.copytree(base / "tokenizer", output / "tokenizer")
        shutil.copy2(base / "agentkernel_lite_encdec_manifest.json", output / "agentkernel_lite_encdec_manifest.json")
    else:
        shutil.copytree(base, output)

    base_state = _load_state(base / "model" / "model.safetensors")
    correction_state = _load_state(correction / "model" / "model.safetensors")
    merged: dict[str, torch.Tensor] = {}
    for key, value in base_state.items():
        other = correction_state.get(key)
        if other is not None and value.is_floating_point() and other.shape == value.shape:
            merged[key] = ((1.0 - alpha) * value.float() + alpha * other.float()).to(dtype=value.dtype)
        else:
            merged[key] = value
    torch.save(merged, output / "model" / "model.safetensors")

    manifest_path = output / "agentkernel_lite_encdec_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_path"] = str(manifest_path)
    manifest["model_dir"] = str(output / "model")
    manifest["tokenizer_dir"] = str(output / "tokenizer")
    if int(args.minimal):
        manifest.pop("browser_bitnet_manifest_path", None)
        manifest.setdefault("runtime_targets", {})["browser"] = "not_exported_for_eval_soup"
    else:
        manifest["browser_bitnet_manifest_path"] = str(output / "browser_bitnet" / "manifest.json")
    manifest["soup"] = {
        "base_bundle": str(base),
        "correction_bundle": str(correction),
        "alpha": alpha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "alpha": alpha}, sort_keys=True))


if __name__ == "__main__":
    main()
