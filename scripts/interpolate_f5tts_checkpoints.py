#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Linearly interpolate or extrapolate two F5TTS checkpoint model_state_dict payloads.")
    parser.add_argument("--base", required=True, help="Checkpoint used at alpha=0.")
    parser.add_argument("--other", required=True, help="Checkpoint used at alpha=1.")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--allow-extrapolate", action="store_true")
    args = parser.parse_args()

    alpha = float(args.alpha)
    if not bool(args.allow_extrapolate) and (alpha < 0.0 or alpha > 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    base_path = Path(args.base)
    other_path = Path(args.other)
    base_payload = torch.load(base_path, map_location="cpu")
    other_payload = torch.load(other_path, map_location="cpu")
    base_state = base_payload["model_state_dict"]
    other_state = other_payload["model_state_dict"]

    missing = sorted(set(base_state) ^ set(other_state))
    if missing:
        raise ValueError(f"checkpoint state keys differ; first mismatch: {missing[:8]}")

    mixed_state = {}
    for name, base_tensor in base_state.items():
        other_tensor = other_state[name]
        if base_tensor.shape != other_tensor.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(base_tensor.shape)} vs {tuple(other_tensor.shape)}")
        if torch.is_floating_point(base_tensor) and torch.is_floating_point(other_tensor):
            mixed_state[name] = base_tensor.mul(1.0 - alpha).add(other_tensor, alpha=alpha)
        else:
            mixed_state[name] = base_tensor.clone()

    output_payload = dict(base_payload)
    output_payload["model_state_dict"] = mixed_state
    output_payload["step"] = int(base_payload.get("step", 0) or 0)
    output_payload["interpolation"] = {
        "base": str(base_path),
        "other": str(other_path),
        "alpha": alpha,
        "label": str(args.label),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)


if __name__ == "__main__":
    main()
