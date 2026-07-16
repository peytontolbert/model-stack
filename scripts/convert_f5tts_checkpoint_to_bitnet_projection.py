#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_INPUT = "/data/transformer_10/checkpoints/f5tts_q4_8step_cfg2_libritts_fullq4_surface_v2/model_q4_12to4_best.pt"
DEFAULT_OUTPUT = "/data/transformer_10/checkpoints/f5tts_bitnet_projection_from_current_best/model_bitnet_projected.pt"
DEFAULT_EXCLUDE = "text_embed.text_embed,mel_spec"


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def should_project(name: str, tensor: torch.Tensor, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    if not name.endswith(".weight"):
        return False
    if not torch.is_floating_point(tensor):
        return False
    if tensor.ndim != 2:
        return False
    if include and not any(item in name for item in include):
        return False
    if exclude and any(item in name for item in exclude):
        return False
    return True


def bitnet_runtime_row_project(weight: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_dtype = weight.dtype
    flat = weight.detach().float()
    row_scale = flat.abs().mean(dim=-1, keepdim=True).clamp_min(float(eps))
    qweight = torch.round(flat / row_scale).clamp_(-1, 1).to(dtype=torch.int8)
    projected = qweight.to(dtype=torch.float32).mul(row_scale).to(dtype=original_dtype)
    return projected, qweight, row_scale.squeeze(-1).to(dtype=torch.float32)


def load_state(checkpoint_path: Path, state_key: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict payload")
    if state_key not in checkpoint:
        raise KeyError(f"{state_key!r} not found; available keys: {sorted(checkpoint.keys())}")
    state = checkpoint[state_key]
    if not isinstance(state, dict):
        raise ValueError(f"{state_key!r} must be a state dict")
    return checkpoint, dict(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project an F5TTS checkpoint's linear weights to runtime-row BitNet ternary weights.")
    parser.add_argument("--checkpoint", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--state-key", default="model_state_dict")
    parser.add_argument("--include", default="")
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    include = split_csv(args.include)
    exclude = split_csv(args.exclude)
    payload, state = load_state(checkpoint_path, args.state_key)

    projected_state: dict[str, torch.Tensor] = {}
    projected_tensors = 0
    projected_params = 0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_values = 0
    max_abs_error = 0.0

    for name, tensor in state.items():
        if should_project(name, tensor, include, exclude):
            projected, _, _ = bitnet_runtime_row_project(tensor, eps=float(args.eps))
            diff = (projected.float() - tensor.float()).abs()
            sq = diff.pow(2)
            values = int(tensor.numel())
            total_abs_error += float(diff.sum().item())
            total_sq_error += float(sq.sum().item())
            total_values += values
            max_abs_error = max(max_abs_error, float(diff.max().item()))
            projected_state[name] = projected.detach().cpu()
            projected_tensors += 1
            projected_params += values
        else:
            projected_state[name] = tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else tensor

    summary = {
        "source_checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "state_key": str(args.state_key),
        "include": list(include),
        "exclude": list(exclude),
        "scheme": "bitnet_runtime_row_ternary_projection",
        "projected_tensors": int(projected_tensors),
        "projected_params": int(projected_params),
        "mae": float(total_abs_error / max(1, total_values)),
        "mse": float(total_sq_error / max(1, total_values)),
        "max_abs_error": float(max_abs_error),
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = dict(payload)
        output_payload[args.state_key] = projected_state
        output_payload["bitnet_projection"] = summary
        torch.save(output_payload, output_path)
        (output_path.parent / "bitnet_projection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
