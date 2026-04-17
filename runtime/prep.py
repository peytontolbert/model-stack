from __future__ import annotations

from dataclasses import dataclass

import torch

from specs.config import ModelConfig


@dataclass(frozen=True)
class RuntimeModelArtifacts:
    cfg: ModelConfig | None
    model: torch.nn.Module
    device: torch.device
    dtype: torch.dtype


def resolve_model_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def resolve_model_dtype(
    dtype: str | torch.dtype | None = None,
    *,
    config_dtype: str | torch.dtype | None = None,
) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(config_dtype, torch.dtype):
        return config_dtype
    resolved_name = str(dtype or config_dtype or "bfloat16").lower()
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(resolved_name, torch.bfloat16)


def prepare_model_for_runtime(
    model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    config_dtype: str | torch.dtype | None = None,
    eval_mode: bool = True,
) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
    resolved_device = resolve_model_device(device)
    resolved_dtype = resolve_model_dtype(dtype, config_dtype=config_dtype)
    model = model.to(device=resolved_device, dtype=resolved_dtype)
    if eval_mode:
        model.eval()
    return model, resolved_device, resolved_dtype


def resolve_model_config(model: torch.nn.Module, *, fallback: ModelConfig | None = None) -> ModelConfig | None:
    cfg = getattr(model, "cfg", None)
    return cfg if isinstance(cfg, ModelConfig) else fallback
