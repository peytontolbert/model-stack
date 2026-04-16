from __future__ import annotations

import importlib
from typing import Optional

import torch

import runtime.checkpoint as runtime_checkpoint_mod


def load_model_dir(
    model_dir: str,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    strict: bool = True,
    task: str = "causal-lm",
    block: str = "llama",
    compress: Optional[dict] = None,
    eval_mode: bool = True,
    **kwargs,
):
    from runtime.factory import build_model
    from runtime.modeling import RuntimeModelArtifacts, prepare_model_for_runtime

    cfg = runtime_checkpoint_mod.load_config(model_dir)
    model = build_model(cfg, task=task, block=block, compress=compress, **kwargs)
    model = runtime_checkpoint_mod.load_pretrained(model, model_dir, strict=bool(strict))
    model, resolved_device, resolved_dtype = prepare_model_for_runtime(
        model,
        device=device,
        dtype=dtype,
        config_dtype=getattr(cfg, "dtype", None),
        eval_mode=eval_mode,
    )
    return RuntimeModelArtifacts(
        cfg=cfg,
        model=model,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def load_model_factory_spec(
    spec: str,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    eval_mode: bool = True,
):
    from runtime.modeling import RuntimeModelArtifacts, prepare_model_for_runtime, resolve_model_config

    mod_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    factory = getattr(module, fn_name)
    out = factory()
    model = out[0] if isinstance(out, tuple) else out
    cfg = resolve_model_config(model)
    model, resolved_device, resolved_dtype = prepare_model_for_runtime(
        model,
        device=device,
        dtype=dtype,
        config_dtype=getattr(cfg, "dtype", None),
        eval_mode=eval_mode,
    )
    return RuntimeModelArtifacts(
        cfg=cfg,
        model=model,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def load_runtime_model(
    *,
    model_dir: str | None = None,
    factory_spec: str | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    strict: bool = True,
    task: str = "causal-lm",
    block: str = "llama",
    compress: Optional[dict] = None,
    eval_mode: bool = True,
    **kwargs,
):
    if (model_dir is None) == (factory_spec is None):
        raise ValueError("Exactly one of model_dir or factory_spec must be provided")
    if model_dir is not None:
        return load_model_dir(
            model_dir,
            device=device,
            dtype=dtype,
            strict=strict,
            task=task,
            block=block,
            compress=compress,
            eval_mode=eval_mode,
            **kwargs,
        )
    return load_model_factory_spec(
        factory_spec,
        device=device,
        dtype=dtype,
        eval_mode=eval_mode,
    )
