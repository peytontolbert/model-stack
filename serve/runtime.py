from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

from specs.config import ModelConfig
from runtime.cache import create_kv_cache, kv_cache_runtime_info, kv_cache_spec_from_config, resolve_kv_cache_backend
from runtime.generation import (
    GenerationConfig,
    RuntimeGenerationSession,
    build_generation_config as runtime_build_generation_config,
    generate as runtime_generate,
    resolve_generation_sampling_mode as runtime_resolve_generation_sampling_mode,
)
from runtime.loader import load_model_dir as runtime_load_model_dir
from runtime.prep import (
    prepare_model_for_runtime as runtime_prepare_model_for_runtime,
    resolve_model_config as runtime_resolve_model_config,
)
from runtime.native import runtime_info as native_runtime_info


@dataclass
class RuntimeConfig:
    model_dir: str
    device: Optional[str] = None  # e.g., "cuda", "cpu"
    dtype: Optional[str] = None   # "float16", "bfloat16", "float32"
    kv_pagesize: int = 512


class ModelRuntime:
    def __init__(self, cfg: ModelConfig, model: torch.nn.Module, device: torch.device, dtype: torch.dtype, kv_pagesize: int) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.dtype = dtype
        self.kv_pagesize = int(kv_pagesize)
        self.native_runtime_info = native_runtime_info()
        self.default_kv_cache_backend = resolve_kv_cache_backend()
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_dir(cls, model_dir: Optional[str] = None, *, device: Optional[str] = None, dtype: Optional[str] = None, kv_pagesize: int = 512) -> "ModelRuntime":
        indir = model_dir or os.environ.get("MODEL_DIR")
        if not indir:
            raise ValueError("MODEL_DIR environment variable not set and model_dir not provided")
        loaded = runtime_load_model_dir(indir, device=device, dtype=dtype)
        return cls(loaded.cfg, loaded.model, loaded.device, loaded.dtype, kv_pagesize)

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        *,
        cfg: ModelConfig | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        kv_pagesize: int = 512,
    ) -> "ModelRuntime":
        resolved_cfg = runtime_resolve_model_config(model, fallback=cfg)
        if resolved_cfg is None:
            raise ValueError("ModelRuntime.from_model requires an explicit cfg or a model.cfg ModelConfig")
        prepared_model, resolved_device, resolved_dtype = runtime_prepare_model_for_runtime(
            model,
            device=device,
            dtype=dtype,
            config_dtype=getattr(resolved_cfg, "dtype", None),
        )
        return cls(resolved_cfg, prepared_model, resolved_device, resolved_dtype, kv_pagesize)

    def cache_spec(self, batch_size: int, *, backend: Optional[str] = None):
        return kv_cache_spec_from_config(
            self.cfg,
            batch_size=int(batch_size),
            dtype=self.dtype,
            device=self.device,
            pagesize=self.kv_pagesize,
            backend=backend,
        )

    def allocate_cache(self, batch_size: int, *, backend: Optional[str] = None):
        return create_kv_cache(self.cache_spec(batch_size, backend=backend))

    def cache_info(self, batch_size: int = 1, *, backend: Optional[str] = None) -> dict[str, object]:
        cache = self.allocate_cache(batch_size=int(batch_size), backend=backend)
        info = kv_cache_runtime_info(cache)
        info["requested_batch"] = int(batch_size)
        return info

    def health_info(self) -> dict[str, object]:
        return {
            "status": "ok",
            "device": str(self.device),
            "dtype": str(self.dtype),
            "kv_cache_backend": str(self.default_kv_cache_backend),
            "native_runtime": dict(self.native_runtime_info),
        }

    def build_generation_config(
        self,
        *,
        max_new_tokens: int = 64,
        do_sample: bool | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
        no_repeat_ngram: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        sliding_window: int | None = None,
    ) -> GenerationConfig:
        resolved_do_sample = runtime_resolve_generation_sampling_mode(
            do_sample=do_sample,
            temperature=float(temperature),
            top_k=top_k,
            top_p=top_p,
        )
        return runtime_build_generation_config(
            max_new_tokens=max_new_tokens,
            do_sample=resolved_do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=eos_id,
            no_repeat_ngram=no_repeat_ngram,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            sliding_window=sliding_window,
        )

    def coerce_input_ids(self, input_ids) -> torch.Tensor:
        try:
            ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids)
        except Exception as exc:
            raise ValueError(f"invalid input_ids: {exc}") from exc
        ids = ids.to(device=self.device, dtype=torch.long)
        if ids.ndim != 2:
            raise ValueError(f"invalid input_ids: expected rank-2 [B, T], got shape {tuple(ids.shape)}")
        return ids

    def coerce_attention_mask(self, attention_mask, *, batch_size: int, seq_len: int) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        try:
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.tensor(attention_mask)
        except Exception as exc:
            raise ValueError(f"invalid attention_mask: {exc}") from exc
        mask = mask.to(device=self.device, dtype=torch.long)
        if mask.ndim != 2 or tuple(mask.shape) != (int(batch_size), int(seq_len)):
            raise ValueError(
                f"invalid attention_mask: expected shape {(int(batch_size), int(seq_len))}, got {tuple(mask.shape)}"
            )
        return mask

    def create_generation_session(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cache=None,
        cache_backend: Optional[str] = None,
    ) -> RuntimeGenerationSession:
        return RuntimeGenerationSession.from_model(
            self.model,
            input_ids.to(self.device),
            attention_mask=(attention_mask.to(self.device) if attention_mask is not None else None),
            cache=cache,
            cache_pagesize=self.kv_pagesize,
            cache_backend=(cache_backend or self.default_kv_cache_backend),
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cache=None,
        config=None,
        sampler=None,
        cache_backend: Optional[str] = None,
    ) -> torch.Tensor:
        return runtime_generate(
            self.model,
            input_ids.to(self.device),
            attention_mask=(attention_mask.to(self.device) if attention_mask is not None else None),
            cache=cache,
            config=config,
            sampler=sampler,
            cache_pagesize=self.kv_pagesize,
            cache_backend=(cache_backend or self.default_kv_cache_backend),
        )

    def generate_token_ids(
        self,
        input_ids,
        *,
        attention_mask=None,
        config: GenerationConfig | None = None,
        cache_backend: Optional[str] = None,
    ) -> torch.Tensor:
        ids = self.coerce_input_ids(input_ids)
        mask = self.coerce_attention_mask(attention_mask, batch_size=int(ids.shape[0]), seq_len=int(ids.shape[1]))
        cache = self.allocate_cache(batch_size=int(ids.shape[0]), backend=cache_backend)
        with torch.inference_mode():
            return self.generate(
                ids,
                attention_mask=mask,
                cache=cache,
                config=config,
                cache_backend=cache_backend,
            )

    def generate_token_lists(
        self,
        input_ids,
        *,
        attention_mask=None,
        config: GenerationConfig | None = None,
        cache_backend: Optional[str] = None,
    ) -> list[list[int]]:
        return self.generate_token_ids(
            input_ids,
            attention_mask=attention_mask,
            config=config,
            cache_backend=cache_backend,
        ).tolist()
