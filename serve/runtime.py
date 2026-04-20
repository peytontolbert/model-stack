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
from .scheduler import GenerationScheduler, SchedulerConfig


@dataclass
class RuntimeConfig:
    model_dir: str
    device: Optional[str] = None  # e.g., "cuda", "cpu"
    dtype: Optional[str] = None   # "float16", "bfloat16", "float32"
    kv_pagesize: int = 512


class ModelRuntime:
    def __init__(
        self,
        cfg: ModelConfig,
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        kv_pagesize: int,
        *,
        draft_model: torch.nn.Module | None = None,
        draft_model_source: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.draft_model = draft_model
        self.draft_model_source = draft_model_source
        self.device = device
        self.dtype = dtype
        self.kv_pagesize = int(kv_pagesize)
        self.native_runtime_info = native_runtime_info()
        self.default_kv_cache_backend = resolve_kv_cache_backend()
        self.scheduler_config = SchedulerConfig.from_env()
        try:
            self.default_num_speculative_tokens = max(int(os.environ.get("MODEL_STACK_NUM_SPECULATIVE_TOKENS", "0")), 0)
        except Exception:
            self.default_num_speculative_tokens = 0
        self.default_speculative_method = str(os.environ.get("MODEL_STACK_SPECULATIVE_METHOD", "")).strip().lower() or None
        self.default_rejection_sample_method = (
            str(
                os.environ.get("MODEL_STACK_SPEC_DECODING_ACCEPTANCE_METHOD")
                or os.environ.get("MODEL_STACK_REJECTION_SAMPLE_METHOD")
                or ""
            ).strip().lower()
            or None
        )
        self._scheduler: GenerationScheduler | None = None
        self.model.to(self.device)
        self.model.eval()
        if self.draft_model is not None:
            self.draft_model.to(self.device)
            self.draft_model.eval()

    @classmethod
    def from_dir(cls, model_dir: Optional[str] = None, *, device: Optional[str] = None, dtype: Optional[str] = None, kv_pagesize: int = 512) -> "ModelRuntime":
        indir = model_dir or os.environ.get("MODEL_DIR")
        if not indir:
            raise ValueError("MODEL_DIR environment variable not set and model_dir not provided")
        loaded = runtime_load_model_dir(indir, device=device, dtype=dtype)
        draft_model = None
        draft_model_source = os.environ.get("MODEL_STACK_DRAFT_MODEL_DIR") or os.environ.get("DRAFT_MODEL_DIR")
        if draft_model_source:
            draft_loaded = runtime_load_model_dir(draft_model_source, device=loaded.device, dtype=loaded.dtype)
            draft_model = draft_loaded.model
        return cls(
            loaded.cfg,
            loaded.model,
            loaded.device,
            loaded.dtype,
            kv_pagesize,
            draft_model=draft_model,
            draft_model_source=draft_model_source,
        )

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        *,
        cfg: ModelConfig | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        kv_pagesize: int = 512,
        draft_model: torch.nn.Module | None = None,
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
        prepared_draft_model = None
        if draft_model is not None:
            draft_cfg = runtime_resolve_model_config(draft_model)
            prepared_draft_model, _, _ = runtime_prepare_model_for_runtime(
                draft_model,
                device=resolved_device,
                dtype=resolved_dtype,
                config_dtype=getattr(draft_cfg, "dtype", None),
            )
        return cls(
            resolved_cfg,
            prepared_model,
            resolved_device,
            resolved_dtype,
            kv_pagesize,
            draft_model=prepared_draft_model,
        )

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
            "scheduler": {
                "enabled": bool(self.scheduler_config.enabled),
                "max_batch_size": int(self.scheduler_config.max_batch_size),
                "max_num_batched_tokens": int(self.scheduler_config.max_num_batched_tokens),
                "max_num_partial_prefills": int(self.scheduler_config.max_num_partial_prefills),
                "max_long_partial_prefills": int(self.scheduler_config.max_long_partial_prefills),
                "long_prefill_token_threshold": int(self.scheduler_config.long_prefill_token_threshold),
                "scheduling_policy": str(self.scheduler_config.scheduling_policy),
                "max_queue_delay_ms": int(self.scheduler_config.max_queue_delay_ms),
                "prefix_cache_size": int(self.scheduler_config.prefix_cache_size),
                "prefix_cache_min_tokens": int(self.scheduler_config.prefix_cache_min_tokens),
                "prefill_chunk_size": self.scheduler_config.prefill_chunk_size,
            },
            "speculative": {
                "draft_model_loaded": bool(self.draft_model is not None),
                "draft_model_source": self.draft_model_source,
                "default_num_speculative_tokens": int(self.default_num_speculative_tokens),
                "default_speculative_method": self.default_speculative_method,
                "default_rejection_sample_method": self.default_rejection_sample_method,
                "supported_methods": ["draft_model", "eagle", "eagle3", "mlp_speculator", "pard", "ngram", "suffix"],
                "supported_acceptance_methods": ["rejection_sampler", "strict", "typical_acceptance_sampler"],
            },
            "native_runtime": dict(self.native_runtime_info),
        }

    @property
    def scheduler(self) -> GenerationScheduler | None:
        if not self.scheduler_config.enabled:
            return None
        if self._scheduler is None:
            self._scheduler = GenerationScheduler(
                model=self.model,
                draft_model=self.draft_model,
                cache_pagesize=self.kv_pagesize,
                default_cache_backend=self.default_kv_cache_backend,
                config=self.scheduler_config,
            )
        return self._scheduler

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
        beam_size: int = 1,
        length_penalty: float = 1.0,
        prefill_chunk_size: int | None = None,
        num_speculative_tokens: int | None = None,
        speculative_method: str | None = None,
        rejection_sample_method: str | None = None,
        prompt_lookup_min: int = 2,
        prompt_lookup_max: int = 4,
        suffix_decoding_max_tree_depth: int = 32,
        suffix_decoding_max_spec_factor: float = 2.0,
        suffix_decoding_min_token_prob: float = 0.0,
        typical_acceptance_sampler_posterior_threshold: float = 0.09,
        typical_acceptance_sampler_posterior_alpha: float = 0.3,
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
            beam_size=beam_size,
            length_penalty=length_penalty,
            prefill_chunk_size=(
                prefill_chunk_size
                if prefill_chunk_size is not None
                else self.scheduler_config.prefill_chunk_size
            ),
            num_speculative_tokens=(
                self.default_num_speculative_tokens
                if num_speculative_tokens is None
                else num_speculative_tokens
            ),
            speculative_method=(
                self.default_speculative_method
                if speculative_method is None
                else speculative_method
            ),
            rejection_sample_method=(
                self.default_rejection_sample_method
                if rejection_sample_method is None
                else rejection_sample_method
            ),
            prompt_lookup_min=prompt_lookup_min,
            prompt_lookup_max=prompt_lookup_max,
            suffix_decoding_max_tree_depth=suffix_decoding_max_tree_depth,
            suffix_decoding_max_spec_factor=suffix_decoding_max_spec_factor,
            suffix_decoding_min_token_prob=suffix_decoding_min_token_prob,
            typical_acceptance_sampler_posterior_threshold=typical_acceptance_sampler_posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=typical_acceptance_sampler_posterior_alpha,
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
            draft_model=self.draft_model,
        )

    def generate_token_ids(
        self,
        input_ids,
        *,
        attention_mask=None,
        config: GenerationConfig | None = None,
        cache_backend: Optional[str] = None,
        priority: int = 0,
    ) -> torch.Tensor:
        ids = self.coerce_input_ids(input_ids)
        mask = self.coerce_attention_mask(attention_mask, batch_size=int(ids.shape[0]), seq_len=int(ids.shape[1]))
        if (
            int(ids.shape[0]) == 1
            and self.scheduler is not None
            and (config is None or int(getattr(config, "beam_size", 1)) <= 1)
        ):
            with torch.inference_mode():
                return self.scheduler.submit(
                    ids,
                    attention_mask=mask,
                    config=(config or self.build_generation_config()),
                    cache_backend=(cache_backend or self.default_kv_cache_backend),
                    priority=int(priority),
                )
        use_cache = config is None or int(getattr(config, "beam_size", 1)) <= 1
        cache = self.allocate_cache(batch_size=int(ids.shape[0]), backend=cache_backend) if use_cache else None
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
        priority: int = 0,
    ) -> list[list[int]]:
        return self.generate_token_ids(
            input_ids,
            attention_mask=attention_mask,
            config=config,
            cache_backend=cache_backend,
            priority=int(priority),
        ).tolist()
