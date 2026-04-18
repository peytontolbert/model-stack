from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from runtime.cache import allocate_model_kv_cache, evict_kv_cache
from runtime.decoding import beam_search, incremental_beam_search
from runtime.kv_cache import reorder_kv_cache_rows_ as runtime_reorder_kv_cache_rows_
from runtime.native import create_native_model_session
from runtime.ops import append_tokens as runtime_append_tokens
from runtime.ops import apply_sampling_mask as runtime_apply_sampling_mask
from runtime.ops import decode_positions as runtime_decode_positions
from runtime.ops import sample_with_policies as runtime_sample_with_policies
from runtime.ops import token_counts as runtime_token_counts
from runtime.sampling import (
    apply_no_repeat_ngram_mask,
    apply_presence_frequency_penalty,
    apply_temperature,
    apply_topk_mask,
    apply_topp_mask,
    apply_transformers_repetition_penalty,
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_id: Optional[int] = None
    no_repeat_ngram: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    sliding_window: Optional[int] = None
    beam_size: int = 1
    length_penalty: float = 1.0


def resolve_generation_sampling_mode(
    *,
    do_sample: bool | None = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> bool:
    if do_sample is not None:
        return bool(do_sample)
    if top_k is not None:
        return True
    if top_p is not None and float(top_p) < 1.0:
        return True
    return float(temperature) != 1.0


def build_generation_config(
    *,
    max_new_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    sliding_window: Optional[int] = None,
    beam_size: int = 1,
    length_penalty: float = 1.0,
) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_k=(int(top_k) if top_k is not None else None),
        top_p=(float(top_p) if top_p is not None else None),
        eos_id=(int(eos_id) if eos_id is not None else None),
        no_repeat_ngram=int(no_repeat_ngram),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        sliding_window=(int(sliding_window) if sliding_window is not None else None),
        beam_size=int(beam_size),
        length_penalty=float(length_penalty),
    )


def _resolve_generation_pad_id(model, cfg) -> int:
    pad_id = getattr(getattr(model, "cfg", object()), "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(cfg, "eos_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def _expand_attention_mask_for_beams(
    attention_mask: torch.Tensor | None,
    beam_size: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError("beam search currently supports rank-2 attention_mask only")
    batch_size, seq_len = attention_mask.shape
    return (
        attention_mask[:, None, :]
        .expand(batch_size, beam_size, seq_len)
        .reshape(batch_size * beam_size, seq_len)
        .contiguous()
    )


def _is_token_attention_mask(
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
) -> bool:
    return attention_mask.ndim == 2 and tuple(attention_mask.shape) == (int(batch_size), int(seq_len))


def _resolve_attention_mask_mode(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
) -> str:
    if attention_mask is None:
        return "none"
    if _is_token_attention_mask(attention_mask, batch_size=batch_size, seq_len=seq_len):
        return "token"
    return "explicit"


def _append_explicit_decode_attention_mask(
    attention_mask: torch.Tensor,
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = attention_mask
    if row_ids is not None:
        row_ids_long = row_ids.to(device=mask.device, dtype=torch.long).contiguous().view(-1)
        if mask.ndim == 3:
            if mask.shape[0] == 1 and row_ids_long.numel() > 1:
                mask = mask.expand(int(row_ids_long.numel()), mask.shape[1], mask.shape[2])
            elif mask.shape[0] > 1:
                mask = mask.index_select(0, row_ids_long)
        elif mask.ndim == 4:
            if mask.shape[0] == 1 and row_ids_long.numel() > 1:
                mask = mask.expand(int(row_ids_long.numel()), mask.shape[1], mask.shape[2], mask.shape[3])
            elif mask.shape[0] > 1:
                mask = mask.index_select(0, row_ids_long)

    src_len = int(mask.shape[-1]) + 1
    if mask.dtype == torch.bool:
        fill = False
    else:
        fill = 0.0

    if mask.ndim == 2:
        return torch.full((1, src_len), fill, dtype=mask.dtype, device=mask.device)
    if mask.ndim == 3:
        return torch.full((mask.shape[0], 1, src_len), fill, dtype=mask.dtype, device=mask.device)
    if mask.ndim == 4:
        return torch.full((mask.shape[0], mask.shape[1], 1, src_len), fill, dtype=mask.dtype, device=mask.device)
    raise ValueError(f"unsupported explicit attention_mask rank for decode append: {mask.ndim}")


def _append_beam_attention_mask(
    attention_mask: torch.Tensor | None,
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2 and attention_mask.shape[0] != attention_mask.shape[1]:
        rows = int(attention_mask.shape[0]) if row_ids is None else int(row_ids.numel())
        next_col = torch.ones(
            (rows, 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        next_mask, _ = runtime_append_tokens(attention_mask, next_col, row_ids=row_ids)
        return next_mask
    return _append_explicit_decode_attention_mask(attention_mask, row_ids=row_ids)


@torch.no_grad()
def _generate_with_full_forward_beam_search(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    config: GenerationConfig,
) -> torch.Tensor:
    if getattr(config, "do_sample", False):
        raise ValueError("beam search does not support sampling; pass do_sample=False and omit sampling-only knobs")

    beam_size = int(getattr(config, "beam_size", 1))
    beam_attention_mask = _expand_attention_mask_for_beams(attention_mask, beam_size)
    eos_id = int(config.eos_id) if config.eos_id is not None else -1
    pad_id = _resolve_generation_pad_id(model, config)

    def _step_fn(beams: torch.Tensor) -> torch.Tensor:
        return beams

    def _logits_fn(beams: torch.Tensor) -> torch.Tensor:
        nonlocal beam_attention_mask
        logits = model(
            beams.contiguous(),
            attn_mask=None,
            attention_mask=beam_attention_mask,
            cache=None,
            return_dict=False,
        )
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        elif logits.dim() != 2:
            raise ValueError(f"beam search logits must be rank-2 or rank-3, got shape {tuple(logits.shape)}")
        logits = apply_sampling_policies(logits, beams, config)
        beam_attention_mask = _append_beam_attention_mask(beam_attention_mask)
        return logits

    return beam_search(
        step_fn=_step_fn,
        logits_fn=_logits_fn,
        input_ids=input_ids,
        beam_size=beam_size,
        max_new_tokens=int(config.max_new_tokens),
        eos_id=eos_id,
        pad_id=pad_id,
        length_penalty=float(getattr(config, "length_penalty", 1.0)),
    )


@torch.no_grad()
def _generate_with_beam_search(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    config: GenerationConfig,
    cache=None,
    cache_pagesize: int = 128,
    cache_backend: str | None = None,
    trace: bool = False,
) -> torch.Tensor:
    if getattr(config, "do_sample", False):
        raise ValueError("beam search does not support sampling; pass do_sample=False and omit sampling-only knobs")
    if int(config.max_new_tokens) <= 0:
        return input_ids

    beam_size = int(getattr(config, "beam_size", 1))
    eos_id = int(config.eos_id) if config.eos_id is not None else -1
    pad_id = _resolve_generation_pad_id(model, config)

    def _fallback() -> torch.Tensor:
        return _generate_with_full_forward_beam_search(
            model,
            input_ids,
            attention_mask=attention_mask,
            config=config,
        )

    try:
        base_session = RuntimeGenerationSession.from_model(
            model,
            input_ids,
            attention_mask=attention_mask,
            cache=cache,
            cache_pagesize=int(cache_pagesize),
            cache_backend=cache_backend,
            trace=trace,
        )
        if base_session.cache is None:
            return _fallback()

        next_logits = base_session.prefill_next_logits()
        if next_logits is None:
            return _fallback()

        batch_size, prompt_len = input_ids.shape
        beams = (
            input_ids[:, None, :]
            .expand(batch_size, beam_size, prompt_len)
            .reshape(batch_size * beam_size, prompt_len)
            .contiguous()
        )

        prompt_logits = (
            next_logits[:, None, :]
            .expand(batch_size, beam_size, next_logits.shape[-1])
            .reshape(batch_size * beam_size, next_logits.shape[-1])
            .contiguous()
        )
        prompt_logits = apply_sampling_policies(prompt_logits, beams, config)
        beam_session: RuntimeGenerationSession | None = None

        def _advance_cached_beam(parent_rows: torch.Tensor, next_beams: torch.Tensor) -> torch.Tensor | None:
            nonlocal beam_session
            max_tokens = getattr(config, "sliding_window", None)
            if beam_session is None:
                prompt_row_ids = torch.div(parent_rows, beam_size, rounding_mode="floor")
                beam_session = RuntimeGenerationSession(
                    model=model,
                    seq=next_beams,
                    attention_mask=None,
                    cache=None,
                    trace=trace,
                )
                next_step_logits = beam_session.advance_beam_decode(
                    next_beams,
                    prompt_row_ids,
                    mask_row_ids=prompt_row_ids,
                    source_attention_mask=attention_mask,
                    source_cache=base_session.cache,
                    max_tokens=max_tokens,
                    policy="sliding-window",
                )
            else:
                next_step_logits = beam_session.advance_beam_decode(
                    next_beams,
                    parent_rows,
                    mask_row_ids=parent_rows,
                    max_tokens=max_tokens,
                    policy="sliding-window",
                )
            if next_step_logits is None:
                return None
            return apply_sampling_policies(next_step_logits, beam_session.seq, config)

        beam_result = incremental_beam_search(
            beams,
            prompt_logits,
            beam_size=beam_size,
            max_new_tokens=int(config.max_new_tokens),
            prompt_length=prompt_len,
            eos_id=eos_id,
            pad_id=pad_id,
            advance_fn=_advance_cached_beam,
        )
        if beam_result is None:
            return _fallback()
        beams, raw_scores, _finished, lengths, _parent_rows = beam_result
    except Exception:
        return _fallback()

    final_scores = raw_scores / lengths.float().pow(float(getattr(config, "length_penalty", 1.0)))
    best = final_scores.argmax(dim=-1)
    final_beams = beams.view(input_ids.shape[0], beam_size, -1)
    return final_beams[torch.arange(input_ids.shape[0], device=input_ids.device), best]


@torch.no_grad()
def greedy_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
    attention_mask: torch.Tensor | None = None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    sliding_window: int | None = None,
    cache_backend: str | None = None,
):
    cfg = build_generation_config(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        eos_id=eos_id,
        no_repeat_ngram=int(no_repeat_ngram),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        sliding_window=(int(sliding_window) if sliding_window is not None else None),
    )
    return generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        config=cfg,
        cache_backend=cache_backend,
    )


@torch.no_grad()
def sample_generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    attention_mask: torch.Tensor | None = None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    sliding_window: int | None = None,
    cache_backend: str | None = None,
):
    cfg = build_generation_config(
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
        no_repeat_ngram=int(no_repeat_ngram),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        sliding_window=(int(sliding_window) if sliding_window is not None else None),
    )
    return generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        config=cfg,
        cache_backend=cache_backend,
    )


def inherit_generation_defaults(model, cfg) -> None:
    try:
        if getattr(cfg, "sliding_window", None) is None:
            sw = getattr(getattr(model, "cfg", object()), "sliding_window", None)
            if sw is not None:
                cfg.sliding_window = int(sw)
    except Exception:
        pass


def apply_sampling_policies(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    cfg,
) -> torch.Tensor:
    x = logits
    if cfg.repetition_penalty is not None and float(cfg.repetition_penalty) != 1.0:
        x = apply_transformers_repetition_penalty(x, input_ids, float(cfg.repetition_penalty))
    if getattr(cfg, "do_sample", False):
        if cfg.temperature is not None and cfg.temperature != 1.0:
            x = apply_temperature(x, cfg.temperature)
    topk_mask = None
    topp_mask = None
    no_repeat_mask = None
    if getattr(cfg, "do_sample", False):
        if cfg.top_k is not None:
            topk_mask = apply_topk_mask(x, cfg.top_k)
        if cfg.top_p is not None:
            topp_mask = apply_topp_mask(x, cfg.top_p)
    if cfg.no_repeat_ngram and cfg.no_repeat_ngram > 0:
        no_repeat_mask = apply_no_repeat_ngram_mask(x, input_ids, cfg.no_repeat_ngram)
    if topk_mask is not None or topp_mask is not None or no_repeat_mask is not None:
        x = runtime_apply_sampling_mask(
            x,
            topk_mask=topk_mask,
            topp_mask=topp_mask,
            no_repeat_mask=no_repeat_mask,
        )
    if cfg.presence_penalty != 0.0 or cfg.frequency_penalty != 0.0:
        counts = runtime_token_counts(
            input_ids,
            vocab_size=x.shape[-1],
            dtype=x.dtype,
        )
        x = apply_presence_frequency_penalty(x, counts, cfg.presence_penalty, cfg.frequency_penalty)
    return x


class RuntimeGenerationSession:
    def __init__(
        self,
        model: torch.nn.Module,
        seq: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: object | None = None,
        trace: bool = False,
    ) -> None:
        self.model = model
        self.trace = bool(trace)
        self._native_session = None
        self._seq = seq
        self._attention_mask = attention_mask
        self._attention_mask_mode = _resolve_attention_mask_mode(
            attention_mask,
            batch_size=int(seq.shape[0]),
            seq_len=int(seq.shape[1]),
        )
        self._cache = cache
        try:
            self._native_session = create_native_model_session(
                model,
                seq,
                attention_mask=attention_mask,
                cache=cache,
                trace=self.trace,
            )
        except Exception:
            self._native_session = None

    @classmethod
    def from_model(
        cls,
        model,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cache=None,
        cache_pagesize: int = 128,
        cache_backend: str | None = None,
        trace: bool = False,
    ) -> "RuntimeGenerationSession":
        resolved_cache = cache
        if resolved_cache is None:
            try:
                resolved_cache = allocate_model_kv_cache(
                    model,
                    batch_size=int(input_ids.shape[0]),
                    pagesize=int(cache_pagesize),
                    backend=cache_backend,
                )
            except Exception:
                resolved_cache = None
        return cls(
            model=model,
            seq=input_ids,
            attention_mask=attention_mask,
            cache=resolved_cache,
            trace=bool(trace),
        )

    @property
    def native_session(self):
        return self._native_session

    @property
    def uses_native_session(self) -> bool:
        return self._native_session is not None

    @property
    def native_executor_kind(self) -> str:
        if self._native_session is None:
            return "python"
        return str(getattr(self._native_session, "native_executor_kind", "python"))

    @property
    def seq(self) -> torch.Tensor:
        if self._native_session is not None:
            return self._native_session.seq
        return self._seq

    @seq.setter
    def seq(self, value: torch.Tensor) -> None:
        self._seq = value
        if self._native_session is not None:
            self._native_session.seq = value

    @property
    def attention_mask(self) -> torch.Tensor | None:
        if self._native_session is not None:
            return self._native_session.attention_mask
        return self._attention_mask

    @property
    def attention_mask_mode(self) -> str:
        return self._attention_mask_mode

    def _set_attention_mask(self, value: torch.Tensor | None, *, mode: str | None = None) -> None:
        self._attention_mask = value
        if mode is None:
            mode = _resolve_attention_mask_mode(
                value,
                batch_size=int(self._seq.shape[0]),
                seq_len=int(self._seq.shape[1]),
            )
        self._attention_mask_mode = str(mode)
        if self._native_session is not None:
            self._native_session.attention_mask = value

    @attention_mask.setter
    def attention_mask(self, value: torch.Tensor | None) -> None:
        self._set_attention_mask(value)

    @property
    def cache(self):
        if self._native_session is not None:
            return self._native_session.cache
        return self._cache

    @cache.setter
    def cache(self, value) -> None:
        self._cache = value
        if self._native_session is not None:
            self._native_session.cache = value

    @property
    def batch_size(self) -> int:
        if self._native_session is not None:
            return int(self._native_session.batch_size)
        return int(self._seq.shape[0])

    @property
    def seq_len(self) -> int:
        if self._native_session is not None:
            return int(self._native_session.seq_len)
        return int(self._seq.shape[1])

    def disable_cache(self) -> None:
        self._cache = None
        if self._native_session is not None:
            self._native_session.disable_cache()
            return
        self.cache = None

    def prefill_next_logits(self) -> torch.Tensor | None:
        if self.cache is None:
            return None
        if self._native_session is not None:
            return self._native_session.prefill_next_logits()
        logits = self.model(
            self.seq.contiguous(),
            attn_mask=None,
            attention_mask=self.attention_mask,
            cache=self.cache,
            return_dict=False,
        )
        return logits[:, -1, :]

    def full_next_logits(self) -> torch.Tensor:
        if self._native_session is not None:
            return self._native_session.full_next_logits()
        logits = self.model(
            self.seq.contiguous(),
            attn_mask=None,
            attention_mask=self.attention_mask,
            cache=None,
            return_dict=False,
        )
        return logits[:, -1, :]

    def append(self, next_id: torch.Tensor) -> None:
        if self._native_session is not None:
            self._native_session.append(next_id)
            return
        if self.attention_mask is not None and self.attention_mask_mode == "explicit":
            next_seq, _ = runtime_append_tokens(self.seq, next_id, None)
            self.seq = next_seq
            self._set_attention_mask(_append_explicit_decode_attention_mask(self.attention_mask), mode="explicit")
            return
        next_seq, next_mask = runtime_append_tokens(self.seq, next_id, self.attention_mask)
        self.seq = next_seq
        self._set_attention_mask(next_mask, mode=("none" if next_mask is None else "token"))

    def decode_positions(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.cache is None:
            return None, None
        if self._native_session is not None:
            pos_ids, cache_pos = self._native_session.decode_positions()
            return pos_ids, cache_pos
        try:
            return runtime_decode_positions(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                reference=self.seq,
            )
        except Exception:
            return None, None

    def decode_next_logits(self) -> torch.Tensor | None:
        if self.cache is None:
            return None
        if self._native_session is not None:
            return self._native_session.decode_next_logits()
        pos_ids, cache_pos = self.decode_positions()
        logits = self.model(
            self.seq[:, -1:].contiguous(),
            attn_mask=None,
            attention_mask=self.attention_mask,
            cache=self.cache,
            position_ids=pos_ids,
            cache_position=cache_pos,
            return_dict=False,
        )
        return logits[:, -1, :]

    def evict_if_needed(self, max_tokens: int | None, *, policy: str = "sliding-window") -> None:
        if self.cache is None or max_tokens is None:
            return
        if self._native_session is not None:
            self._native_session.evict_if_needed(int(max_tokens), policy=policy)
            return
        evict_kv_cache(self.cache, int(max_tokens), policy=policy)

    def advance_beam_decode(
        self,
        next_beams: torch.Tensor,
        cache_row_ids: torch.Tensor,
        *,
        mask_row_ids: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        source_cache=None,
        max_tokens: int | None = None,
        policy: str = "sliding-window",
    ) -> torch.Tensor | None:
        if self._native_session is not None:
            max_tokens_arg = -1 if max_tokens is None else int(max_tokens)
            return self._native_session.advance_beam_decode(
                next_beams,
                cache_row_ids,
                mask_row_ids,
                source_attention_mask,
                source_cache,
                max_tokens_arg,
                policy,
            )
        self.seq = next_beams
        mask_in = self.attention_mask if source_attention_mask is None else source_attention_mask
        source_mode = self.attention_mask_mode if source_attention_mask is None else (
            "explicit"
            if self.attention_mask_mode == "explicit"
            else _resolve_attention_mask_mode(
                source_attention_mask,
                batch_size=int(source_attention_mask.shape[0]),
                seq_len=int(source_attention_mask.shape[-1]),
            )
        )
        next_mask = _append_beam_attention_mask(mask_in, mask_row_ids)
        self._set_attention_mask(next_mask, mode=("explicit" if source_mode == "explicit" else ("none" if next_mask is None else "token")))
        cache_in = self.cache if source_cache is None else source_cache
        if cache_in is None:
            self.cache = None
        else:
            self.cache = runtime_reorder_kv_cache_rows_(cache_in, cache_row_ids)
        self.evict_if_needed(max_tokens, policy=policy)
        return self.decode_next_logits()


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    cache=None,
    attention_mask: Optional[torch.Tensor] = None,
    config: Optional[GenerationConfig] = None,
    sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    cache_pagesize: int = 128,
    cache_backend: str | None = None,
) -> torch.Tensor:
    import os
    import traceback

    trace = (os.getenv("GEN_TRACE", "0") == "1")
    cfg = config or GenerationConfig()
    inherit_generation_defaults(model, cfg)

    if int(getattr(cfg, "beam_size", 1)) > 1:
        if sampler is not None:
            raise ValueError("beam search does not support a custom sampler")
        return _generate_with_beam_search(
            model,
            input_ids,
            attention_mask=attention_mask,
            config=cfg,
            cache=cache,
            cache_pagesize=int(cache_pagesize),
            cache_backend=cache_backend,
            trace=trace,
        )

    session = RuntimeGenerationSession.from_model(
        model,
        input_ids,
        attention_mask=attention_mask,
        cache=cache,
        cache_pagesize=int(cache_pagesize),
        cache_backend=cache_backend,
        trace=trace,
    )

    if trace:
        try:
            p = next(model.parameters())
            print(f"[engine] start: device={p.device} dtype={p.dtype} init_len={session.seq_len} do_sample={getattr(cfg,'do_sample',False)}")
        except Exception:
            pass

    next_logits = None
    if session.cache is not None:
        try:
            next_logits = session.prefill_next_logits()
        except Exception:
            if trace:
                print(f"[engine] exception in prompt prefill seq_len={session.seq_len}")
                traceback.print_exc()
            session.disable_cache()

    for step in range(int(cfg.max_new_tokens)):
        if next_logits is None:
            try:
                next_logits = session.full_next_logits()
            except Exception:
                if trace:
                    print(f"[engine] exception in model forward at step={step} seq_len={session.seq_len}")
                    traceback.print_exc()
                raise
        if sampler is not None:
            sampled_logits = apply_sampling_policies(next_logits, session.seq, cfg)
            next_id = sampler(sampled_logits)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)
        else:
            next_id = runtime_sample_with_policies(
                next_logits,
                session.seq,
                do_sample=getattr(cfg, "do_sample", False),
                temperature=float(cfg.temperature),
                top_k=(int(cfg.top_k) if cfg.top_k is not None else None),
                top_p=(float(cfg.top_p) if cfg.top_p is not None else None),
                no_repeat_ngram=int(cfg.no_repeat_ngram),
                repetition_penalty=float(cfg.repetition_penalty),
                presence_penalty=float(cfg.presence_penalty),
                frequency_penalty=float(cfg.frequency_penalty),
            )
        if trace:
            try:
                print(f"[engine] step={step} seq_len={session.seq_len} logits_dtype={next_logits.dtype} next_id={int(next_id[0,0].item())}")
            except Exception:
                pass
        session.append(next_id)
        if cfg.eos_id is not None and (next_id == int(cfg.eos_id)).all():
            break
        next_logits = None
        if session.cache is not None:
            try:
                next_logits = session.decode_next_logits()
            except Exception:
                if trace:
                    print(f"[engine] exception in decode step={step} seq_len={session.seq_len}")
                    traceback.print_exc()
                session.disable_cache()
            try:
                session.evict_if_needed(getattr(cfg, "sliding_window", None), policy="sliding-window")
            except Exception:
                pass
    return session.seq
