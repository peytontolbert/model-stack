from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from runtime.cache import allocate_model_kv_cache, evict_kv_cache
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
    )


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


@dataclass
class RuntimeGenerationSession:
    model: torch.nn.Module
    seq: torch.Tensor
    attention_mask: torch.Tensor | None = None
    cache: object | None = None
    trace: bool = False

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
    def batch_size(self) -> int:
        return int(self.seq.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.seq.shape[1])

    def disable_cache(self) -> None:
        self.cache = None

    def prefill_next_logits(self) -> torch.Tensor | None:
        if self.cache is None:
            return None
        logits = self.model(
            self.seq.contiguous(),
            attn_mask=None,
            attention_mask=self.attention_mask,
            cache=self.cache,
            return_dict=False,
        )
        return logits[:, -1, :]

    def full_next_logits(self) -> torch.Tensor:
        logits = self.model(
            self.seq.contiguous(),
            attn_mask=None,
            attention_mask=self.attention_mask,
            cache=None,
            return_dict=False,
        )
        return logits[:, -1, :]

    def append(self, next_id: torch.Tensor) -> None:
        self.seq, self.attention_mask = runtime_append_tokens(self.seq, next_id, self.attention_mask)

    def decode_positions(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.cache is None:
            return None, None
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
        evict_kv_cache(self.cache, int(max_tokens), policy=policy)


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
