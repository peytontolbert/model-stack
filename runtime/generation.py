from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Optional

import torch

from runtime.cache import allocate_model_kv_cache, evict_kv_cache
from runtime.decoding import beam_search, incremental_beam_search
from runtime.kv_cache import (
    concat_kv_caches,
    reorder_kv_cache_rows_ as runtime_reorder_kv_cache_rows_,
    split_kv_cache_rows,
    truncate_kv_cache_prefix,
)
from runtime.native import create_native_model_session
from runtime.ops import append_tokens as runtime_append_tokens
from runtime.ops import apply_sampling_mask as runtime_apply_sampling_mask
from runtime.ops import decode_positions as runtime_decode_positions
from runtime.ops import speculative_accept as runtime_speculative_accept
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
    prefill_chunk_size: Optional[int] = None
    num_speculative_tokens: int = 0
    speculative_method: str | None = None
    rejection_sample_method: str = "strict"
    prompt_lookup_min: int = 2
    prompt_lookup_max: int = 4
    suffix_decoding_max_tree_depth: int = 32
    suffix_decoding_max_spec_factor: float = 2.0
    suffix_decoding_min_token_prob: float = 0.0
    typical_acceptance_sampler_posterior_threshold: float = 0.09
    typical_acceptance_sampler_posterior_alpha: float = 0.3


@dataclass
class SpeculativeDecodeResult:
    seq: torch.Tensor
    attention_mask: torch.Tensor | None
    cache: object | None
    emitted_tokens: int
    accepted_tokens: int
    drafted_tokens: int
    ready_logits: torch.Tensor | None = None


@dataclass
class SpeculativeProposal:
    tokens: torch.Tensor
    draft_probs: torch.Tensor
    method: str


@dataclass
class SpeculativeBatchRequest:
    seq: torch.Tensor
    attention_mask: torch.Tensor | None
    cache: object | None
    config: GenerationConfig
    ready_logits: torch.Tensor | None = None
    prompt_seq: torch.Tensor | None = None
    cache_backend: str | None = None
    remaining_new_tokens: int | None = None


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
    prefill_chunk_size: Optional[int] = None,
    num_speculative_tokens: int = 0,
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
        prefill_chunk_size=(int(prefill_chunk_size) if prefill_chunk_size is not None else None),
        num_speculative_tokens=max(int(num_speculative_tokens), 0),
        speculative_method=(None if speculative_method is None else str(speculative_method).strip().lower() or None),
        rejection_sample_method=str(rejection_sample_method or ("rejection_sampler" if bool(do_sample) else "strict")).strip().lower(),
        prompt_lookup_min=max(int(prompt_lookup_min), 1),
        prompt_lookup_max=max(int(prompt_lookup_max), 1),
        suffix_decoding_max_tree_depth=max(int(suffix_decoding_max_tree_depth), 1),
        suffix_decoding_max_spec_factor=max(float(suffix_decoding_max_spec_factor), 1.0),
        suffix_decoding_min_token_prob=max(float(suffix_decoding_min_token_prob), 0.0),
        typical_acceptance_sampler_posterior_threshold=max(float(typical_acceptance_sampler_posterior_threshold), 0.0),
        typical_acceptance_sampler_posterior_alpha=max(float(typical_acceptance_sampler_posterior_alpha), 0.0),
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


def _select_last_token_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if logits.dim() == 2:
        return logits
    if attention_mask is not None and attention_mask.ndim == 2 and tuple(attention_mask.shape) == tuple(logits.shape[:2]):
        last_idx = attention_mask.to(torch.long).sum(dim=-1).clamp_min(1).sub(1)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        return logits.gather(1, gather_idx).squeeze(1)
    return logits[:, -1, :]


def _slice_prefill_attention_mask(
    attention_mask: torch.Tensor | None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        return attention_mask[:, int(start): int(end)].contiguous()
    return None


def _append_token_attention_mask(
    attention_mask: torch.Tensor | None,
    count: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError("speculative decoding only supports rank-2 token attention masks")
    append = torch.ones(
        (int(attention_mask.shape[0]), int(count)),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, append], dim=1).contiguous()


def _append_generated_tokens(
    seq: torch.Tensor,
    attention_mask: torch.Tensor | None,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if int(tokens.numel()) == 0:
        return seq.contiguous(), (attention_mask.contiguous() if attention_mask is not None else None)
    next_seq = torch.cat([seq, tokens.contiguous()], dim=1).contiguous()
    next_attention_mask = _append_token_attention_mask(attention_mask, int(tokens.shape[1]))
    return next_seq, next_attention_mask


def _sample_next_token(
    logits: torch.Tensor,
    seq: torch.Tensor,
    cfg,
) -> torch.Tensor:
    next_id = runtime_sample_with_policies(
        logits,
        seq,
        do_sample=bool(getattr(cfg, "do_sample", False)),
        temperature=float(getattr(cfg, "temperature", 1.0)),
        top_k=(int(cfg.top_k) if getattr(cfg, "top_k", None) is not None else None),
        top_p=(float(cfg.top_p) if getattr(cfg, "top_p", None) is not None else None),
        no_repeat_ngram=int(getattr(cfg, "no_repeat_ngram", 0)),
        repetition_penalty=float(getattr(cfg, "repetition_penalty", 1.0)),
        presence_penalty=float(getattr(cfg, "presence_penalty", 0.0)),
        frequency_penalty=float(getattr(cfg, "frequency_penalty", 0.0)),
    )
    if next_id.ndim == 1:
        next_id = next_id.unsqueeze(-1)
    return next_id.contiguous()


def _supports_speculative_decode(
    draft_model,
    seq: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache,
    cfg,
) -> bool:
    method = _resolve_speculative_method(cfg, draft_model)
    if cache is None or method is None:
        return False
    if int(getattr(cfg, "num_speculative_tokens", 0)) <= 0:
        return False
    if int(getattr(cfg, "beam_size", 1)) > 1:
        return False
    if seq.ndim != 2 or int(seq.shape[0]) != 1:
        return False
    if attention_mask is not None and attention_mask.ndim != 2:
        return False
    if _speculative_method_requires_draft_model(method) and draft_model is None:
        return False
    return not _token_mask_has_padding(seq, attention_mask)


_MODEL_BASED_SPECULATIVE_METHODS = {"draft_model", "eagle", "eagle3", "mlp_speculator", "pard"}
_LOOKUP_SPECULATIVE_METHODS = {"ngram", "suffix"}


def _normalize_speculative_method(method: str | None) -> str | None:
    if method is None:
        return None
    text = str(method).strip().lower()
    if not text:
        return None
    aliases = {
        "draft": "draft_model",
        "draft-model": "draft_model",
        "n-gram": "ngram",
        "mlp": "mlp_speculator",
    }
    return aliases.get(text, text)


def _resolve_speculative_method(cfg, draft_model) -> str | None:
    method = _normalize_speculative_method(getattr(cfg, "speculative_method", None))
    if method is not None:
        return method
    if int(getattr(cfg, "num_speculative_tokens", 0)) <= 0:
        return None
    if draft_model is not None:
        return "draft_model"
    return None


def _speculative_method_requires_draft_model(method: str | None) -> bool:
    return method in _MODEL_BASED_SPECULATIVE_METHODS


def _speculative_accept_method(cfg) -> str:
    method = str(getattr(cfg, "rejection_sample_method", "")).strip().lower()
    if not method:
        return "rejection_sampler" if bool(getattr(cfg, "do_sample", False)) else "strict"
    aliases = {
        "probabilistic": "rejection_sampler",
        "rejection": "rejection_sampler",
        "rejection_sampler": "rejection_sampler",
        "strict": "strict",
        "typical": "typical_acceptance_sampler",
        "typical_acceptance_sampler": "typical_acceptance_sampler",
    }
    return aliases.get(method, method)


def _policy_logits(logits: torch.Tensor, seq: torch.Tensor, cfg) -> torch.Tensor:
    return apply_sampling_policies(logits.contiguous(), seq, cfg)


def _sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    sampled = torch.multinomial(probs.to(torch.float32), num_samples=1)
    return sampled.to(device=probs.device, dtype=torch.long)


def _one_hot_probs(token: torch.Tensor, vocab_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    out = torch.zeros((int(token.shape[0]), int(vocab_size)), dtype=dtype, device=device)
    out.scatter_(1, token.to(torch.long), 1.0)
    return out


def _policy_probs(logits: torch.Tensor, seq: torch.Tensor, cfg) -> torch.Tensor:
    adjusted = _policy_logits(logits, seq, cfg)
    if bool(getattr(cfg, "do_sample", False)):
        probs = torch.softmax(adjusted.to(torch.float32), dim=-1)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    token = torch.argmax(adjusted, dim=-1, keepdim=True)
    return _one_hot_probs(token, adjusted.shape[-1], dtype=torch.float32, device=adjusted.device)


def _policy_token_and_probs(logits: torch.Tensor, seq: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    probs = _policy_probs(logits, seq, cfg)
    if bool(getattr(cfg, "do_sample", False)):
        return _sample_from_probs(probs), probs
    token = torch.argmax(probs, dim=-1, keepdim=True)
    return token, probs


def _counts_to_probs(
    vocab_size: int,
    counts: dict[int, int],
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if not counts:
        return None
    probs = torch.zeros((1, int(vocab_size)), dtype=torch.float32, device=device)
    total = 0.0
    for token_id, count in counts.items():
        if 0 <= int(token_id) < int(vocab_size) and int(count) > 0:
            probs[0, int(token_id)] = float(count)
            total += float(count)
    if total <= 0.0:
        return None
    return probs / total


def _extract_1d_tokens(x: torch.Tensor | None, fallback: torch.Tensor) -> list[int]:
    base = fallback if x is None else x
    return [int(token) for token in base.view(-1).detach().to(torch.long).cpu().tolist()]


def _build_model_based_speculative_proposal(
    draft_model,
    seq: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    config: GenerationConfig,
    proposal_budget: int,
    cache_pagesize: int,
    cache_backend: str | None,
) -> SpeculativeProposal | None:
    draft_session = RuntimeGenerationSession.from_model(
        draft_model,
        seq,
        attention_mask=attention_mask,
        cache=None,
        cache_pagesize=int(cache_pagesize),
        cache_backend=cache_backend,
        trace=False,
    )
    draft_logits = (
        draft_session.prefill_next_logits(chunk_size=getattr(config, "prefill_chunk_size", None))
        if draft_session.cache is not None
        else draft_session.full_next_logits()
    )
    if draft_logits is None:
        return None

    tokens: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    for _ in range(int(proposal_budget)):
        token, token_probs = _policy_token_and_probs(draft_logits, draft_session.seq, config)
        tokens.append(token)
        probs.append(token_probs.unsqueeze(1))
        draft_session.append(token)
        if config.eos_id is not None and bool((token == int(config.eos_id)).all().item()):
            break
        if len(tokens) >= int(proposal_budget):
            break
        draft_logits = (
            draft_session.decode_next_logits()
            if draft_session.cache is not None
            else draft_session.full_next_logits()
        )
        if draft_logits is None:
            break
    if not tokens:
        return None
    return SpeculativeProposal(
        tokens=torch.cat(tokens, dim=1).contiguous(),
        draft_probs=torch.cat(probs, dim=1).contiguous(),
        method=str(_resolve_speculative_method(config, draft_model) or "draft_model"),
    )


def _build_ngram_speculative_proposal(
    seq: torch.Tensor,
    *,
    prompt_seq: torch.Tensor | None,
    config: GenerationConfig,
    proposal_budget: int,
    vocab_size: int,
) -> SpeculativeProposal | None:
    source = _extract_1d_tokens(prompt_seq, seq)
    working = _extract_1d_tokens(None, seq)
    tokens: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    min_n = max(min(int(getattr(config, "prompt_lookup_min", 2)), int(getattr(config, "prompt_lookup_max", 4))), 1)
    max_n = max(int(getattr(config, "prompt_lookup_max", 4)), min_n)
    device = seq.device

    for _ in range(int(proposal_budget)):
        chosen_probs = None
        max_window = min(max_n, len(working), max(len(source) - 1, 0))
        for n in range(max_window, min_n - 1, -1):
            pattern = working[-n:]
            counts: dict[int, int] = {}
            for idx in range(0, len(source) - n):
                if source[idx: idx + n] != pattern:
                    continue
                next_token = int(source[idx + n])
                counts[next_token] = counts.get(next_token, 0) + 1
            chosen_probs = _counts_to_probs(vocab_size, counts, device=device)
            if chosen_probs is not None:
                break
        if chosen_probs is None:
            break
        if bool(getattr(config, "do_sample", False)):
            token = _sample_from_probs(chosen_probs)
            token_probs = chosen_probs
        else:
            token = torch.argmax(chosen_probs, dim=-1, keepdim=True)
            token_probs = _one_hot_probs(token, vocab_size, dtype=torch.float32, device=device)
        tokens.append(token)
        probs.append(token_probs.unsqueeze(1))
        working.append(int(token.item()))
        if config.eos_id is not None and int(token.item()) == int(config.eos_id):
            break
    if not tokens:
        return None
    return SpeculativeProposal(
        tokens=torch.cat(tokens, dim=1).contiguous(),
        draft_probs=torch.cat(probs, dim=1).contiguous(),
        method="ngram",
    )


def _build_suffix_speculative_proposal(
    seq: torch.Tensor,
    *,
    prompt_seq: torch.Tensor | None,
    config: GenerationConfig,
    proposal_budget: int,
    vocab_size: int,
) -> SpeculativeProposal | None:
    base_source = _extract_1d_tokens(prompt_seq, seq)
    working = _extract_1d_tokens(None, seq)
    tokens: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    device = seq.device
    max_depth = max(int(getattr(config, "suffix_decoding_max_tree_depth", 32)), 1)
    max_factor = max(float(getattr(config, "suffix_decoding_max_spec_factor", 2.0)), 1.0)
    min_prob = max(float(getattr(config, "suffix_decoding_min_token_prob", 0.0)), 0.0)

    for _ in range(int(proposal_budget)):
        source = base_source + working[len(base_source):]
        chosen_probs = None
        chosen_width = 0
        max_window = min(max_depth, len(working) - 1 if len(working) > 1 else 0)
        for n in range(max_window, 0, -1):
            if len(tokens) >= max(1, int(round(float(n) * max_factor))):
                continue
            pattern = working[-n:]
            counts: dict[int, int] = {}
            for idx in range(0, len(source) - n):
                if source[idx: idx + n] != pattern:
                    continue
                next_token = int(source[idx + n])
                counts[next_token] = counts.get(next_token, 0) + 1
            probs_candidate = _counts_to_probs(vocab_size, counts, device=device)
            if probs_candidate is None:
                continue
            max_token_prob = float(probs_candidate.max().item()) if probs_candidate.numel() > 0 else 0.0
            if max_token_prob < min_prob:
                continue
            chosen_probs = probs_candidate
            chosen_width = n
            break
        if chosen_probs is None or chosen_width <= 0:
            break
        if bool(getattr(config, "do_sample", False)):
            token = _sample_from_probs(chosen_probs)
            token_probs = chosen_probs
        else:
            token = torch.argmax(chosen_probs, dim=-1, keepdim=True)
            token_probs = _one_hot_probs(token, vocab_size, dtype=torch.float32, device=device)
        tokens.append(token)
        probs.append(token_probs.unsqueeze(1))
        working.append(int(token.item()))
        if config.eos_id is not None and int(token.item()) == int(config.eos_id):
            break
    if not tokens:
        return None
    return SpeculativeProposal(
        tokens=torch.cat(tokens, dim=1).contiguous(),
        draft_probs=torch.cat(probs, dim=1).contiguous(),
        method="suffix",
    )


def _build_speculative_proposal(
    draft_model,
    seq: torch.Tensor,
    *,
    prompt_seq: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    current_logits: torch.Tensor,
    config: GenerationConfig,
    proposal_budget: int,
    cache_pagesize: int,
    cache_backend: str | None,
) -> SpeculativeProposal | None:
    method = _resolve_speculative_method(config, draft_model)
    if method is None or proposal_budget <= 0:
        return None
    if method in _MODEL_BASED_SPECULATIVE_METHODS:
        return _build_model_based_speculative_proposal(
            draft_model,
            seq,
            attention_mask=attention_mask,
            config=config,
            proposal_budget=proposal_budget,
            cache_pagesize=cache_pagesize,
            cache_backend=cache_backend,
        )
    vocab_size = int(current_logits.shape[-1])
    if method == "ngram":
        return _build_ngram_speculative_proposal(
            seq,
            prompt_seq=prompt_seq,
            config=config,
            proposal_budget=proposal_budget,
            vocab_size=vocab_size,
        )
    if method == "suffix":
        return _build_suffix_speculative_proposal(
            seq,
            prompt_seq=prompt_seq,
            config=config,
            proposal_budget=proposal_budget,
            vocab_size=vocab_size,
        )
    return None


def _token_mask_lengths(
    seq: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if not _is_token_attention_mask(
        attention_mask,
        batch_size=int(seq.shape[0]),
        seq_len=int(seq.shape[1]),
    ):
        return None
    return attention_mask.to(torch.long).sum(dim=-1).clamp_min(1)


def _token_mask_has_padding(
    seq: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> bool:
    lengths = _token_mask_lengths(seq, attention_mask)
    if lengths is None:
        return False
    return not bool((lengths == int(seq.shape[1])).all().item())


def _resolve_decode_step_inputs(
    seq: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    lengths = _token_mask_lengths(seq, attention_mask)
    if lengths is None:
        return seq[:, -1:].contiguous(), None, None
    last_idx = lengths.sub(1).to(torch.long)
    decode_tokens = seq.gather(1, last_idx.view(-1, 1)).contiguous()
    position_ids = last_idx.view(-1, 1).contiguous()
    return decode_tokens, position_ids, last_idx.contiguous()


def _cache_length(cache) -> int:
    if cache is None or not hasattr(cache, "layer"):
        return 0
    try:
        return int(cache.layer(0).length())
    except Exception:
        return 0


def _native_decode_graph_enabled_by_env() -> bool:
    return os.getenv("MODEL_STACK_DISABLE_NATIVE_DECODE_GRAPH", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _try_build_native_decode_graph_replay(native_session) -> tuple[Callable[[], torch.Tensor | None] | None, torch.Tensor | None]:
    if native_session is None or not torch.cuda.is_available():
        return None, None
    decode_graph_eligible = getattr(native_session, "decode_graph_eligible", None)
    set_decode_graph_enabled = getattr(native_session, "set_decode_graph_enabled", None)
    if not callable(decode_graph_eligible) or not callable(set_decode_graph_enabled):
        return None, None
    try:
        if not bool(decode_graph_eligible()):
            return None, None
        set_decode_graph_enabled(True)
        from tensor.compile import cuda_graph_warmup, graph_replay_step

        warm = cuda_graph_warmup(native_session.decode_next_logits)
        replay = graph_replay_step(native_session.decode_next_logits, ())
        return replay, warm
    except Exception:
        try:
            set_decode_graph_enabled(False)
        except Exception:
            pass
        return None, None


def _build_target_speculative_probs(
    seq: torch.Tensor,
    current_logits: torch.Tensor,
    verify_logits: torch.Tensor,
    proposal: SpeculativeProposal,
    config: GenerationConfig,
    remaining_new_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    drafted_tokens = int(proposal.tokens.shape[1])
    target_probs: list[torch.Tensor] = []
    for token_idx in range(drafted_tokens):
        if token_idx == 0:
            step_logits = current_logits.contiguous()
            step_seq = seq
        else:
            step_logits = verify_logits[:, token_idx - 1, :].contiguous()
            step_seq, _ = _append_generated_tokens(seq, None, proposal.tokens[:, :token_idx].contiguous())
        target_probs.append(_policy_probs(step_logits, step_seq, config).unsqueeze(1))
    batch_target_probs = torch.cat(target_probs, dim=1).contiguous()

    bonus_enabled = torch.tensor([False], device=seq.device, dtype=torch.bool)
    if drafted_tokens <= 0:
        return batch_target_probs, None, bonus_enabled
    last_token = proposal.tokens[:, drafted_tokens - 1 : drafted_tokens].contiguous()
    eos_hit = config.eos_id is not None and bool((last_token == int(config.eos_id)).all().item())
    if int(remaining_new_tokens) <= drafted_tokens or eos_hit:
        return batch_target_probs, None, bonus_enabled
    full_seq, _ = _append_generated_tokens(seq, None, proposal.tokens)
    bonus_probs = _policy_probs(verify_logits[:, drafted_tokens - 1, :].contiguous(), full_seq, config).contiguous()
    bonus_enabled = torch.tensor([True], device=seq.device, dtype=torch.bool)
    return batch_target_probs, bonus_probs, bonus_enabled


def _finalize_speculative_result(
    seq: torch.Tensor,
    attention_mask: torch.Tensor | None,
    verified_cache,
    verify_logits: torch.Tensor,
    proposal: SpeculativeProposal,
    emitted_tokens: torch.Tensor,
    emitted_length: int,
    accepted_tokens: int,
) -> SpeculativeDecodeResult:
    emitted = emitted_tokens[:, :int(emitted_length)].contiguous() if int(emitted_length) > 0 else emitted_tokens[:, :0].contiguous()
    next_seq, next_attention_mask = _append_generated_tokens(seq, attention_mask, emitted)
    drafted_tokens = int(proposal.tokens.shape[1])
    keep_tokens = int(seq.shape[1]) + int(accepted_tokens)
    next_cache = truncate_kv_cache_prefix(verified_cache, keep_tokens)
    ready_logits = None
    if accepted_tokens == drafted_tokens and drafted_tokens > 0 and int(emitted_length) == drafted_tokens:
        ready_logits = verify_logits[:, drafted_tokens - 1, :].contiguous()
        next_cache = truncate_kv_cache_prefix(verified_cache, int(seq.shape[1]) + drafted_tokens)
    return SpeculativeDecodeResult(
        seq=next_seq,
        attention_mask=next_attention_mask,
        cache=next_cache,
        emitted_tokens=int(emitted_length),
        accepted_tokens=int(accepted_tokens),
        drafted_tokens=drafted_tokens,
        ready_logits=ready_logits,
    )


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

        next_logits = base_session.prefill_next_logits(
            chunk_size=getattr(config, "prefill_chunk_size", None),
        )
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
        self._native_decode_graph_replay: Callable[[], torch.Tensor | None] | None = None
        self._seq = seq
        self._attention_mask = attention_mask
        self._attention_mask_mode = _resolve_attention_mask_mode(
            attention_mask,
            batch_size=int(seq.shape[0]),
            seq_len=int(seq.shape[1]),
        )
        self._cache = cache
        if not _token_mask_has_padding(seq, attention_mask):
            try:
                native_session = create_native_model_session(
                    model,
                    seq,
                    attention_mask=attention_mask,
                    cache=cache,
                    trace=self.trace,
                )
                if (
                    native_session is not None
                    and getattr(native_session, "native_executor_kind", "causal_lm") != "causal_lm"
                ):
                    native_session = None
                self._native_session = native_session
            except Exception:
                self._native_session = None

    def _clear_native_decode_graph(self) -> None:
        self._native_decode_graph_replay = None
        if self._native_session is None:
            return
        disable = getattr(self._native_session, "set_decode_graph_enabled", None)
        if callable(disable):
            try:
                disable(False)
            except Exception:
                pass

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
            self._clear_native_decode_graph()
            self._native_session.seq = value
            if _token_mask_has_padding(self._seq, self._attention_mask):
                self._native_session = None

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
            self._clear_native_decode_graph()
            self._native_session.attention_mask = value
            if _token_mask_has_padding(self._seq, self._attention_mask):
                self._native_session = None

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
            self._clear_native_decode_graph()
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
            self._clear_native_decode_graph()
            self._native_session.disable_cache()
            return
        self.cache = None

    def prefill_next_logits(self, *, chunk_size: int | None = None) -> torch.Tensor | None:
        if self.cache is None:
            return None
        chunk = None if chunk_size is None else max(int(chunk_size), 0)
        if self._native_session is not None and (chunk is None or chunk <= 0 or self.seq_len <= chunk):
            return self._native_session.prefill_next_logits()
        if chunk is not None and chunk > 0 and self.seq_len > chunk and self.attention_mask_mode != "explicit":
            if self._native_session is not None:
                full_seq = self.seq.contiguous()
                full_seq_len = int(full_seq.shape[1])
                full_attention_mask = self.attention_mask
                last_logits = None
                try:
                    for start in range(0, full_seq_len, chunk):
                        end = min(start + chunk, full_seq_len)
                        self.seq = full_seq[:, start:end].contiguous()
                        self.attention_mask = _slice_prefill_attention_mask(full_attention_mask, start, end)
                        last_logits = self._native_session.prefill_next_logits()
                    return last_logits
                finally:
                    self.seq = full_seq
                    self.attention_mask = full_attention_mask
            last_logits = None
            for start in range(0, self.seq_len, chunk):
                end = min(start + chunk, self.seq_len)
                chunk_seq = self.seq[:, start:end].contiguous()
                chunk_mask = _slice_prefill_attention_mask(self.attention_mask, start, end)
                logits = self.model(
                    chunk_seq,
                    attn_mask=None,
                    attention_mask=chunk_mask,
                    cache=self.cache,
                    return_dict=False,
                )
                last_logits = _select_last_token_logits(logits, chunk_mask)
            return last_logits
        logits = self.model(
            self.seq.contiguous(),
            attn_mask=None,
            attention_mask=self.attention_mask,
            cache=self.cache,
            return_dict=False,
        )
        return _select_last_token_logits(logits, self.attention_mask)

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
        return _select_last_token_logits(logits, self.attention_mask)

    def append(self, next_id: torch.Tensor) -> None:
        if self._native_session is not None:
            try:
                self._native_session.append(next_id)
                return
            except Exception:
                self._clear_native_decode_graph()
                self._seq = self._native_session.seq
                self._attention_mask = self._native_session.attention_mask
                self._cache = self._native_session.cache
                self._native_session = None
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
        _decode_tokens, position_ids, cache_position = _resolve_decode_step_inputs(self.seq, self.attention_mask)
        if position_ids is not None and cache_position is not None:
            return position_ids, cache_position
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
            if self._native_decode_graph_replay is None and _native_decode_graph_enabled_by_env():
                replay, warm = _try_build_native_decode_graph_replay(self._native_session)
                if replay is not None:
                    self._native_decode_graph_replay = replay
                    if warm is not None:
                        return warm
                    return replay()
            if self._native_decode_graph_replay is not None:
                try:
                    return self._native_decode_graph_replay()
                except Exception:
                    self._clear_native_decode_graph()
            return self._native_session.decode_next_logits()
        decode_tokens, pos_ids, cache_pos = _resolve_decode_step_inputs(self.seq, self.attention_mask)
        if pos_ids is None or cache_pos is None:
            pos_ids, cache_pos = self.decode_positions()
        decode_attention_mask = self.attention_mask
        if (
            decode_attention_mask is not None
            and self.attention_mask_mode == "token"
            and decode_attention_mask.ndim == 2
        ):
            decode_attention_mask = torch.ones(
                (self.batch_size, int(decode_tokens.shape[1])),
                dtype=decode_attention_mask.dtype,
                device=decode_attention_mask.device,
            )
        logits = self.model(
            decode_tokens,
            attn_mask=None,
            attention_mask=decode_attention_mask,
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
            self._clear_native_decode_graph()
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
def speculative_decode_batch(
    model,
    draft_model,
    *,
    requests: list[SpeculativeBatchRequest],
    cache_pagesize: int = 128,
    cache_backend: str | None = None,
) -> list[SpeculativeDecodeResult | None]:
    if not requests:
        return []

    results: list[SpeculativeDecodeResult | None] = [None for _ in requests]
    prepared: list[dict[str, object]] = []
    for request_idx, request in enumerate(requests):
        seq = request.seq.contiguous()
        attention_mask = request.attention_mask.contiguous() if request.attention_mask is not None else None
        if not _supports_speculative_decode(draft_model, seq, attention_mask, request.cache, request.config):
            continue
        remaining = (
            int(request.config.max_new_tokens)
            if request.remaining_new_tokens is None
            else max(int(request.remaining_new_tokens), 0)
        )
        if remaining <= 0:
            continue
        base_cache = request.cache
        current_logits = request.ready_logits
        if current_logits is None:
            target_session = RuntimeGenerationSession(
                model=model,
                seq=seq,
                attention_mask=attention_mask,
                cache=base_cache,
                trace=False,
            )
            current_logits = target_session.decode_next_logits()
            base_cache = target_session.cache
        if current_logits is None or base_cache is None:
            continue
        proposal_budget = min(int(getattr(request.config, "num_speculative_tokens", 0)), remaining)
        if proposal_budget <= 0:
            continue
        resolved_backend = request.cache_backend or cache_backend
        proposal = _build_speculative_proposal(
            draft_model,
            seq,
            prompt_seq=request.prompt_seq,
            attention_mask=attention_mask,
            current_logits=current_logits,
            config=request.config,
            proposal_budget=proposal_budget,
            cache_pagesize=int(cache_pagesize),
            cache_backend=resolved_backend,
        )
        if proposal is None or int(proposal.tokens.shape[1]) <= 0:
            continue
        prepared.append(
            {
                "request_idx": request_idx,
                "seq": seq,
                "attention_mask": attention_mask,
                "cache": base_cache,
                "config": request.config,
                "current_logits": current_logits.contiguous(),
                "proposal": proposal,
                "remaining": int(remaining),
                "cache_backend": resolved_backend,
                "accept_method": _speculative_accept_method(request.config),
            }
        )

    grouped: dict[tuple[str, int, str, float, float], list[dict[str, object]]] = {}
    dispatch_order: list[tuple[str, int, str, float, float]] = []
    for item in prepared:
        proposal = item["proposal"]
        signature = (
            str(item["cache_backend"] or ""),
            int(proposal.tokens.shape[1]),
            str(item["accept_method"]),
            float(getattr(item["config"], "typical_acceptance_sampler_posterior_threshold", 0.09)),
            float(getattr(item["config"], "typical_acceptance_sampler_posterior_alpha", 0.3)),
        )
        if signature not in grouped:
            grouped[signature] = []
            dispatch_order.append(signature)
        grouped[signature].append(item)

    for signature in dispatch_order:
        group = grouped[signature]
        batch_tokens = torch.cat([item["proposal"].tokens for item in group], dim=0).contiguous()
        batch_cache = concat_kv_caches([item["cache"] for item in group])
        if batch_cache is None:
            continue
        verify_attention_mask = torch.ones(
            (len(group), int(batch_tokens.shape[1])),
            dtype=torch.long,
            device=batch_tokens.device,
        )
        verify_logits = model(
            batch_tokens,
            attn_mask=None,
            attention_mask=verify_attention_mask,
            cache=batch_cache,
            return_dict=False,
        )
        if verify_logits.dim() == 2:
            verify_logits = verify_logits.unsqueeze(1)
        if verify_logits.dim() != 3:
            raise ValueError(f"speculative decode verify logits must be rank-3, got shape {tuple(verify_logits.shape)}")

        target_prob_rows: list[torch.Tensor] = []
        draft_prob_rows: list[torch.Tensor] = []
        draft_token_rows: list[torch.Tensor] = []
        bonus_prob_rows: list[torch.Tensor] = []
        bonus_enabled_rows: list[torch.Tensor] = []
        for row_idx, item in enumerate(group):
            row_verify = verify_logits[row_idx : row_idx + 1].contiguous()
            target_probs, bonus_probs, bonus_enabled = _build_target_speculative_probs(
                item["seq"],
                item["current_logits"],
                row_verify,
                item["proposal"],
                item["config"],
                int(item["remaining"]),
            )
            target_prob_rows.append(target_probs)
            draft_prob_rows.append(item["proposal"].draft_probs.contiguous())
            draft_token_rows.append(item["proposal"].tokens.contiguous())
            if bonus_probs is None:
                bonus_prob_rows.append(
                    torch.zeros(
                        (1, int(target_probs.shape[-1])),
                        dtype=torch.float32,
                        device=target_probs.device,
                    )
                )
            else:
                bonus_prob_rows.append(bonus_probs.contiguous())
            bonus_enabled_rows.append(bonus_enabled.to(device=target_probs.device, dtype=torch.bool).view(1))

        accepted_tokens, emitted_lengths, accepted_counts = runtime_speculative_accept(
            torch.cat(target_prob_rows, dim=0).contiguous(),
            torch.cat(draft_prob_rows, dim=0).contiguous(),
            torch.cat(draft_token_rows, dim=0).contiguous(),
            bonus_probs=torch.cat(bonus_prob_rows, dim=0).contiguous(),
            bonus_enabled=torch.cat(bonus_enabled_rows, dim=0).contiguous(),
            method=str(signature[2]),
            posterior_threshold=float(signature[3]),
            posterior_alpha=float(signature[4]),
        )
        cache_rows = split_kv_cache_rows(batch_cache, [1 for _ in group])
        for row_idx, item in enumerate(group):
            results[int(item["request_idx"])] = _finalize_speculative_result(
                item["seq"],
                item["attention_mask"],
                cache_rows[row_idx],
                verify_logits[row_idx : row_idx + 1].contiguous(),
                item["proposal"],
                accepted_tokens[row_idx : row_idx + 1].contiguous(),
                int(emitted_lengths[row_idx].item()),
                int(accepted_counts[row_idx].item()),
            )

    return results


@torch.no_grad()
def speculative_decode_step(
    model,
    draft_model,
    seq: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    cache,
    config: GenerationConfig,
    ready_logits: torch.Tensor | None = None,
    prompt_seq: torch.Tensor | None = None,
    remaining_new_tokens: int | None = None,
    cache_pagesize: int = 128,
    cache_backend: str | None = None,
) -> SpeculativeDecodeResult | None:
    results = speculative_decode_batch(
        model,
        draft_model,
        requests=[
            SpeculativeBatchRequest(
                seq=seq,
                attention_mask=attention_mask,
                cache=cache,
                config=config,
                ready_logits=ready_logits,
                prompt_seq=prompt_seq,
                cache_backend=cache_backend,
                remaining_new_tokens=remaining_new_tokens,
            )
        ],
        cache_pagesize=int(cache_pagesize),
        cache_backend=cache_backend,
    )
    return results[0] if results else None


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
    draft_model=None,
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

    prompt_seq = input_ids.contiguous()
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
            next_logits = session.prefill_next_logits(
                chunk_size=getattr(cfg, "prefill_chunk_size", None),
            )
        except Exception:
            if trace:
                print(f"[engine] exception in prompt prefill seq_len={session.seq_len}")
                traceback.print_exc()
            session.disable_cache()

    generated_tokens = 0
    while generated_tokens < int(cfg.max_new_tokens):
        step = generated_tokens
        remaining_new_tokens = int(cfg.max_new_tokens) - generated_tokens
        if next_logits is None:
            try:
                if generated_tokens > 0 and session.cache is not None and _cache_length(session.cache) < session.seq_len:
                    next_logits = session.decode_next_logits()
                else:
                    next_logits = session.full_next_logits()
            except Exception:
                if trace:
                    print(f"[engine] exception in model forward at step={step} seq_len={session.seq_len}")
                    traceback.print_exc()
                raise
        if sampler is None:
            speculative = speculative_decode_step(
                model,
                draft_model,
                session.seq,
                attention_mask=session.attention_mask,
                cache=session.cache,
                config=cfg,
                ready_logits=next_logits,
                prompt_seq=prompt_seq,
                remaining_new_tokens=remaining_new_tokens,
                cache_pagesize=int(cache_pagesize),
                cache_backend=cache_backend,
            )
            if speculative is not None and speculative.emitted_tokens > 0:
                session = RuntimeGenerationSession(
                    model=model,
                    seq=speculative.seq,
                    attention_mask=speculative.attention_mask,
                    cache=speculative.cache,
                    trace=trace,
                )
                generated_tokens += int(speculative.emitted_tokens)
                next_logits = speculative.ready_logits
                if trace:
                    print(
                        f"[engine] speculative step={step} drafted={speculative.drafted_tokens} "
                        f"accepted={speculative.accepted_tokens} emitted={speculative.emitted_tokens}"
                    )
                if cfg.eos_id is not None and bool((session.seq[:, -int(speculative.emitted_tokens):] == int(cfg.eos_id)).any().item()):
                    break
                if generated_tokens >= int(cfg.max_new_tokens):
                    break
                try:
                    session.evict_if_needed(getattr(cfg, "sliding_window", None), policy="sliding-window")
                except Exception:
                    pass
                continue
        if sampler is not None:
            sampled_logits = apply_sampling_policies(next_logits, session.seq, cfg)
            next_id = sampler(sampled_logits)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)
        else:
            next_id = _sample_next_token(next_logits, session.seq, cfg)
        if trace:
            try:
                print(f"[engine] step={step} seq_len={session.seq_len} logits_dtype={next_logits.dtype} next_id={int(next_id[0,0].item())}")
            except Exception:
                pass
        session.append(next_id)
        generated_tokens += int(next_id.shape[1])
        if cfg.eos_id is not None and (next_id == int(cfg.eos_id)).all():
            break
        if generated_tokens >= int(cfg.max_new_tokens):
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
