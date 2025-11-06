from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from tensor.sampling import (
    apply_temperature,
    apply_topk_mask,
    apply_topp_mask,
    apply_presence_frequency_penalty,
    apply_no_repeat_ngram_mask,
)
from attn.kv_cache import init_kv_cache, kv_cache_evict


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    # Match Transformers: greedy by default unless do_sample=True
    do_sample: bool = False
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_id: Optional[int] = None
    no_repeat_ngram: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # Optional sliding-window length for KV cache eviction during generation
    sliding_window: Optional[int] = None


def _apply_sampling_policies(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    cfg: GenerationConfig,
) -> torch.Tensor:
    x = logits
    # Only apply temperature/top-k/top-p when sampling is enabled
    if getattr(cfg, "do_sample", False):
        if cfg.temperature is not None and cfg.temperature != 1.0:
            x = apply_temperature(x, cfg.temperature)
    mask = None
    if getattr(cfg, "do_sample", False):
        if cfg.top_k is not None:
            m = apply_topk_mask(x, cfg.top_k)
            mask = m if mask is None else (mask | m)
        if cfg.top_p is not None:
            m = apply_topp_mask(x, cfg.top_p)
            mask = m if mask is None else (mask | m)
    if cfg.no_repeat_ngram and cfg.no_repeat_ngram > 0:
        m = apply_no_repeat_ngram_mask(x, input_ids, cfg.no_repeat_ngram)
        mask = m if mask is None else (mask | m)
    if mask is not None:
        min_val = torch.finfo(x.dtype).min if x.dtype.is_floating_point else -1e9
        x = x.masked_fill(mask, min_val)
    if cfg.presence_penalty != 0.0 or cfg.frequency_penalty != 0.0:
        # Build counts (simplified): frequency per token over the sequence
        B, V = x.shape[0], x.shape[-1]
        counts = torch.zeros(B, V, dtype=x.dtype, device=x.device)
        for b in range(B):
            ids, c = input_ids[b].unique(return_counts=True)
            counts[b].index_add_(0, ids, c.to(x.dtype))
        x = apply_presence_frequency_penalty(x, counts, cfg.presence_penalty, cfg.frequency_penalty)
    return x


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    cache=None,
    attention_mask: Optional[torch.Tensor] = None,
    config: Optional[GenerationConfig] = None,
    sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    import os, traceback
    trace = (os.getenv("GEN_TRACE", "0") == "1")
    cfg = config or GenerationConfig()
    # If not specified on the generation config, inherit from model.cfg
    try:
        if getattr(cfg, "sliding_window", None) is None:
            sw = getattr(getattr(model, "cfg", object()), "sliding_window", None)
            if sw is not None:
                cfg.sliding_window = int(sw)
    except Exception:
        pass
    seq = input_ids
    # Initialize a simple KV cache for efficient decoding
    try:
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        n_layers = len(getattr(model, "blocks"))
        # Prefer explicit head_dim; else derive
        head_dim = int(getattr(model.cfg, "head_dim", None) or (int(model.cfg.d_model) // int(model.cfg.n_heads)))
        n_kv = int(getattr(getattr(model.blocks[0], "attn"), "n_kv_heads"))
        cache = init_kv_cache(batch=seq.shape[0], n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim, pagesize=128, dtype=dt, device=dev)
    except Exception:
        cache = None

    if trace:
        try:
            p = next(model.parameters())
            print(f"[engine] start: device={p.device} dtype={p.dtype} init_len={seq.shape[1]} do_sample={getattr(cfg,'do_sample',False)}")
        except Exception:
            pass
    for step in range(int(cfg.max_new_tokens)):
        # Greedy step with last token and KV cache if available
        # Compute absolute position ids for the next token
        try:
            # Absolute position id for the new token (L-1)
            pos_ids = torch.arange(seq.shape[1] - 1, seq.shape[1], device=seq.device, dtype=torch.long)
            pos_ids = pos_ids.view(1, 1).expand(seq.shape[0], -1)
            cache_pos = pos_ids.view(-1)  # (B,)
        except Exception:
            pos_ids = None
            cache_pos = None
        try:
            logits = model(
                seq[:, -1:].contiguous(),
                attn_mask=None,
                attention_mask=attention_mask,
                cache=cache,
                position_ids=pos_ids,
                cache_position=cache_pos,
                return_dict=False,
            )
        except Exception:
            if trace:
                print(f"[engine] exception in model forward at step={step} seq_len={seq.shape[1]}")
                traceback.print_exc()
            raise
        next_logits = logits[:, -1, :]
        sampled_logits = _apply_sampling_policies(next_logits, seq, cfg)
        if sampler is not None:
            next_id = sampler(sampled_logits)
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)
        else:
            if getattr(cfg, "do_sample", False):
                probs = F.softmax(sampled_logits.float(), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(sampled_logits, dim=-1, keepdim=True)
        if trace:
            try:
                print(f"[engine] step={step} seq_len={seq.shape[1]} logits_dtype={next_logits.dtype} next_id={int(next_id[0,0].item())}")
            except Exception:
                pass
        seq = torch.cat([seq, next_id], dim=1)
        # Extend attention_mask by 1 token (mark as non-pad)
        if attention_mask is not None:
            ones = torch.ones(next_id.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, ones], dim=1)
        # Apply sliding-window cache eviction if configured
        try:
            if cache is not None and getattr(cfg, "sliding_window", None):
                kv_cache_evict(cache, int(cfg.sliding_window), policy="sliding-window")
        except Exception:
            pass
        if cfg.eos_id is not None:
            if (next_id == int(cfg.eos_id)).all():
                break
    return seq


