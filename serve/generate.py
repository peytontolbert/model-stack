import torch

from runtime.generation import build_generation_config as runtime_build_generation_config
from runtime.generation import resolve_generation_sampling_mode as runtime_resolve_generation_sampling_mode

from .engine import generate as _engine_generate


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=64,
    cache=None,
    sampler=None,
    attention_mask=None,
    attn_mask=None,
    cache_backend=None,
    do_sample: bool | None = None,
    **kwargs,
):
    resolved_attention_mask = attention_mask if attention_mask is not None else attn_mask
    resolved_do_sample = runtime_resolve_generation_sampling_mode(
        do_sample=do_sample,
        temperature=float(kwargs.get("temperature", 1.0)),
        top_k=kwargs.get("top_k"),
        top_p=kwargs.get("top_p"),
    )
    cfg = runtime_build_generation_config(
        max_new_tokens=int(max_new_tokens),
        do_sample=resolved_do_sample,
        **kwargs,
    )
    return _engine_generate(
        model,
        input_ids,
        cache=cache,
        attention_mask=resolved_attention_mask,
        config=cfg,
        sampler=sampler,
        cache_backend=cache_backend,
    )
