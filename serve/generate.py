import torch

from .engine import generate as _engine_generate, GenerationConfig


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64, cache=None, sampler=None, **kwargs):
    cfg = GenerationConfig(max_new_tokens=int(max_new_tokens), **kwargs)
    return _engine_generate(model, input_ids, cache=cache, config=cfg, sampler=sampler)
