from typing import Callable, Optional

import torch

from runtime.generation import (
    GenerationConfig,
    build_generation_config as runtime_build_generation_config,
    generate as runtime_generate,
)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    cache=None,
    attention_mask: Optional[torch.Tensor] = None,
    config: Optional[GenerationConfig] = None,
    sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    cache_backend: str | None = None,
) -> torch.Tensor:
    cfg = config or runtime_build_generation_config()
    return runtime_generate(
        model,
        input_ids,
        cache=cache,
        attention_mask=attention_mask,
        config=cfg,
        sampler=sampler,
        cache_backend=cache_backend,
    )
