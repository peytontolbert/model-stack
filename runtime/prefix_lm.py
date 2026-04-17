import torch

from runtime.causal import CausalLM
from specs.config import ModelConfig
from tensor.masking import build_prefix_lm_mask


class PrefixCausalLM(CausalLM):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        block_variant: str = "llama",
        drop_path_max: float = 0.0,
        init_recipe: str | None = None,
        tie_weights: bool = True,
        **overrides,
    ):
        super().__init__(
            cfg,
            block_variant=block_variant,
            drop_path_max=drop_path_max,
            init_recipe=init_recipe,
            tie_weights=tie_weights,
            **overrides,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        prefix_lengths: int | torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        if attn_mask is None and prefix_lengths is not None:
            B, T = input_ids.shape
            device = input_ids.device
            if isinstance(prefix_lengths, int):
                attn_mask = build_prefix_lm_mask(T, prefix_lengths, device=device)
            else:
                masks = []
                for b in range(B):
                    prefix_len = int(prefix_lengths[b].item())
                    masks.append(build_prefix_lm_mask(T, prefix_len, device=device))
                attn_mask = torch.stack(masks, dim=0)
        return super().forward(input_ids=input_ids, attn_mask=attn_mask, cache=cache)


__all__ = ["PrefixCausalLM"]
