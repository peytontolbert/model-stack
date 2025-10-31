import torch
import torch.nn as nn

from specs.config import ModelConfig
from compress import apply_compression
from blocks.factory import build_block_stack


class CausalLM(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        block_variant: str = "llama",
        drop_path_max: float = 0.0,
        init_recipe: str | None = None,
        tie_weights: bool = True,
        compress: dict | None = None,
        **overrides,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = build_block_stack(
            cfg,
            variant=block_variant,
            drop_path_max=drop_path_max,
            init_recipe=init_recipe,
            **overrides,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        # Optional compression hooks
        if compress is not None:
            lora_cfg = compress.get("lora") if isinstance(compress, dict) else None
            quant_cfg = compress.get("quant") if isinstance(compress, dict) else None
            self._compression_summary = apply_compression(self, lora=lora_cfg, quant=quant_cfg)
        else:
            self._compression_summary = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        if cache is None:
            for blk in self.blocks:
                x = blk(x, attn_mask, None)
        else:
            for i, blk in enumerate(self.blocks):
                layer_cache = cache.layer(i)
                x = blk(x, attn_mask, layer_cache)
        x = self.norm(x)
        return self.lm_head(x)


