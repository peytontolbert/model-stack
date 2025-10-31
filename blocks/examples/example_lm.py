import torch
import torch.nn as nn

from specs.config import ModelConfig
from blocks.llama_block import LlamaBlock
from blocks.init import init_transformer_stack


class ExampleTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig, block: str = "llama"):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        blocks: list[nn.Module] = []
        for _ in range(cfg.n_layers):
            if block == "llama":
                blocks.append(LlamaBlock(cfg))
            else:
                from blocks.gpt_block import GPTBlock
                blocks.append(GPTBlock(cfg))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model)  # final norm
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_transformer_stack(self.blocks, recipe=("llama" if block == "llama" else "gpt"))

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        x = self.embed(input_ids)
        for i, blk in enumerate(self.blocks):
            c = None if cache is None else cache.layer(i)
            x = blk(x, attn_mask, c)
        x = self.norm(x)
        return self.lm_head(x)


