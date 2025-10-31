import torch
import torch.nn as nn

from specs.config import ModelConfig
from blocks.encoder_block import EncoderBlock
from blocks.schedules import drop_path_linear
from blocks.init import init_transformer_stack


class EncoderModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        drop_path_max: float = 0.0,
        init_recipe: str | None = None,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        schedule = drop_path_linear(cfg.n_layers, drop_path_max)
        blocks: list[nn.Module] = []
        for i in range(cfg.n_layers):
            blocks.append(EncoderBlock(cfg, drop_path=schedule[i]))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.d_model)
        if init_recipe is not None:
            init_transformer_stack(self.blocks, recipe=init_recipe)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, padding_mask)
        x = self.norm(x)
        return x


