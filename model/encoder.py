import torch
import torch.nn as nn

from runtime.blocks import apply_native_norm, execute_encoder_stack
from runtime.ops import embedding as runtime_embedding
from specs.config import ModelConfig
from compress import apply_compression
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
        compress: dict | None = None,
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
        if compress is not None:
            lora_cfg = compress.get("lora") if isinstance(compress, dict) else None
            quant_cfg = compress.get("quant") if isinstance(compress, dict) else None
            self._compression_summary = apply_compression(self, lora=lora_cfg, quant=quant_cfg)
        else:
            self._compression_summary = None

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = runtime_embedding(self.embed.weight, input_ids, self.embed.padding_idx)
        x = execute_encoder_stack(self.blocks, x, padding_mask)
        x = apply_native_norm(x, self.norm)
        return x
