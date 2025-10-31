import torch
import torch.nn as nn

from specs.config import ModelConfig
from blocks.encoder_block import EncoderBlock
from blocks.decoder_block import DecoderBlock
from blocks.schedules import drop_path_linear
from blocks.init import init_transformer_stack


class EncoderDecoderLM(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        drop_path_max_enc: float = 0.0,
        drop_path_max_dec: float = 0.0,
        init_recipe_enc: str | None = None,
        init_recipe_dec: str | None = None,
        tie_embeddings: bool = False,
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        V = vocab_size if vocab_size is not None else cfg.vocab_size
        # Embeddings
        self.enc_embed = nn.Embedding(V, cfg.d_model)
        self.dec_embed = nn.Embedding(V, cfg.d_model)
        if tie_embeddings:
            self.dec_embed.weight = self.enc_embed.weight
        # Encoder stack
        sch_e = drop_path_linear(cfg.n_layers, drop_path_max_enc)
        self.encoder = nn.ModuleList([EncoderBlock(cfg, drop_path=sch_e[i]) for i in range(cfg.n_layers)])
        # Decoder stack
        sch_d = drop_path_linear(cfg.n_layers, drop_path_max_dec)
        self.decoder = nn.ModuleList([DecoderBlock(cfg, drop_path=sch_d[i]) for i in range(cfg.n_layers)])
        # Norms and head
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, V, bias=False)
        # Optional init
        if init_recipe_enc is not None:
            init_transformer_stack(self.encoder, recipe=init_recipe_enc)
        if init_recipe_dec is not None:
            init_transformer_stack(self.decoder, recipe=init_recipe_dec)

    def encode(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.enc_embed(input_ids)
        for blk in self.encoder:
            x = blk(x, padding_mask)
        return self.enc_norm(x)

    def decode(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        x = self.dec_embed(input_ids)
        if cache is None:
            for blk in self.decoder:
                x = blk(x, memory, self_mask, memory_mask, None)
        else:
            for i, blk in enumerate(self.decoder):
                layer_cache = cache.layer(i)
                x = blk(x, memory, self_mask, memory_mask, layer_cache)
        return self.dec_norm(x)

    def forward(
        self,
        enc_input_ids: torch.Tensor,
        dec_input_ids: torch.Tensor,
        enc_padding_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        mem = self.encode(enc_input_ids, enc_padding_mask)
        x = self.decode(dec_input_ids, mem, dec_self_mask, enc_padding_mask, cache)
        return self.lm_head(x)


