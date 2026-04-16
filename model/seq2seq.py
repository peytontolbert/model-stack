import torch
import torch.nn as nn

from runtime.blocks import apply_native_norm, execute_decoder_stack, execute_encoder_stack
from runtime.ops import embedding as runtime_embedding
from runtime.ops import linear as runtime_linear
from specs.config import ModelConfig
from compress import apply_compression
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
        compress: dict | None = None,
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
        if compress is not None:
            lora_cfg = compress.get("lora") if isinstance(compress, dict) else None
            quant_cfg = compress.get("quant") if isinstance(compress, dict) else None
            self._compression_summary = apply_compression(self, lora=lora_cfg, quant=quant_cfg)
        else:
            self._compression_summary = None

    def encode(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = runtime_embedding(self.enc_embed.weight, input_ids, self.enc_embed.padding_idx)
        x = execute_encoder_stack(self.encoder, x, padding_mask)
        return apply_native_norm(x, self.enc_norm)

    def decode(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        x = runtime_embedding(self.dec_embed.weight, input_ids, self.dec_embed.padding_idx)
        x = execute_decoder_stack(
            self.decoder,
            x,
            memory,
            self_mask,
            memory_mask,
            cache,
        )
        return apply_native_norm(x, self.dec_norm)

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
        return runtime_linear(x, self.lm_head.weight, self.lm_head.bias)
