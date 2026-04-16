import torch
import torch.nn as nn

from typing import Optional

from runtime.blocks import execute_block_stack, execute_decoder_stack, execute_encoder_stack


class TransformerStack(nn.Module):
    def __init__(self, blocks: nn.ModuleList, ln_f: Optional[nn.Module] = None):
        super().__init__()
        self.blocks = blocks
        self.ln_f = ln_f

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        x = execute_block_stack(self.blocks, x, attn_mask, cache)
        if self.ln_f is not None:
            x = self.ln_f(x)
        return x


class EncoderDecoderStack(nn.Module):
    def __init__(self, encoder: nn.ModuleList, decoder: nn.ModuleList, ln_f: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ln_f = ln_f

    def forward(self, enc_x: torch.Tensor, dec_x: torch.Tensor, enc_pad_mask: torch.Tensor | None = None, dec_attn_mask: torch.Tensor | None = None, enc_cache=None, dec_cache=None) -> torch.Tensor:
        x_enc = execute_encoder_stack(self.encoder, enc_x, enc_pad_mask)
        x_dec = execute_decoder_stack(self.decoder, dec_x, x_enc, dec_attn_mask, None, dec_cache)
        if self.ln_f is not None:
            x_dec = self.ln_f(x_dec)
        return x_dec

