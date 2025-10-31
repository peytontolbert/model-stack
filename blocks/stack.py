import torch
import torch.nn as nn

from typing import Optional


class TransformerStack(nn.Module):
    def __init__(self, blocks: nn.ModuleList, ln_f: Optional[nn.Module] = None):
        super().__init__()
        self.blocks = blocks
        self.ln_f = ln_f

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, cache=None) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            c = None if cache is None else cache.layer(i)
            x = blk(x, attn_mask, c)
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
        x_enc = enc_x
        for i, blk in enumerate(self.encoder):
            x_enc = blk(x_enc, padding_mask=enc_pad_mask)
        x_dec = dec_x
        for i, blk in enumerate(self.decoder):
            c = None if dec_cache is None else dec_cache.layer(i)
            # Expect DecoderBlock signature
            x_dec = blk(x_dec, x_enc, self_mask=dec_attn_mask, enc_mask=None, cache=c)
        if self.ln_f is not None:
            x_dec = self.ln_f(x_dec)
        return x_dec


