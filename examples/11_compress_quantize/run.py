from __future__ import annotations

import torch

from specs.config import ModelConfig
from model.lm import TransformerLM
from compress.apply import apply_compression


def main() -> None:
    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl="flash")
    model = TransformerLM(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    summary = apply_compression(model, quant={})
    print("quantized modules:", int(summary.get("quant", {}).get("num", 0)))


if __name__ == "__main__":
    main()


