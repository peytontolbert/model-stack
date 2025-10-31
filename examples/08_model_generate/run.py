from __future__ import annotations

import torch

from specs.config import ModelConfig
from model.lm import TransformerLM
from model.generate import sample_generate


def main() -> None:
    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl="flash")
    model = TransformerLM(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long, device=device)
    out = sample_generate(model, prompt, max_new_tokens=32, temperature=0.8, top_p=0.9)
    print("generated shape:", tuple(out.shape))


if __name__ == "__main__":
    main()


