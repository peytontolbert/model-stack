from __future__ import annotations

import torch

from specs.config import ModelConfig
from model.lm import TransformerLM
from interpret.logit_lens import logit_lens


def main() -> None:
    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl="flash")
    model = TransformerLM(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    x = torch.randint(0, cfg.vocab_size, (1, 32), dtype=torch.long, device=device)
    res = logit_lens(model, x, layer_ids=[0, cfg.n_layers // 2, cfg.n_layers - 1], topk=5)
    # Print top-5 ids for last token per selected layer
    for lid, (idx, val) in res.items():
        print(f"layer {lid}: ids={idx.tolist()} scores={[float(v) for v in val.tolist()]}")


if __name__ == "__main__":
    main()


