from __future__ import annotations

import time
import torch

from specs.config import ModelConfig
from model.lm import TransformerLM


def bench_forward(attn_impl: str) -> float:
    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl=attn_impl)
    model = TransformerLM(cfg).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randint(0, cfg.vocab_size, (4, 256), dtype=torch.long, device=device)
    # warmup
    for _ in range(3):
        _ = model(x, None)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    for _ in range(10):
        _ = model(x, None)
    torch.cuda.synchronize() if device.type == "cuda" else None
    return (time.time() - t0) / 10.0


def main() -> None:
    candidates = ["flash", "reference"]
    scores = {impl: bench_forward(impl) for impl in candidates}
    best = min(scores, key=scores.get)
    print({"latency_s_per_iter": scores, "best_attn_impl": best})


if __name__ == "__main__":
    main()


