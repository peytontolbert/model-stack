from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from specs.config import ModelConfig
from model.lm import TransformerLM
from eval.loop import evaluate_lm_next_token
from data.loader import build_dataloader
from data.batch import Batch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _synthetic_loader(*, batch_size: int, seq_len: int, num_batches: int) -> Iterator[Batch]:
    vocab_size = 32000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(int(num_batches)):
        x = torch.randint(0, int(vocab_size), (int(batch_size), int(seq_len)), dtype=torch.long, device=device)
        yield Batch(input_ids=x, attn_mask=None)  # type: ignore[arg-type]


def main() -> None:
    root = _repo_root()
    shards = root / "corpus" / "shards"

    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl="flash")
    model = TransformerLM(cfg)

    try:
        loader = build_dataloader(str(shards), batch_size=8, seq_len=256)
    except Exception:
        loader = _synthetic_loader(batch_size=8, seq_len=256, num_batches=50)

    res = evaluate_lm_next_token(model, loader, max_batches=50, report_accuracy=True)
    print({"nll": res.nll, "ppl": res.ppl, "acc": res.acc, "num_tokens": res.num_tokens})


if __name__ == "__main__":
    main()


