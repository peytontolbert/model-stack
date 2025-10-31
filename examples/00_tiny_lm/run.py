from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import torch

from specs.config import ModelConfig
from specs.dist import DistConfig
from model.lm import TransformerLM
from model.generate import greedy_generate
from train.trainer import Trainer, TrainConfig
from data.loader import build_dataloader
from data.batch import Batch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _try_build_loader(shards_dir: Path, batch_size: int, seq_len: int) -> Iterator[Batch]:
    try:
        return build_dataloader(str(shards_dir), batch_size=batch_size, seq_len=seq_len)
    except Exception:
        return _synthetic_loader(batch_size=batch_size, seq_len=seq_len, num_batches=200)


def _synthetic_loader(*, batch_size: int, seq_len: int, num_batches: int) -> Iterator[Batch]:
    vocab_size = 32000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(int(num_batches)):
        x = torch.randint(0, int(vocab_size), (int(batch_size), int(seq_len)), dtype=torch.long, device=device)
        yield Batch(input_ids=x, attn_mask=None)  # type: ignore[arg-type]


def main() -> None:
    root = _repo_root()
    shards = root / "corpus" / "shards"

    cfg = ModelConfig(
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
        vocab_size=32000,
        attn_impl="flash",
    )
    model = TransformerLM(cfg)

    dist = DistConfig(
        backend=("nccl" if torch.cuda.is_available() else "gloo"),
        strategy="FSDP",
        precision=("bf16" if torch.cuda.is_available() else "fp32"),
        grad_ckpt=True,
    )
    tc = TrainConfig(total_steps=500, warmup_steps=50, log_interval=50)

    trainer = Trainer(model, dist_cfg=dist, train_cfg=tc)
    loader = _try_build_loader(shards, batch_size=8, seq_len=256)

    trainer.train_steps(loader, max_steps=tc.total_steps)

    # Greedy generate from a dummy prompt (zeros) for a quick smoke test
    device = next(model.parameters()).device
    prompt = torch.zeros((1, 8), dtype=torch.long, device=device)
    out = greedy_generate(trainer._unwrap_model(), prompt, max_new_tokens=32)  # noqa: SLF001
    print("generated shape:", tuple(out.shape))


if __name__ == "__main__":
    main()


