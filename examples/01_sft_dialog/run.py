from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import torch

from specs.config import ModelConfig
from specs.dist import DistConfig
from model.lm import TransformerLM
from train.trainer import Trainer, TrainConfig
from data.batch import Batch
from data.tokenizer import get_tokenizer


@dataclass
class Dialog:
    prompt: str
    response: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _toy_dialogs() -> List[Dialog]:
    return [
        Dialog("Hello, who are you?", "I am a small language model."),
        Dialog("What is 2 + 2?", "2 + 2 equals 4."),
        Dialog("Name a color of the sky.", "Blue."),
    ]


def _tokenize_pairs(pairs: Sequence[Dialog]) -> List[List[int]]:
    tok = get_tokenizer(None)
    sequences: List[List[int]] = []
    for d in pairs:
        ids = tok.encode(d.prompt) + [1] + tok.encode(d.response)  # simple separator id=1
        sequences.append(ids)
    return sequences


def _batchify(seqs: List[List[int]], *, batch_size: int, seq_len: int, num_batches: int) -> Iterator[Batch]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(int(num_batches)):
        batch = []
        for b in range(int(batch_size)):
            s = seqs[(i * batch_size + b) % len(seqs)]
            # pad or trim to seq_len
            arr = (s + [0] * max(0, seq_len - len(s)))[:seq_len]
            batch.append(torch.tensor(arr, dtype=torch.long, device=device))
        x = torch.stack(batch, dim=0)
        yield Batch(input_ids=x, attn_mask=None)  # type: ignore[arg-type]


def main() -> None:
    _ = _repo_root()  # reserved for future data paths
    seqs = _tokenize_pairs(_toy_dialogs())

    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=65536, attn_impl="flash")
    model = TransformerLM(cfg)

    dist = DistConfig(backend=("nccl" if torch.cuda.is_available() else "gloo"), strategy="FSDP", precision=("bf16" if torch.cuda.is_available() else "fp32"))
    tc = TrainConfig(total_steps=300, warmup_steps=30, log_interval=50)
    trainer = Trainer(model, dist_cfg=dist, train_cfg=tc)

    loader = _batchify(seqs, batch_size=8, seq_len=128, num_batches=tc.total_steps)
    trainer.train_steps(loader, max_steps=tc.total_steps)
    print("SFT toy training complete.")


if __name__ == "__main__":
    main()


