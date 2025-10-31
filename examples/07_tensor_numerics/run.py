from __future__ import annotations

import torch

from tensor.metrics import token_accuracy


def main() -> None:
    torch.manual_seed(0)
    # Fake logits and targets to demo metrics
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    acc = token_accuracy(logits, targets, mask=None)
    print({"token_accuracy": float(acc)})


if __name__ == "__main__":
    main()


