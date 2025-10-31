import torch
import torch.nn as nn


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # hidden_states: (B,T,D), attention_mask: (B,T) with 1 token / 0 pad
        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)  # (B,T,1)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        return self.proj(pooled)


class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B,T,D)
        return self.proj(hidden_states)


