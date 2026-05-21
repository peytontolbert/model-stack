from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class AsrDecodeOptions:
    eos_token_id: int | None = None
    no_speech_token_id: int | None = None
    suppress_token_ids: tuple[int, ...] = ()
    timestamp_token_begin: int | None = None
    max_tokens: int = 448
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")


@dataclass
class AsrDecodeState:
    tokens: list[int] = field(default_factory=list)
    completed: bool = False
    no_speech_prob: float | None = None
    avg_logprob: float | None = None

    def append(self, token_id: int, options: AsrDecodeOptions) -> None:
        self.tokens.append(int(token_id))
        if options.eos_token_id is not None and int(token_id) == options.eos_token_id:
            self.completed = True
        if len(self.tokens) >= options.max_tokens:
            self.completed = True


def apply_asr_logit_policy(logits: torch.Tensor, options: AsrDecodeOptions) -> torch.Tensor:
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    masked = logits.clone()
    vocab = masked.shape[-1]
    for token_id in options.suppress_token_ids:
        if 0 <= token_id < vocab:
            masked[..., token_id] = -torch.inf
    return masked


def greedy_decode_step(
    logits: torch.Tensor,
    state: AsrDecodeState | None = None,
    options: AsrDecodeOptions | None = None,
) -> tuple[int, AsrDecodeState]:
    options = options or AsrDecodeOptions()
    state = state or AsrDecodeState()
    masked = apply_asr_logit_policy(logits, options)
    token = int(torch.argmax(masked, dim=-1).reshape(-1)[0].item())
    state.append(token, options)
    return token, state
