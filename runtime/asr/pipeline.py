from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from runtime.asr.config import AsrFeatureConfig
from runtime.asr.decode import AsrDecodeOptions
from runtime.asr.features import pcm_to_log_mel


@dataclass(frozen=True)
class AsrTranscript:
    text: str
    tokens: tuple[int, ...] = ()
    language: str | None = None
    confidence: float | None = None
    segments: tuple[dict, ...] = field(default_factory=tuple)


class AsrBackend(Protocol):
    def transcribe_features(self, log_mel: torch.Tensor, options: AsrDecodeOptions) -> AsrTranscript:
        ...


class AsrRuntime:
    """ASR runtime façade around model-stack audio features and pluggable backends."""

    def __init__(
        self,
        backend: AsrBackend,
        *,
        feature_config: AsrFeatureConfig | None = None,
        decode_options: AsrDecodeOptions | None = None,
    ):
        self.backend = backend
        self.feature_config = feature_config or AsrFeatureConfig()
        self.decode_options = decode_options or AsrDecodeOptions()

    def features_from_pcm(self, pcm: torch.Tensor) -> torch.Tensor:
        return pcm_to_log_mel(pcm, self.feature_config)

    def transcribe_pcm(self, pcm: torch.Tensor, options: AsrDecodeOptions | None = None) -> AsrTranscript:
        log_mel = self.features_from_pcm(pcm)
        return self.backend.transcribe_features(log_mel, options or self.decode_options)

    def transcribe_features(self, log_mel: torch.Tensor, options: AsrDecodeOptions | None = None) -> AsrTranscript:
        return self.backend.transcribe_features(log_mel, options or self.decode_options)
