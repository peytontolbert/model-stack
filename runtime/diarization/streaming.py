from __future__ import annotations

from dataclasses import dataclass

import torch

from runtime.diarization.config import DiarizationStreamConfig


@dataclass(frozen=True)
class DiarizationAudioChunk:
    pcm: torch.Tensor
    start_sample: int
    end_sample: int
    sample_rate: int
    is_final: bool = False

    @property
    def duration_seconds(self) -> float:
        return float(self.pcm.numel()) / float(self.sample_rate)


class DiarizationStreamState:
    """Streaming PCM chunker for online diarization frontends.

    This mirrors the ASR chunker but uses longer overlapped windows by default
    so speaker assignment has enough context to remain stable across updates.
    """

    def __init__(self, config: DiarizationStreamConfig | None = None):
        self.config = config or DiarizationStreamConfig()
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._buffer_start_sample = 0

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.numel())

    @property
    def next_sample(self) -> int:
        return self._buffer_start_sample + self.buffered_samples

    def append(self, pcm: torch.Tensor) -> None:
        if not pcm.dtype.is_floating_point:
            raise TypeError("pcm must be floating point")
        if pcm.ndim != 1:
            raise ValueError("streaming append expects mono pcm with shape (samples,)")
        if pcm.numel() == 0:
            return
        chunk = pcm.detach().to(device="cpu", dtype=torch.float32)
        self._buffer = torch.cat([self._buffer, chunk], dim=0)

    def pop_chunk(self) -> DiarizationAudioChunk | None:
        if self.buffered_samples < self.config.chunk_samples:
            return None
        start = self._buffer_start_sample
        end = start + self.config.chunk_samples
        pcm = self._buffer[: self.config.chunk_samples].clone()
        advance = self.config.step_samples
        self._buffer = self._buffer[advance:].clone()
        self._buffer_start_sample += advance
        return DiarizationAudioChunk(
            pcm=pcm,
            start_sample=start,
            end_sample=end,
            sample_rate=self.config.sample_rate,
            is_final=False,
        )

    def flush(self) -> DiarizationAudioChunk | None:
        if self.buffered_samples < self.config.min_flush_samples:
            return None
        start = self._buffer_start_sample
        end = start + self.buffered_samples
        pcm = self._buffer.clone()
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._buffer_start_sample = end
        return DiarizationAudioChunk(
            pcm=pcm,
            start_sample=start,
            end_sample=end,
            sample_rate=self.config.sample_rate,
            is_final=True,
        )

    def reset(self) -> None:
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._buffer_start_sample = 0
