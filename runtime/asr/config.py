from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AsrFeatureConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float | None = None
    center: bool = True
    log_floor: float = 1.0e-10
    dynamic_range_db: float = 80.0
    whisper_normalize: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError("n_fft, hop_length, and win_length must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot exceed n_fft")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.f_min < 0:
            raise ValueError("f_min cannot be negative")
        if self.f_max is not None and self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")


@dataclass(frozen=True)
class AsrStreamConfig:
    sample_rate: int = 16_000
    chunk_seconds: float = 6.0
    overlap_seconds: float = 1.0
    min_flush_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        if self.overlap_seconds < 0:
            raise ValueError("overlap_seconds cannot be negative")
        if self.overlap_seconds >= self.chunk_seconds:
            raise ValueError("overlap_seconds must be less than chunk_seconds")
        if self.min_flush_seconds < 0:
            raise ValueError("min_flush_seconds cannot be negative")

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_seconds * self.sample_rate))

    @property
    def overlap_samples(self) -> int:
        return int(round(self.overlap_seconds * self.sample_rate))

    @property
    def step_samples(self) -> int:
        return self.chunk_samples - self.overlap_samples

    @property
    def min_flush_samples(self) -> int:
        return int(round(self.min_flush_seconds * self.sample_rate))
