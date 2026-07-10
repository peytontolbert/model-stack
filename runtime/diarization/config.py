from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiarizationStreamConfig:
    sample_rate: int = 16_000
    chunk_seconds: float = 8.0
    overlap_seconds: float = 3.0
    min_flush_seconds: float = 0.5

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


@dataclass(frozen=True)
class DiarizationVadConfig:
    frame_seconds: float = 0.4
    hop_seconds: float = 0.2
    min_speech_seconds: float = 0.45
    merge_gap_seconds: float = 0.25
    absolute_energy_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.frame_seconds <= 0:
            raise ValueError("frame_seconds must be positive")
        if self.hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        if self.hop_seconds > self.frame_seconds:
            raise ValueError("hop_seconds cannot exceed frame_seconds")
        if self.min_speech_seconds < 0:
            raise ValueError("min_speech_seconds cannot be negative")
        if self.merge_gap_seconds < 0:
            raise ValueError("merge_gap_seconds cannot be negative")
        if self.absolute_energy_floor <= 0:
            raise ValueError("absolute_energy_floor must be positive")


@dataclass(frozen=True)
class DiarizationReferenceConfig:
    band_count: int = 12
    same_speaker_threshold: float = 0.82
    prototype_momentum: float = 0.85
    max_speakers: int = 8
    analysis_window_seconds: float = 0.8
    analysis_hop_seconds: float = 0.4

    def __post_init__(self) -> None:
        if self.band_count <= 0:
            raise ValueError("band_count must be positive")
        if not 0.0 < self.same_speaker_threshold <= 1.0:
            raise ValueError("same_speaker_threshold must be in (0, 1]")
        if not 0.0 <= self.prototype_momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1)")
        if self.max_speakers <= 0:
            raise ValueError("max_speakers must be positive")
        if self.analysis_window_seconds <= 0:
            raise ValueError("analysis_window_seconds must be positive")
        if self.analysis_hop_seconds <= 0:
            raise ValueError("analysis_hop_seconds must be positive")
        if self.analysis_hop_seconds > self.analysis_window_seconds:
            raise ValueError("analysis_hop_seconds cannot exceed analysis_window_seconds")
