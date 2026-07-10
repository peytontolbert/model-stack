from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from runtime.diarization.config import (
    DiarizationReferenceConfig,
    DiarizationStreamConfig,
    DiarizationVadConfig,
)
from runtime.diarization.streaming import DiarizationAudioChunk


@dataclass(frozen=True)
class DiarizationSegment:
    speaker_id: str
    start_sample: int
    end_sample: int
    confidence: float

    @property
    def duration_samples(self) -> int:
        return max(0, int(self.end_sample - self.start_sample))


@dataclass
class SpeakerTrackState:
    speaker_id: str
    prototype: torch.Tensor
    total_weight: float
    last_seen_sample: int


@dataclass
class OnlineDiarizationState:
    speaker_tracks: list[SpeakerTrackState] = field(default_factory=list)
    committed_until_sample: int = 0


@dataclass(frozen=True)
class StreamingDiarizationResult:
    chunk: DiarizationAudioChunk
    segments: tuple[DiarizationSegment, ...]
    committed_until_sample: int
    active_speakers: tuple[str, ...]


def _frame_signal(audio: torch.Tensor, frame_samples: int, hop_samples: int) -> torch.Tensor:
    if audio.numel() <= frame_samples:
        padded = torch.zeros(frame_samples, dtype=audio.dtype)
        padded[: audio.numel()] = audio
        return padded.unsqueeze(0)
    return audio.unfold(0, frame_samples, hop_samples).contiguous()


def _rms_frames(audio: torch.Tensor, frame_samples: int, hop_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    frames = _frame_signal(audio, frame_samples, hop_samples)
    rms = torch.sqrt(frames.pow(2).mean(dim=1).clamp_min(1.0e-12))
    return frames, rms


def _speech_regions(audio: torch.Tensor, sample_rate: int, config: DiarizationVadConfig) -> list[tuple[int, int]]:
    frame_samples = max(1, int(round(config.frame_seconds * sample_rate)))
    hop_samples = max(1, int(round(config.hop_seconds * sample_rate)))
    _frames, rms = _rms_frames(audio, frame_samples, hop_samples)
    if rms.numel() == 0:
        return []

    noise_floor = float(torch.quantile(rms, 0.2).item()) if rms.numel() > 1 else float(rms[0].item())
    peak = float(torch.quantile(rms, 0.9).item()) if rms.numel() > 1 else float(rms.max().item())
    threshold = max(
        config.absolute_energy_floor,
        noise_floor + 0.10 * max(0.0, peak - noise_floor),
        peak * 0.30,
    )
    speech_mask = rms >= threshold
    if not bool(speech_mask.any()):
        return []

    min_speech_samples = int(round(config.min_speech_seconds * sample_rate))
    merge_gap_samples = int(round(config.merge_gap_seconds * sample_rate))
    regions: list[tuple[int, int]] = []
    start_sample: int | None = None

    for index, is_speech in enumerate(speech_mask.tolist()):
        frame_start = index * hop_samples
        if is_speech and start_sample is None:
            start_sample = frame_start
        elif not is_speech and start_sample is not None:
            regions.append((start_sample, frame_start))
            start_sample = None

    if start_sample is not None:
        regions.append((start_sample, int(audio.numel())))

    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] <= merge_gap_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return [(start, end) for start, end in merged if end - start >= min_speech_samples]


def _segment_embedding(segment_pcm: torch.Tensor, band_count: int) -> torch.Tensor:
    signal = segment_pcm.detach().to(dtype=torch.float32)
    if signal.numel() == 0:
        return F.normalize(torch.ones(band_count + 3), dim=0)
    signal = signal - signal.mean()
    window = torch.hann_window(signal.numel(), periodic=False, dtype=signal.dtype)
    spectrum = torch.fft.rfft(signal * window).abs().clamp_min(1.0e-8)
    edges = torch.linspace(0, spectrum.numel(), band_count + 1, dtype=torch.int64)
    band_values = []
    for idx in range(band_count):
        start = int(edges[idx].item())
        end = max(start + 1, int(edges[idx + 1].item()))
        band_values.append(torch.log1p(spectrum[start:end].mean()))
    band_tensor = torch.stack(band_values)
    zcr = (signal[:-1] * signal[1:] < 0).to(torch.float32).mean() if signal.numel() > 1 else torch.tensor(0.0)
    rms = torch.sqrt(signal.pow(2).mean().clamp_min(1.0e-12))
    freqs = torch.linspace(0.0, 1.0, spectrum.numel(), dtype=spectrum.dtype)
    centroid = (freqs * spectrum).sum() / spectrum.sum().clamp_min(1.0e-8)
    features = torch.cat([band_tensor, torch.tensor([zcr.item(), torch.log1p(rms).item(), centroid.item()])])
    return F.normalize(features, dim=0)


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _subsegment_region_by_embedding(
    pcm: torch.Tensor,
    region_start: int,
    sample_rate: int,
    config: DiarizationReferenceConfig,
) -> list[tuple[int, int, torch.Tensor]]:
    window_samples = max(1, int(round(config.analysis_window_seconds * sample_rate)))
    hop_samples = max(1, int(round(config.analysis_hop_seconds * sample_rate)))
    if pcm.numel() <= window_samples:
        return [(region_start, region_start + int(pcm.numel()), _segment_embedding(pcm, config.band_count))]

    windows: list[tuple[int, int, torch.Tensor]] = []
    start = 0
    last_start = max(0, int(pcm.numel()) - window_samples)
    while start <= last_start:
        end = start + window_samples
        windows.append((region_start + start, region_start + end, _segment_embedding(pcm[start:end], config.band_count)))
        start += hop_samples
    if not windows or windows[-1][1] < region_start + int(pcm.numel()):
        tail_start = max(0, int(pcm.numel()) - window_samples)
        windows.append((region_start + tail_start, region_start + int(pcm.numel()), _segment_embedding(pcm[tail_start:], config.band_count)))

    merged: list[list[object]] = []
    for start_sample, end_sample, embedding in windows:
        if not merged:
            merged.append([start_sample, end_sample, embedding])
            continue
        previous_embedding = merged[-1][2]
        similarity = _cosine_similarity(embedding, previous_embedding)
        if similarity >= config.same_speaker_threshold:
            merged[-1][1] = end_sample
            merged[-1][2] = F.normalize((previous_embedding + embedding) / 2.0, dim=0)
        else:
            merged.append([start_sample, end_sample, embedding])

    return [(int(start), int(end), embedding) for start, end, embedding in merged]


class ReferenceStreamingDiarizationRuntime:
    """CPU reference online diarization runtime.

    This is intentionally lightweight and model-free. It provides the online
    session mechanics we need first: overlapped windowing, speech gating,
    speaker-prototype tracking, and stable speaker stitching across updates.
    """

    def __init__(
        self,
        *,
        stream_config: DiarizationStreamConfig | None = None,
        vad_config: DiarizationVadConfig | None = None,
        reference_config: DiarizationReferenceConfig | None = None,
        state: OnlineDiarizationState | None = None,
    ):
        self.stream_config = stream_config or DiarizationStreamConfig()
        self.vad_config = vad_config or DiarizationVadConfig()
        self.reference_config = reference_config or DiarizationReferenceConfig()
        self.state = state or OnlineDiarizationState()

    def reset(self) -> None:
        self.state = OnlineDiarizationState()

    def _match_speaker(self, embedding: torch.Tensor, end_sample: int) -> tuple[str, float]:
        best_track: SpeakerTrackState | None = None
        best_score = -1.0
        for track in self.state.speaker_tracks:
            score = _cosine_similarity(embedding, track.prototype)
            if score > best_score:
                best_track = track
                best_score = score

        if best_track is not None and best_score >= self.reference_config.same_speaker_threshold:
            momentum = self.reference_config.prototype_momentum
            updated = F.normalize(momentum * best_track.prototype + (1.0 - momentum) * embedding, dim=0)
            best_track.prototype = updated
            best_track.total_weight += 1.0
            best_track.last_seen_sample = end_sample
            return best_track.speaker_id, best_score

        if len(self.state.speaker_tracks) < self.reference_config.max_speakers:
            speaker_id = f"speaker-{len(self.state.speaker_tracks) + 1}"
            self.state.speaker_tracks.append(
                SpeakerTrackState(
                    speaker_id=speaker_id,
                    prototype=embedding.clone(),
                    total_weight=1.0,
                    last_seen_sample=end_sample,
                )
            )
            return speaker_id, 1.0

        fallback = max(self.state.speaker_tracks, key=lambda track: _cosine_similarity(embedding, track.prototype))
        fallback.last_seen_sample = end_sample
        return fallback.speaker_id, max(best_score, 0.0)

    def process_chunk(self, chunk: DiarizationAudioChunk) -> StreamingDiarizationResult:
        relative_regions = _speech_regions(chunk.pcm, chunk.sample_rate, self.vad_config)
        raw_segments: list[DiarizationSegment] = []
        for rel_start, rel_end in relative_regions:
            region_pcm = chunk.pcm[rel_start:rel_end]
            for abs_start, abs_end, embedding in _subsegment_region_by_embedding(
                region_pcm,
                chunk.start_sample + rel_start,
                chunk.sample_rate,
                self.reference_config,
            ):
                speaker_id, confidence = self._match_speaker(embedding, abs_end)
                raw_segments.append(DiarizationSegment(
                    speaker_id=speaker_id,
                    start_sample=abs_start,
                    end_sample=abs_end,
                    confidence=confidence,
                ))

        commit_end = chunk.end_sample if chunk.is_final else max(chunk.start_sample, chunk.end_sample - self.stream_config.overlap_samples)
        committed_segments: list[DiarizationSegment] = []
        for segment in raw_segments:
            if segment.end_sample <= self.state.committed_until_sample:
                continue
            if segment.start_sample >= commit_end:
                continue
            start_sample = max(segment.start_sample, self.state.committed_until_sample)
            end_sample = min(segment.end_sample, commit_end)
            if end_sample <= start_sample:
                continue
            committed_segments.append(DiarizationSegment(
                speaker_id=segment.speaker_id,
                start_sample=start_sample,
                end_sample=end_sample,
                confidence=segment.confidence,
            ))

        self.state.committed_until_sample = max(self.state.committed_until_sample, commit_end)
        active_speakers = tuple(track.speaker_id for track in self.state.speaker_tracks)
        return StreamingDiarizationResult(
            chunk=chunk,
            segments=tuple(committed_segments),
            committed_until_sample=self.state.committed_until_sample,
            active_speakers=active_speakers,
        )
