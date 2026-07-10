import math

import torch

from runtime.diarization.config import (
    DiarizationReferenceConfig,
    DiarizationStreamConfig,
    DiarizationVadConfig,
)
from runtime.diarization.pipeline import ReferenceStreamingDiarizationRuntime
from runtime.diarization.streaming import DiarizationAudioChunk


def _tone(sample_rate: int, seconds: float, frequency: float, amplitude: float = 0.4) -> torch.Tensor:
    t = torch.arange(int(round(sample_rate * seconds)), dtype=torch.float32) / float(sample_rate)
    return amplitude * torch.sin(2.0 * math.pi * frequency * t)


def test_reference_runtime_assigns_stable_speaker_ids_across_overlapped_chunks():
    sample_rate = 100
    runtime = ReferenceStreamingDiarizationRuntime(
        stream_config=DiarizationStreamConfig(sample_rate=sample_rate, chunk_seconds=4.0, overlap_seconds=1.0),
        vad_config=DiarizationVadConfig(
            frame_seconds=0.4,
            hop_seconds=0.2,
            min_speech_seconds=0.3,
            merge_gap_seconds=0.15,
            absolute_energy_floor=1.0e-5,
        ),
        reference_config=DiarizationReferenceConfig(same_speaker_threshold=0.70),
    )

    chunk_one = torch.cat([
        _tone(sample_rate, 1.0, 3.0),
        torch.zeros(int(sample_rate * 0.2)),
        _tone(sample_rate, 1.0, 8.0),
        torch.zeros(int(sample_rate * 1.8)),
    ])
    chunk_two = torch.cat([
        torch.zeros(int(sample_rate * 1.0)),
        _tone(sample_rate, 1.0, 8.0),
        torch.zeros(int(sample_rate * 0.2)),
        _tone(sample_rate, 1.0, 3.0),
        torch.zeros(int(sample_rate * 0.8)),
    ])

    first = runtime.process_chunk(DiarizationAudioChunk(
        pcm=chunk_one,
        start_sample=0,
        end_sample=chunk_one.numel(),
        sample_rate=sample_rate,
        is_final=False,
    ))
    second = runtime.process_chunk(DiarizationAudioChunk(
        pcm=chunk_two,
        start_sample=300,
        end_sample=300 + chunk_two.numel(),
        sample_rate=sample_rate,
        is_final=True,
    ))

    assert len(first.segments) >= 2
    assert len(second.segments) >= 2

    first_ids = {segment.speaker_id for segment in first.segments}
    second_ids = {segment.speaker_id for segment in second.segments}
    assert len(first_ids) == 2
    assert len(second_ids) == 2
    assert first_ids == second_ids


def test_reference_runtime_holds_back_overlap_until_commit_horizon():
    sample_rate = 20
    runtime = ReferenceStreamingDiarizationRuntime(
        stream_config=DiarizationStreamConfig(sample_rate=sample_rate, chunk_seconds=4.0, overlap_seconds=1.0),
        vad_config=DiarizationVadConfig(
            frame_seconds=0.5,
            hop_seconds=0.25,
            min_speech_seconds=0.25,
            merge_gap_seconds=0.1,
            absolute_energy_floor=1.0e-5,
        ),
    )

    pcm = _tone(sample_rate, 4.0, 2.0)
    result = runtime.process_chunk(DiarizationAudioChunk(
        pcm=pcm,
        start_sample=0,
        end_sample=pcm.numel(),
        sample_rate=sample_rate,
        is_final=False,
    ))

    assert result.committed_until_sample == 60
    assert all(segment.end_sample <= 60 for segment in result.segments)
