import torch

from runtime.asr.config import AsrStreamConfig
from runtime.asr.streaming import AsrStreamState


def test_stream_state_emits_overlapped_chunks():
    config = AsrStreamConfig(sample_rate=10, chunk_seconds=1.0, overlap_seconds=0.2)
    state = AsrStreamState(config)
    state.append(torch.arange(18, dtype=torch.float32))

    first = state.pop_chunk()
    second = state.pop_chunk()

    assert first is not None
    assert first.start_sample == 0
    assert first.end_sample == 10
    assert first.sample_rate == 10
    assert first.duration_seconds == 1.0
    assert first.pcm.tolist() == list(range(10))
    assert second is not None
    assert second.start_sample == 8
    assert second.end_sample == 18
    assert second.pcm.tolist() == list(range(8, 18))
    assert state.pop_chunk() is None


def test_stream_flush_returns_remainder_and_resets_buffer():
    config = AsrStreamConfig(sample_rate=10, chunk_seconds=2.0, overlap_seconds=0.5, min_flush_seconds=0.2)
    state = AsrStreamState(config)
    state.append(torch.ones(5))

    chunk = state.flush()

    assert chunk is not None
    assert chunk.is_final
    assert chunk.start_sample == 0
    assert chunk.end_sample == 5
    assert state.buffered_samples == 0
    assert state.next_sample == 5
