import torch

from runtime.asr.config import AsrFeatureConfig
from runtime.asr.features import create_mel_filterbank, expected_num_frames, pcm_to_log_mel


def test_pcm_to_log_mel_emits_finite_whisper_features():
    config = AsrFeatureConfig(n_mels=80)
    samples = torch.linspace(-1.0, 1.0, config.sample_rate)

    mel = pcm_to_log_mel(samples, config)

    assert mel.shape == (80, expected_num_frames(config.sample_rate, config))
    assert torch.isfinite(mel).all()


def test_pcm_to_log_mel_supports_batches_and_is_deterministic():
    config = AsrFeatureConfig(n_mels=40)
    pcm = torch.randn(2, config.sample_rate // 2)

    first = pcm_to_log_mel(pcm, config)
    second = pcm_to_log_mel(pcm, config)

    assert first.shape[0] == 2
    assert first.shape[1] == 40
    assert torch.allclose(first, second)


def test_create_mel_filterbank_has_expected_shape_and_nonnegative_weights():
    config = AsrFeatureConfig(n_mels=32, n_fft=256, win_length=256)

    filters = create_mel_filterbank(config)

    assert filters.shape == (32, 129)
    assert torch.all(filters >= 0)
    assert torch.count_nonzero(filters).item() > 0
