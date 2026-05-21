from __future__ import annotations

import math

import torch

from runtime.asr.config import AsrFeatureConfig


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)


def create_mel_filterbank(
    config: AsrFeatureConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a triangular mel filterbank shaped ``(n_mels, n_fft // 2 + 1)``."""

    if not dtype.is_floating_point:
        raise TypeError("mel filterbank dtype must be floating point")
    device = torch.device(device) if device is not None else torch.device("cpu")
    n_freqs = config.n_fft // 2 + 1
    f_max = float(config.f_max if config.f_max is not None else config.sample_rate / 2)
    if f_max > config.sample_rate / 2:
        raise ValueError("f_max cannot exceed Nyquist frequency")

    fft_freqs = torch.linspace(0.0, config.sample_rate / 2, n_freqs, device=device, dtype=dtype)
    mel_min = _hz_to_mel(torch.tensor(float(config.f_min), device=device, dtype=dtype))
    mel_max = _hz_to_mel(torch.tensor(f_max, device=device, dtype=dtype))
    mel_points = torch.linspace(mel_min, mel_max, config.n_mels + 2, device=device, dtype=dtype)
    hz_points = _mel_to_hz(mel_points)

    lower = hz_points[:-2].unsqueeze(1)
    center = hz_points[1:-1].unsqueeze(1)
    upper = hz_points[2:].unsqueeze(1)
    freqs = fft_freqs.unsqueeze(0)

    left = (freqs - lower) / (center - lower).clamp_min(torch.finfo(dtype).eps)
    right = (upper - freqs) / (upper - center).clamp_min(torch.finfo(dtype).eps)
    filters = torch.minimum(left, right).clamp_min(0.0)

    enorm = 2.0 / (upper.squeeze(1) - lower.squeeze(1)).clamp_min(torch.finfo(dtype).eps)
    filters = filters * enorm.unsqueeze(1)
    return filters


def pcm_to_log_mel(pcm: torch.Tensor, config: AsrFeatureConfig | None = None) -> torch.Tensor:
    """Convert mono PCM audio to Whisper-style log-mel features.

    Input shape can be ``(samples,)`` or ``(batch, samples)``. The caller is
    responsible for resampling to ``config.sample_rate`` before this step.
    """

    config = config or AsrFeatureConfig()
    if not pcm.dtype.is_floating_point:
        raise TypeError("pcm must be a floating point tensor in [-1, 1]")
    if pcm.ndim == 1:
        audio = pcm.unsqueeze(0)
    elif pcm.ndim == 2:
        audio = pcm
    else:
        raise ValueError("pcm must have shape (samples,) or (batch, samples)")
    if audio.shape[-1] == 0:
        raise ValueError("pcm cannot be empty")

    dtype = audio.dtype if audio.dtype in (torch.float32, torch.float64) else torch.float32
    work = audio.to(dtype=dtype)
    window = torch.hann_window(config.win_length, device=work.device, dtype=dtype)
    stft = torch.stft(
        work,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        pad_mode="reflect",
        return_complex=True,
    )
    power_spec = stft.abs().pow(2.0)
    mel_fb = create_mel_filterbank(config, device=work.device, dtype=dtype)
    mel = torch.matmul(mel_fb.unsqueeze(0), power_spec).clamp_min(config.log_floor)
    log_mel = torch.log10(mel)

    if config.whisper_normalize:
        floor = log_mel.amax(dim=(-2, -1), keepdim=True) - (config.dynamic_range_db / 10.0)
        log_mel = torch.maximum(log_mel, floor)
        log_mel = (log_mel + 4.0) / 4.0

    if pcm.ndim == 1:
        return log_mel.squeeze(0)
    return log_mel


def expected_num_frames(num_samples: int, config: AsrFeatureConfig | None = None) -> int:
    config = config or AsrFeatureConfig()
    if num_samples <= 0:
        return 0
    if config.center:
        return 1 + num_samples // config.hop_length
    return max(0, 1 + math.floor((num_samples - config.n_fft) / config.hop_length))
