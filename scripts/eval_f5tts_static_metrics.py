#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def band_ratio(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
    if audio.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
    power = spectrum * spectrum
    freqs = np.fft.rfftfreq(audio.size, 1.0 / float(sample_rate))
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    return float(power[freqs >= float(cutoff_hz)].sum() / total)


def spectral_flatness_high(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> float:
    if audio.size == 0:
        return 0.0
    stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256, win_length=1024)) + 1e-10
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
    high = stft[freqs >= float(cutoff_hz), :]
    if high.size == 0:
        return 0.0
    geometric = np.exp(np.mean(np.log(high), axis=0))
    arithmetic = np.mean(high, axis=0)
    return float(np.mean(geometric / np.maximum(arithmetic, 1e-10)))


def zero_crossing_rate(audio: np.ndarray) -> float:
    if audio.size < 2:
        return 0.0
    signs = np.signbit(audio)
    return float(np.mean(signs[1:] != signs[:-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure high-frequency static proxies for F5TTS samples.")
    parser.add_argument("--audio", action="append", required=True)
    parser.add_argument("--cutoff-hz", type=float, default=7000.0)
    args = parser.parse_args()

    rows = []
    for item in args.audio:
        path = Path(item)
        audio, sample_rate = load_audio(path)
        rows.append(
            {
                "file": str(path),
                "seconds": round(float(audio.size) / float(sample_rate), 4) if sample_rate else 0.0,
                "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
                "rms": float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0,
                "high_band_ratio": band_ratio(audio, sample_rate, float(args.cutoff_hz)),
                "high_band_flatness": spectral_flatness_high(audio, sample_rate, float(args.cutoff_hz)),
                "zero_crossing_rate": zero_crossing_rate(audio),
            }
        )
    print(json.dumps({"cutoff_hz": float(args.cutoff_hz), "results": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
