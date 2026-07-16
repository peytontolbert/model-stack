#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    n_fft = 1024
    hop_length = 256
    if audio.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: audio.size] = audio.astype(np.float64)
        frames = padded[None, :]
    else:
        starts = range(0, audio.size - n_fft + 1, hop_length)
        frames = np.stack([audio[start : start + n_fft] for start in starts], axis=0).astype(np.float64)
    window = np.hanning(n_fft).astype(np.float64)
    stft = np.abs(np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)).T + 1e-10
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sample_rate))
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


def score(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> dict:
    return {
        "seconds": round(float(audio.size) / float(sample_rate), 4) if sample_rate else 0.0,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0,
        "high_band_ratio": band_ratio(audio, sample_rate, cutoff_hz),
        "high_band_flatness": spectral_flatness_high(audio, sample_rate, cutoff_hz),
        "zero_crossing_rate": zero_crossing_rate(audio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure static proxies by segment for F5TTS manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--cutoff-hz", type=float, default=7000.0)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = []
    for output in manifest.get("outputs", []):
        audio, sample_rate = load_audio(Path(output["path"]))
        item = {
            "index": int(output.get("index", len(rows))),
            "path": output["path"],
            "text": output.get("text", ""),
            "full": score(audio, sample_rate, float(args.cutoff_hz)),
            "segments": [],
        }
        count = max(1, int(args.segments))
        for segment in range(count):
            start = int(round(audio.size * segment / count))
            end = int(round(audio.size * (segment + 1) / count))
            item["segments"].append({"segment": segment, **score(audio[start:end], sample_rate, float(args.cutoff_hz))})
        rows.append(item)
    print(json.dumps({"cutoff_hz": float(args.cutoff_hz), "segments": int(args.segments), "results": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
