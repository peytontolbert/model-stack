#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def dehiss(audio: np.ndarray, sample_rate: int, cutoff_hz: float, order: int, wet: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    nyquist = 0.5 * float(sample_rate)
    cutoff = min(float(cutoff_hz), nyquist * 0.95)
    if cutoff <= 0:
        return audio
    sos = scipy.signal.butter(int(order), cutoff / nyquist, btype="lowpass", output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, audio.astype(np.float64)).astype(np.float32)
    wet = min(1.0, max(0.0, float(wet)))
    mixed = (1.0 - wet) * audio.astype(np.float32) + wet * filtered
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a lightweight de-hiss filter to all WAVs in an F5TTS manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cutoff-hz", type=float, default=9000.0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--wet", type=float, default=0.65)
    args = parser.parse_args()

    source = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir) / str(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for row in source.get("outputs", []):
        path = Path(row["path"])
        audio, sample_rate = load_audio(path)
        filtered = dehiss(audio, sample_rate, float(args.cutoff_hz), int(args.order), float(args.wet))
        out_path = out_dir / path.name
        sf.write(out_path, filtered, sample_rate, format="WAV", subtype="PCM_16")
        next_row = dict(row)
        next_row["source_path"] = str(path)
        next_row["path"] = str(out_path)
        next_row["dehiss"] = {
            "cutoff_hz": float(args.cutoff_hz),
            "order": int(args.order),
            "wet": float(args.wet),
        }
        next_row["raw_peak"] = float(np.max(np.abs(filtered))) if filtered.size else 0.0
        next_row["samples"] = int(filtered.size)
        next_row["sample_rate"] = int(sample_rate)
        outputs.append(next_row)

    manifest = dict(source)
    manifest["label"] = str(args.label)
    manifest["post_filter"] = {
        "type": "butterworth_lowpass_mix",
        "cutoff_hz": float(args.cutoff_hz),
        "order": int(args.order),
        "wet": float(args.wet),
    }
    manifest["outputs"] = outputs
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
