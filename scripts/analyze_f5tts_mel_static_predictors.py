#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return float(power[freqs >= cutoff_hz].sum() / total)


def segment_bounds(length: int, segments: int) -> list[tuple[int, int]]:
    return [
        (int(round(length * index / segments)), int(round(length * (index + 1) / segments)))
        for index in range(segments)
    ]


def safe_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def mel_features(mel: np.ndarray, high_start_bin: int) -> dict[str, float]:
    mel = np.asarray(mel, dtype=np.float64)
    if mel.ndim != 2 or mel.size == 0:
        return {}
    low = mel[:, :high_start_bin]
    high = mel[:, high_start_bin:]
    frame_energy = np.mean(np.abs(mel), axis=1)
    high_energy = np.mean(np.abs(high), axis=1) if high.size else np.zeros_like(frame_energy)
    temporal = np.diff(mel, axis=0) if mel.shape[0] > 1 else np.zeros((0, mel.shape[1]))
    high_temporal = np.diff(high, axis=0) if high.shape[0] > 1 else np.zeros((0, high.shape[1]))
    second = np.diff(mel, n=2, axis=0) if mel.shape[0] > 2 else np.zeros((0, mel.shape[1]))
    return {
        "mel_mean": float(np.mean(mel)),
        "mel_abs_mean": float(np.mean(np.abs(mel))),
        "mel_std": float(np.std(mel)),
        "mel_p01": float(np.quantile(mel, 0.01)),
        "mel_p99": float(np.quantile(mel, 0.99)),
        "mel_range_p01_p99": float(np.quantile(mel, 0.99) - np.quantile(mel, 0.01)),
        "low_abs_mean": float(np.mean(np.abs(low))) if low.size else 0.0,
        "high_abs_mean": float(np.mean(np.abs(high))) if high.size else 0.0,
        "high_to_low_abs": float(np.mean(np.abs(high)) / max(1e-8, np.mean(np.abs(low)))) if low.size and high.size else 0.0,
        "frame_energy_mean": float(np.mean(frame_energy)),
        "frame_energy_std": float(np.std(frame_energy)),
        "frame_energy_p95": float(np.quantile(frame_energy, 0.95)),
        "high_frame_energy_mean": float(np.mean(high_energy)),
        "high_frame_energy_std": float(np.std(high_energy)),
        "temporal_abs_mean": float(np.mean(np.abs(temporal))) if temporal.size else 0.0,
        "high_temporal_abs_mean": float(np.mean(np.abs(high_temporal))) if high_temporal.size else 0.0,
        "second_abs_mean": float(np.mean(np.abs(second))) if second.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find mel-side predictors for F5TTS/Vocos high-band static.")
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--student-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--audio-cutoff-hz", type=float, default=7000.0)
    parser.add_argument("--high-start-bin", type=int, default=80)
    args = parser.parse_args()

    teacher = load_manifest(Path(args.teacher_manifest))
    student = load_manifest(Path(args.student_manifest))
    rows = []
    for teacher_out, student_out in zip(teacher["outputs"], student["outputs"], strict=True):
        teacher_mel = np.load(teacher_out["mel_path"])
        student_mel = np.load(student_out["mel_path"])
        teacher_audio, teacher_sr = load_audio(Path(teacher_out["path"]))
        student_audio, student_sr = load_audio(Path(student_out["path"]))
        if teacher_sr != student_sr:
            raise ValueError(f"sample-rate mismatch: {teacher_sr} vs {student_sr}")
        for segment, ((tm0, tm1), (sm0, sm1), (ta0, ta1), (sa0, sa1)) in enumerate(
            zip(
                segment_bounds(len(teacher_mel), int(args.segments)),
                segment_bounds(len(student_mel), int(args.segments)),
                segment_bounds(len(teacher_audio), int(args.segments)),
                segment_bounds(len(student_audio), int(args.segments)),
                strict=True,
            )
        ):
            teacher_audio_high = band_ratio(teacher_audio[ta0:ta1], teacher_sr, float(args.audio_cutoff_hz))
            student_audio_high = band_ratio(student_audio[sa0:sa1], student_sr, float(args.audio_cutoff_hz))
            teacher_features = mel_features(teacher_mel[tm0:tm1], int(args.high_start_bin))
            student_features = mel_features(student_mel[sm0:sm1], int(args.high_start_bin))
            row = {
                "index": int(student_out["index"]),
                "segment": int(segment),
                "text": student_out.get("text", ""),
                "teacher_audio_high_ratio": teacher_audio_high,
                "student_audio_high_ratio": student_audio_high,
                "audio_high_ratio_delta": student_audio_high - teacher_audio_high,
            }
            for key in sorted(student_features):
                row[f"teacher_{key}"] = teacher_features.get(key, 0.0)
                row[f"student_{key}"] = student_features[key]
                row[f"delta_{key}"] = student_features[key] - teacher_features.get(key, 0.0)
                denom = max(1e-8, abs(teacher_features.get(key, 0.0)))
                row[f"rel_delta_{key}"] = (student_features[key] - teacher_features.get(key, 0.0)) / denom
            rows.append(row)

    target_keys = ["student_audio_high_ratio", "audio_high_ratio_delta"]
    feature_keys = sorted(key for key in rows[0] if key.startswith(("delta_", "rel_delta_", "student_")))
    correlations = []
    for target in target_keys:
        ys = [float(row[target]) for row in rows]
        for key in feature_keys:
            if key in target_keys:
                continue
            xs = [float(row[key]) for row in rows]
            correlations.append({"target": target, "feature": key, "corr": safe_corr(xs, ys)})
    correlations.sort(key=lambda row: abs(row["corr"]), reverse=True)
    worst = sorted(rows, key=lambda row: row["student_audio_high_ratio"], reverse=True)[:12]
    payload = {
        "teacher_manifest": str(args.teacher_manifest),
        "student_manifest": str(args.student_manifest),
        "segments": int(args.segments),
        "audio_cutoff_hz": float(args.audio_cutoff_hz),
        "high_start_bin": int(args.high_start_bin),
        "top_abs_correlations": correlations[:40],
        "worst_student_audio_high_ratio": worst,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "top_abs_correlations": correlations[:12]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
