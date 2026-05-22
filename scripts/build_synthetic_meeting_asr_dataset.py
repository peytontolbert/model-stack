#!/usr/bin/env python3
"""Build Parquet artifacts for synthetic conversational ASR stress data.

The script has two modes:

1. `plan-f5`: create a Parquet render-job table for F5TTS. Each row names the
   text, speaker reference WAV, and local F5 bundle that should be rendered.
2. `mix-rendered`: read rendered utterance WAVs and mix them into overlapped,
   noisy meeting-style audio examples with segment metadata.

The F5 renderer itself currently lives in /data/agent_kernel_lite JS/WASM
examples. Keeping the render jobs and mixed examples in Parquet lets ASR
training stay storage-efficient and avoids large JSONL artifacts.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


DEFAULT_F5_BUNDLE = "/data/agent_kernel_lite/artifacts/hf_releases/f5tts-4bit-distill"
DEFAULT_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class RenderedUtterance:
    audio_path: Path
    text: str
    speaker_id: str
    voice_id: str


def _read_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    return lines


def _read_texts_from_parquet(path: Path, text_column: str) -> list[str]:
    table = pq.read_table(path, columns=[text_column])
    texts = []
    for row in table.to_pylist():
        text = str(row.get(text_column) or "").strip()
        if text:
            texts.append(text)
    return texts


def _read_speaker_refs_from_manifest(path: Path) -> list[tuple[Path, str, str]]:
    table = pq.read_table(path)
    refs: list[tuple[Path, str, str]] = []
    for index, row in enumerate(table.to_pylist()):
        raw_path = row.get("speaker_reference_path") or row.get("audio_path") or row.get("path")
        if not raw_path:
            continue
        ref_path = Path(str(raw_path))
        speaker_id = str(row.get("speaker_id") or row.get("voice_id") or ref_path.stem or f"speaker_{index}")
        reference_text = str(row.get("speaker_reference_text") or row.get("reference_text") or "")
        refs.append((ref_path, speaker_id, reference_text))
    return refs


def _write_table(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output, compression="zstd")


def _load_audio_mono(path: Path, target_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if int(sample_rate) != int(target_sample_rate):
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("resampling requires librosa when sample rates differ") from exc
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=int(target_sample_rate))
    return np.asarray(audio, dtype=np.float32)


def _peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current <= 1e-6:
        return audio
    return audio * min(1.0, peak / current)


def _add_noise(audio: np.ndarray, rng: random.Random, snr_min: float, snr_max: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    snr_db = rng.uniform(snr_min, snr_max)
    signal_power = float(np.mean(np.square(audio))) + 1e-9
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.default_rng(rng.randrange(2**32)).normal(
        0.0, math.sqrt(noise_power), size=audio.shape
    )
    return audio + noise.astype(np.float32)


def _row_value(row: dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _load_rendered_utterances(path: Path) -> list[RenderedUtterance]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    utterances: list[RenderedUtterance] = []
    for index, row in enumerate(rows):
        audio_path = _row_value(row, ("audio_path", "path", "wav_path"))
        text = _row_value(row, ("text", "transcript", "reference_text", "teacher_text"))
        speaker_id = _row_value(row, ("speaker_id", "speaker", "voice_id"), f"speaker_{index % 8}")
        voice_id = _row_value(row, ("voice_id", "speaker_id", "speaker"), speaker_id)
        if not audio_path or not text:
            continue
        utterances.append(
            RenderedUtterance(
                audio_path=Path(audio_path),
                text=text,
                speaker_id=speaker_id,
                voice_id=voice_id,
            )
        )
    if not utterances:
        raise ValueError(f"no usable rendered utterances found in {path}")
    return utterances


def plan_f5(args: argparse.Namespace) -> None:
    if args.texts_parquet:
        texts = _read_texts_from_parquet(Path(args.texts_parquet), args.text_column)
    elif args.texts:
        texts = _read_lines(Path(args.texts))
    else:
        raise ValueError("plan-f5 requires either --texts or --texts-parquet")

    if args.speaker_reference_manifest:
        refs = _read_speaker_refs_from_manifest(Path(args.speaker_reference_manifest))
    else:
        refs = [(Path(item), Path(item).stem, "") for item in sorted(glob.glob(args.speaker_reference_glob))]
    if not texts:
        raise ValueError("no non-empty texts found")
    if not refs:
        raise ValueError("no speaker references found")

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for index in range(args.count):
        text = texts[index % len(texts)] if not args.shuffle_texts else rng.choice(texts)
        reference, speaker_id, reference_text = refs[index % len(refs)] if not args.shuffle_speakers else rng.choice(refs)
        voice_id = speaker_id or reference.stem
        output_wav = Path(args.render_output_dir) / f"f5_{index:08d}_{voice_id}.wav"
        rows.append(
            {
                "job_id": f"f5_{index:08d}",
                "text": text,
                "speaker_reference_path": str(reference),
                "speaker_reference_text": reference_text,
                "speaker_id": speaker_id,
                "voice_id": voice_id,
                "f5_bundle_path": str(Path(args.f5_bundle).resolve()),
                "output_wav_path": str(output_wav),
                "sample_rate": int(args.sample_rate),
            }
        )
    _write_table(rows, Path(args.output))
    print(f"wrote {len(rows)} F5 render jobs to {args.output}")


def mix_rendered(args: argparse.Namespace) -> None:
    utterances = _load_rendered_utterances(Path(args.rendered_utterances))
    utterances_by_speaker: dict[str, list[RenderedUtterance]] = {}
    for utterance in utterances:
        utterances_by_speaker.setdefault(utterance.speaker_id, []).append(utterance)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []

    for meeting_index in range(args.meetings):
        turn_count = rng.randint(args.min_turns, args.max_turns)
        if args.prefer_distinct_speakers and len(utterances_by_speaker) > 1:
            speakers = list(utterances_by_speaker)
            rng.shuffle(speakers)
            selected = [
                rng.choice(utterances_by_speaker[speaker])
                for speaker in speakers[: min(turn_count, len(speakers))]
            ]
            while len(selected) < turn_count:
                selected.append(rng.choice(utterances))
        else:
            selected = [rng.choice(utterances) for _ in range(turn_count)]
        cursor_seconds = 0.0
        placements: list[tuple[RenderedUtterance, np.ndarray, float, float]] = []

        for utterance in selected:
            audio = _load_audio_mono(utterance.audio_path, args.sample_rate)
            if args.turn_gain_db_min != 0.0 or args.turn_gain_db_max != 0.0:
                gain_db = rng.uniform(args.turn_gain_db_min, args.turn_gain_db_max)
                audio = audio * float(10.0 ** (gain_db / 20.0))

            duration = len(audio) / float(args.sample_rate)
            if placements and rng.random() < args.overlap_probability:
                gap = -rng.uniform(args.overlap_min_seconds, args.overlap_max_seconds)
            else:
                gap = rng.uniform(args.gap_min_seconds, args.gap_max_seconds)
            start = max(0.0, cursor_seconds + gap)
            end = start + duration
            placements.append((utterance, audio, start, end))
            cursor_seconds = max(cursor_seconds, end)

        total_samples = max(1, int(math.ceil((cursor_seconds + 0.2) * args.sample_rate)))
        mix = np.zeros(total_samples, dtype=np.float32)
        segments: list[dict[str, Any]] = []
        transcript_parts: list[str] = []

        for utterance, audio, start, end in placements:
            start_sample = int(round(start * args.sample_rate))
            end_sample = min(total_samples, start_sample + len(audio))
            mix[start_sample:end_sample] += audio[: end_sample - start_sample]
            segments.append(
                {
                    "speaker_id": utterance.speaker_id,
                    "voice_id": utterance.voice_id,
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "text": utterance.text,
                    "audio_path": str(utterance.audio_path),
                }
            )
            transcript_parts.append(f"[{utterance.speaker_id}] {utterance.text}")

        if rng.random() < args.noise_probability:
            mix = _add_noise(mix, rng, args.noise_snr_db_min, args.noise_snr_db_max)
        mix = _peak_normalize(mix)

        output_wav = Path(args.audio_output_dir) / f"synthetic_meeting_{meeting_index:08d}.wav"
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), mix, args.sample_rate, subtype="PCM_16")

        rows.append(
            {
                "audio": str(output_wav),
                "audio_path": str(output_wav),
                "text": " ".join(item.text for item, *_ in placements),
                "speaker_labeled_text": "\n".join(transcript_parts),
                "segments_json": json.dumps(segments, ensure_ascii=True),
                "duration_seconds": round(len(mix) / float(args.sample_rate), 3),
                "source": str(args.source_label),
                "overlap_probability": float(args.overlap_probability),
                "noise_probability": float(args.noise_probability),
                "sample_rate": int(args.sample_rate),
            }
        )

    _write_table(rows, Path(args.output))
    print(f"wrote {len(rows)} synthetic meeting rows to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    plan = subparsers.add_parser("plan-f5", help="write Parquet F5TTS render jobs")
    plan.add_argument("--texts", default="", help="newline-delimited utterance text file")
    plan.add_argument("--texts-parquet", default="", help="Parquet with utterance text")
    plan.add_argument("--text-column", default="text")
    plan.add_argument("--speaker-reference-glob", default="")
    plan.add_argument("--speaker-reference-manifest", default="", help="Parquet with speaker_reference_path")
    plan.add_argument("--output", required=True)
    plan.add_argument("--render-output-dir", required=True)
    plan.add_argument("--f5-bundle", default=DEFAULT_F5_BUNDLE)
    plan.add_argument("--count", type=int, default=1000)
    plan.add_argument("--sample-rate", type=int, default=24000)
    plan.add_argument("--seed", type=int, default=7)
    plan.add_argument("--shuffle-texts", action="store_true")
    plan.add_argument("--shuffle-speakers", action="store_true")
    plan.set_defaults(func=plan_f5)

    mix = subparsers.add_parser("mix-rendered", help="mix rendered utterance WAVs into meetings")
    mix.add_argument("--rendered-utterances", required=True, help="Parquet with audio_path/text/speaker_id")
    mix.add_argument("--output", required=True)
    mix.add_argument("--audio-output-dir", required=True)
    mix.add_argument("--meetings", type=int, default=1000)
    mix.add_argument("--min-turns", type=int, default=5)
    mix.add_argument("--max-turns", type=int, default=14)
    mix.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    mix.add_argument("--seed", type=int, default=11)
    mix.add_argument("--overlap-probability", type=float, default=0.25)
    mix.add_argument("--overlap-min-seconds", type=float, default=0.15)
    mix.add_argument("--overlap-max-seconds", type=float, default=0.9)
    mix.add_argument("--gap-min-seconds", type=float, default=0.05)
    mix.add_argument("--gap-max-seconds", type=float, default=1.4)
    mix.add_argument("--noise-probability", type=float, default=0.35)
    mix.add_argument("--noise-snr-db-min", type=float, default=8.0)
    mix.add_argument("--noise-snr-db-max", type=float, default=24.0)
    mix.add_argument("--turn-gain-db-min", type=float, default=-4.0)
    mix.add_argument("--turn-gain-db-max", type=float, default=4.0)
    mix.add_argument("--prefer-distinct-speakers", action="store_true")
    mix.add_argument("--source-label", default="synthetic_conversation_mix")
    mix.set_defaults(func=mix_rendered)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
