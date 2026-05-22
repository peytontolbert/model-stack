#!/usr/bin/env python3
"""Collect real utterance text and speaker references for ASR synthesis.

This script samples Hugging Face ASR datasets into compact Parquet source
tables. It writes:

- utterances Parquet: text prompts that F5TTS can render;
- speaker reference Parquet: short real WAV clips for voice conditioning.

The generated tables are inputs to `build_synthetic_meeting_asr_dataset.py
plan-f5`, keeping the synthetic ASR pipeline Parquet-first.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, load_dataset


def _parse_dataset_spec(spec: str) -> tuple[str, str | None, str, str, str, str]:
    parts = spec.split(":")
    if len(parts) < 4:
        raise ValueError(
            "dataset spec must be name:config:split:text_column[:audio_column[:speaker_column]]"
        )
    name = parts[0]
    config = parts[1] or None
    if len(parts) >= 6:
        split = ":".join(parts[2:-3])
        text_column = parts[-3]
        audio_column = parts[-2] or "audio"
        speaker_column = parts[-1] or "speaker_id"
    elif len(parts) == 5:
        split = ":".join(parts[2:-2])
        text_column = parts[-2]
        audio_column = parts[-1] or "audio"
        speaker_column = "speaker_id"
    else:
        split = ":".join(parts[2:-1])
        text_column = parts[-1]
        audio_column = "audio"
        speaker_column = "speaker_id"
    return name, config, split, text_column, audio_column, speaker_column


def _clean_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:80] or "unknown"


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def _audio_duration(audio: dict[str, Any]) -> float:
    array = np.asarray(audio["array"], dtype=np.float32)
    rate = float(audio["sampling_rate"])
    return float(len(array) / max(rate, 1.0))


def _write_wav(audio: dict[str, Any], output_path: Path, target_sample_rate: int) -> None:
    array = np.asarray(audio["array"], dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    sample_rate = int(audio["sampling_rate"])
    if sample_rate != target_sample_rate:
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("resampling speaker references requires librosa") from exc
        array = librosa.resample(array, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    sf.write(str(output_path), array, sample_rate, subtype="PCM_16")


def collect(args: argparse.Namespace) -> None:
    utterance_rows: list[dict[str, Any]] = []
    speaker_rows: list[dict[str, Any]] = []
    seen_speaker_refs: set[str] = set()
    speaker_utterance_counts: dict[str, int] = {}

    for spec_index, spec in enumerate(args.dataset):
        name, config, split, text_column, audio_column, speaker_column = _parse_dataset_spec(spec)
        load_kwargs = {"split": split, "streaming": bool(args.streaming)}
        dataset = load_dataset(name, config, **load_kwargs) if config else load_dataset(name, **load_kwargs)
        if audio_column in dataset.column_names:
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=args.sample_rate))

        accepted = 0
        for row_index, row in enumerate(dataset):
            text = str(row.get(text_column) or "").strip()
            if not text or _word_count(text) < args.min_words:
                continue
            audio = row.get(audio_column)
            duration = _audio_duration(audio) if isinstance(audio, dict) and "array" in audio else 0.0
            if duration and duration < args.min_duration_seconds:
                continue
            if duration and duration > args.max_duration_seconds:
                continue

            speaker_id = str(row.get(speaker_column) or row.get("speaker") or row.get("file_id") or f"{spec_index}_{row_index}")
            if args.max_utterances_per_speaker > 0 and speaker_utterance_counts.get(speaker_id, 0) >= args.max_utterances_per_speaker:
                continue
            speaker_utterance_counts[speaker_id] = speaker_utterance_counts.get(speaker_id, 0) + 1
            source_id = f"{name}:{config or 'default'}:{split}"
            utterance_audio_path = ""
            if args.utterance_audio_dir and audio:
                utterance_name = (
                    f"{_clean_id(name)}_{_clean_id(config or 'default')}_"
                    f"{_clean_id(speaker_id)}_{row_index:08d}.wav"
                )
                utterance_audio_path = str(Path(args.utterance_audio_dir) / utterance_name)
                _write_wav(audio, Path(utterance_audio_path), args.sample_rate)
            utterance_rows.append(
                {
                    "text": text,
                    "audio_path": utterance_audio_path,
                    "speaker_id": speaker_id,
                    "source_dataset": source_id,
                    "source_row_index": int(row_index),
                    "duration_seconds": float(duration),
                }
            )

            if len(seen_speaker_refs) < args.max_speaker_refs and audio and speaker_id not in seen_speaker_refs:
                ref_name = f"{_clean_id(name)}_{_clean_id(config or 'default')}_{_clean_id(speaker_id)}.wav"
                ref_path = Path(args.speaker_ref_dir) / ref_name
                _write_wav(audio, ref_path, args.sample_rate)
                speaker_rows.append(
                    {
                        "speaker_id": speaker_id,
                        "speaker_reference_path": str(ref_path),
                        "speaker_reference_text": text,
                        "source_dataset": source_id,
                        "source_row_index": int(row_index),
                        "duration_seconds": float(duration),
                    }
                )
                seen_speaker_refs.add(speaker_id)

            accepted += 1
            if args.limit_per_dataset > 0 and accepted >= args.limit_per_dataset:
                break

    if not utterance_rows:
        raise ValueError("no utterances collected")
    if not speaker_rows:
        raise ValueError("no speaker references collected")

    Path(args.utterances_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.speaker_refs_output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(utterance_rows), args.utterances_output, compression="zstd")
    pq.write_table(pa.Table.from_pylist(speaker_rows), args.speaker_refs_output, compression="zstd")
    print(f"wrote {len(utterance_rows)} utterances to {args.utterances_output}")
    print(f"wrote {len(speaker_rows)} speaker refs to {args.speaker_refs_output}")
    if args.streaming:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="name:config:split:text_column[:audio_column[:speaker_column]]",
    )
    parser.add_argument("--utterances-output", required=True)
    parser.add_argument("--speaker-refs-output", required=True)
    parser.add_argument("--speaker-ref-dir", required=True)
    parser.add_argument("--utterance-audio-dir", default="", help="Optional dir to save accepted utterance WAVs")
    parser.add_argument("--limit-per-dataset", type=int, default=1000)
    parser.add_argument("--max-speaker-refs", type=int, default=256)
    parser.add_argument("--max-utterances-per-speaker", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--min-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-duration-seconds", type=float, default=18.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--streaming", action="store_true", help="Stream HF rows instead of downloading the whole split")
    return parser


def main() -> None:
    collect(build_parser().parse_args())


if __name__ == "__main__":
    main()
