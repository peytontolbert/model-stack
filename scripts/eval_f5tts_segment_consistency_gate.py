#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_f5tts_audio_gate import (  # noqa: E402
    audio_stats,
    features,
    load_audio,
    make_transcriber,
    path_distance,
    phonetic_word_error_rate,
    word_error_rate,
)


def read_manifest_audio(manifest_path: Path) -> tuple[Path, str]:
    manifest = json.loads(manifest_path.read_text())
    outputs = manifest.get("outputs") or []
    if len(outputs) != 1:
        raise ValueError(f"{manifest_path} must contain exactly one output for segmented scoring")
    row = outputs[0]
    return Path(row["path"]), str(row.get("speech_text") or row.get("text") or "")


def split_text_even_words(text: str, parts: int) -> list[str]:
    words = text.split()
    if not words:
        return [""] * parts
    segments = []
    for index in range(parts):
        start = round(index * len(words) / parts)
        end = round((index + 1) * len(words) / parts)
        segments.append(" ".join(words[start:end]))
    return segments


def split_audio_by_text_lengths(audio: np.ndarray, segments: list[str]) -> list[np.ndarray]:
    weights = np.asarray([max(1, len(segment)) for segment in segments], dtype=np.float64)
    offsets = np.rint(np.cumsum(weights) / float(weights.sum()) * len(audio)).astype(np.int64)
    chunks = []
    start = 0
    for end in offsets:
        chunks.append(audio[start:int(end)])
        start = int(end)
    return chunks


def score_segments(
    *,
    teacher_audio: np.ndarray,
    candidate_audio: np.ndarray,
    sample_rate: int,
    segments: list[str],
    transcribe,
) -> list[dict[str, object]]:
    teacher_chunks = split_audio_by_text_lengths(teacher_audio, segments)
    candidate_chunks = split_audio_by_text_lengths(candidate_audio, segments)
    rows = []
    for index, (text, teacher_chunk, candidate_chunk) in enumerate(
        zip(segments, teacher_chunks, candidate_chunks, strict=True)
    ):
        teacher_mfcc, teacher_logmel = features(teacher_chunk, sample_rate)
        candidate_mfcc, candidate_logmel = features(candidate_chunk, sample_rate)
        row: dict[str, object] = {
            "segment": index,
            "text": text,
            "teacher": audio_stats(teacher_chunk, sample_rate),
            "candidate": audio_stats(candidate_chunk, sample_rate),
            "mfcc_dtw_to_teacher": path_distance(teacher_mfcc, candidate_mfcc),
            "logmel_dtw_to_teacher": path_distance(teacher_logmel, candidate_logmel),
        }
        if transcribe is not None:
            hyp = transcribe(candidate_chunk, sample_rate)
            row["asr_hypothesis"] = hyp
            row["asr_wer"] = word_error_rate(text, hyp)
            row["asr_phonetic_wer"] = phonetic_word_error_rate(text, hyp)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score long F5TTS samples by segment instead of whole-file average.")
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--candidate-manifest", action="append", required=True)
    parser.add_argument("--segment-text", action="append", default=[])
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--asr-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--asr-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    teacher_wav, teacher_text = read_manifest_audio(Path(args.teacher_manifest))
    segment_texts = [str(item).strip() for item in args.segment_text if str(item).strip()]
    if not segment_texts:
        segment_texts = split_text_even_words(teacher_text, int(args.segments))

    teacher_audio, teacher_sr = load_audio(teacher_wav)
    transcribe = None
    if args.asr_model:
        transcribe = make_transcriber(
            str(args.asr_model),
            local_files_only=bool(args.asr_local_files_only),
            device=str(args.asr_device),
        )

    result = {
        "teacher_manifest": str(args.teacher_manifest),
        "teacher_wav": str(teacher_wav),
        "segments": segment_texts,
        "candidates": [],
    }
    for manifest_arg in args.candidate_manifest:
        candidate_wav, _ = read_manifest_audio(Path(manifest_arg))
        candidate_audio, candidate_sr = load_audio(candidate_wav)
        if candidate_sr != teacher_sr:
            raise ValueError(f"sample-rate mismatch: teacher={teacher_sr}, candidate={candidate_sr}")
        rows = score_segments(
            teacher_audio=teacher_audio,
            candidate_audio=candidate_audio,
            sample_rate=teacher_sr,
            segments=segment_texts,
            transcribe=transcribe,
        )
        result["candidates"].append(
            {
                "manifest": str(manifest_arg),
                "wav": str(candidate_wav),
                "segments": rows,
                "max_segment_asr_wer": max((float(row.get("asr_wer", 0.0)) for row in rows), default=0.0),
                "max_segment_asr_phonetic_wer": max(
                    (float(row.get("asr_phonetic_wer", 0.0)) for row in rows),
                    default=0.0,
                ),
                "max_segment_logmel_dtw": max((float(row["logmel_dtw_to_teacher"]) for row in rows), default=0.0),
            }
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
