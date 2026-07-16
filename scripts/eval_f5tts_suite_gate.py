#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval_f5tts_audio_gate import (
    audio_stats,
    features,
    load_audio,
    make_transcriber,
    phonetic_word_error_rate,
    path_distance,
    word_error_rate,
)


_WORD_RE = re.compile(r"[a-z0-9']+")


def repetition_stats(text: str) -> dict[str, int]:
    words = _WORD_RE.findall(text.lower())
    max_run = 0
    repeated_runs = 0
    current_word = ""
    current_run = 0
    for word in words:
        if word == current_word:
            current_run += 1
        else:
            if current_run >= 3:
                repeated_runs += 1
            current_word = word
            current_run = 1
        max_run = max(max_run, current_run)
    if current_run >= 3:
        repeated_runs += 1

    phrase_repeat_hits = 0
    max_phrase_count = 0
    for ngram_size in range(2, 7):
        counts: dict[tuple[str, ...], int] = {}
        for offset in range(0, max(0, len(words) - ngram_size + 1)):
            ngram = tuple(words[offset : offset + ngram_size])
            counts[ngram] = counts.get(ngram, 0) + 1
        for ngram, count in counts.items():
            # Single function words repeat naturally; repeated multiword phrases
            # are a strong signature of diffusion collapse.
            if count >= 3 and len(set(ngram)) > 1:
                phrase_repeat_hits += 1
                max_phrase_count = max(max_phrase_count, count)

    # Whisper often renders audio collapse as stuttered characters or repeated
    # short fragments. Keep this strict enough to avoid penalizing normal speech.
    stutter_fragments = len(re.findall(r"\b([a-z])(?:[- ]\1){2,}\b", text.lower()))
    repeated_syllables = len(re.findall(r"([a-z]{2,4})\1{3,}", text.lower()))
    return {
        "asr_max_consecutive_word_repeat": int(max_run),
        "asr_repeated_word_runs": int(repeated_runs),
        "asr_repeated_phrase_hits": int(phrase_repeat_hits),
        "asr_max_repeated_phrase_count": int(max_phrase_count),
        "asr_stutter_fragments": int(stutter_fragments),
        "asr_repeated_syllables": int(repeated_syllables),
        "asr_repetition_flag": int(
            max_run >= 4
            or repeated_runs > 0
            or phrase_repeat_hits > 0
            or stutter_fragments > 0
            or repeated_syllables > 0
        ),
    }


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_candidate_runtime(candidate_manifest: dict, min_cuda_total_gb: float) -> None:
    if float(min_cuda_total_gb) <= 0.0:
        return
    runtime = candidate_manifest.get("runtime") or {}
    if str(runtime.get("device", "")).startswith("cuda"):
        total_gb = float(runtime.get("cuda_total_memory_gb", 0.0))
        if total_gb < float(min_cuda_total_gb):
            raise RuntimeError(
                f"candidate render was produced on a CUDA device below the promotion/parity memory floor: "
                f"{runtime.get('cuda_name')} has {total_gb:.2f} GB, "
                f"required {float(min_cuda_total_gb):.2f} GB"
            )


def summarize(rows: list[dict], outputs: list[dict] | None = None) -> dict:
    count = max(1, len(rows))
    summary = {
        "samples": len(rows),
        "mean_logmel_dtw_to_teacher": sum(float(row["logmel_dtw_to_teacher"]) for row in rows) / count,
        "mean_mfcc_dtw_to_teacher": sum(float(row["mfcc_dtw_to_teacher"]) for row in rows) / count,
        "total_clipped": int(sum(int(row["clipped"]) for row in rows)),
        "mean_peak": sum(float(row["peak"]) for row in rows) / count,
        "mean_rms": sum(float(row["rms"]) for row in rows) / count,
    }
    if rows and "asr_wer" in rows[0]:
        summary["mean_asr_wer"] = sum(float(row["asr_wer"]) for row in rows) / count
    if rows and "asr_phonetic_wer" in rows[0]:
        summary["mean_asr_phonetic_wer"] = sum(float(row["asr_phonetic_wer"]) for row in rows) / count
    if rows and "duration_ratio_to_teacher" in rows[0]:
        duration_ratios = [float(row["duration_ratio_to_teacher"]) for row in rows]
        duration_shortfalls = [float(row["duration_shortfall_seconds"]) for row in rows]
        duration_deltas = [float(row["duration_delta_seconds"]) for row in rows]
        summary["mean_duration_ratio_to_teacher"] = sum(duration_ratios) / count
        summary["min_duration_ratio_to_teacher"] = min(duration_ratios)
        summary["mean_duration_delta_seconds"] = sum(duration_deltas) / count
        summary["mean_duration_shortfall_seconds"] = sum(duration_shortfalls) / count
        summary["max_duration_shortfall_seconds"] = max(duration_shortfalls)
    if rows and "asr_repetition_flag" in rows[0]:
        summary["repetition_flagged_outputs"] = int(sum(int(row["asr_repetition_flag"]) for row in rows))
        summary["max_asr_consecutive_word_repeat"] = int(max(int(row["asr_max_consecutive_word_repeat"]) for row in rows))
        summary["total_asr_repeated_phrase_hits"] = int(sum(int(row["asr_repeated_phrase_hits"]) for row in rows))
        summary["max_asr_repeated_phrase_count"] = int(max(int(row["asr_max_repeated_phrase_count"]) for row in rows))
        summary["total_asr_stutter_fragments"] = int(sum(int(row["asr_stutter_fragments"]) for row in rows))
        summary["total_asr_repeated_syllables"] = int(sum(int(row["asr_repeated_syllables"]) for row in rows))
    if outputs:
        output_count = max(1, len(outputs))
        raw_peaks = [float(row.get("raw_peak", 0.0)) for row in outputs]
        summary["mean_raw_peak"] = sum(raw_peaks) / output_count
        summary["max_raw_peak"] = max(raw_peaks) if raw_peaks else 0.0
        summary["raw_peak_over_1_outputs"] = int(sum(1 for peak in raw_peaks if peak > 1.0))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two rendered F5TTS suite manifests.")
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--asr-model", default="")
    parser.add_argument("--asr-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--asr-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--asr-reference-field",
        choices=("text", "speech_text"),
        default="text",
        help="Manifest field to use as the ASR reference. Use speech_text for normalized F5TTS renders.",
    )
    parser.add_argument(
        "--require-candidate-min-cuda-gb",
        type=float,
        default=0.0,
        help="Abort if candidate manifest was rendered on a CUDA GPU below this memory floor.",
    )
    args = parser.parse_args()

    teacher_manifest = load_manifest(Path(args.teacher_manifest))
    candidate_manifest = load_manifest(Path(args.candidate_manifest))
    validate_candidate_runtime(candidate_manifest, float(args.require_candidate_min_cuda_gb))
    if len(teacher_manifest["outputs"]) != len(candidate_manifest["outputs"]):
        raise ValueError("teacher and candidate manifests have different output counts")

    transcribe = None
    if args.asr_model:
        transcribe = make_transcriber(
            args.asr_model,
            local_files_only=bool(args.asr_local_files_only),
            device=str(args.asr_device),
        )

    rows = []
    teacher_rows = []
    for teacher_output, candidate_output in zip(teacher_manifest["outputs"], candidate_manifest["outputs"], strict=True):
        text = str(teacher_output["text"])
        if text != str(candidate_output["text"]):
            raise ValueError(f"text mismatch at index {teacher_output['index']}")
        asr_reference = str(teacher_output.get(args.asr_reference_field) or text)
        candidate_reference = str(candidate_output.get(args.asr_reference_field) or candidate_output["text"])
        if asr_reference != candidate_reference:
            raise ValueError(f"ASR reference mismatch at index {teacher_output['index']}")

        teacher_audio, teacher_sr = load_audio(Path(teacher_output["path"]))
        candidate_audio, candidate_sr = load_audio(Path(candidate_output["path"]))
        if candidate_sr != teacher_sr:
            raise ValueError(f"sample-rate mismatch at index {teacher_output['index']}")

        teacher_mfcc, teacher_logmel = features(teacher_audio, teacher_sr)
        candidate_mfcc, candidate_logmel = features(candidate_audio, candidate_sr)

        teacher_stats = audio_stats(teacher_audio, teacher_sr)
        candidate_stats = audio_stats(candidate_audio, candidate_sr)
        teacher_seconds = float(teacher_stats["seconds"])
        candidate_seconds = float(candidate_stats["seconds"])
        duration_delta = candidate_seconds - teacher_seconds

        teacher_row = {
            "index": int(teacher_output["index"]),
            "text": text,
            "asr_reference": asr_reference,
            "file": str(teacher_output["path"]),
            **teacher_stats,
            "mfcc_dtw_to_teacher": 0.0,
            "logmel_dtw_to_teacher": 0.0,
        }
        row = {
            "index": int(candidate_output["index"]),
            "text": text,
            "asr_reference": asr_reference,
            "file": str(candidate_output["path"]),
            **candidate_stats,
            "teacher_seconds": teacher_seconds,
            "duration_delta_seconds": duration_delta,
            "duration_ratio_to_teacher": candidate_seconds / max(1e-6, teacher_seconds),
            "duration_shortfall_seconds": max(0.0, -duration_delta),
            "mfcc_dtw_to_teacher": path_distance(teacher_mfcc, candidate_mfcc),
            "logmel_dtw_to_teacher": path_distance(teacher_logmel, candidate_logmel),
        }
        if transcribe is not None:
            teacher_hypothesis = transcribe(teacher_audio, teacher_sr)
            candidate_hypothesis = transcribe(candidate_audio, candidate_sr)
            teacher_row["asr_hypothesis"] = teacher_hypothesis
            teacher_row["asr_wer"] = word_error_rate(asr_reference, teacher_hypothesis)
            teacher_row["asr_phonetic_wer"] = phonetic_word_error_rate(asr_reference, teacher_hypothesis)
            teacher_row.update(repetition_stats(teacher_hypothesis))
            row["asr_hypothesis"] = candidate_hypothesis
            row["asr_wer"] = word_error_rate(asr_reference, candidate_hypothesis)
            row["asr_phonetic_wer"] = phonetic_word_error_rate(asr_reference, candidate_hypothesis)
            row.update(repetition_stats(candidate_hypothesis))
            row["beats_teacher_asr_wer"] = row["asr_wer"] < teacher_row["asr_wer"]
            row["beats_teacher_asr_phonetic_wer"] = row["asr_phonetic_wer"] < teacher_row["asr_phonetic_wer"]
        rows.append(row)
        teacher_rows.append(teacher_row)

    result = {
        "teacher": {
            "manifest": str(Path(args.teacher_manifest)),
            "summary": summarize(teacher_rows, teacher_manifest.get("outputs")),
            "results": teacher_rows,
        },
        "candidate": {
            "manifest": str(Path(args.candidate_manifest)),
            "runtime": candidate_manifest.get("runtime"),
            "summary": summarize(rows, candidate_manifest.get("outputs")),
            "results": rows,
        },
    }
    if "mean_asr_wer" in result["candidate"]["summary"]:
        result["candidate"]["summary"]["beats_teacher_mean_asr_wer"] = (
            result["candidate"]["summary"]["mean_asr_wer"] < result["teacher"]["summary"]["mean_asr_wer"]
        )
    if "mean_asr_phonetic_wer" in result["candidate"]["summary"]:
        result["candidate"]["summary"]["beats_teacher_mean_asr_phonetic_wer"] = (
            result["candidate"]["summary"]["mean_asr_phonetic_wer"]
            < result["teacher"]["summary"]["mean_asr_phonetic_wer"]
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
