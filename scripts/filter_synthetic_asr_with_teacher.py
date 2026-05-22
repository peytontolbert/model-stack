#!/usr/bin/env python3
"""Filter synthetic ASR Parquet with a stronger teacher transcript.

F5TTS synthetic audio should not be used just because it rendered. This script
transcribes each synthetic row with a teacher ASR model and only accepts rows
where the teacher transcript is close to the intended target text. Rejected rows
are written to a separate Parquet for debugging.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    row = list(range(len(right) + 1))
    for i, left_word in enumerate(left, 1):
        prev, row[0] = row[0], i
        for j, right_word in enumerate(right, 1):
            old = row[j]
            row[j] = prev if left_word == right_word else min(prev, row[j], row[j - 1]) + 1
            prev = old
    return row[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return edit_distance(ref_words, hyp_words) / max(1, len(ref_words))


def is_repetitive(text: str) -> bool:
    words = normalize_text(text).split()
    if len(words) < 16:
        return False
    if len(set(words)) / max(1, len(words)) <= 0.2:
        return True
    for gram_size in range(1, 8):
        index = 0
        while index + gram_size * 3 <= len(words):
            phrase = words[index : index + gram_size]
            repeats = 1
            while (
                index + repeats * gram_size + gram_size <= len(words)
                and words[index + repeats * gram_size : index + (repeats + 1) * gram_size] == phrase
            ):
                repeats += 1
            if repeats >= 4 or (gram_size >= 3 and repeats >= 3):
                return True
            index += max(1, repeats * gram_size)
    return False


def transcribe_teacher(
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    sample_rate: int,
    *,
    max_new_tokens: int,
    language: str,
    task: str,
) -> str:
    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    inputs = {
        key: value.to(model.device, dtype=dtype) if key == "input_features" else value.to(model.device)
        for key, value in inputs.items()
    }
    generate_kwargs: dict[str, Any] = {}
    if language or task:
        if language:
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task
    with torch.inference_mode():
        base_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.0,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        try:
            predicted = model.generate(inputs["input_features"], **base_kwargs, **generate_kwargs)
        except ValueError:
            predicted = model.generate(inputs["input_features"], **base_kwargs)
    return processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()


def write_rows(rows: list[dict[str, Any]], output: Path, sample_rate: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=sample_rate)).to_parquet(
            str(output)
        )
        return
    table = pa.Table.from_pydict(
        {
            "audio": [],
            "teacher_text": [],
            "reference_text": [],
            "duration_seconds": [],
            "teacher_wer": [],
            "source": [],
        }
    )
    pq.write_table(table, output, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Synthetic Parquet with audio/text")
    parser.add_argument("--output", required=True, help="Accepted trainable Parquet")
    parser.add_argument("--rejected-output", default="", help="Optional rejected Parquet")
    parser.add_argument("--teacher-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--target-column", default="text")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-wer", type=float, default=0.25)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files=args.input, split="train")
    if "audio" not in dataset.column_names and "audio_path" in dataset.column_names:
        dataset = dataset.rename_column("audio_path", "audio")
    if "audio" not in dataset.column_names:
        raise ValueError("input must include audio or audio_path")
    if args.target_column not in dataset.column_names:
        raise ValueError(f"input missing target column {args.target_column!r}")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    processor = AutoProcessor.from_pretrained(args.teacher_model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.teacher_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        target_text = str(row.get(args.target_column) or "").strip()
        audio = row["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio.get("sampling_rate") or args.sample_rate)
        duration_seconds = len(audio_array) / float(sample_rate)
        teacher_text = transcribe_teacher(
            model,
            processor,
            audio_array,
            sample_rate,
            max_new_tokens=args.max_new_tokens,
            language=args.language,
            task=args.task,
        )
        teacher_wer = word_error_rate(target_text, teacher_text)
        reason = ""
        if len(normalize_text(target_text).split()) < args.min_words:
            reason = "target_too_short"
        elif duration_seconds > args.max_duration_seconds:
            reason = "too_long"
        elif not teacher_text:
            reason = "empty_teacher"
        elif is_repetitive(teacher_text):
            reason = "repetitive_teacher"
        elif teacher_wer > args.max_wer:
            reason = "teacher_target_mismatch"

        output_row = {
            "audio": {"array": audio_array, "sampling_rate": sample_rate},
            "teacher_text": teacher_text,
            "reference_text": target_text,
            "duration_seconds": float(duration_seconds),
            "teacher_wer": float(teacher_wer),
            "source": str(row.get("source") or "synthetic"),
        }
        if reason:
            rejected.append({**output_row, "reject_reason": reason})
        else:
            accepted.append(output_row)
        if (index + 1) % 25 == 0:
            print(json.dumps({"seen": index + 1, "accepted": len(accepted), "rejected": len(rejected)}))

    output = Path(args.output)
    write_rows(accepted, output, args.sample_rate)
    if args.rejected_output:
        rejected_output = Path(args.rejected_output)
        if rejected:
            Dataset.from_list(rejected).cast_column("audio", Audio(sampling_rate=args.sample_rate)).to_parquet(
                str(rejected_output)
            )
        else:
            table = pa.Table.from_pydict(
                {
                    "audio": [],
                    "teacher_text": [],
                    "reference_text": [],
                    "duration_seconds": [],
                    "teacher_wer": [],
                    "source": [],
                    "reject_reason": [],
                }
            )
            rejected_output.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, rejected_output, compression="zstd")
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "teacher_model": args.teacher_model,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "max_wer": args.max_wer,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
