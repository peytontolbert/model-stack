#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def parse_dataset_spec(spec: str) -> tuple[str, str | None, str, str]:
    parts = spec.split(":")
    if len(parts) == 3:
        name, split, text_column = parts
        return name, None, split, text_column
    if len(parts) >= 4:
        name = parts[0]
        config = parts[1]
        split = ":".join(parts[2:-1])
        text_column = parts[-1]
        return name, config or None, split, text_column
    raise ValueError(
        "Dataset spec must be name:split:text_column or name:config:split:text_column"
    )


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_repetitive(text: str, *, max_unique_ratio: float, min_words: int) -> bool:
    words = normalize_text(text).split()
    if len(words) < min_words:
        return False
    unique_ratio = len(set(words)) / max(1, len(words))
    if unique_ratio <= max_unique_ratio:
        return True
    for gram_size in range(1, 9):
        index = 0
        while index + gram_size * 3 <= len(words):
            phrase = words[index : index + gram_size]
            repeats = 1
            while (
                index + repeats * gram_size + gram_size <= len(words)
                and words[index + repeats * gram_size : index + (repeats + 1) * gram_size]
                == phrase
            ):
                repeats += 1
            if repeats >= 4 or (gram_size >= 3 and repeats >= 3):
                return True
            index += max(1, repeats * gram_size)
    return False


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


def build_conversation_windows(
    dataset: Dataset,
    *,
    group_columns: list[str],
    max_duration_seconds: float,
    max_gap_seconds: float,
    min_words: int,
    sample_rate: int,
) -> Dataset:
    if max_duration_seconds <= 0:
        return dataset
    if "begin_time" not in dataset.column_names or "end_time" not in dataset.column_names:
        return dataset

    existing_group_columns = [column for column in group_columns if column in dataset.column_names]

    def sort_key(index: int):
        row = dataset[index]
        group = tuple(str(row.get(column, "")) for column in existing_group_columns)
        return (*group, float(row["begin_time"] or 0), float(row["end_time"] or 0))

    rows: list[dict[str, Any]] = []
    active_group = None
    active_audio: list[np.ndarray] = []
    active_text: list[str] = []
    active_duration = 0.0
    active_end = 0.0
    active_source_rows: list[int] = []

    def flush() -> None:
        nonlocal active_audio, active_text, active_duration, active_source_rows
        text = " ".join(piece.strip() for piece in active_text if piece.strip()).strip()
        if active_audio and len(normalize_text(text).split()) >= min_words:
            rows.append(
                {
                    "audio": {
                        "array": np.concatenate(active_audio).astype(np.float32),
                        "sampling_rate": sample_rate,
                    },
                    "reference_text": text,
                    "source_rows": list(active_source_rows),
                    "duration_seconds": active_duration,
                }
            )
        active_audio = []
        active_text = []
        active_duration = 0.0
        active_source_rows = []

    for index in sorted(range(len(dataset)), key=sort_key):
        row = dataset[index]
        group = tuple(str(row.get(column, "")) for column in existing_group_columns)
        begin = float(row["begin_time"] or 0)
        end = float(row["end_time"] or begin)
        audio = row["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        duration = len(audio_array) / float(audio.get("sampling_rate") or sample_rate)
        gap = max(0.0, begin - active_end) if active_audio else 0.0
        should_flush = bool(active_audio) and (
            group != active_group
            or gap > max_gap_seconds
            or active_duration + min(gap, max_gap_seconds) + duration > max_duration_seconds
        )
        if should_flush:
            flush()
        if active_audio and gap > 0:
            active_audio.append(np.zeros(int(min(gap, max_gap_seconds) * sample_rate), dtype=np.float32))
            active_duration += min(gap, max_gap_seconds)
        active_group = group
        active_audio.append(audio_array)
        active_text.append(str(row["reference_text"] or ""))
        active_duration += duration
        active_end = end
        active_source_rows.append(index)
    flush()
    return Dataset.from_list(rows)


def load_audio_rows(
    specs: list[str],
    *,
    limit_per_dataset: int,
    sample_rate: int,
    seed: int,
    streaming: bool,
    conversation_window_seconds: float,
    conversation_window_gap_seconds: float,
    conversation_window_group_columns: list[str],
    conversation_window_min_words: int,
) -> Dataset:
    datasets = []
    for spec in specs:
        name, config, split, text_column = parse_dataset_spec(spec)
        if streaming and conversation_window_seconds > 0:
            raise ValueError("Streaming teacher loading is only supported for row-level sources, not timestamp windows")
        dataset = (
            load_dataset(name, config, split=split, streaming=streaming)
            if config
            else load_dataset(name, split=split, streaming=streaming)
        )
        if "audio" not in dataset.column_names:
            raise ValueError(f"{spec} does not include an 'audio' column")
        if text_column not in dataset.column_names:
            raise ValueError(f"{spec} does not include text column {text_column!r}")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        if streaming:
            rows: list[dict[str, Any]] = []
            for row in dataset:
                rows.append(
                    {
                        "audio": row["audio"],
                        "reference_text": str(row.get(text_column) or ""),
                    }
                )
                if limit_per_dataset > 0 and len(rows) >= limit_per_dataset:
                    break
            dataset = Dataset.from_list(rows)
            datasets.append(dataset)
            continue
        if limit_per_dataset > 0 and conversation_window_seconds <= 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(limit_per_dataset, len(dataset))))
        if text_column != "reference_text":
            dataset = dataset.rename_column(text_column, "reference_text")
        dataset = build_conversation_windows(
            dataset,
            group_columns=conversation_window_group_columns,
            max_duration_seconds=conversation_window_seconds,
            max_gap_seconds=conversation_window_gap_seconds,
            min_words=conversation_window_min_words,
            sample_rate=sample_rate,
        )
        if limit_per_dataset > 0 and conversation_window_seconds > 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(limit_per_dataset, len(dataset))))
        keep = {"audio", "reference_text", "source_rows", "duration_seconds"}
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in keep])
        datasets.append(dataset)
    if not datasets:
        raise ValueError("At least one dataset is required")
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def transcribe_teacher(
    model: Any,
    processor: Any,
    audio_array: np.ndarray,
    sample_rate: int,
    *,
    max_new_tokens: int,
) -> str:
    inputs = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    dtype = next(model.parameters()).dtype
    inputs = {
        key: value.to(model.device, dtype=dtype) if key == "input_features" else value.to(model.device)
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        predicted = model.generate(
            inputs["input_features"],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=0.0,
            condition_on_prev_tokens=False,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
    return processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Parquet pseudo-label dataset from a stronger Whisper teacher."
    )
    parser.add_argument("--teacher-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=200)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--streaming", action="store_true", help="Stream HF row-level sources instead of downloading the whole split")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--min-duration-seconds", type=float, default=0.0)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--max-reference-teacher-wer", type=float, default=0.0)
    parser.add_argument("--max-repetitive-unique-ratio", type=float, default=0.2)
    parser.add_argument("--conversation-window-seconds", type=float, default=0.0)
    parser.add_argument("--conversation-window-gap-seconds", type=float, default=1.0)
    parser.add_argument("--conversation-window-group-columns", default="meeting_id,microphone_id")
    parser.add_argument("--conversation-window-min-words", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = load_audio_rows(
        args.dataset,
        limit_per_dataset=args.limit_per_dataset,
        sample_rate=args.sample_rate,
        seed=args.seed,
        streaming=bool(args.streaming),
        conversation_window_seconds=args.conversation_window_seconds,
        conversation_window_gap_seconds=args.conversation_window_gap_seconds,
        conversation_window_group_columns=[
            column.strip()
            for column in args.conversation_window_group_columns.split(",")
            if column.strip()
        ],
        conversation_window_min_words=args.conversation_window_min_words,
    )
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
    rejected = {"too_short": 0, "too_long": 0, "too_short_audio": 0, "repetitive": 0, "empty": 0, "disagreement": 0}
    for idx, row in enumerate(rows):
        audio = row["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio.get("sampling_rate") or args.sample_rate)
        duration = len(audio_array) / float(sample_rate)
        if args.min_duration_seconds > 0 and duration < args.min_duration_seconds:
            rejected["too_short_audio"] += 1
            continue
        if args.max_duration_seconds > 0 and duration > args.max_duration_seconds:
            rejected["too_long"] += 1
            continue
        teacher_text = transcribe_teacher(
            model,
            processor,
            audio_array,
            sample_rate,
            max_new_tokens=args.max_new_tokens,
        )
        word_count = len(normalize_text(teacher_text).split())
        if not teacher_text:
            rejected["empty"] += 1
            continue
        if word_count < args.min_words:
            rejected["too_short"] += 1
            continue
        if is_repetitive(
            teacher_text,
            max_unique_ratio=args.max_repetitive_unique_ratio,
            min_words=max(args.min_words, 12),
        ):
            rejected["repetitive"] += 1
            continue
        reference_text = str(row.get("reference_text") or "")
        reference_teacher_wer = word_error_rate(reference_text, teacher_text) if reference_text else None
        if (
            args.max_reference_teacher_wer > 0
            and reference_teacher_wer is not None
            and reference_teacher_wer > args.max_reference_teacher_wer
        ):
            rejected["disagreement"] += 1
            continue
        accepted.append(
            {
                "audio": {"array": audio_array, "sampling_rate": sample_rate},
                "teacher_text": teacher_text,
                "reference_text": reference_text,
                "reference_teacher_wer": reference_teacher_wer,
                "source_rows": row.get("source_rows"),
                "duration_seconds": duration,
            }
        )
        if (idx + 1) % 25 == 0:
            print(json.dumps({"seen": idx + 1, "accepted": len(accepted), "rejected": rejected}))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(accepted).cast_column("audio", Audio(sampling_rate=args.sample_rate)).to_parquet(
        str(output)
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "teacher_model": args.teacher_model,
                "accepted": len(accepted),
                "rejected": rejected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
