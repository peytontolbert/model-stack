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


def load_audio_rows(
    specs: list[str],
    *,
    limit_per_dataset: int,
    sample_rate: int,
    seed: int,
) -> Dataset:
    datasets = []
    for spec in specs:
        name, config, split, text_column = parse_dataset_spec(spec)
        dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        if "audio" not in dataset.column_names:
            raise ValueError(f"{spec} does not include an 'audio' column")
        if text_column not in dataset.column_names:
            raise ValueError(f"{spec} does not include text column {text_column!r}")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        if limit_per_dataset > 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(limit_per_dataset, len(dataset))))
        if text_column != "reference_text":
            dataset = dataset.rename_column(text_column, "reference_text")
        keep = {"audio", "reference_text"}
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
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--max-repetitive-unique-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = load_audio_rows(
        args.dataset,
        limit_per_dataset=args.limit_per_dataset,
        sample_rate=args.sample_rate,
        seed=args.seed,
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
    rejected = {"too_short": 0, "too_long": 0, "repetitive": 0, "empty": 0}
    for idx, row in enumerate(rows):
        audio = row["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio.get("sampling_rate") or args.sample_rate)
        duration = len(audio_array) / float(sample_rate)
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
        accepted.append(
            {
                "audio": {"array": audio_array, "sampling_rate": sample_rate},
                "teacher_text": teacher_text,
                "reference_text": str(row.get("reference_text") or ""),
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
