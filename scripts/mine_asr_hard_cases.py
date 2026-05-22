#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


QUESTION_PATTERNS = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "can you",
    "could you",
    "would you",
    "do you",
    "did you",
    "are you",
    "is there",
)

CRITICAL_PATTERNS = (
    "not sure",
    "don't know",
    "do not know",
    "unclear",
    "confused",
    "explain",
    "clarify",
    "repeat",
    "concern",
    "issue",
    "problem",
    "expensive",
    "budget",
    "timeline",
    "risk",
    "blocked",
    "maybe",
    "depends",
)


def normalize_text(text: str) -> str:
    text = text.lower().replace("mister", "mr")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    raise ValueError("Dataset spec must be name:split:text_column or name:config:split:text_column")


def load_audio_dataset(specs: list[str], *, sample_rate: int, limit_per_dataset: int, seed: int) -> Dataset:
    datasets = []
    for spec in specs:
        name, config, split, text_column = parse_dataset_spec(spec)
        if name == "parquet":
            if not config:
                raise ValueError("Parquet specs must be parquet:/path/file.parquet:split:text_column")
            dataset = load_dataset("parquet", data_files=config, split=split)
        else:
            dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        if "audio" not in dataset.column_names:
            raise ValueError(f"{spec} does not include an 'audio' column")
        if text_column not in dataset.column_names:
            raise ValueError(f"{spec} does not include text column {text_column!r}")
        if limit_per_dataset > 0:
            dataset = dataset.shuffle(seed=seed).select(range(min(limit_per_dataset, len(dataset))))
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")
        keep = {"audio", "text", "duration_seconds", "source_rows", "reference_text"}
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in keep])
        datasets.append(dataset.cast_column("audio", Audio(sampling_rate=sample_rate)))
    if not datasets:
        raise ValueError("At least one dataset is required")
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, cols):
        dp[0][j] = j
        back[0][j] = "insert"
    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "match"
                continue
            choices = [
                (dp[i - 1][j] + 1, "delete"),
                (dp[i][j - 1] + 1, "insert"),
                (dp[i - 1][j - 1] + 1, "substitute"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])
    i = len(ref_words)
    j = len(hyp_words)
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": len(ref_words)}
    while i > 0 or j > 0:
        action = back[i][j]
        if action == "match":
            i -= 1
            j -= 1
        elif action == "substitute":
            counts["substitutions"] += 1
            i -= 1
            j -= 1
        elif action == "delete":
            counts["deletions"] += 1
            i -= 1
        elif action == "insert":
            counts["insertions"] += 1
            j -= 1
        else:
            break
    return counts


def pattern_hits(text: str, patterns: tuple[str, ...]) -> set[str]:
    normalized = normalize_text(text)
    return {pattern for pattern in patterns if pattern in normalized}


def transcribe(model: Any, processor: Any, audio: dict[str, Any], *, max_new_tokens: int, device: str, dtype: torch.dtype) -> str:
    audio_array = np.asarray(audio["array"], dtype=np.float32)
    inputs = processor(audio_array, sampling_rate=int(audio.get("sampling_rate") or 16_000), return_tensors="pt")
    inputs = {
        key: value.to(device, dtype=dtype) if key == "input_features" else value.to(device)
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


def hard_case_score(reference: str, hypothesis: str, counts: dict[str, int]) -> float:
    ref_words = max(1, int(counts["ref_words"]))
    wer = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / ref_words
    deletion_rate = counts["deletions"] / ref_words
    question_miss = 1.0 if pattern_hits(reference, QUESTION_PATTERNS) and not pattern_hits(hypothesis, QUESTION_PATTERNS) else 0.0
    critical_miss = 1.0 if pattern_hits(reference, CRITICAL_PATTERNS) and not pattern_hits(hypothesis, CRITICAL_PATTERNS) else 0.0
    return wer + 0.75 * deletion_rate + 0.35 * question_miss + 0.25 * critical_miss


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard ASR rows into a focused Parquet training set.")
    parser.add_argument("--model", default="distil-whisper/distil-medium.en")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=800)
    parser.add_argument("--select", type=int, default=160)
    parser.add_argument("--repeat-selected", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--min-duration-seconds", type=float, default=4.0)
    parser.add_argument("--max-duration-seconds", type=float, default=24.0)
    parser.add_argument("--min-wer", type=float, default=0.12)
    parser.add_argument("--max-wer", type=float, default=1.2)
    parser.add_argument("--max-insertion-rate", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    dataset = load_audio_dataset(
        args.dataset,
        sample_rate=args.sample_rate,
        limit_per_dataset=args.limit_per_dataset,
        seed=args.seed,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        reference = str(row.get("text") or "").strip()
        word_count = len(normalize_text(reference).split())
        audio = row["audio"]
        duration = len(audio["array"]) / float(audio.get("sampling_rate") or args.sample_rate)
        if word_count < args.min_words:
            continue
        if args.min_duration_seconds > 0 and duration < args.min_duration_seconds:
            continue
        if args.max_duration_seconds > 0 and duration > args.max_duration_seconds:
            continue
        hypothesis = transcribe(
            model,
            processor,
            audio,
            max_new_tokens=args.max_new_tokens,
            device=device,
            dtype=dtype,
        )
        counts = edit_counts(reference, hypothesis)
        ref_words = max(1, int(counts["ref_words"]))
        wer = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / ref_words
        insertion_rate = counts["insertions"] / ref_words
        if wer < args.min_wer or wer > args.max_wer or insertion_rate > args.max_insertion_rate:
            continue
        score = hard_case_score(reference, hypothesis, counts)
        candidates.append(
            {
                "audio": audio,
                "text": reference,
                "reference_text": row.get("reference_text"),
                "source_rows": row.get("source_rows"),
                "duration_seconds": duration,
                "word_count": word_count,
                "hypothesis": hypothesis,
                "hard_case_score": score,
                "wer": wer,
                "deletion_rate": counts["deletions"] / ref_words,
                "substitution_rate": counts["substitutions"] / ref_words,
                "insertion_rate": insertion_rate,
            }
        )
        if (index + 1) % 50 == 0:
            print(json.dumps({"seen": index + 1, "candidates": len(candidates)}))

    candidates.sort(key=lambda row: float(row["hard_case_score"]), reverse=True)
    selected = candidates[: args.select] if args.select > 0 else candidates
    rng = random.Random(args.seed)
    repeated: list[dict[str, Any]] = []
    for row in selected:
        for _ in range(max(1, args.repeat_selected)):
            repeated.append(dict(row))
    rng.shuffle(repeated)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(repeated).cast_column("audio", Audio(sampling_rate=args.sample_rate)).to_parquet(str(output))
    print(
        json.dumps(
            {
                "output": str(output),
                "model": args.model,
                "device": device,
                "scored": len(candidates),
                "selected": len(selected),
                "written": len(repeated),
                "top": [
                    {
                        "score": round(float(row["hard_case_score"]), 4),
                        "wer": round(float(row["wer"]), 4),
                        "deletion_rate": round(float(row["deletion_rate"]), 4),
                        "text": row["text"][:160],
                        "hypothesis": row["hypothesis"][:160],
                    }
                    for row in selected[:10]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
