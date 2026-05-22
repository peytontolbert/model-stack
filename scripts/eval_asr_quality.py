#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import torch
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def normalize_text(text: str) -> str:
    text = text.lower().replace("mister", "mr")
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


def word_error_counts(reference: str, hypothesis: str) -> dict[str, int]:
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


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return edit_distance(ref_words, hyp_words) / max(1, len(ref_words))


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

CONVERSATION_CRITICAL_PATTERNS = (
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


def pattern_hits(text: str, patterns: tuple[str, ...]) -> set[str]:
    normalized = normalize_text(text)
    return {pattern for pattern in patterns if pattern in normalized}


def pattern_recall(reference: str, hypothesis: str, patterns: tuple[str, ...]) -> float | None:
    ref_hits = pattern_hits(reference, patterns)
    if not ref_hits:
        return None
    hyp_hits = pattern_hits(hypothesis, patterns)
    return len(ref_hits & hyp_hits) / len(ref_hits)


def is_repetitive_transcript(text: str) -> bool:
    words = normalize_text(text).split()
    if len(words) < 16:
        return False
    if len(set(words)) / max(1, len(words)) <= 0.2:
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


def add_white_noise(audio, snr_db: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, len(audio)).astype("float32")
    signal_power = float((audio.astype("float32") ** 2).mean()) + 1.0e-12
    noise_power = float((noise**2).mean()) + 1.0e-12
    noise *= (signal_power / (noise_power * (10.0 ** (snr_db / 10.0)))) ** 0.5
    return (audio + noise).clip(-1.0, 1.0).astype("float32")


def speed_audio(audio, factor: float):
    return resample_poly(audio, 100, int(round(100 * factor))).clip(-1.0, 1.0).astype("float32")


@dataclass(frozen=True)
class EvalCase:
    name: str
    audio: object
    sample_rate: int
    reference: str


def build_cases(dataset, *, perturb: bool) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for idx, example in enumerate(dataset):
        audio = example["audio"]
        array = audio["array"].astype("float32")
        sample_rate = int(audio["sampling_rate"])
        reference = str(example["text"])
        case_id = str(example.get("case_id") or f"clean_{idx}")
        cases.append(EvalCase(case_id, array, sample_rate, reference))
        if perturb:
            cases.append(EvalCase(f"noise20db_{case_id}", add_white_noise(array, 20.0, idx), sample_rate, reference))
            cases.append(EvalCase(f"noise10db_{case_id}", add_white_noise(array, 10.0, idx), sample_rate, reference))
            cases.append(EvalCase(f"fast115_{case_id}", speed_audio(array, 1.15), sample_rate, reference))
    return cases


def build_conversation_window_cases(
    dataset,
    *,
    target_seconds: float,
    gap_seconds: float,
    perturb: bool,
) -> list[EvalCase]:
    import numpy as np

    cases: list[EvalCase] = []
    window_audio: list[object] = []
    window_text: list[str] = []
    window_seconds = 0.0
    window_start = 0
    window_index = 0
    current_sample_rate = 16_000

    def flush() -> None:
        nonlocal window_audio, window_text, window_seconds, window_start, window_index, current_sample_rate
        if not window_audio or not window_text:
            return
        gap = np.zeros(max(0, int(round(gap_seconds * current_sample_rate))), dtype="float32")
        chunks = []
        for audio in window_audio:
            if chunks and len(gap):
                chunks.append(gap)
            chunks.append(audio)
        combined = np.concatenate(chunks).clip(-1.0, 1.0).astype("float32")
        reference = " ".join(window_text)
        name = f"window_{window_index}_rows_{window_start}_{window_start + len(window_text) - 1}"
        cases.append(EvalCase(name, combined, current_sample_rate, reference))
        if perturb:
            cases.append(EvalCase(f"noise20db_{name}", add_white_noise(combined, 20.0, window_index), current_sample_rate, reference))
            cases.append(EvalCase(f"noise10db_{name}", add_white_noise(combined, 10.0, window_index), current_sample_rate, reference))
            cases.append(EvalCase(f"fast115_{name}", speed_audio(combined, 1.15), current_sample_rate, reference))
        window_audio = []
        window_text = []
        window_seconds = 0.0
        window_index += 1

    for idx, example in enumerate(dataset):
        audio = example["audio"]
        array = audio["array"].astype("float32")
        sample_rate = int(audio["sampling_rate"])
        seconds = len(array) / float(sample_rate)
        if sample_rate != current_sample_rate:
            flush()
            current_sample_rate = sample_rate
        if not window_audio:
            window_start = idx
        if window_audio and window_seconds + gap_seconds + seconds > target_seconds:
            flush()
            window_start = idx
        window_audio.append(array)
        window_text.append(str(example["text"]))
        window_seconds += seconds if len(window_audio) == 1 else gap_seconds + seconds
    flush()
    return cases


def filter_dataset(
    dataset,
    *,
    min_words: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
):
    if min_words <= 0 and min_duration_seconds <= 0 and max_duration_seconds <= 0:
        return dataset

    def keep(example):
        text = str(example["text"] or "")
        if len(normalize_text(text).split()) < min_words:
            return False
        audio = example["audio"]
        duration = len(audio["array"]) / float(audio.get("sampling_rate") or 16_000)
        if min_duration_seconds > 0 and duration < min_duration_seconds:
            return False
        if max_duration_seconds > 0 and duration > max_duration_seconds:
            return False
        return True

    return dataset.filter(keep, desc="Filtering ASR eval rows")


def summarize(rows: list[dict]) -> dict:
    wers = [float(row["wer"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    ref_words = sum(int(row["ref_words"]) for row in rows)
    question_recalls = [
        float(row["question_recall"])
        for row in rows
        if row["question_recall"] is not None
    ]
    critical_recalls = [
        float(row["critical_recall"])
        for row in rows
        if row["critical_recall"] is not None
    ]
    if not rows:
        return {"n": 0}
    p90_index = max(0, min(len(rows) - 1, math.ceil(len(rows) * 0.9) - 1))
    return {
        "n": len(rows),
        "mean_wer": round(sum(wers) / len(wers), 4),
        "median_wer": round(statistics.median(wers), 4),
        "p90_wer": round(sorted(wers)[p90_index], 4),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "p90_latency_ms": round(sorted(latencies)[p90_index], 1),
        "substitution_rate": round(sum(int(row["substitutions"]) for row in rows) / max(1, ref_words), 4),
        "deletion_rate": round(sum(int(row["deletions"]) for row in rows) / max(1, ref_words), 4),
        "insertion_rate": round(sum(int(row["insertions"]) for row in rows) / max(1, ref_words), 4),
        "repetition_rate": round(sum(1 for row in rows if row["repetitive"]) / len(rows), 4),
        "question_recall": round(sum(question_recalls) / len(question_recalls), 4)
        if question_recalls
        else None,
        "critical_term_recall": round(sum(critical_recalls) / len(critical_recalls), 4)
        if critical_recalls
        else None,
    }


def case_group(name: str) -> str:
    return name.rsplit("_", 1)[0]


def transcribe_cases(
    model_id: str,
    cases: Iterable[EvalCase],
    *,
    max_new_tokens: int,
    device: str,
    num_beams: int,
) -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    rows = []
    for case in cases:
        inputs = processor(case.audio, sampling_rate=case.sample_rate, return_tensors="pt")
        inputs = {
            key: value.to(device, dtype=dtype) if key == "input_features" else value.to(device)
            for key, value in inputs.items()
        }
        started = time.perf_counter()
        with torch.inference_mode():
            predicted_ids = model.generate(
                inputs["input_features"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                temperature=0.0,
                condition_on_prev_tokens=False,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        counts = word_error_counts(case.reference, hypothesis)
        rows.append(
            {
                "case": case.name,
                "seconds": round(len(case.audio) / case.sample_rate, 2),
                "latency_ms": round(latency_ms, 1),
                "wer": round(word_error_rate(case.reference, hypothesis), 4),
                **counts,
                "repetitive": is_repetitive_transcript(hypothesis),
                "question_recall": pattern_recall(case.reference, hypothesis, QUESTION_PATTERNS),
                "critical_recall": pattern_recall(case.reference, hypothesis, CONVERSATION_CRITICAL_PATTERNS),
                "ref": case.reference,
                "hyp": hypothesis,
            }
        )
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(case_group(str(row["case"])), []).append(row)
    return {
        "model": model_id,
        "device": device,
        "num_beams": num_beams,
        "params_m": round(params / 1_000_000, 1),
        "summary": summarize(rows),
        "groups": {key: summarize(value) for key, value in sorted(by_group.items())},
        "worst": sorted(rows, key=lambda row: float(row["wer"]), reverse=True)[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR model quality on labeled Hugging Face audio datasets.")
    parser.add_argument("--model", action="append", required=True, help="HF ASR model id. Repeat for comparison.")
    parser.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.0)
    parser.add_argument("--max-duration-seconds", type=float, default=0.0)
    parser.add_argument("--perturb", action="store_true", help="Add 20dB noise, 10dB noise, and 1.15x speed cases.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-json", default="", help="Optional path to write the full eval JSON report.")
    parser.add_argument(
        "--conversation-window-seconds",
        type=float,
        default=0.0,
        help="Concatenate adjacent utterances into approximate conversation windows before evaluating.",
    )
    parser.add_argument(
        "--conversation-window-gap-seconds",
        type=float,
        default=0.25,
        help="Silence inserted between utterances when --conversation-window-seconds is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "parquet":
        dataset = load_dataset("parquet", data_files=args.config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, args.config, split=args.split)
    if "audio" not in dataset.column_names and "audio_path" in dataset.column_names:
        dataset = dataset.rename_column("audio_path", "audio")
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    if args.text_column != "text":
        dataset = dataset.rename_column(args.text_column, "text")
    dataset = filter_dataset(
        dataset,
        min_words=args.min_words,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
    )
    if args.limit > 0 and args.conversation_window_seconds <= 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.conversation_window_seconds > 0:
        cases = build_conversation_window_cases(
            dataset,
            target_seconds=args.conversation_window_seconds,
            gap_seconds=args.conversation_window_gap_seconds,
            perturb=bool(args.perturb),
        )
        if args.limit > 0:
            cases = cases[: args.limit]
    else:
        cases = build_cases(dataset, perturb=bool(args.perturb))
    results = [
        transcribe_cases(
            model_id,
            cases,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            num_beams=args.num_beams,
        )
        for model_id in args.model
    ]
    report = {"dataset": args.dataset, "config": args.config, "split": args.split, "cases": len(cases), "results": results}
    rendered = json.dumps(report, indent=2)
    if args.output_json:
        from pathlib import Path

        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
