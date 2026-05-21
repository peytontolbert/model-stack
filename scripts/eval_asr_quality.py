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
from datasets import load_dataset
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


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return edit_distance(ref_words, hyp_words) / max(1, len(ref_words))


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
        cases.append(EvalCase(f"clean_{idx}", array, sample_rate, reference))
        if perturb:
            cases.append(EvalCase(f"noise20db_{idx}", add_white_noise(array, 20.0, idx), sample_rate, reference))
            cases.append(EvalCase(f"noise10db_{idx}", add_white_noise(array, 10.0, idx), sample_rate, reference))
            cases.append(EvalCase(f"fast115_{idx}", speed_audio(array, 1.15), sample_rate, reference))
    return cases


def summarize(rows: list[dict]) -> dict:
    wers = [float(row["wer"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
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
    }


def case_group(name: str) -> str:
    return name.rsplit("_", 1)[0]


def transcribe_cases(model_id: str, cases: Iterable[EvalCase], *, max_new_tokens: int) -> dict:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    rows = []
    for case in cases:
        inputs = processor(case.audio, sampling_rate=case.sample_rate, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            predicted_ids = model.generate(inputs.input_features, max_new_tokens=max_new_tokens)
        latency_ms = (time.perf_counter() - started) * 1000.0
        hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        rows.append(
            {
                "case": case.name,
                "seconds": round(len(case.audio) / case.sample_rate, 2),
                "latency_ms": round(latency_ms, 1),
                "wer": round(word_error_rate(case.reference, hypothesis), 4),
                "ref": case.reference,
                "hyp": hypothesis,
            }
        )
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(case_group(str(row["case"])), []).append(row)
    return {
        "model": model_id,
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
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--perturb", action="store_true", help="Add 20dB noise, 10dB noise, and 1.15x speed cases.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    cases = build_cases(dataset, perturb=bool(args.perturb))
    results = [transcribe_cases(model_id, cases, max_new_tokens=args.max_new_tokens) for model_id in args.model]
    print(json.dumps({"dataset": args.dataset, "config": args.config, "split": args.split, "cases": len(cases), "results": results}, indent=2))


if __name__ == "__main__":
    main()
