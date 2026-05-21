#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


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


def build_conversation_windows(
    dataset,
    *,
    group_columns: list[str],
    max_duration_seconds: float,
    max_gap_seconds: float,
    min_words: int,
    sample_rate: int,
):
    if max_duration_seconds <= 0:
        return dataset
    if "begin_time" not in dataset.column_names or "end_time" not in dataset.column_names:
        return dataset
    existing_group_columns = [column for column in group_columns if column in dataset.column_names]

    def sort_key(index: int):
        row = dataset[index]
        group = tuple(str(row.get(column, "")) for column in existing_group_columns)
        return (*group, float(row["begin_time"] or 0), float(row["end_time"] or 0))

    rows = []
    active_group = None
    active_audio: list[np.ndarray] = []
    active_text: list[str] = []
    active_duration = 0.0
    active_end = 0.0

    def flush() -> None:
        nonlocal active_audio, active_text, active_duration
        text = " ".join(piece.strip() for piece in active_text if piece.strip()).strip()
        if active_audio and len(text.split()) >= min_words:
            audio_array = np.concatenate(active_audio).astype(np.float32)
            rows.append(
                {
                    "audio": {"array": audio_array, "sampling_rate": sample_rate},
                    "text": text,
                }
            )
        active_audio = []
        active_text = []
        active_duration = 0.0

    for index in sorted(range(len(dataset)), key=sort_key):
        row = dataset[index]
        group = tuple(str(row.get(column, "")) for column in existing_group_columns)
        begin = float(row["begin_time"] or 0)
        end = float(row["end_time"] or begin)
        audio = row["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        duration = len(audio_array) / float(audio.get("sampling_rate") or sample_rate)
        gap = max(0.0, begin - active_end) if active_audio else 0.0
        should_flush = (
            bool(active_audio)
            and (
                group != active_group
                or gap > max_gap_seconds
                or active_duration + min(gap, max_gap_seconds) + duration > max_duration_seconds
            )
        )
        if should_flush:
            flush()
        if active_audio and gap > 0:
            active_audio.append(np.zeros(int(min(gap, max_gap_seconds) * sample_rate), dtype=np.float32))
            active_duration += min(gap, max_gap_seconds)
        active_group = group
        active_audio.append(audio_array)
        active_text.append(str(row["text"] or ""))
        active_duration += duration
        active_end = end
    flush()
    return Dataset.from_list(rows)


def load_training_dataset(
    specs: list[str],
    *,
    limit_per_dataset: int,
    sample_rate: int,
    shuffle_seed: int,
    conversation_window_seconds: float,
    conversation_window_gap_seconds: float,
    conversation_window_group_columns: list[str],
    conversation_window_min_words: int,
):
    datasets = []
    for spec in specs:
        name, config, split, text_column = parse_dataset_spec(spec)
        dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        if "audio" not in dataset.column_names:
            raise ValueError(f"{spec} does not include an 'audio' column")
        if text_column not in dataset.column_names:
            raise ValueError(f"{spec} does not include text column {text_column!r}")
        if limit_per_dataset > 0 and conversation_window_seconds <= 0:
            dataset = dataset.shuffle(seed=shuffle_seed).select(
                range(min(limit_per_dataset, len(dataset)))
            )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")
        dataset = build_conversation_windows(
            dataset,
            group_columns=conversation_window_group_columns,
            max_duration_seconds=conversation_window_seconds,
            max_gap_seconds=conversation_window_gap_seconds,
            min_words=conversation_window_min_words,
            sample_rate=sample_rate,
        )
        if limit_per_dataset > 0 and conversation_window_seconds > 0:
            dataset = dataset.shuffle(seed=shuffle_seed).select(
                range(min(limit_per_dataset, len(dataset)))
            )
        keep = {"audio", "text"}
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in keep])
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        datasets.append(dataset)
    if not datasets:
        raise ValueError("At least one dataset is required")
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def filter_transcript_quality(
    dataset,
    *,
    sample_rate: int,
    min_words: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
    min_chars_per_second: float,
    max_chars_per_second: float,
):
    if (
        min_words <= 0
        and min_duration_seconds <= 0
        and max_duration_seconds <= 0
        and min_chars_per_second <= 0
        and max_chars_per_second <= 0
    ):
        return dataset

    def keep(example):
        text = str(example["text"] or "").strip()
        word_count = len(text.split())
        if word_count < min_words:
            return False
        audio = example["audio"]
        duration_seconds = len(audio["array"]) / float(audio.get("sampling_rate") or sample_rate)
        if duration_seconds <= 0:
            return False
        if min_duration_seconds > 0 and duration_seconds < min_duration_seconds:
            return False
        if max_duration_seconds > 0 and duration_seconds > max_duration_seconds:
            return False
        chars_per_second = len(text) / duration_seconds
        if min_chars_per_second > 0 and chars_per_second < min_chars_per_second:
            return False
        if max_chars_per_second > 0 and chars_per_second > max_chars_per_second:
            return False
        return True

    return dataset.filter(keep, desc="Filtering transcript quality")


def mix_with_offset(base: np.ndarray, overlay: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    if len(base) == 0 or len(overlay) == 0:
        return base
    output = base.copy()
    if len(overlay) > len(output):
        start = int(rng.integers(0, len(overlay) - len(output) + 1))
        output += overlay[start : start + len(output)]
        return output
    start = int(rng.integers(0, len(output) - len(overlay) + 1))
    output[start : start + len(overlay)] += overlay
    return output


def apply_train_audio_augmentation(
    audio_array: np.ndarray,
    *,
    rng: np.random.Generator,
    gain_db_min: float,
    gain_db_max: float,
    noise_prob: float,
    noise_snr_db_min: float,
    noise_snr_db_max: float,
) -> np.ndarray:
    augmented = audio_array.astype(np.float32, copy=True)
    if gain_db_min != 0.0 or gain_db_max != 0.0:
        gain_db = float(rng.uniform(gain_db_min, gain_db_max))
        augmented *= float(10 ** (gain_db / 20.0))
    if noise_prob > 0 and rng.random() < noise_prob and len(augmented) > 0:
        signal_rms = float(np.sqrt(np.mean(np.square(augmented))) + 1e-8)
        snr_db = float(rng.uniform(noise_snr_db_min, noise_snr_db_max))
        noise_rms = signal_rms / float(10 ** (snr_db / 20.0))
        augmented += rng.normal(0.0, noise_rms, size=augmented.shape).astype(np.float32)
    return np.clip(augmented, -1.0, 1.0)


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def evaluate_samples(model, processor, dataset, *, max_eval_samples: int, max_new_tokens: int) -> dict:
    model.eval()
    rows = []
    for idx in range(min(max_eval_samples, len(dataset))):
        example = dataset[idx]
        dtype = next(model.parameters()).dtype
        inputs = torch.tensor(example["input_features"], dtype=dtype).unsqueeze(0).to(model.device)
        with torch.inference_mode():
            predicted = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
                condition_on_prev_tokens=False,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
        hypothesis = processor.batch_decode(predicted, skip_special_tokens=True)[0]
        label_ids = [token for token in example["labels"] if token != -100]
        reference = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
        rows.append(
            {
                "wer": word_error_rate(reference, hypothesis),
                "ref": reference,
                "hyp": hypothesis,
            }
        )
    wers = [row["wer"] for row in rows]
    return {
        "n": len(rows),
        "mean_wer": round(sum(wers) / max(1, len(wers)), 4),
        "median_wer": round(sorted(wers)[len(wers) // 2], 4) if wers else None,
        "worst": sorted(
            [
                {"wer": round(row["wer"], 4), "ref": row["ref"], "hyp": row["hyp"]}
                for row in rows
            ],
            key=lambda row: row["wer"],
            reverse=True,
        )[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper ASR with LoRA adapters.")
    parser.add_argument("--model", default="openai/whisper-small.en")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument(
        "--eval-dataset",
        action="append",
        help="Optional held-out dataset spec. Uses a train/test split from --dataset when omitted.",
    )
    parser.add_argument("--output-dir", default="/data/model/bddy-whisper-asr-lora")
    parser.add_argument("--merged-output-dir", default="")
    parser.add_argument("--limit-per-dataset", type=int, default=600)
    parser.add_argument("--eval-limit-per-dataset", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.0)
    parser.add_argument("--max-duration-seconds", type=float, default=0.0)
    parser.add_argument("--min-chars-per-second", type=float, default=0.0)
    parser.add_argument("--max-chars-per-second", type=float, default=0.0)
    parser.add_argument(
        "--conversation-window-seconds",
        type=float,
        default=0.0,
        help="Join timestamped adjacent utterances into ASR windows up to this duration.",
    )
    parser.add_argument("--conversation-window-gap-seconds", type=float, default=1.0)
    parser.add_argument(
        "--conversation-window-group-columns",
        default="meeting_id,microphone_id",
        help="Comma-separated metadata columns that must match when joining utterances.",
    )
    parser.add_argument("--conversation-window-min-words", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument("--max-label-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--train-gain-db-min", type=float, default=0.0)
    parser.add_argument("--train-gain-db-max", type=float, default=0.0)
    parser.add_argument("--train-noise-prob", type=float, default=0.0)
    parser.add_argument("--train-noise-snr-db-min", type=float, default=18.0)
    parser.add_argument("--train-noise-snr-db-max", type=float, default=30.0)
    parser.add_argument("--train-overlap-prob", type=float, default=0.0)
    parser.add_argument("--train-overlap-gain-db-min", type=float, default=-18.0)
    parser.add_argument("--train-overlap-gain-db-max", type=float, default=-8.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the base model on the prepared eval split without LoRA training.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=17,
        help="Seed used when shuffling before per-dataset row limits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = load_training_dataset(
        args.dataset,
        limit_per_dataset=args.limit_per_dataset,
        sample_rate=args.sample_rate,
        shuffle_seed=args.selection_seed,
        conversation_window_seconds=args.conversation_window_seconds,
        conversation_window_gap_seconds=args.conversation_window_gap_seconds,
        conversation_window_group_columns=[
            column.strip()
            for column in args.conversation_window_group_columns.split(",")
            if column.strip()
        ],
        conversation_window_min_words=args.conversation_window_min_words,
    )
    dataset = filter_transcript_quality(
        dataset,
        sample_rate=args.sample_rate,
        min_words=args.min_words,
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        min_chars_per_second=args.min_chars_per_second,
        max_chars_per_second=args.max_chars_per_second,
    )
    if args.eval_dataset:
        train_split = dataset
        eval_split = load_training_dataset(
            args.eval_dataset,
            limit_per_dataset=args.eval_limit_per_dataset or args.eval_size,
            sample_rate=args.sample_rate,
            shuffle_seed=args.selection_seed + 1,
            conversation_window_seconds=args.conversation_window_seconds,
            conversation_window_gap_seconds=args.conversation_window_gap_seconds,
            conversation_window_group_columns=[
                column.strip()
                for column in args.conversation_window_group_columns.split(",")
                if column.strip()
            ],
            conversation_window_min_words=args.conversation_window_min_words,
        )
        eval_split = filter_transcript_quality(
            eval_split,
            sample_rate=args.sample_rate,
            min_words=args.min_words,
            min_duration_seconds=args.min_duration_seconds,
            max_duration_seconds=args.max_duration_seconds,
            min_chars_per_second=args.min_chars_per_second,
            max_chars_per_second=args.max_chars_per_second,
        )
    else:
        split = dataset.train_test_split(
            test_size=min(args.eval_size, max(1, len(dataset) // 5)),
            seed=args.seed,
        )
        train_split = split["train"]
        eval_split = split["test"]
    processor = AutoProcessor.from_pretrained(args.model)

    def prepare(example, index=None, *, augment: bool = False):
        audio = example["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
        if augment and index is not None:
            rng = np.random.default_rng(args.seed + int(index))
            if args.train_overlap_prob > 0 and rng.random() < args.train_overlap_prob and len(train_split) > 1:
                overlay_index = int(rng.integers(0, len(train_split) - 1))
                if overlay_index >= int(index):
                    overlay_index += 1
                overlay_example = train_split[overlay_index]["audio"]
                overlay = np.asarray(overlay_example["array"], dtype=np.float32)
                overlay_gain_db = float(
                    rng.uniform(args.train_overlap_gain_db_min, args.train_overlap_gain_db_max)
                )
                overlay *= float(10 ** (overlay_gain_db / 20.0))
                audio_array = mix_with_offset(audio_array, overlay, rng=rng)
            audio_array = apply_train_audio_augmentation(
                audio_array,
                rng=rng,
                gain_db_min=args.train_gain_db_min,
                gain_db_max=args.train_gain_db_max,
                noise_prob=args.train_noise_prob,
                noise_snr_db_min=args.train_noise_snr_db_min,
                noise_snr_db_max=args.train_noise_snr_db_max,
            )
        if len(audio_array) < 480:
            audio_array = np.pad(audio_array, (0, 480 - len(audio_array)))
        features = processor.feature_extractor(
            audio_array,
            sampling_rate=audio["sampling_rate"],
        )
        labels = processor.tokenizer(
            example["text"],
            truncation=True,
            max_length=args.max_label_length,
        ).input_ids
        return {
            "input_features": features["input_features"][0],
            "labels": labels,
        }

    train_dataset = train_split.map(
        lambda example, index: prepare(example, index, augment=True),
        with_indices=True,
        remove_columns=train_split.column_names,
        desc="Preparing train audio",
    )
    eval_dataset = eval_split.map(
        lambda example: prepare(example, augment=False),
        remove_columns=eval_split.column_names,
        desc="Preparing eval audio",
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            module.strip()
            for module in args.lora_target_modules.split(",")
            if module.strip()
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=max(1, math.ceil(args.max_steps * 0.05)),
        max_steps=args.max_steps,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=args.max_steps,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperDataCollator(processor),
        processing_class=processor,
    )
    before = evaluate_samples(
        model,
        processor,
        eval_dataset,
        max_eval_samples=min(args.eval_samples, len(eval_dataset)),
        max_new_tokens=args.max_new_tokens,
    )
    if args.eval_only:
        print(
            json.dumps(
                {
                    "output_dir": None,
                    "merged_output_dir": None,
                    "before": before,
                    "after": before,
                },
                indent=2,
            )
        )
        return
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    merged_output_dir = args.merged_output_dir
    if merged_output_dir:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
        processor.save_pretrained(merged_output_dir)
        model = merged_model
    after = evaluate_samples(
        model,
        processor,
        eval_dataset,
        max_eval_samples=min(args.eval_samples, len(eval_dataset)),
        max_new_tokens=args.max_new_tokens,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "merged_output_dir": merged_output_dir or None,
                "before": before,
                "after": after,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
