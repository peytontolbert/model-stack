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
from datasets import Audio, concatenate_datasets, load_dataset
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
    if len(parts) == 4:
        name, config, split, text_column = parts
        return name, config or None, split, text_column
    raise ValueError(
        "Dataset spec must be name:split:text_column or name:config:split:text_column"
    )


def load_training_dataset(specs: list[str], *, limit_per_dataset: int, sample_rate: int):
    datasets = []
    for spec in specs:
        name, config, split, text_column = parse_dataset_spec(spec)
        dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        if limit_per_dataset > 0:
            dataset = dataset.select(range(min(limit_per_dataset, len(dataset))))
        if "audio" not in dataset.column_names:
            raise ValueError(f"{spec} does not include an 'audio' column")
        if text_column not in dataset.column_names:
            raise ValueError(f"{spec} does not include text column {text_column!r}")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
        keep = {"audio", text_column}
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in keep])
        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")
        datasets.append(dataset)
    if not datasets:
        raise ValueError("At least one dataset is required")
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


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
            predicted = model.generate(inputs, max_new_tokens=max_new_tokens)
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
    parser.add_argument("--output-dir", default="/data/model/bddy-whisper-asr-lora")
    parser.add_argument("--limit-per-dataset", type=int, default=600)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-label-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = load_training_dataset(
        args.dataset,
        limit_per_dataset=args.limit_per_dataset,
        sample_rate=args.sample_rate,
    )
    split = dataset.train_test_split(
        test_size=min(args.eval_size, max(1, len(dataset) // 5)),
        seed=args.seed,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    def prepare(example):
        audio = example["audio"]
        audio_array = np.asarray(audio["array"], dtype=np.float32)
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

    train_dataset = split["train"].map(
        prepare,
        remove_columns=split["train"].column_names,
        desc="Preparing train audio",
    )
    eval_dataset = split["test"].map(
        prepare,
        remove_columns=split["test"].column_names,
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
        target_modules=["q_proj", "v_proj"],
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
        max_eval_samples=min(16, len(eval_dataset)),
        max_new_tokens=args.max_new_tokens,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    after = evaluate_samples(
        model,
        processor,
        eval_dataset,
        max_eval_samples=min(32, len(eval_dataset)),
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({"output_dir": args.output_dir, "before": before, "after": after}, indent=2))


if __name__ == "__main__":
    main()
