#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Audio, Dataset, load_dataset


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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_slug(split: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", split.split("[", 1)[0]).strip("_") or "split"


def load_audio_dataset(spec: str, *, sample_rate: int):
    name, config, split, text_column = parse_dataset_spec(spec)
    dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    if "audio" not in dataset.column_names:
        raise ValueError(f"{spec} does not include an 'audio' column")
    if text_column not in dataset.column_names:
        raise ValueError(f"{spec} does not include text column {text_column!r}")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    return dataset, {
        "name": name,
        "config": config or "",
        "split": split,
        "text_column": text_column,
    }


def audio_duration_seconds(row: dict[str, Any], *, sample_rate: int) -> float:
    audio = row["audio"]
    return len(audio["array"]) / float(audio.get("sampling_rate") or sample_rate)


def short_rows(
    dataset,
    source: dict[str, str],
    *,
    limit: int,
    min_words: int,
    min_duration_seconds: float,
    max_duration_seconds: float,
    sample_rate: int,
    seed: int,
) -> Dataset:
    text_column = source["text_column"]

    def keep(row: dict[str, Any]) -> bool:
        text = str(row.get(text_column) or "")
        word_count = len(normalize_text(text).split())
        if word_count < min_words:
            return False
        duration = audio_duration_seconds(row, sample_rate=sample_rate)
        if min_duration_seconds > 0 and duration < min_duration_seconds:
            return False
        if max_duration_seconds > 0 and duration > max_duration_seconds:
            return False
        return True

    filtered = dataset.filter(keep, desc=f"Filtering {source['config'] or source['name']} short eval rows")
    if limit > 0:
        filtered = filtered.shuffle(seed=seed).select(range(min(limit, len(filtered))))

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(filtered):
        text = str(row.get(text_column) or "").strip()
        duration = audio_duration_seconds(row, sample_rate=sample_rate)
        rows.append(
            {
                "case_id": f"{source['config'] or source['name']}_short_{idx:05d}",
                "source_dataset": source["name"],
                "source_config": source["config"],
                "source_split": source["split"],
                "source_rows": [int(row.get("id", idx))] if str(row.get("id", "")).isdigit() else [idx],
                "group_key": group_key(row, default=""),
                "audio": row["audio"],
                "text": text,
                "duration_seconds": duration,
                "word_count": len(normalize_text(text).split()),
            }
        )
    return Dataset.from_list(rows)


def group_key(row: dict[str, Any], *, default: str) -> str:
    parts = []
    for column in ("meeting_id", "microphone_id", "speaker_id", "session_id", "file_id"):
        if column in row and row.get(column) is not None:
            parts.append(f"{column}={row[column]}")
    return "|".join(parts) if parts else default


def conversation_window_rows(
    dataset,
    source: dict[str, str],
    *,
    limit: int,
    max_duration_seconds: float,
    max_gap_seconds: float,
    min_words: int,
    sample_rate: int,
    seed: int,
    group_columns: list[str],
) -> Dataset:
    if "begin_time" not in dataset.column_names or "end_time" not in dataset.column_names:
        raise ValueError(f"{source['name']}:{source['config']} lacks begin_time/end_time for fixed window eval")

    text_column = source["text_column"]
    existing_group_columns = [column for column in group_columns if column in dataset.column_names]
    begin_times = dataset["begin_time"]
    end_times = dataset["end_time"]
    text_values = dataset[text_column]
    group_values = {column: dataset[column] for column in existing_group_columns}

    def sort_key(index: int):
        group = tuple(str(group_values[column][index] or "") for column in existing_group_columns)
        return (*group, float(begin_times[index] or 0.0), float(end_times[index] or 0.0), index)

    window_specs: list[dict[str, Any]] = []
    active_group: tuple[str, ...] | None = None
    active_text: list[str] = []
    active_duration = 0.0
    active_end = 0.0
    active_pieces: list[tuple[int, float]] = []
    active_group_key = ""

    def flush() -> None:
        nonlocal active_text, active_duration, active_pieces, active_group_key
        text = " ".join(piece.strip() for piece in active_text if piece.strip()).strip()
        word_count = len(normalize_text(text).split())
        if active_pieces and word_count >= min_words:
            row_index = len(window_specs)
            window_specs.append(
                {
                    "case_id": f"{source['config'] or source['name']}_window_{row_index:05d}",
                    "source_dataset": source["name"],
                    "source_config": source["config"],
                    "source_split": source["split"],
                    "source_rows": [index for index, _ in active_pieces],
                    "pieces": list(active_pieces),
                    "group_key": active_group_key,
                    "text": text,
                    "duration_seconds": active_duration,
                    "word_count": word_count,
                }
            )
        active_text = []
        active_duration = 0.0
        active_pieces = []
        active_group_key = ""

    for index in sorted(range(len(dataset)), key=sort_key):
        group = tuple(str(group_values[column][index] or "") for column in existing_group_columns)
        begin = float(begin_times[index] or 0.0)
        end = float(end_times[index] or begin)
        duration = max(0.0, end - begin)
        gap = max(0.0, begin - active_end) if active_pieces else 0.0
        should_flush = bool(active_pieces) and (
            group != active_group
            or gap > max_gap_seconds
            or active_duration + min(gap, max_gap_seconds) + duration > max_duration_seconds
        )
        if should_flush:
            flush()
        active_group = group
        active_group_key = "|".join(f"{column}={value}" for column, value in zip(existing_group_columns, group))
        gap_before = min(gap, max_gap_seconds) if active_pieces else 0.0
        active_pieces.append((index, gap_before))
        active_text.append(str(text_values[index] or ""))
        active_duration += gap_before + duration
        active_end = end
    flush()

    if limit > 0 and len(window_specs) > limit:
        rng = random.Random(seed)
        window_specs = rng.sample(window_specs, limit)
        window_specs.sort(key=lambda item: item["case_id"])

    rows: list[dict[str, Any]] = []
    for spec in window_specs:
        chunks: list[np.ndarray] = []
        duration_seconds = 0.0
        for index, gap_before in spec.pop("pieces"):
            if gap_before > 0:
                gap = np.zeros(int(round(gap_before * sample_rate)), dtype=np.float32)
                chunks.append(gap)
                duration_seconds += gap_before
            audio = dataset[index]["audio"]
            audio_array = np.asarray(audio["array"], dtype=np.float32)
            chunks.append(audio_array)
            duration_seconds += len(audio_array) / float(audio.get("sampling_rate") or sample_rate)
        rows.append(
            {
                **spec,
                "audio": {
                    "array": np.concatenate(chunks).astype(np.float32),
                    "sampling_rate": sample_rate,
                },
                "duration_seconds": duration_seconds,
            }
        )
    return Dataset.from_list(rows)


def write_dataset(dataset: Dataset, output: Path, *, sample_rate: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    dataset.to_parquet(str(output))
    return {
        "path": str(output),
        "rows": len(dataset),
        "total_seconds": round(sum(float(row["duration_seconds"]) for row in dataset), 2),
        "total_words": int(sum(int(row["word_count"]) for row in dataset)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed Parquet ASR eval suites for Bddy meeting transcription.")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--output-dir", default="/data/model/bddy-asr-eval/v14")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--short-limit-per-source", type=int, default=120)
    parser.add_argument("--window-limit-per-source", type=int, default=60)
    parser.add_argument("--min-short-words", type=int, default=4)
    parser.add_argument("--min-short-duration-seconds", type=float, default=1.0)
    parser.add_argument("--max-short-duration-seconds", type=float, default=24.0)
    parser.add_argument("--window-seconds", type=float, default=18.0)
    parser.add_argument("--window-gap-seconds", type=float, default=1.0)
    parser.add_argument("--window-min-words", type=int, default=16)
    parser.add_argument("--window-group-columns", default="meeting_id,microphone_id,speaker_id")
    args = parser.parse_args()

    specs = args.dataset or [
        "edinburghcstr/ami:sdm:validation:text",
        "edinburghcstr/ami:ihm:validation:text",
    ]
    output_dir = Path(args.output_dir)
    group_columns = [column.strip() for column in args.window_group_columns.split(",") if column.strip()]

    manifest: dict[str, Any] = {
        "output_dir": str(output_dir),
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "datasets": [],
        "artifacts": [],
    }

    for spec in specs:
        dataset, source = load_audio_dataset(spec, sample_rate=args.sample_rate)
        source_slug = "_".join(part for part in (source["config"], split_slug(source["split"])) if part)
        short_output = output_dir / f"ami_{source_slug}_short.parquet" if source["name"] == "edinburghcstr/ami" else output_dir / f"{source_slug}_short.parquet"
        window_output = output_dir / f"ami_{source_slug}_windows.parquet" if source["name"] == "edinburghcstr/ami" else output_dir / f"{source_slug}_windows.parquet"

        short_dataset = short_rows(
            dataset,
            source,
            limit=args.short_limit_per_source,
            min_words=args.min_short_words,
            min_duration_seconds=args.min_short_duration_seconds,
            max_duration_seconds=args.max_short_duration_seconds,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
        window_dataset = conversation_window_rows(
            dataset,
            source,
            limit=args.window_limit_per_source,
            max_duration_seconds=args.window_seconds,
            max_gap_seconds=args.window_gap_seconds,
            min_words=args.window_min_words,
            sample_rate=args.sample_rate,
            seed=args.seed,
            group_columns=group_columns,
        )

        manifest["datasets"].append(source)
        manifest["artifacts"].append(write_dataset(short_dataset, short_output, sample_rate=args.sample_rate))
        manifest["artifacts"].append(write_dataset(window_dataset, window_output, sample_rate=args.sample_rate))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "artifacts": manifest["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
