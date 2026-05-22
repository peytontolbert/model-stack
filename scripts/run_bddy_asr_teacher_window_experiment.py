#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, skip: bool = False) -> None:
    print(json.dumps({"command": command, "skip": skip}))
    if skip:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def merge_lora(*, base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    merged_dir.mkdir(parents=True, exist_ok=True)
    base = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoProcessor.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))


def parquet_num_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return 1
    return int(pq.ParquetFile(path).metadata.num_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bddy conversational ASR teacher-window training loop."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--work-dir", default="/data/model/bddy-asr-runs")
    parser.add_argument("--teacher-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--base-model", default="distil-whisper/distil-medium.en")
    parser.add_argument(
        "--source-dataset",
        action="append",
        default=[],
        help="Teacher-label timestamp/window source dataset spec. Repeat for diverse meeting corpora.",
    )
    parser.add_argument(
        "--streaming-row-source-dataset",
        action="append",
        default=[],
        help="Large row-level teacher source dataset spec to stream separately, such as Earnings22.",
    )
    parser.add_argument("--eval-dataset", default="edinburghcstr/ami:sdm:validation[:1500]:text")
    parser.add_argument("--limit-windows", type=int, default=500)
    parser.add_argument("--limit-streaming-rows", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,out_proj",
        help=(
            "Comma-separated LoRA module names. For higher-capacity ASR runs, "
            "include decoder MLP modules such as fc1,fc2 after validating memory."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--train-gain-db-min", type=float, default=-2.0)
    parser.add_argument("--train-gain-db-max", type=float, default=2.0)
    parser.add_argument("--train-noise-prob", type=float, default=0.1)
    parser.add_argument("--train-noise-snr-db-min", type=float, default=16.0)
    parser.add_argument("--train-noise-snr-db-max", type=float, default=28.0)
    parser.add_argument("--train-overlap-prob", type=float, default=0.0)
    parser.add_argument("--train-overlap-gain-db-min", type=float, default=-18.0)
    parser.add_argument("--train-overlap-gain-db-max", type=float, default=-8.0)
    parser.add_argument("--eval-limit", type=int, default=40)
    parser.add_argument("--window-eval-limit", type=int, default=12)
    parser.add_argument("--short-eval-parquet", default="", help="Optional fixed short-form eval Parquet built by build_bddy_asr_eval_suite.py")
    parser.add_argument("--window-eval-parquet", default="", help="Optional fixed conversation-window eval Parquet built by build_bddy_asr_eval_suite.py")
    parser.add_argument("--window-seconds", type=float, default=18.0)
    parser.add_argument("--window-gap-seconds", type=float, default=1.0)
    parser.add_argument("--window-min-words", type=int, default=20)
    parser.add_argument(
        "--extra-train-dataset",
        action="append",
        default=[],
        help="Extra supervised/robustness dataset spec. Repeat to add more sources.",
    )
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-row-teacher", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_datasets = args.source_dataset or ["edinburghcstr/ami:sdm:train[:8000]:text"]
    streaming_row_source_datasets = args.streaming_row_source_dataset
    extra_train_datasets = args.extra_train_dataset or [
        "parquet:/data/model/bddy-real-mix-asr/libritts_conversation_mix_v2.parquet:train:text"
    ]
    run_dir = Path(args.work_dir) / args.run_name
    teacher_path = run_dir / "teacher_windows.parquet"
    row_teacher_path = run_dir / "teacher_rows_streaming.parquet"
    adapter_dir = run_dir / "adapter"
    merged_dir = run_dir / "merged"
    reports_dir = run_dir / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_name": args.run_name,
        "teacher_model": args.teacher_model,
        "base_model": args.base_model,
        "source_datasets": source_datasets,
        "streaming_row_source_datasets": streaming_row_source_datasets,
        "eval_dataset": args.eval_dataset,
        "extra_train_datasets": extra_train_datasets,
        "limit_windows": args.limit_windows,
        "limit_streaming_rows": args.limit_streaming_rows,
        "train_steps": args.train_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_gain_db_min": args.train_gain_db_min,
        "train_gain_db_max": args.train_gain_db_max,
        "train_noise_prob": args.train_noise_prob,
        "train_noise_snr_db_min": args.train_noise_snr_db_min,
        "train_noise_snr_db_max": args.train_noise_snr_db_max,
        "train_overlap_prob": args.train_overlap_prob,
        "train_overlap_gain_db_min": args.train_overlap_gain_db_min,
        "train_overlap_gain_db_max": args.train_overlap_gain_db_max,
        "window_seconds": args.window_seconds,
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    teacher_command = [
            sys.executable,
            "scripts/build_whisper_teacher_dataset.py",
            "--teacher-model",
            args.teacher_model,
            "--output",
            str(teacher_path),
            "--limit-per-dataset",
            str(args.limit_windows),
            "--conversation-window-seconds",
            str(args.window_seconds),
            "--conversation-window-gap-seconds",
            str(args.window_gap_seconds),
            "--conversation-window-min-words",
            str(args.window_min_words),
            "--min-words",
            "8",
            "--min-duration-seconds",
            "5",
            "--max-duration-seconds",
            "24",
            "--max-new-tokens",
            "192",
        ]
    for dataset_spec in source_datasets:
        teacher_command.extend(["--dataset", dataset_spec])
    run(
        teacher_command,
        skip=args.skip_teacher and teacher_path.exists(),
    )

    if streaming_row_source_datasets:
        row_teacher_command = [
            sys.executable,
            "scripts/build_whisper_teacher_dataset.py",
            "--streaming",
            "--teacher-model",
            args.teacher_model,
            "--output",
            str(row_teacher_path),
            "--limit-per-dataset",
            str(args.limit_streaming_rows),
            "--conversation-window-seconds",
            "0",
            "--min-words",
            "8",
            "--min-duration-seconds",
            "5",
            "--max-duration-seconds",
            "24",
            "--max-new-tokens",
            "192",
        ]
        for dataset_spec in streaming_row_source_datasets:
            row_teacher_command.extend(["--dataset", dataset_spec])
        run(
            row_teacher_command,
            skip=args.skip_row_teacher and row_teacher_path.exists(),
        )

    train_command = [
        sys.executable,
        "scripts/train_whisper_asr_lora.py",
        "--model",
        args.base_model,
        "--dataset",
        f"parquet:{teacher_path}:train:teacher_text",
    ]
    if streaming_row_source_datasets and parquet_num_rows(row_teacher_path) > 0:
        train_command.extend(["--dataset", f"parquet:{row_teacher_path}:train:teacher_text"])
    for dataset_spec in extra_train_datasets:
        train_command.extend(["--dataset", dataset_spec])
    train_command.extend(
        [
            "--eval-dataset",
            args.eval_dataset,
            "--limit-per-dataset",
            str(args.limit_windows),
            "--eval-limit-per-dataset",
            "80",
            "--eval-samples",
            "24",
            "--min-words",
            "8",
            "--min-duration-seconds",
            "5",
            "--max-duration-seconds",
            "24",
            "--max-steps",
            str(args.train_steps),
            "--batch-size",
            str(args.batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--lora-target-modules",
            args.lora_target_modules,
            "--train-gain-db-min",
            str(args.train_gain_db_min),
            "--train-gain-db-max",
            str(args.train_gain_db_max),
            "--train-noise-prob",
            str(args.train_noise_prob),
            "--train-noise-snr-db-min",
            str(args.train_noise_snr_db_min),
            "--train-noise-snr-db-max",
            str(args.train_noise_snr_db_max),
            "--train-overlap-prob",
            str(args.train_overlap_prob),
            "--train-overlap-gain-db-min",
            str(args.train_overlap_gain_db_min),
            "--train-overlap-gain-db-max",
            str(args.train_overlap_gain_db_max),
            "--max-label-length",
            "256",
            "--max-new-tokens",
            "192",
            "--output-dir",
            str(adapter_dir),
        ]
    )
    run(train_command, skip=args.skip_train)

    if not args.skip_train:
        merge_lora(base_model=args.base_model, adapter_dir=adapter_dir, merged_dir=merged_dir)

    if not args.skip_eval:
        short_eval_command = [
            sys.executable,
            "scripts/eval_asr_quality.py",
            "--model",
            args.base_model,
            "--model",
            str(merged_dir),
            "--text-column",
            "text",
            "--limit",
            str(args.eval_limit),
            "--max-new-tokens",
            "128",
            "--output-json",
            str(reports_dir / "short_ami_eval.json"),
        ]
        if args.short_eval_parquet:
            short_eval_command.extend(["--dataset", "parquet", "--config", args.short_eval_parquet, "--split", "train"])
        else:
            short_eval_command.extend(
                [
                    "--dataset",
                    "edinburghcstr/ami",
                    "--config",
                    "sdm",
                    "--split",
                    "validation[:1500]",
                    "--min-words",
                    "4",
                    "--min-duration-seconds",
                    "1.0",
                    "--max-duration-seconds",
                    "24",
                ]
            )
        run(short_eval_command)

        window_eval_command = [
            sys.executable,
            "scripts/eval_asr_quality.py",
            "--model",
            args.base_model,
            "--model",
            str(merged_dir),
            "--text-column",
            "text",
            "--limit",
            str(args.window_eval_limit),
            "--max-new-tokens",
            "192",
            "--output-json",
            str(reports_dir / "window_ami_eval.json"),
        ]
        if args.window_eval_parquet:
            window_eval_command.extend(["--dataset", "parquet", "--config", args.window_eval_parquet, "--split", "train"])
        else:
            window_eval_command.extend(
                [
                    "--dataset",
                    "edinburghcstr/ami",
                    "--config",
                    "sdm",
                    "--split",
                    "validation[:1200]",
                    "--min-words",
                    "1",
                    "--min-duration-seconds",
                    "0.5",
                    "--max-duration-seconds",
                    "12",
                    "--conversation-window-seconds",
                    str(args.window_seconds),
                ]
            )
        run(window_eval_command)

    print(json.dumps({"run_dir": str(run_dir), "teacher_path": str(teacher_path), "adapter_dir": str(adapter_dir), "merged_dir": str(merged_dir)}, indent=2))


if __name__ == "__main__":
    main()
