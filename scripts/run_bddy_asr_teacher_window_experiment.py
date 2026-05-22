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
        help="Teacher-label source dataset spec. Repeat for diverse corpora.",
    )
    parser.add_argument("--eval-dataset", default="edinburghcstr/ami:sdm:validation[:1500]:text")
    parser.add_argument("--limit-windows", type=int, default=500)
    parser.add_argument("--train-steps", type=int, default=180)
    parser.add_argument("--eval-limit", type=int, default=40)
    parser.add_argument("--window-eval-limit", type=int, default=12)
    parser.add_argument("--window-seconds", type=float, default=18.0)
    parser.add_argument("--window-gap-seconds", type=float, default=1.0)
    parser.add_argument("--window-min-words", type=int, default=20)
    parser.add_argument(
        "--extra-train-dataset",
        action="append",
        default=[],
        help="Extra supervised/robustness dataset spec. Repeat to add more sources.",
    )
    parser.add_argument("--streaming-teacher-sources", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_datasets = args.source_dataset or ["edinburghcstr/ami:sdm:train[:8000]:text"]
    extra_train_datasets = args.extra_train_dataset or [
        "parquet:/data/model/bddy-real-mix-asr/libritts_conversation_mix_v2.parquet:train:text"
    ]
    run_dir = Path(args.work_dir) / args.run_name
    teacher_path = run_dir / "teacher_windows.parquet"
    adapter_dir = run_dir / "adapter"
    merged_dir = run_dir / "merged"
    reports_dir = run_dir / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_name": args.run_name,
        "teacher_model": args.teacher_model,
        "base_model": args.base_model,
        "source_datasets": source_datasets,
        "eval_dataset": args.eval_dataset,
        "extra_train_datasets": extra_train_datasets,
        "limit_windows": args.limit_windows,
        "train_steps": args.train_steps,
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
    if args.streaming_teacher_sources:
        teacher_command.append("--streaming")
    run(
        teacher_command,
        skip=args.skip_teacher and teacher_path.exists(),
    )

    train_command = [
        sys.executable,
        "scripts/train_whisper_asr_lora.py",
        "--model",
        args.base_model,
        "--dataset",
        f"parquet:{teacher_path}:train:teacher_text",
    ]
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
            "2",
            "--gradient-accumulation-steps",
            "8",
            "--learning-rate",
            "4e-6",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
            "--lora-dropout",
            "0.05",
            "--lora-target-modules",
            "q_proj,k_proj,v_proj,out_proj",
            "--train-gain-db-min",
            "-2",
            "--train-gain-db-max",
            "2",
            "--train-noise-prob",
            "0.1",
            "--train-noise-snr-db-min",
            "16",
            "--train-noise-snr-db-max",
            "28",
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
        run(
            [
                sys.executable,
                "scripts/eval_asr_quality.py",
                "--model",
                args.base_model,
                "--model",
                str(merged_dir),
                "--dataset",
                "edinburghcstr/ami",
                "--config",
                "sdm",
                "--split",
                "validation[:1500]",
                "--text-column",
                "text",
                "--limit",
                str(args.eval_limit),
                "--min-words",
                "4",
                "--min-duration-seconds",
                "1.0",
                "--max-duration-seconds",
                "24",
                "--max-new-tokens",
                "128",
                "--output-json",
                str(reports_dir / "short_ami_eval.json"),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/eval_asr_quality.py",
                "--model",
                args.base_model,
                "--model",
                str(merged_dir),
                "--dataset",
                "edinburghcstr/ami",
                "--config",
                "sdm",
                "--split",
                "validation[:1200]",
                "--text-column",
                "text",
                "--limit",
                str(args.window_eval_limit),
                "--min-words",
                "1",
                "--min-duration-seconds",
                "0.5",
                "--max-duration-seconds",
                "12",
                "--conversation-window-seconds",
                str(args.window_seconds),
                "--max-new-tokens",
                "192",
                "--output-json",
                str(reports_dir / "window_ami_eval.json"),
            ]
        )

    print(json.dumps({"run_dir": str(run_dir), "teacher_path": str(teacher_path), "adapter_dir": str(adapter_dir), "merged_dir": str(merged_dir)}, indent=2))


if __name__ == "__main__":
    main()
