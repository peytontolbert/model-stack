#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distill_f5tts_12_to_4_q4 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_VOCAB,
    build_loss_mask,
    build_model,
    load_checkpoint_state,
    make_batch,
    make_time_grid,
    rollout_sample,
    row_to_item,
    stream_hf_rows,
    text_to_ids,
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def cache_sample(
    *,
    item: dict[str, Any],
    teacher,
    device: torch.device,
    args: argparse.Namespace,
    sample_index: int,
) -> dict[str, Any] | None:
    batch = make_batch([item])
    mel = batch["mel"].permute(0, 2, 1).to(device)
    lens = batch["mel_lengths"].to(device).clamp_max(mel.shape[1])
    text_ids = text_to_ids(teacher, batch["text"], device)
    cond_frames = min(int(args.cond_frames), max(1, int(lens.min().item()) - int(args.min_gen_frames)))
    cond = torch.zeros_like(mel)
    cond[:, :cond_frames, :] = mel[:, :cond_frames, :]
    torch.manual_seed(int(args.seed) + int(sample_index))
    noise = torch.randn_like(mel)
    loss_mask = build_loss_mask(lens, cond_frames, mel.shape[1], device)
    if not bool(loss_mask.any()):
        return None

    with torch.no_grad():
        teacher_y, _, _, states, state_times = rollout_sample(
            teacher,
            noise=noise,
            cond=cond,
            text_ids=text_ids,
            steps=int(args.teacher_steps),
            cfg_strength=float(args.teacher_cfg_strength),
            sway_sampling_coef=float(args.sway_sampling_coef),
            cond_frames=cond_frames,
            return_states=True,
        )

    sample_path = Path(args.output_dir) / "samples" / f"sample_{sample_index:06d}.pt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_index": int(sample_index),
        "text": [str(value) for value in batch["text"]],
        "mel": mel.detach().cpu().to(torch.float16),
        "lens": lens.detach().cpu().to(torch.int32),
        "cond": cond.detach().cpu().to(torch.float16),
        "noise": noise.detach().cpu().to(torch.float16),
        "text_ids": text_ids.detach().cpu().to(torch.int32),
        "cond_frames": int(cond_frames),
        "loss_mask": loss_mask.detach().cpu(),
        "teacher_y": teacher_y.detach().cpu().to(torch.float16),
        "teacher_states": [state.detach().cpu().to(torch.float16) for state in states],
        "teacher_times": [float(value) for value in state_times],
    }
    tmp_path = sample_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(sample_path)
    return {
        "sample_index": int(sample_index),
        "path": str(sample_path.relative_to(args.output_dir)),
        "frames": int(mel.shape[1]),
        "cond_frames": int(cond_frames),
        "generated_frames": int(loss_mask.sum().item()),
        "text": payload["text"][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache FP32 F5TTS teacher trajectories for stable 2-step distillation.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="train.100")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=6.0)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--cond-frames", type=int, default=256)
    parser.add_argument("--min-gen-frames", type=int, default=48)
    parser.add_argument("--shuffle-buffer", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--teacher-cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "metadata.jsonl"
    manifest_path.unlink(missing_ok=True)
    config_path = output_dir / "cache_config.json"

    device = torch.device(args.device)
    teacher = build_model(Path(args.vocab), device)
    load_checkpoint_state(teacher, Path(args.checkpoint))
    teacher.eval().requires_grad_(False)

    teacher_times = make_time_grid(int(args.teacher_steps), float(args.sway_sampling_coef), device, torch.float32)
    config_path.write_text(
        json.dumps(
            {
                "mode": "f5tts_fp32_teacher_trajectory_cache",
                "checkpoint": str(args.checkpoint),
                "vocab": str(args.vocab),
                "dataset": str(args.dataset),
                "config": str(args.config),
                "split": str(args.split),
                "teacher_steps": int(args.teacher_steps),
                "teacher_cfg_strength": float(args.teacher_cfg_strength),
                "sway_sampling_coef": float(args.sway_sampling_coef),
                "teacher_times": [float(value) for value in teacher_times.cpu()],
                "max_frames": int(args.max_frames),
                "cond_frames": int(args.cond_frames),
                "seed": int(args.seed),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = iter(stream_hf_rows(args))
    saved = 0
    scanned = 0
    started = time.time()
    while saved < int(args.max_samples):
        try:
            row = next(rows)
        except StopIteration:
            break
        scanned += 1
        item = row_to_item(row, args=args, model=teacher, device=device)
        if item is None:
            continue
        cache_row = cache_sample(item=item, teacher=teacher, device=device, args=args, sample_index=saved)
        if cache_row is None:
            continue
        append_jsonl(manifest_path, cache_row)
        saved += 1
        print(json.dumps({"event": "cache", "saved": saved, "scanned": scanned, **cache_row}), flush=True)

    summary = {"event": "done", "saved": saved, "scanned": scanned, "seconds": round(time.time() - started, 3)}
    append_jsonl(manifest_path, summary)
    print(json.dumps(summary), flush=True)
    # Some dataset/audio extension stacks can crash during interpreter teardown
    # after all cache files are safely written. Exit directly to keep batch jobs
    # from being marked failed because of shutdown-only native cleanup.
    os._exit(0)


if __name__ == "__main__":
    main()
