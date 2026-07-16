#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distill_f5tts_12_to_4_q4 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_VOCAB,
    audio_array_to_item,
    build_loss_mask,
    build_model,
    cfg_flow_with_delta,
    load_checkpoint_state,
    make_batch,
    make_time_grid,
    rollout_sample,
    text_to_ids,
)


def masked_cosine(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    left_flat = left[mask].float().reshape(1, -1)
    right_flat = right[mask].float().reshape(1, -1)
    return float(F.cosine_similarity(left_flat, right_flat, dim=-1).item())


def masked_norm_ratio(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    left_norm = left[mask].float().norm()
    right_norm = right[mask].float().norm().clamp_min(1e-8)
    return float((left_norm / right_norm).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare F5TTS student flow against FP32 teacher at deployment timesteps.")
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--cache-sample", default="", help="Optional cached teacher trajectory sample .pt to diagnose.")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--ref-audio", default="/data/resumebot/voice_profiles/Peyton/sample_0.wav")
    parser.add_argument(
        "--ref-text",
        default="Hi, I'm recording this sample to create a digital copy of my voice. I want it to sound natural and conversational, just like how I normally speak.",
    )
    parser.add_argument("--gen-text", default="This is Peyton speaking from Agent Kernel Lite. The voice should be clear natural and easy to understand.")
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--student-steps", type=int, default=2)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--cond-frames", type=int, default=256)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--min-gen-frames", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cuda-visible-devices", default="")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)

    teacher = build_model(Path(args.vocab), device)
    student = build_model(Path(args.vocab), device)
    load_checkpoint_state(teacher, Path(args.teacher_checkpoint))
    load_checkpoint_state(student, Path(args.student_checkpoint))
    teacher.eval().requires_grad_(False)
    student.eval().requires_grad_(False)

    if args.cache_sample:
        payload = torch.load(args.cache_sample, map_location="cpu")
        cond = payload["cond"].to(device=device, dtype=torch.float32)
        text_ids = payload["text_ids"].to(device=device)
        loss_mask = payload["loss_mask"].to(device=device).bool()
        teacher_states = [state.to(device=device, dtype=torch.float32) for state in payload["teacher_states"]]
        teacher_times = [float(value) for value in payload["teacher_times"]]
        cond_frames = int(payload["cond_frames"])
        frames = int(cond.shape[1])
    else:
        audio, sample_rate = sf.read(args.ref_audio, always_2d=False, dtype="float32")
        item = audio_array_to_item(audio, args.ref_text + " " + args.gen_text, int(sample_rate), args=args, model=teacher, device=device)
        if item is None:
            raise RuntimeError("reference audio did not produce a valid diagnostic item")
        batch = make_batch([item])
        mel = batch["mel"].permute(0, 2, 1).to(device)
        lens = batch["mel_lengths"].to(device).clamp_max(mel.shape[1])
        text_ids = text_to_ids(teacher, batch["text"], device)
        cond_frames = min(int(args.cond_frames), max(1, int(lens.min().item()) - int(args.min_gen_frames)))
        cond = torch.zeros_like(mel)
        cond[:, :cond_frames, :] = mel[:, :cond_frames, :]
        noise = torch.randn_like(mel)
        loss_mask = build_loss_mask(lens, cond_frames, mel.shape[1], device)
        with torch.no_grad():
            _teacher_y, _, _, teacher_states, teacher_times = rollout_sample(
                teacher,
                noise=noise,
                cond=cond,
                text_ids=text_ids,
                steps=int(args.teacher_steps),
                cfg_strength=float(args.cfg_strength),
                sway_sampling_coef=float(args.sway_sampling_coef),
                cond_frames=cond_frames,
                return_states=True,
            )
        frames = int(mel.shape[1])

    student_times = [float(value) for value in make_time_grid(int(args.student_steps), float(args.sway_sampling_coef), device, torch.float32).cpu()]
    rows = []
    for index in range(int(args.student_steps)):
        t0 = student_times[index]
        t1 = student_times[index + 1]
        dt = float(t1 - t0)
        teacher_idx0 = min(range(len(teacher_times)), key=lambda item_idx: abs(float(teacher_times[item_idx]) - t0))
        teacher_idx1 = min(range(len(teacher_times)), key=lambda item_idx: abs(float(teacher_times[item_idx]) - t1))
        y0 = teacher_states[teacher_idx0].detach()
        y1 = teacher_states[teacher_idx1].detach()
        target_flow = (y1 - y0) / dt
        time_tensor = torch.full((1,), t0, device=device, dtype=y0.dtype)
        with torch.no_grad():
            teacher_flow, teacher_delta = cfg_flow_with_delta(teacher, y0, cond, text_ids, time_tensor, float(args.cfg_strength))
            student_flow, student_delta = cfg_flow_with_delta(student, y0, cond, text_ids, time_tensor, float(args.cfg_strength))
        rows.append(
            {
                "interval": index,
                "t0": t0,
                "t1": t1,
                "teacher_state_index_0": teacher_idx0,
                "teacher_state_index_1": teacher_idx1,
                "student_vs_avg_flow_cosine": masked_cosine(student_flow, target_flow, loss_mask),
                "student_vs_avg_flow_norm_ratio": masked_norm_ratio(student_flow, target_flow, loss_mask),
                "teacher_instant_vs_avg_flow_cosine": masked_cosine(teacher_flow, target_flow, loss_mask),
                "teacher_instant_vs_avg_flow_norm_ratio": masked_norm_ratio(teacher_flow, target_flow, loss_mask),
                "student_vs_teacher_instant_cosine": masked_cosine(student_flow, teacher_flow, loss_mask),
                "student_vs_teacher_instant_norm_ratio": masked_norm_ratio(student_flow, teacher_flow, loss_mask),
                "student_cfg_delta_vs_teacher_cosine": masked_cosine(student_delta, teacher_delta, loss_mask),
                "student_cfg_delta_vs_teacher_norm_ratio": masked_norm_ratio(student_delta, teacher_delta, loss_mask),
            }
        )
    print(
        json.dumps(
            {
                "student_checkpoint": str(args.student_checkpoint),
                "teacher_checkpoint": str(args.teacher_checkpoint),
                "cond_frames": int(cond_frames),
                "frames": frames,
                "generated_frames": int(loss_mask.sum().item()),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
