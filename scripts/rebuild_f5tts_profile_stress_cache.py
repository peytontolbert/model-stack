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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402

from build_f5tts_teacher12_voiceclone_cache import maybe_profile_transform_ref_mel  # noqa: E402
from distill_f5tts_12_to_4_q4 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_VOCAB,
    build_loss_mask,
    build_model,
    cfg_flow_with_delta,
    load_checkpoint_state,
    make_time_grid,
    text_to_ids,
)


def iter_metadata(cache_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cache_dir in cache_dirs:
        metadata_path = cache_dir / "metadata.jsonl"
        if not metadata_path.exists():
            continue
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "done" or not row.get("path"):
                continue
            row["_cache_dir"] = str(cache_dir)
            rows.append(row)
    return rows


def nearest_state_index(times: list[float], target: float) -> int:
    return min(range(len(times)), key=lambda idx: abs(float(times[idx]) - float(target)))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate FP32 F5TTS teacher cache rows with profile-stressed reference mels.")
    parser.add_argument("--source-cache-dir", action="append", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--student-steps", type=int, default=6)
    parser.add_argument("--teacher-cfg-strength", type=float, default=2.0)
    parser.add_argument("--student-cfg-strength", type=float, default=1.15)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ref-mel-bias", type=float, default=0.0)
    parser.add_argument("--ref-mel-gain", type=float, default=1.0)
    parser.add_argument("--ref-mel-high-band-gain", type=float, default=1.0)
    parser.add_argument("--ref-mel-high-band-start-bin", type=int, default=46)
    parser.add_argument("--ref-mel-smooth-kernel", type=int, default=1)
    parser.add_argument("--ref-mel-noise-std", type=float, default=0.0)
    parser.add_argument("--ref-mel-high-band-noise-std", type=float, default=0.0)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.unlink(missing_ok=True)
    (output_dir / "cache_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = iter_metadata([Path(value) for value in args.source_cache_dir])
    if bool(args.shuffle):
        random.shuffle(rows)
    rows = rows[: int(args.max_samples)]
    if not rows:
        raise RuntimeError("No source cache rows found")

    device = torch.device(args.device)
    teacher = build_model(Path(args.vocab), device)
    load_checkpoint_state(teacher, Path(args.checkpoint))
    teacher.eval().requires_grad_(False)

    started = time.time()
    saved = 0
    teacher_times = [float(value) for value in make_time_grid(int(args.teacher_steps), float(args.sway_sampling_coef), device, torch.float32).cpu()]
    student_time_tensor = make_time_grid(int(args.student_steps), float(args.sway_sampling_coef), device, torch.float32)
    student_times = [float(value) for value in student_time_tensor.detach().cpu()]
    for source_index, row in enumerate(rows):
        source_cache = Path(row["_cache_dir"])
        payload = torch.load(source_cache / str(row["path"]), map_location="cpu")
        cond = payload["cond"].to(device=device, dtype=torch.float32)
        duration = int(payload["lens"].view(-1)[0].item())
        ref_frames = int(payload.get("cond_frames", row.get("cond_frames", 0)))
        if ref_frames <= 0 or duration <= ref_frames:
            continue
        ref_mel = cond[:, :ref_frames, :]
        torch.manual_seed(int(args.seed) + source_index)
        ref_mel, ref_profile_transform = maybe_profile_transform_ref_mel(ref_mel, args)
        ref_text = str(payload.get("ref_text") or row.get("ref_text") or "").strip()
        gen_text = str(payload.get("gen_text") or row.get("gen_text") or "").strip()
        if not ref_text or not gen_text:
            continue
        final_text_list = convert_char_to_pinyin([ref_text + " " + gen_text])
        text_ids = text_to_ids(teacher, final_text_list, device)
        with torch.no_grad():
            sampled, trajectory = teacher.sample(
                cond=ref_mel,
                text=final_text_list,
                duration=duration,
                steps=int(args.teacher_steps),
                cfg_strength=float(args.teacher_cfg_strength),
                sway_sampling_coef=float(args.sway_sampling_coef),
                seed=int(args.seed) + source_index,
            )
        new_cond = torch.zeros_like(sampled)
        new_cond[:, :ref_frames, :] = ref_mel
        lens = torch.full((1,), duration, device=device, dtype=torch.long)
        loss_mask = build_loss_mask(lens, ref_frames, duration, device)
        teacher_grid_flows: list[torch.Tensor] = []
        teacher_grid_cond_deltas: list[torch.Tensor] = []
        for interval, t0 in enumerate(student_times[:-1]):
            idx0 = nearest_state_index(teacher_times, t0)
            y0 = trajectory[idx0].detach()
            time_tensor = torch.full((y0.shape[0],), float(t0), device=device, dtype=y0.dtype)
            flow, cond_delta = cfg_flow_with_delta(
                teacher,
                y0,
                new_cond,
                text_ids,
                time_tensor,
                float(args.student_cfg_strength),
                detach_null_grad=True,
            )
            teacher_grid_flows.append(flow.detach().cpu().to(torch.float16))
            teacher_grid_cond_deltas.append(cond_delta.detach().cpu().to(torch.float16))
        sample_path = output_dir / "samples" / f"sample_{saved:06d}.pt"
        torch.save(
            {
                "sample_index": saved,
                "seed": int(args.seed) + source_index,
                "mode": "voiceclone_teacher_profile_stress",
                "text": [ref_text + " " + gen_text],
                "ref_text": ref_text,
                "gen_text": gen_text,
                "original_gen_text": payload.get("original_gen_text", row.get("original_gen_text", gen_text)),
                "normalized_f5tts_text": bool(payload.get("normalized_f5tts_text", row.get("normalized_f5tts_text", False))),
                "ref_profile_transform": ref_profile_transform,
                "cond": new_cond.detach().cpu().to(torch.float16),
                "noise": trajectory[0].detach().cpu().to(torch.float16),
                "text_ids": text_ids.detach().cpu().to(torch.int32),
                "lens": lens.detach().cpu().to(torch.int32),
                "cond_frames": ref_frames,
                "loss_mask": loss_mask.detach().cpu(),
                "teacher_y": sampled.detach().cpu().to(torch.float16),
                "teacher_states": [trajectory[index].detach().cpu().to(torch.float16) for index in range(trajectory.shape[0])],
                "teacher_times": teacher_times,
                "student_times": student_times,
                "teacher_grid_flows": teacher_grid_flows,
                "teacher_grid_cond_deltas": teacher_grid_cond_deltas,
                "source_cache_dir": str(source_cache),
                "source_path": str(row["path"]),
            },
            sample_path,
        )
        cache_row = {
            "sample_index": saved,
            "seed": int(args.seed) + source_index,
            "path": str(sample_path.relative_to(output_dir)),
            "frames": duration,
            "cond_frames": ref_frames,
            "generated_frames": int(loss_mask.sum().item()),
            "ref_text": ref_text,
            "gen_text": gen_text,
            "ref_profile_transform": ref_profile_transform,
            "source_cache_dir": str(source_cache),
            "source_path": str(row["path"]),
        }
        append_jsonl(metadata_path, cache_row)
        saved += 1
        print(json.dumps({"event": "cache", "saved": saved, "source_index": source_index, **cache_row}), flush=True)
    summary = {"event": "done", "saved": saved, "source_rows": len(rows), "seconds": round(time.time() - started, 3)}
    append_jsonl(metadata_path, summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
