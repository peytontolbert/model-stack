#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402

from distill_f5tts_12_to_4_q4 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_VOCAB,
    audio_array_to_item,
    build_loss_mask,
    build_model,
    cfg_flow_with_delta,
    load_checkpoint_state,
    make_time_grid,
    row_to_item,
    stream_hf_rows,
    text_to_ids,
)
from tts_text_normalizer import normalize_f5tts_speech_text  # noqa: E402


SPACED_LETTER_RE = re.compile(r"(?:\b[A-Z]\b\s+){4,}\b[A-Z]\b")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def bump_reject(args: argparse.Namespace, reason: str) -> None:
    counts = getattr(args, "_reject_counts", None)
    if isinstance(counts, dict):
        counts[reason] = int(counts.get(reason, 0)) + 1


def text_is_usable(text: Any, args: argparse.Namespace) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    text_bytes = len(value.encode("utf-8"))
    if text_bytes < int(args.min_text_bytes) or text_bytes > int(args.max_text_bytes):
        return False
    if bool(args.reject_spaced_letters) and SPACED_LETTER_RE.search(value):
        return False
    return True


def row_group_key(row: dict[str, Any], args: argparse.Namespace) -> str:
    speaker = row.get(args.speaker_column)
    chapter = row.get(args.chapter_column)
    if bool(args.same_chapter_pair) and speaker is not None and chapter is not None:
        return f"{speaker}:{chapter}"
    if speaker is not None:
        return str(speaker)
    return "__unknown__"


def row_to_voiceclone_item(row: dict[str, Any], *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    text = row.get(args.text_column)
    if not text_is_usable(text, args):
        return None
    item = row_to_item(row, args=args, model=model, device=device)
    if item is None:
        return None
    frames = int(item["mel_spec"].shape[-1])
    if frames < int(args.min_ref_frames) or frames > int(args.max_ref_frames):
        return None
    item["speaker_id"] = row.get(args.speaker_column)
    item["chapter_id"] = row.get(args.chapter_column)
    item["utterance_id"] = row.get("id") or row.get("path") or row.get("file")
    item["group_key"] = row_group_key(row, args)
    return item


def load_synthetic_gen_texts(path: str) -> list[str]:
    if not path:
        return []
    texts: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            texts.append(value)
    return texts


def nearest_state_index(times: list[float], target: float) -> int:
    return min(range(len(times)), key=lambda idx: abs(float(times[idx]) - float(target)))


def maybe_profile_transform_ref_mel(ref_mel: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if float(getattr(args, "ref_mel_bias", 0.0)) != 0.0:
        ref_mel = ref_mel + float(args.ref_mel_bias)
        metadata["ref_mel_bias"] = float(args.ref_mel_bias)
    if float(getattr(args, "ref_mel_gain", 1.0)) != 1.0:
        ref_mel = ref_mel * float(args.ref_mel_gain)
        metadata["ref_mel_gain"] = float(args.ref_mel_gain)
    if float(getattr(args, "ref_mel_high_band_gain", 1.0)) != 1.0:
        start_bin = int(getattr(args, "ref_mel_high_band_start_bin", 46))
        if 0 <= start_bin < ref_mel.shape[-1]:
            ref_mel = ref_mel.clone()
            ref_mel[..., start_bin:] = ref_mel[..., start_bin:] * float(args.ref_mel_high_band_gain)
            metadata["ref_mel_high_band_gain"] = float(args.ref_mel_high_band_gain)
            metadata["ref_mel_high_band_start_bin"] = int(start_bin)
    smooth_kernel = int(getattr(args, "ref_mel_smooth_kernel", 1))
    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        import torch.nn.functional as F

        original_dtype = ref_mel.dtype
        x = ref_mel.transpose(1, 2).float()
        x = F.avg_pool1d(x, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel // 2)
        ref_mel = x.transpose(1, 2).to(original_dtype)
        metadata["ref_mel_smooth_kernel"] = int(smooth_kernel)
    if float(getattr(args, "ref_mel_noise_std", 0.0)) > 0.0:
        ref_mel = ref_mel + torch.randn_like(ref_mel) * float(args.ref_mel_noise_std)
        metadata["ref_mel_noise_std"] = float(args.ref_mel_noise_std)
    if float(getattr(args, "ref_mel_high_band_noise_std", 0.0)) > 0.0:
        start_bin = int(getattr(args, "ref_mel_high_band_start_bin", 46))
        if 0 <= start_bin < ref_mel.shape[-1]:
            ref_mel = ref_mel.clone()
            ref_mel[..., start_bin:] = ref_mel[..., start_bin:] + torch.randn_like(ref_mel[..., start_bin:]) * float(args.ref_mel_high_band_noise_std)
            metadata["ref_mel_high_band_noise_std"] = float(args.ref_mel_high_band_noise_std)
            metadata["ref_mel_high_band_start_bin"] = int(start_bin)
    return ref_mel, metadata


def cache_pair(
    *,
    ref_item: dict[str, Any],
    gen_item: dict[str, Any],
    teacher,
    device: torch.device,
    args: argparse.Namespace,
    sample_index: int,
    seed_offset: int = 0,
) -> dict[str, Any] | None:
    ref_mel = ref_item["mel_spec"].transpose(0, 1).unsqueeze(0).to(device)
    if int(args.cond_frames) > 0 and ref_mel.shape[1] > int(args.cond_frames):
        ref_mel = ref_mel[:, : int(args.cond_frames), :]
    ref_mel, ref_profile_transform = maybe_profile_transform_ref_mel(ref_mel, args)
    ref_frames = int(ref_mel.shape[1])
    ref_text = str(ref_item["text"]).strip()
    gen_text = str(gen_item["text"]).strip()
    original_gen_text = gen_text
    if bool(getattr(args, "normalize_f5tts_text", False)):
        gen_text = normalize_f5tts_speech_text(gen_text)
    if not ref_text or not gen_text:
        bump_reject(args, "empty_ref_or_gen_text")
        return None
    local_speed = float(args.speed)
    if len(gen_text.encode("utf-8")) < 10:
        local_speed = min(local_speed, 0.3)
    ref_text_len = max(1, len(ref_text.encode("utf-8")))
    gen_text_len = max(1, len(gen_text.encode("utf-8")))
    estimated_duration = ref_frames + int(ref_frames / ref_text_len * gen_text_len / local_speed)
    duration = max(estimated_duration, ref_frames + int(args.min_gen_frames))
    if duration > int(args.max_frames):
        bump_reject(args, "estimated_duration_over_max_frames")
        return None
    if duration <= ref_frames:
        bump_reject(args, "duration_not_longer_than_reference")
        return None
    generated_frames = int(duration - ref_frames)
    if generated_frames < int(args.min_generated_frames) or generated_frames > int(args.max_generated_frames):
        bump_reject(args, "generated_frames_out_of_range")
        return None

    text_list = [ref_text + " " + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)
    text_ids = text_to_ids(teacher, final_text_list, device)
    sample_seed = int(args.seed) + int(seed_offset)
    torch.manual_seed(sample_seed)
    with torch.no_grad():
        sampled, trajectory = teacher.sample(
            cond=ref_mel,
            text=final_text_list,
            duration=int(duration),
            steps=int(args.teacher_steps),
            cfg_strength=float(args.teacher_cfg_strength),
            sway_sampling_coef=float(args.sway_sampling_coef),
            seed=sample_seed,
        )
    cond = torch.zeros_like(sampled)
    cond[:, :ref_frames, :] = ref_mel[:, :ref_frames, :]
    lens = torch.full((1,), int(duration), device=device, dtype=torch.long)
    loss_mask = build_loss_mask(lens, ref_frames, int(duration), device)
    if not bool(loss_mask.any()):
        bump_reject(args, "empty_loss_mask")
        return None

    teacher_states = [trajectory[index].detach().cpu().to(torch.float16) for index in range(trajectory.shape[0])]
    teacher_times = [float(value) for value in make_time_grid(int(args.teacher_steps), float(args.sway_sampling_coef), device, torch.float32).cpu()]
    teacher_grid_flows: list[torch.Tensor] = []
    teacher_grid_cond_deltas: list[torch.Tensor] = []
    student_times: list[float] = []
    if bool(args.cache_student_grid):
        student_time_tensor = make_time_grid(int(args.student_steps), float(args.sway_sampling_coef), device, torch.float32)
        student_times = [float(value) for value in student_time_tensor.detach().cpu()]
        for interval in range(int(args.student_steps)):
            t0 = student_times[interval]
            idx0 = nearest_state_index(teacher_times, t0)
            y0 = trajectory[idx0].detach()
            time_tensor = torch.full((y0.shape[0],), float(t0), device=device, dtype=y0.dtype)
            flow, cond_delta = cfg_flow_with_delta(
                teacher,
                y0,
                cond,
                text_ids,
                time_tensor,
                float(args.student_cfg_strength),
                detach_null_grad=True,
            )
            teacher_grid_flows.append(flow.detach().cpu().to(torch.float16))
            teacher_grid_cond_deltas.append(cond_delta.detach().cpu().to(torch.float16))
    sample_path = Path(args.output_dir) / "samples" / f"sample_{sample_index:06d}.pt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_index": int(sample_index),
        "seed": int(sample_seed),
        "mode": "voiceclone_teacher_sample",
        "text": text_list,
        "ref_text": ref_text,
        "gen_text": gen_text,
        "original_gen_text": original_gen_text,
        "normalized_f5tts_text": bool(gen_text != original_gen_text),
        "ref_profile_transform": ref_profile_transform,
        "cond": cond.detach().cpu().to(torch.float16),
        "noise": trajectory[0].detach().cpu().to(torch.float16),
        "text_ids": text_ids.detach().cpu().to(torch.int32),
        "lens": lens.detach().cpu().to(torch.int32),
        "cond_frames": int(ref_frames),
        "loss_mask": loss_mask.detach().cpu(),
        "teacher_y": sampled.detach().cpu().to(torch.float16),
        "teacher_states": teacher_states,
        "teacher_times": teacher_times,
        "student_times": student_times,
        "teacher_grid_flows": teacher_grid_flows,
        "teacher_grid_cond_deltas": teacher_grid_cond_deltas,
    }
    tmp_path = sample_path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(sample_path)
    return {
        "sample_index": int(sample_index),
        "seed": int(sample_seed),
        "path": str(sample_path.relative_to(args.output_dir)),
        "frames": int(duration),
        "cond_frames": int(ref_frames),
        "generated_frames": int(loss_mask.sum().item()),
        "speaker_id": ref_item.get("speaker_id"),
        "chapter_id": ref_item.get("chapter_id"),
        "ref_utterance_id": ref_item.get("utterance_id"),
        "gen_utterance_id": gen_item.get("utterance_id"),
        "group_key": ref_item.get("group_key"),
        "ref_text": ref_text,
        "gen_text": gen_text,
        "original_gen_text": original_gen_text,
        "normalized_f5tts_text": bool(gen_text != original_gen_text),
        "ref_profile_transform": ref_profile_transform,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache F5TTS FP32 teacher trajectories in the same reference-audio voice-cloning setup used at inference.")
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
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--cond-frames", type=int, default=256)
    parser.add_argument("--min-gen-frames", type=int, default=48)
    parser.add_argument("--min-generated-frames", type=int, default=48)
    parser.add_argument("--max-generated-frames", type=int, default=1600)
    parser.add_argument("--min-ref-frames", type=int, default=160)
    parser.add_argument("--max-ref-frames", type=int, default=1600)
    parser.add_argument("--min-text-bytes", type=int, default=8)
    parser.add_argument("--max-text-bytes", type=int, default=260)
    parser.add_argument("--reject-spaced-letters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--chapter-column", default="chapter_id")
    parser.add_argument("--same-chapter-pair", action="store_true")
    parser.add_argument("--max-pair-buffer-per-group", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--seed-repeats",
        type=int,
        default=1,
        help="Cache multiple teacher trajectories per reference/text pair with different noise seeds.",
    )
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--teacher-cfg-strength", type=float, default=2.0)
    parser.add_argument("--student-steps", type=int, default=2)
    parser.add_argument("--student-cfg-strength", type=float, default=2.0)
    parser.add_argument("--cache-student-grid", action="store_true")
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--ode-method", default="", help="Optional torchdiffeq ODE method override, e.g. euler or midpoint.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fixed-ref-audio", default="", help="Optional fixed reference audio for diagnostic/profile cache builds.")
    parser.add_argument("--fixed-ref-text", default="", help="Text for --fixed-ref-audio.")
    parser.add_argument("--fixed-gen-text", action="append", default=[], help="Generation text for fixed-reference diagnostic cache builds.")
    parser.add_argument("--ref-mel-bias", type=float, default=0.0, help="Additive mel transform for profile-stress teacher caches.")
    parser.add_argument("--ref-mel-gain", type=float, default=1.0, help="Multiplicative mel transform for profile-stress teacher caches.")
    parser.add_argument("--ref-mel-high-band-gain", type=float, default=1.0, help="Scale conditioning mel bins at/above --ref-mel-high-band-start-bin.")
    parser.add_argument("--ref-mel-high-band-start-bin", type=int, default=46)
    parser.add_argument("--ref-mel-smooth-kernel", type=int, default=1, help="Odd temporal smoothing kernel for conditioning mels.")
    parser.add_argument("--ref-mel-noise-std", type=float, default=0.0)
    parser.add_argument("--ref-mel-high-band-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--normalize-f5tts-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize F5TTS/WebGPU/WASM/int4-style tokens before teacher trajectory generation.",
    )
    parser.add_argument(
        "--synthetic-gen-text-file",
        default="",
        help="Optional text file. In dataset mode, pair same-speaker references with these generated texts instead of another dataset utterance.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    args._reject_counts = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.unlink(missing_ok=True)
    (output_dir / "cache_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    device = torch.device(args.device)
    teacher = build_model(Path(args.vocab), device)
    load_checkpoint_state(teacher, Path(args.checkpoint))
    if str(args.ode_method).strip():
        teacher.odeint_kwargs = {**getattr(teacher, "odeint_kwargs", {}), "method": str(args.ode_method).strip()}
    teacher.eval().requires_grad_(False)

    saved = 0
    scanned = 0
    started = time.time()
    if args.fixed_ref_audio and args.fixed_ref_text:
        resumebot_root = Path("/data/resumebot")
        if str(resumebot_root) not in sys.path:
            sys.path.insert(0, str(resumebot_root))
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        ref_audio_path, ref_text = preprocess_ref_audio_text(args.fixed_ref_audio, args.fixed_ref_text)
        audio, sample_rate = sf.read(ref_audio_path, always_2d=False, dtype="float32")
        audio_tensor = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
        rms = torch.sqrt(torch.mean(torch.square(audio_tensor)))
        target_rms = 0.1
        if float(rms) > 0.0 and float(rms) < target_rms:
            audio = (audio_tensor * (target_rms / rms)).cpu().numpy()
        ref_item = audio_array_to_item(
            audio,
            str(ref_text),
            int(sample_rate),
            args=args,
            model=teacher,
            device=device,
        )
        if ref_item is None:
            raise RuntimeError("--fixed-ref-audio did not produce a valid reference item")

        def fixed_gen_text_rows():
            if args.fixed_gen_text:
                for text in args.fixed_gen_text:
                    yield {"text": str(text)}
                return
            for row in stream_hf_rows(args):
                text = row.get(args.text_column)
                if text is not None:
                    yield {"text": str(text)}

        for gen_text_index, gen_row in enumerate(fixed_gen_text_rows()):
            gen_text = str(gen_row["text"]).strip()
            if not gen_text:
                continue
            for repeat_index in range(max(1, int(args.seed_repeats))):
                cache_row = cache_pair(
                    ref_item=ref_item,
                    gen_item={"text": gen_text},
                    teacher=teacher,
                    device=device,
                    args=args,
                    sample_index=saved,
                    seed_offset=gen_text_index * max(1, int(args.seed_repeats)) + repeat_index,
                )
                if cache_row is None:
                    continue
                append_jsonl(metadata_path, cache_row)
                saved += 1
                print(json.dumps({"event": "cache", "saved": saved, "scanned": scanned, **cache_row}), flush=True)
                if saved >= int(args.max_samples):
                    break
            if saved >= int(args.max_samples):
                break
        summary = {"event": "done", "saved": saved, "scanned": scanned, "seconds": round(time.time() - started, 3)}
        if getattr(args, "_reject_counts", None):
            summary["reject_counts"] = dict(sorted(args._reject_counts.items()))
        append_jsonl(metadata_path, summary)
        print(json.dumps(summary), flush=True)
        os._exit(0)

    synthetic_gen_texts = load_synthetic_gen_texts(args.synthetic_gen_text_file)
    group_items: dict[str, list[dict[str, Any]]] = {}
    for row in stream_hf_rows(args):
        scanned += 1
        item = row_to_voiceclone_item(row, args=args, model=teacher, device=device)
        if item is None:
            bump_reject(args, "invalid_reference_item")
            continue
        group_key = str(item.get("group_key") or "__unknown__")
        candidates = group_items.setdefault(group_key, [])
        if not candidates:
            candidates.append(item)
            bump_reject(args, "first_item_for_group")
            continue
        ref_item = random.choice(candidates)
        if synthetic_gen_texts:
            gen_text = synthetic_gen_texts[(saved + scanned) % len(synthetic_gen_texts)]
            gen_item = {
                "text": gen_text,
                "speaker_id": ref_item.get("speaker_id"),
                "chapter_id": ref_item.get("chapter_id"),
                "utterance_id": f"synthetic:{saved}:{(saved + scanned) % len(synthetic_gen_texts)}",
                "group_key": ref_item.get("group_key"),
            }
        else:
            gen_item = item
        if ref_item.get("utterance_id") == gen_item.get("utterance_id"):
            candidates.append(item)
            bump_reject(args, "same_utterance_pair")
            continue
        for repeat_index in range(max(1, int(args.seed_repeats))):
            cache_row = cache_pair(
                ref_item=ref_item,
                gen_item=gen_item,
                teacher=teacher,
                device=device,
                args=args,
                sample_index=saved,
                seed_offset=saved,
            )
            if cache_row is None:
                continue
            append_jsonl(metadata_path, cache_row)
            saved += 1
            print(json.dumps({"event": "cache", "saved": saved, "scanned": scanned, **cache_row}), flush=True)
            if saved >= int(args.max_samples):
                break
        candidates.append(item)
        if len(candidates) > int(args.max_pair_buffer_per_group):
            del candidates[: len(candidates) - int(args.max_pair_buffer_per_group)]
        if saved >= int(args.max_samples):
            break
    summary = {"event": "done", "saved": saved, "scanned": scanned, "seconds": round(time.time() - started, 3)}
    if getattr(args, "_reject_counts", None):
        summary["reject_counts"] = dict(sorted(args._reject_counts.items()))
    append_jsonl(metadata_path, summary)
    print(json.dumps(summary), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
