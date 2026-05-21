#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

# CUDA must be masked before importing torch. Override from the shell with
# CUDA_VISIBLE_DEVICES=... when a different training GPU is intended.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


DEFAULT_CHECKPOINT = "/data/resumebot/checkpoints/final_finetuned_model.pt"
DEFAULT_VOCAB = "/data/resumebot/checkpoints/F5TTS_Base_vocab.txt"
DEFAULT_OUTPUT = "/data/transformer_10/checkpoints/f5tts_q4_12to4_distill"


class RowwiseQ4STE(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        original_dtype = weight.dtype
        flat = weight.float().reshape(weight.shape[0], -1)
        scale = flat.detach().abs().amax(dim=1).clamp_min(1e-8) / 7.0
        quantized = torch.round(flat / scale[:, None]).clamp(-8, 7)
        dequantized = (quantized * scale[:, None]).reshape_as(weight).to(original_dtype)
        return weight + (dequantized - weight).detach()


def rowwise_q4_dequantize(weight: torch.Tensor) -> torch.Tensor:
    original_dtype = weight.dtype
    flat = weight.detach().float().reshape(weight.shape[0], -1)
    scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 7.0
    quantized = torch.round(flat / scale[:, None]).clamp(-8, 7)
    return (quantized * scale[:, None]).reshape_as(weight).to(original_dtype)


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def load_vocab(path: Path) -> tuple[dict[str, int], int]:
    vocab = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    vocab_char_map = {char: idx for idx, char in enumerate(vocab) if char}
    return vocab_char_map, len(vocab_char_map) + 1


def build_model(vocab_path: Path, device: torch.device):
    from f5_tts.model import CFM, DiT

    vocab_char_map, vocab_size = load_vocab(vocab_path)
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    return model.to(device)


def load_checkpoint_state(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"missing": missing, "unexpected": unexpected}, indent=2))
    del checkpoint


def should_q4_module(name: str, module: nn.Module, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return False
    if include and not any(item in name for item in include):
        return False
    if exclude and any(item in name for item in exclude):
        return False
    return hasattr(module, "weight") and module.weight.ndim >= 2


def apply_q4_parametrizations(
    model: nn.Module,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    ste_include: tuple[str, ...] = (),
) -> tuple[int, int]:
    modules = 0
    params = 0
    for name, module in model.named_modules():
        if should_q4_module(name, module, include, exclude):
            if not ste_include or any(item in name for item in ste_include):
                parametrize.register_parametrization(module, "weight", RowwiseQ4STE())
            else:
                with torch.no_grad():
                    module.weight.copy_(rowwise_q4_dequantize(module.weight))
            modules += 1
            params += int(module.weight.numel())
    return modules, params


def configure_trainable_parameters(
    model: nn.Module,
    *,
    train_include: tuple[str, ...],
    train_exclude: tuple[str, ...],
) -> tuple[int, int]:
    tensors = 0
    params = 0
    for name, parameter in model.named_parameters():
        trainable = True
        if train_include:
            trainable = any(item in name for item in train_include)
        if train_exclude and any(item in name for item in train_exclude):
            trainable = False
        parameter.requires_grad_(trainable)
        if trainable:
            tensors += 1
            params += int(parameter.numel())
    return tensors, params


def trainable_anchor_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def anchor_weight_loss(model: nn.Module, anchors: dict[str, torch.Tensor]) -> torch.Tensor:
    loss: torch.Tensor | None = None
    terms = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in anchors:
            continue
        term = F.mse_loss(parameter, anchors[name].to(device=parameter.device, dtype=parameter.dtype))
        loss = term if loss is None else loss + term
        terms += 1
    if loss is None:
        return next(model.parameters()).new_zeros(())
    return loss / float(max(1, terms))


def materialized_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if ".parametrizations.weight." not in name:
            state[name] = tensor.detach().cpu()
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            key = f"{name}.weight" if name else "weight"
            state[key] = module.weight.detach().cpu()
    return state


def save_checkpoint(model: nn.Module, path: Path, *, step: int, args: argparse.Namespace) -> None:
    payload = {
        "model_state_dict": materialized_state_dict(model),
        "step": int(step),
        "args": vars(args),
        "distillation": {
            "teacher_steps": int(args.teacher_steps),
            "student_steps": int(args.student_steps),
            "cfg_strength": float(args.cfg_strength),
            "teacher_cfg_strength": float(args.teacher_cfg_strength),
            "student_cfg_strength": float(args.student_cfg_strength),
            "mode": "f5tts_q4_12to4_rollout_distill",
        },
    }
    torch.save(payload, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def scheduled_weight(base: float, *, step: int, max_steps: int, warmup_steps: int = 0, final: float | None = None) -> float:
    if final is None or int(warmup_steps) <= 0:
        return float(base)
    progress = min(1.0, max(0.0, float(step) / float(max(1, min(int(warmup_steps), int(max_steps))))))
    return float(base) + (float(final) - float(base)) * progress


def stream_hf_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {"split": args.split, "streaming": True}
    dataset = load_dataset(args.dataset, args.config or None, **kwargs)
    if args.audio_column:
        dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sample_rate))
    if int(args.shuffle_buffer) > 0:
        dataset = dataset.shuffle(buffer_size=int(args.shuffle_buffer), seed=int(args.seed))
    return dataset


def audio_array_to_item(array: Any, text: str, sample_rate: int, *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    if array is None:
        return None
    audio = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)
    duration = float(audio.numel()) / float(sample_rate)
    if duration < args.min_duration or duration > args.max_duration:
        return None
    if sample_rate != args.sample_rate:
        audio = F.interpolate(
            audio.view(1, 1, -1),
            size=int(round(audio.numel() * args.sample_rate / sample_rate)),
            mode="linear",
            align_corners=False,
        ).view(-1)
    with torch.no_grad():
        mel = model.mel_spec(audio.to(device).view(1, -1)).detach().cpu().squeeze(0)
    if mel.shape[-1] < int(args.cond_frames) + int(args.min_gen_frames):
        return None
    if mel.shape[-1] > int(args.max_frames):
        mel = mel[:, : int(args.max_frames)].contiguous()
    return {"mel_spec": mel, "text": str(text)}


def row_to_item(row: dict[str, Any], *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    audio_obj = row.get(args.audio_column)
    text = row.get(args.text_column)
    if not audio_obj or text is None:
        return None
    return audio_array_to_item(
        audio_obj.get("array"),
        str(text),
        int(audio_obj.get("sampling_rate") or args.sample_rate),
        args=args,
        model=model,
        device=device,
    )


def load_local_samples(samples_path: str) -> list[tuple[Path, str]]:
    path = Path(samples_path)
    if not path.exists():
        return []
    rows: list[tuple[Path, str]] = []
    base_dir = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "|" not in line:
            continue
        audio_ref, text = line.split("|", 1)
        audio_path = Path(audio_ref)
        if not audio_path.exists():
            basename = PureWindowsPath(audio_ref).name if "\\" in audio_ref else audio_path.name
            audio_path = base_dir / basename
        if audio_path.exists():
            rows.append((audio_path, text.strip()))
    return rows


def local_row_to_item(row: tuple[Path, str], *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    audio_path, text = row
    array, sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
    return audio_array_to_item(array, text, int(sample_rate), args=args, model=model, device=device)


def make_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    from f5_tts.model.dataset import collate_fn

    return collate_fn(items)


def text_to_ids(model, text, device: torch.device) -> torch.Tensor:
    from f5_tts.model.utils import list_str_to_idx, list_str_to_tensor

    if isinstance(text, torch.Tensor):
        return text.to(device)
    if isinstance(text, list):
        if getattr(model, "vocab_char_map", None):
            return list_str_to_idx(text, model.vocab_char_map).to(device)
        return list_str_to_tensor(text).to(device)
    raise TypeError(f"unsupported text batch type: {type(text)!r}")


def make_time_grid(steps: int, sway_sampling_coef: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = []
    for index in range(int(steps) + 1):
        t = index / float(steps)
        t = t + float(sway_sampling_coef) * (math.cos((math.pi / 2.0) * t) - 1.0 + t)
        values.append(t)
    return torch.tensor(values, device=device, dtype=dtype)


def cfg_flow(
    model,
    y: torch.Tensor,
    cond: torch.Tensor,
    text_ids: torch.Tensor,
    time: torch.Tensor,
    cfg_strength: float,
    *,
    detach_null_grad: bool = False,
) -> torch.Tensor:
    pred = model.transformer(
        x=y,
        cond=cond,
        text=text_ids,
        time=time,
        drop_audio_cond=False,
        drop_text=False,
    )
    if abs(float(cfg_strength)) < 1e-8:
        return pred
    null_pred = model.transformer(
        x=y,
        cond=cond,
        text=text_ids,
        time=time,
        drop_audio_cond=True,
        drop_text=True,
    )
    if detach_null_grad:
        null_pred = null_pred.detach()
    return pred + (pred - null_pred) * float(cfg_strength)


def rollout_sample(
    model,
    *,
    noise: torch.Tensor,
    cond: torch.Tensor,
    text_ids: torch.Tensor,
    steps: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    cond_frames: int,
    collect_flow_loss: bool = False,
    teacher_model=None,
    teacher_cfg_strength: float | None = None,
    loss_mask: torch.Tensor | None = None,
    detach_null_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = noise.clone()
    times = make_time_grid(steps, sway_sampling_coef, y.device, y.dtype)
    flow_loss = y.new_zeros(())
    flow_terms = 0
    for step in range(int(steps)):
        t = times[step]
        dt = times[step + 1] - times[step]
        time = torch.full((y.shape[0],), float(t.detach().cpu()), device=y.device, dtype=y.dtype)
        flow = cfg_flow(model, y, cond, text_ids, time, cfg_strength, detach_null_grad=detach_null_grad)
        if collect_flow_loss and teacher_model is not None and loss_mask is not None:
            with torch.no_grad():
                teacher_flow = cfg_flow(
                    teacher_model,
                    y.detach(),
                    cond,
                    text_ids,
                    time,
                    float(cfg_strength if teacher_cfg_strength is None else teacher_cfg_strength),
                )
            flow_loss = flow_loss + F.mse_loss(flow[loss_mask], teacher_flow[loss_mask])
            flow_terms += 1
        y = y + dt * flow
    if cond_frames > 0:
        y[:, :cond_frames, :] = cond[:, :cond_frames, :]
    if flow_terms:
        flow_loss = flow_loss / float(flow_terms)
    return y, flow_loss


def build_loss_mask(lens: torch.Tensor, cond_frames: int, frames: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(frames, device=device)[None, :]
    valid = positions < lens.to(device)[:, None]
    generated = positions >= int(cond_frames)
    return valid & generated


def temporal_delta_loss(student_y: torch.Tensor, teacher_y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    first_mask = loss_mask[:, 1:] & loss_mask[:, :-1]
    if not bool(first_mask.any()):
        return student_y.new_zeros(())
    student_delta = student_y[:, 1:, :] - student_y[:, :-1, :]
    teacher_delta = teacher_y[:, 1:, :] - teacher_y[:, :-1, :]
    loss = F.mse_loss(student_delta[first_mask], teacher_delta[first_mask])

    second_mask = first_mask[:, 1:] & first_mask[:, :-1]
    if bool(second_mask.any()):
        student_accel = student_delta[:, 1:, :] - student_delta[:, :-1, :]
        teacher_accel = teacher_delta[:, 1:, :] - teacher_delta[:, :-1, :]
        loss = loss + 0.5 * F.mse_loss(student_accel[second_mask], teacher_accel[second_mask])
    return loss


def mel_energy_loss(student_y: torch.Tensor, teacher_y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if not bool(loss_mask.any()):
        return student_y.new_zeros(())
    student_energy = student_y.float().pow(2).mean(dim=-1)
    teacher_energy = teacher_y.float().pow(2).mean(dim=-1)
    return F.mse_loss(student_energy[loss_mask], teacher_energy[loss_mask])


def high_mel_excess_loss(student_y: torch.Tensor, teacher_y: torch.Tensor, loss_mask: torch.Tensor, start_bin: int = 80) -> torch.Tensor:
    if not bool(loss_mask.any()):
        return student_y.new_zeros(())
    start = max(0, min(int(start_bin), int(student_y.shape[-1]) - 1))
    student_band = student_y[..., start:].float().pow(2).mean(dim=-1)
    teacher_band = teacher_y[..., start:].float().pow(2).mean(dim=-1)
    excess = F.relu(student_band - teacher_band)
    return (excess[loss_mask]).pow(2).mean()


def high_mel_match_loss(student_y: torch.Tensor, teacher_y: torch.Tensor, loss_mask: torch.Tensor, start_bin: int = 80) -> torch.Tensor:
    if not bool(loss_mask.any()):
        return student_y.new_zeros(())
    start = max(0, min(int(start_bin), int(student_y.shape[-1]) - 1))
    student_band = student_y[..., start:].float()
    teacher_band = teacher_y[..., start:].float()
    return F.smooth_l1_loss(student_band[loss_mask], teacher_band[loss_mask], beta=0.25)


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = build_model(Path(args.vocab), device)
    student = build_model(Path(args.vocab), device)
    load_checkpoint_state(teacher, Path(args.checkpoint))
    load_checkpoint_state(student, Path(args.student_checkpoint or args.checkpoint))
    teacher.eval().requires_grad_(False)

    q4_modules, q4_params = apply_q4_parametrizations(
        student,
        include=split_csv(args.q4_include),
        exclude=split_csv(args.q4_exclude),
        ste_include=split_csv(args.q4_ste_include),
    )
    train_tensors, train_params = configure_trainable_parameters(
        student,
        train_include=split_csv(args.train_include),
        train_exclude=split_csv(args.train_exclude),
    )
    student.train()
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters selected")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    anchor_parameters = trainable_anchor_parameters(student) if float(args.anchor_weight_loss_weight) > 0.0 else {}

    local_rows = load_local_samples(args.local_samples)
    use_hf_stream = bool(args.dataset) and float(args.local_sample_prob) < 1.0
    row_iter = iter(stream_hf_rows(args)) if use_hf_stream else iter(())
    setup_row = {
        "device": str(device),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "use_hf_stream": use_hf_stream,
        "q4_modules": q4_modules,
        "q4_params": q4_params,
        "train_tensors": train_tensors,
        "train_params": train_params,
        "teacher_steps": args.teacher_steps,
        "student_steps": args.student_steps,
        "teacher_checkpoint": args.checkpoint,
        "student_checkpoint": args.student_checkpoint or args.checkpoint,
        "cfg_strength": args.cfg_strength,
        "teacher_cfg_strength": args.teacher_cfg_strength,
        "student_cfg_strength": args.student_cfg_strength,
        "anchor_weight_loss_weight": args.anchor_weight_loss_weight,
        "temporal_delta_loss_weight": args.temporal_delta_loss_weight,
        "mel_energy_loss_weight": args.mel_energy_loss_weight,
        "high_mel_excess_loss_weight": args.high_mel_excess_loss_weight,
        "high_mel_match_loss_weight": args.high_mel_match_loss_weight,
        "high_mel_start_bin": args.high_mel_start_bin,
        "detach_null_grad": bool(args.detach_null_grad),
        "local_profile_samples_available": len(local_rows),
        "local_sample_prob": float(args.local_sample_prob),
        "local_profile_samples_enabled": bool(local_rows) and float(args.local_sample_prob) > 0.0,
    }
    print(json.dumps(setup_row, indent=2), flush=True)
    metrics_path = output_dir / "metrics.jsonl"
    append_jsonl(metrics_path, {"event": "setup", "time": time.time(), **setup_row})
    if bool(args.dry_run_init):
        return

    pending: list[dict[str, Any]] = []
    best_loss = float("inf")
    step = 0
    while step < int(args.max_steps):
        use_local = bool(local_rows) and random.random() < float(args.local_sample_prob)
        if use_local:
            item = local_row_to_item(random.choice(local_rows), args=args, model=student, device=device)
        else:
            try:
                item = row_to_item(next(row_iter), args=args, model=student, device=device)
            except StopIteration:
                if not local_rows:
                    break
                item = local_row_to_item(random.choice(local_rows), args=args, model=student, device=device)
        if item is None:
            continue
        pending.append(item)
        if len(pending) < int(args.batch_size):
            continue

        batch = make_batch(pending)
        pending = []
        mel = batch["mel"].permute(0, 2, 1).to(device)
        lens = batch["mel_lengths"].to(device).clamp_max(mel.shape[1])
        text_ids = text_to_ids(student, batch["text"], device)
        cond_frames = min(int(args.cond_frames), max(1, int(lens.min().item()) - int(args.min_gen_frames)))
        cond = torch.zeros_like(mel)
        cond[:, :cond_frames, :] = mel[:, :cond_frames, :]
        noise = torch.randn_like(mel)
        loss_mask = build_loss_mask(lens, cond_frames, mel.shape[1], device)
        if not bool(loss_mask.any()):
            continue

        with torch.no_grad():
            teacher_y, _ = rollout_sample(
                teacher,
                noise=noise,
                cond=cond,
                text_ids=text_ids,
                steps=int(args.teacher_steps),
                cfg_strength=float(args.teacher_cfg_strength),
                sway_sampling_coef=float(args.sway_sampling_coef),
                cond_frames=cond_frames,
            )
        student_y, flow_loss = rollout_sample(
            student,
            noise=noise,
            cond=cond,
            text_ids=text_ids,
            steps=int(args.student_steps),
            cfg_strength=float(args.student_cfg_strength),
            sway_sampling_coef=float(args.sway_sampling_coef),
            cond_frames=cond_frames,
            collect_flow_loss=float(args.teacher_flow_loss_weight) > 0,
            teacher_model=teacher,
            teacher_cfg_strength=float(args.teacher_cfg_strength),
            loss_mask=loss_mask,
            detach_null_grad=bool(args.detach_null_grad),
        )
        rollout_loss = F.mse_loss(student_y[loss_mask], teacher_y[loss_mask])
        real_mel_loss = F.l1_loss(student_y[loss_mask], mel[loss_mask])
        delta_loss = temporal_delta_loss(student_y, teacher_y, loss_mask)
        energy_loss = mel_energy_loss(student_y, teacher_y, loss_mask)
        high_mel_loss = high_mel_excess_loss(student_y, teacher_y, loss_mask, start_bin=int(args.high_mel_start_bin))
        high_mel_match = high_mel_match_loss(student_y, teacher_y, loss_mask, start_bin=int(args.high_mel_start_bin))
        rollout_weight = scheduled_weight(float(args.rollout_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.rollout_loss_weight_final)
        teacher_flow_weight = scheduled_weight(float(args.teacher_flow_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.teacher_flow_loss_weight_final)
        real_mel_weight = scheduled_weight(float(args.real_mel_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.real_mel_loss_weight_final)
        anchor_weight = scheduled_weight(float(args.anchor_weight_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.anchor_weight_loss_weight_final)
        delta_weight = scheduled_weight(float(args.temporal_delta_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.temporal_delta_loss_weight_final)
        energy_weight = scheduled_weight(float(args.mel_energy_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.mel_energy_loss_weight_final)
        high_mel_weight = scheduled_weight(float(args.high_mel_excess_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.high_mel_excess_loss_weight_final)
        high_mel_match_weight = scheduled_weight(float(args.high_mel_match_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.high_mel_match_loss_weight_final)
        parameter_anchor_loss = anchor_weight_loss(student, anchor_parameters) if anchor_parameters else mel.new_zeros(())
        loss = (
            rollout_weight * rollout_loss
            + teacher_flow_weight * flow_loss
            + real_mel_weight * real_mel_loss
            + delta_weight * delta_loss
            + energy_weight * energy_loss
            + high_mel_weight * high_mel_loss
            + high_mel_match_weight * high_mel_match
            + anchor_weight * parameter_anchor_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.max_grad_norm))
        optimizer.step()

        step += 1
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            if args.save_best:
                save_checkpoint(student, output_dir / "model_q4_12to4_best.pt", step=step, args=args)
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss_value,
                            "rollout_loss": float(rollout_loss.detach().cpu()),
                            "teacher_flow_loss": float(flow_loss.detach().cpu()),
                            "real_mel_loss": float(real_mel_loss.detach().cpu()),
                            "temporal_delta_loss": float(delta_loss.detach().cpu()),
                            "mel_energy_loss": float(energy_loss.detach().cpu()),
                            "high_mel_excess_loss": float(high_mel_loss.detach().cpu()),
                            "high_mel_match_loss": float(high_mel_match.detach().cpu()),
                            "rollout_weight": rollout_weight,
                            "teacher_flow_weight": teacher_flow_weight,
                            "real_mel_weight": real_mel_weight,
                            "temporal_delta_weight": delta_weight,
                            "mel_energy_weight": energy_weight,
                            "high_mel_excess_weight": high_mel_weight,
                            "high_mel_match_weight": high_mel_match_weight,
                            "parameter_anchor_loss": float(parameter_anchor_loss.detach().cpu()),
                            "anchor_weight": anchor_weight,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        if step % int(args.log_every) == 0:
            metrics_row = {
                "step": step,
                "loss": loss_value,
                "rollout_loss": float(rollout_loss.detach().cpu()),
                "teacher_flow_loss": float(flow_loss.detach().cpu()),
                "real_mel_loss": float(real_mel_loss.detach().cpu()),
                "temporal_delta_loss": float(delta_loss.detach().cpu()),
                "mel_energy_loss": float(energy_loss.detach().cpu()),
                "high_mel_excess_loss": float(high_mel_loss.detach().cpu()),
                "high_mel_match_loss": float(high_mel_match.detach().cpu()),
                "rollout_weight": rollout_weight,
                "teacher_flow_weight": teacher_flow_weight,
                "real_mel_weight": real_mel_weight,
                "temporal_delta_weight": delta_weight,
                "mel_energy_weight": energy_weight,
                "high_mel_excess_weight": high_mel_weight,
                "high_mel_match_weight": high_mel_match_weight,
                "parameter_anchor_loss": float(parameter_anchor_loss.detach().cpu()),
                "anchor_weight": anchor_weight,
                "best_loss": best_loss,
                "frames": int(mel.shape[1]),
                "cond_frames": int(cond_frames),
            }
            print(json.dumps(metrics_row), flush=True)
            append_jsonl(metrics_path, {"event": "train", "time": time.time(), **metrics_row})
        if step % int(args.save_every) == 0:
            path = output_dir / f"model_q4_12to4_step_{step}.pt"
            save_checkpoint(student, path, step=step, args=args)
            print(f"saved={path}", flush=True)
            append_jsonl(metrics_path, {"event": "save", "time": time.time(), "step": step, "path": str(path)})

    final_path = output_dir / "model_q4_12to4_last.pt"
    save_checkpoint(student, final_path, step=step, args=args)
    print(f"saved={final_path}", flush=True)
    append_jsonl(metrics_path, {"event": "save_final", "time": time.time(), "step": step, "path": str(final_path)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill F5TTS Q4 from 12-step CFG teacher quality into 4-step CFG rollout.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--student-checkpoint", default="")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="librispeech_asr")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="train.100")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=384)
    parser.add_argument("--cond-frames", type=int, default=256)
    parser.add_argument("--min-gen-frames", type=int, default=48)
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--student-steps", type=int, default=4)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--teacher-cfg-strength", type=float, default=None)
    parser.add_argument("--student-cfg-strength", type=float, default=None)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-flow-loss-weight", type=float, default=0.25)
    parser.add_argument("--real-mel-loss-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight-loss-weight", type=float, default=0.0)
    parser.add_argument("--temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--mel-energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-start-bin", type=int, default=80)
    parser.add_argument("--loss-schedule-steps", type=int, default=0)
    parser.add_argument("--rollout-loss-weight-final", type=float, default=None)
    parser.add_argument("--teacher-flow-loss-weight-final", type=float, default=None)
    parser.add_argument("--real-mel-loss-weight-final", type=float, default=None)
    parser.add_argument("--anchor-weight-loss-weight-final", type=float, default=None)
    parser.add_argument("--temporal-delta-loss-weight-final", type=float, default=None)
    parser.add_argument("--mel-energy-loss-weight-final", type=float, default=None)
    parser.add_argument("--high-mel-excess-loss-weight-final", type=float, default=None)
    parser.add_argument("--high-mel-match-loss-weight-final", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--q4-include", default="")
    parser.add_argument("--q4-exclude", default="text_embed.text_embed,mel_spec")
    parser.add_argument("--q4-ste-include", default="", help="Optional module-name filter for STE fake quant. Other Q4 modules are materialized once to reduce memory.")
    parser.add_argument("--local-samples", default="/data/resumebot/voice_profiles/Peyton/samples.txt")
    parser.add_argument(
        "--local-sample-prob",
        type=float,
        default=0.0,
        help="Optional probability of sampling a voice-profile manifest. Keep 0 for base model distillation; use profile samples for eval/adaptation only.",
    )
    parser.add_argument(
        "--train-include",
        default="transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out",
    )
    parser.add_argument("--train-exclude", default="")
    parser.add_argument("--detach-null-grad", action="store_true", help="For CFG student training, detach the null branch to reduce activation memory while preserving CFG inference math.")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run-init", action="store_true", help="Initialize model/data/training state, print the setup, then exit.")
    args = parser.parse_args()
    if args.teacher_cfg_strength is None:
        args.teacher_cfg_strength = float(args.cfg_strength)
    if args.student_cfg_strength is None:
        args.student_cfg_strength = float(args.cfg_strength)
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    train(args)


if __name__ == "__main__":
    main()
