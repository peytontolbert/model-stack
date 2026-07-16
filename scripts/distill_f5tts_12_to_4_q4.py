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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    benign_unexpected = tuple(
        name
        for name in unexpected
        if name.endswith(".spin_signs") or name.endswith(".spin_enabled_flag")
    )
    actionable_unexpected = tuple(name for name in unexpected if name not in benign_unexpected)
    if missing or actionable_unexpected:
        print(json.dumps({"missing": missing, "unexpected": list(actionable_unexpected)}, indent=2))
    elif benign_unexpected:
        print(json.dumps({"ignored_unexpected_bitnet_state_tensors": len(benign_unexpected)}, indent=2))
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
    ste_disabled = any(item == "__none__" for item in ste_include)
    for name, module in model.named_modules():
        if should_q4_module(name, module, include, exclude):
            if not ste_disabled and (not ste_include or any(item in name for item in ste_include)):
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
        runtime_weight = getattr(module, "runtime_weight", None)
        if callable(runtime_weight):
            key = f"{name}.weight" if name else "weight"
            state[key] = runtime_weight(dtype=torch.float32, device="cpu").detach().cpu()
            runtime_bias = getattr(module, "runtime_bias", None)
            if callable(runtime_bias):
                bias = runtime_bias(dtype=torch.float32, device="cpu")
                if bias is not None:
                    state[f"{name}.bias" if name else "bias"] = bias.detach().cpu()
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module, "weight"):
            key = f"{name}.weight" if name else "weight"
            state[key] = module.weight.detach().cpu()
    return state


def checkpoint_name(args: argparse.Namespace, suffix: str) -> str:
    return f"model_q4_{int(args.teacher_steps)}to{int(args.student_steps)}_{suffix}.pt"


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
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def scheduled_weight(base: float, *, step: int, max_steps: int, warmup_steps: int = 0, final: float | None = None) -> float:
    if final is None or int(warmup_steps) <= 0:
        return float(base)
    progress = min(1.0, max(0.0, float(step) / float(max(1, min(int(warmup_steps), int(max_steps))))))
    return float(base) + (float(final) - float(base)) * progress


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def stream_hf_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    from datasets import Audio, interleave_datasets, load_dataset

    splits = split_csv(args.split)
    if not splits:
        raise ValueError("--split must name at least one split")
    datasets = [
        load_dataset(args.dataset, args.config or None, split=split, streaming=True)
        for split in splits
    ]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        probabilities = parse_float_list(getattr(args, "split_probabilities", ""))
        if probabilities and len(probabilities) != len(datasets):
            raise ValueError("--split-probabilities must match the number of comma-separated splits")
        if probabilities:
            total = sum(probabilities)
            if total <= 0:
                raise ValueError("--split-probabilities must sum to a positive value")
            probabilities = [value / total for value in probabilities]
        else:
            probabilities = None
        dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            seed=int(args.seed),
            stopping_strategy="all_exhausted",
        )
    if args.audio_column:
        dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.sample_rate))
    if int(args.shuffle_buffer) > 0:
        dataset = dataset.shuffle(buffer_size=int(args.shuffle_buffer), seed=int(args.seed))
    return dataset


def audio_to_tensor(array: Any, sample_rate: int, *, args: argparse.Namespace, enforce_duration: bool = True) -> torch.Tensor | None:
    if array is None:
        return None
    audio = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)
    duration = float(audio.numel()) / float(sample_rate)
    if enforce_duration and (duration < args.min_duration or duration > args.max_duration):
        return None
    if sample_rate != args.sample_rate:
        audio = F.interpolate(
            audio.view(1, 1, -1),
            size=int(round(audio.numel() * args.sample_rate / sample_rate)),
            mode="linear",
            align_corners=False,
        ).view(-1)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    target_rms = 0.1
    if float(rms) > 0.0 and float(rms) < target_rms:
        audio = audio * (target_rms / rms)
    return audio


def audio_array_to_item(array: Any, text: str, sample_rate: int, *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    audio = audio_to_tensor(array, sample_rate, args=args)
    if audio is None:
        return None
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
    if not str(samples_path).strip() or not path.exists() or path.is_dir():
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


def load_local_pairs(pairs_path: str) -> list[tuple[Path, str, Path, str]]:
    path = Path(pairs_path)
    if not str(pairs_path).strip() or not path.exists():
        return []
    rows: list[tuple[Path, str, Path, str]] = []
    base_dir = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        ref_audio_ref, ref_text, target_audio_ref, target_text = [part.strip() for part in parts]
        ref_audio = Path(ref_audio_ref)
        target_audio = Path(target_audio_ref)
        if not ref_audio.exists():
            basename = PureWindowsPath(ref_audio_ref).name if "\\" in ref_audio_ref else ref_audio.name
            ref_audio = base_dir / basename
        if not target_audio.exists():
            basename = PureWindowsPath(target_audio_ref).name if "\\" in target_audio_ref else target_audio.name
            target_audio = base_dir / basename
        if ref_audio.exists() and target_audio.exists() and ref_text and target_text:
            rows.append((ref_audio, ref_text, target_audio, target_text))
    return rows


def local_row_to_item(row: tuple[Path, str], *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    audio_path, text = row
    array, sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
    return audio_array_to_item(array, text, int(sample_rate), args=args, model=model, device=device)


def local_pair_to_item(row: tuple[Path, str, Path, str], *, args: argparse.Namespace, model, device: torch.device) -> dict[str, Any] | None:
    ref_audio_path, ref_text, target_audio_path, target_text = row
    ref_array, ref_sample_rate = sf.read(ref_audio_path, always_2d=False, dtype="float32")
    target_array, target_sample_rate = sf.read(target_audio_path, always_2d=False, dtype="float32")
    ref_audio = audio_to_tensor(ref_array, int(ref_sample_rate), args=args, enforce_duration=False)
    target_audio = audio_to_tensor(target_array, int(target_sample_rate), args=args)
    if ref_audio is None or target_audio is None:
        return None
    with torch.no_grad():
        ref_mel = model.mel_spec(ref_audio.to(device).view(1, -1)).detach().cpu().squeeze(0)
        target_mel = model.mel_spec(target_audio.to(device).view(1, -1)).detach().cpu().squeeze(0)
    cond_frames = int(args.cond_frames)
    if ref_mel.shape[-1] < cond_frames or target_mel.shape[-1] < int(args.min_gen_frames):
        return None
    max_target_frames = max(int(args.min_gen_frames), int(args.max_frames) - cond_frames)
    target_mel = target_mel[:, :max_target_frames].contiguous()
    mel = torch.cat([ref_mel[:, :cond_frames].contiguous(), target_mel], dim=-1)
    return {"mel_spec": mel, "text": f"{ref_text.strip()} {target_text.strip()}".strip()}


def make_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    from f5_tts.model.dataset import collate_fn

    return collate_fn(items)


def text_to_ids(model, text, device: torch.device, *, convert_to_pinyin: bool = False) -> torch.Tensor:
    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx, list_str_to_tensor

    if isinstance(text, torch.Tensor):
        return text.to(device)
    if isinstance(text, list):
        if convert_to_pinyin:
            text = convert_char_to_pinyin(text)
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
    flow, _ = cfg_flow_with_delta(
        model,
        y,
        cond,
        text_ids,
        time,
        cfg_strength,
        detach_null_grad=detach_null_grad,
    )
    return flow


def cfg_flow_with_delta(
    model,
    y: torch.Tensor,
    cond: torch.Tensor,
    text_ids: torch.Tensor,
    time: torch.Tensor,
    cfg_strength: float,
    *,
    detach_null_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = model.transformer(
        x=y,
        cond=cond,
        text=text_ids,
        time=time,
        drop_audio_cond=False,
        drop_text=False,
    )
    if abs(float(cfg_strength)) < 1e-8:
        return pred, torch.zeros_like(pred)
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
    cond_delta = pred - null_pred
    return pred + cond_delta * float(cfg_strength), cond_delta


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
    return_states: bool = False,
    collect_cond_delta_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[float]]:
    y = noise.clone()
    times = make_time_grid(steps, sway_sampling_coef, y.device, y.dtype)
    flow_loss = y.new_zeros(())
    cond_delta_loss = y.new_zeros(())
    flow_terms = 0
    cond_delta_terms = 0
    states: list[torch.Tensor] = [y.detach()] if return_states else []
    state_times: list[float] = [0.0] if return_states else []
    for step in range(int(steps)):
        t = times[step]
        dt = times[step + 1] - times[step]
        time = torch.full((y.shape[0],), float(t.detach().cpu()), device=y.device, dtype=y.dtype)
        flow, student_cond_delta = cfg_flow_with_delta(
            model,
            y,
            cond,
            text_ids,
            time,
            cfg_strength,
            detach_null_grad=detach_null_grad,
        )
        if collect_flow_loss and teacher_model is not None and loss_mask is not None:
            with torch.no_grad():
                teacher_flow, teacher_cond_delta = cfg_flow_with_delta(
                    teacher_model,
                    y.detach(),
                    cond,
                    text_ids,
                    time,
                    float(cfg_strength if teacher_cfg_strength is None else teacher_cfg_strength),
                )
            flow_loss = flow_loss + F.mse_loss(flow[loss_mask], teacher_flow[loss_mask])
            flow_terms += 1
            if collect_cond_delta_loss:
                cond_delta_loss = cond_delta_loss + F.mse_loss(student_cond_delta[loss_mask], teacher_cond_delta[loss_mask])
                cond_delta_terms += 1
        y = y + dt * flow
        if return_states:
            states.append(y)
            state_times.append(float(times[step + 1].detach().cpu()))
    if cond_frames > 0:
        y[:, :cond_frames, :] = cond[:, :cond_frames, :]
    if flow_terms:
        flow_loss = flow_loss / float(flow_terms)
    if cond_delta_terms:
        cond_delta_loss = cond_delta_loss / float(cond_delta_terms)
    if return_states:
        return y, flow_loss, cond_delta_loss, states, state_times
    return y, flow_loss, cond_delta_loss


def teacher_state_at_time(teacher_states: list[torch.Tensor], teacher_times: list[float], target_time: float) -> torch.Tensor:
    if not teacher_states:
        raise ValueError("teacher states are required for trajectory loss")
    index = min(range(len(teacher_times)), key=lambda item: abs(float(teacher_times[item]) - float(target_time)))
    return teacher_states[index]


def trajectory_state_loss(
    student_states: list[torch.Tensor],
    student_times: list[float],
    teacher_states: list[torch.Tensor],
    teacher_times: list[float],
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if not student_states or not bool(loss_mask.any()):
        return loss_mask.new_zeros((), dtype=torch.float32)
    terms = []
    for student_state, student_time in zip(student_states, student_times, strict=True):
        teacher_state = teacher_state_at_time(teacher_states, teacher_times, student_time)
        terms.append(F.mse_loss(student_state[loss_mask], teacher_state[loss_mask]))
    return torch.stack(terms).mean()


def segment_flow_loss(
    model,
    teacher_states: list[torch.Tensor],
    teacher_times: list[float],
    *,
    cond: torch.Tensor,
    text_ids: torch.Tensor,
    student_steps: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    loss_mask: torch.Tensor,
    detach_null_grad: bool = False,
) -> torch.Tensor:
    if not teacher_states or not bool(loss_mask.any()):
        return loss_mask.new_zeros((), dtype=torch.float32)
    times = make_time_grid(student_steps, sway_sampling_coef, cond.device, cond.dtype)
    terms: list[torch.Tensor] = []
    for step in range(int(student_steps)):
        t0 = float(times[step].detach().cpu())
        t1 = float(times[step + 1].detach().cpu())
        dt = float(t1 - t0)
        if abs(dt) < 1e-8:
            continue
        y0 = teacher_state_at_time(teacher_states, teacher_times, t0).detach()
        y1 = teacher_state_at_time(teacher_states, teacher_times, t1).detach()
        target_flow = (y1 - y0) / dt
        time_tensor = torch.full((cond.shape[0],), t0, device=cond.device, dtype=cond.dtype)
        student_flow = cfg_flow(
            model,
            y0,
            cond,
            text_ids,
            time_tensor,
            cfg_strength,
            detach_null_grad=detach_null_grad,
        )
        terms.append(F.mse_loss(student_flow[loss_mask], target_flow[loss_mask]))
    if not terms:
        return loss_mask.new_zeros((), dtype=torch.float32)
    return torch.stack(terms).mean()


def build_loss_mask(lens: torch.Tensor, cond_frames: int, frames: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(frames, device=device)[None, :]
    valid = positions < lens.to(device)[:, None]
    generated = positions >= int(cond_frames)
    return valid & generated


def apply_generated_frame_jitter(
    mel: torch.Tensor,
    lens: torch.Tensor,
    *,
    cond_frames: int,
    min_gen_frames: int,
    max_frames: int,
    jitter_frames: int,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(jitter_frames) <= 0 or float(probability) <= 0.0 or random.random() >= float(probability):
        return mel, lens
    batch, frames, channels = mel.shape
    min_total_frames = int(cond_frames) + int(min_gen_frames)
    target_lens = lens.detach().cpu().clone()
    changed = False
    for index in range(batch):
        current_len = int(target_lens[index].item())
        if current_len <= min_total_frames:
            continue
        delta = random.randint(-int(jitter_frames), int(jitter_frames))
        if delta == 0:
            continue
        new_len = max(min_total_frames, min(int(max_frames), current_len + delta))
        if new_len != current_len:
            target_lens[index] = int(new_len)
            changed = True
    if not changed:
        return mel, lens

    new_frames = int(target_lens.max().item())
    if new_frames == frames:
        return mel, target_lens.to(lens.device, dtype=lens.dtype)

    if new_frames < frames:
        return mel[:, :new_frames, :].contiguous(), target_lens.to(lens.device, dtype=lens.dtype)

    padded = mel.new_zeros((batch, new_frames, channels))
    padded[:, :frames, :] = mel
    for index in range(batch):
        old_len = int(lens[index].item())
        new_len = int(target_lens[index].item())
        if new_len > old_len and old_len > 0:
            padded[index, old_len:new_len, :] = mel[index, old_len - 1 : old_len, :]
    return padded.contiguous(), target_lens.to(lens.device, dtype=lens.dtype)


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


def mel_energy_envelope(y: torch.Tensor) -> torch.Tensor:
    return torch.log1p(y.float().pow(2).mean(dim=-1) * 16.0)


def energy_envelope_loss(student_y: torch.Tensor, teacher_y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if not bool(loss_mask.any()):
        return student_y.new_zeros(())
    student_energy = mel_energy_envelope(student_y)
    teacher_energy = mel_energy_envelope(teacher_y)
    return F.smooth_l1_loss(student_energy[loss_mask], teacher_energy[loss_mask], beta=0.08)


def silence_envelope_loss(student_y: torch.Tensor, teacher_y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if not bool(loss_mask.any()):
        return student_y.new_zeros(())
    student_energy = mel_energy_envelope(student_y)
    teacher_energy = mel_energy_envelope(teacher_y)
    active_teacher_energy = teacher_energy[loss_mask]
    if active_teacher_energy.numel() == 0:
        return student_y.new_zeros(())
    quiet_threshold = torch.quantile(active_teacher_energy.detach().float(), 0.35).to(teacher_energy.dtype)
    quiet_mask = loss_mask & (teacher_energy <= quiet_threshold)
    if not bool(quiet_mask.any()):
        return student_y.new_zeros(())
    excess = F.relu(student_energy[quiet_mask] - teacher_energy[quiet_mask] - 0.02)
    return excess.mean()


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


def high_mel_ratio_loss(student_y: torch.Tensor, teacher_y: torch.Tensor, loss_mask: torch.Tensor, start_bin: int = 72) -> torch.Tensor:
    if not bool(loss_mask.any()):
        return student_y.new_zeros(())
    start = max(0, min(int(start_bin), int(student_y.shape[-1]) - 1))
    student_power = student_y.float().pow(2)
    teacher_power = teacher_y.float().pow(2)
    student_total = student_power.mean(dim=-1).clamp_min(1e-6)
    teacher_total = teacher_power.mean(dim=-1).clamp_min(1e-6)
    student_ratio = student_power[..., start:].mean(dim=-1) / student_total
    teacher_ratio = teacher_power[..., start:].mean(dim=-1) / teacher_total
    excess = F.relu(student_ratio - teacher_ratio - 0.02)
    return F.smooth_l1_loss(excess[loss_mask], torch.zeros_like(excess[loss_mask]), beta=0.05)


def high_mel_temporal_loss(
    student_y: torch.Tensor,
    teacher_y: torch.Tensor,
    loss_mask: torch.Tensor,
    start_bin: int = 64,
) -> torch.Tensor:
    first_mask = loss_mask[:, 1:] & loss_mask[:, :-1]
    if not bool(first_mask.any()):
        return student_y.new_zeros(())
    start = max(0, min(int(start_bin), int(student_y.shape[-1]) - 1))
    student_delta = student_y[:, 1:, start:].float() - student_y[:, :-1, start:].float()
    teacher_delta = teacher_y[:, 1:, start:].float() - teacher_y[:, :-1, start:].float()
    match = F.smooth_l1_loss(student_delta[first_mask], teacher_delta[first_mask], beta=0.08)
    excess_delta = F.relu(student_delta.abs().mean(dim=-1) - teacher_delta.abs().mean(dim=-1) - 0.01)
    excess = F.smooth_l1_loss(excess_delta[first_mask], torch.zeros_like(excess_delta[first_mask]), beta=0.02)

    second_mask = first_mask[:, 1:] & first_mask[:, :-1]
    if bool(second_mask.any()):
        student_accel = student_delta[:, 1:, :] - student_delta[:, :-1, :]
        teacher_accel = teacher_delta[:, 1:, :] - teacher_delta[:, :-1, :]
        accel = F.smooth_l1_loss(student_accel[second_mask], teacher_accel[second_mask], beta=0.08)
    else:
        accel = student_y.new_zeros(())
    return match + 0.5 * accel + 0.35 * excess


def low_mid_mel_body_loss(
    student_y: torch.Tensor,
    teacher_y: torch.Tensor,
    loss_mask: torch.Tensor,
    end_bin: int = 80,
) -> torch.Tensor:
    if not bool(loss_mask.any()):
        return student_y.new_zeros(())
    end = max(1, min(int(end_bin), int(student_y.shape[-1])))
    student_band = student_y[..., :end].float()
    teacher_band = teacher_y[..., :end].float()
    band_loss = F.smooth_l1_loss(student_band[loss_mask], teacher_band[loss_mask], beta=0.20)
    student_energy = student_band.pow(2).mean(dim=-1).clamp_min(1e-6)
    teacher_energy = teacher_band.pow(2).mean(dim=-1).clamp_min(1e-6)
    energy_loss = F.smooth_l1_loss(
        torch.log(student_energy[loss_mask]),
        torch.log(teacher_energy[loss_mask]),
        beta=0.10,
    )
    return band_loss + 0.25 * energy_loss


def corrupt_text_ids(text_ids: torch.Tensor, mode: str = "reverse") -> torch.Tensor:
    if text_ids.ndim != 2 or text_ids.shape[1] <= 2:
        return text_ids
    mode = str(mode or "reverse").strip().lower()
    corrupted = text_ids.clone()
    for row in range(corrupted.shape[0]):
        valid = torch.nonzero(corrupted[row] >= 0, as_tuple=False).flatten()
        if valid.numel() <= 2:
            continue
        if mode in {"local", "adjacent", "hard-local"}:
            # Keep the corruption close to the original text so contrastive loss
            # remains active for word-order and near-pronunciation confusions.
            pairs = valid[1::4]
            next_pairs = pairs + 1
            next_pairs = next_pairs[next_pairs < valid[-1]]
            pairs = pairs[: next_pairs.numel()]
            if pairs.numel() > 0:
                left = corrupted[row, pairs].clone()
                corrupted[row, pairs] = corrupted[row, next_pairs]
                corrupted[row, next_pairs] = left
            else:
                corrupted[row, valid] = corrupted[row, valid].roll(1)
        elif mode in {"roll", "shift"}:
            corrupted[row, valid] = corrupted[row, valid].roll(1)
        else:
            values = corrupted[row, valid].flip(0)
            corrupted[row, valid] = values
    return corrupted


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
    teacher.eval().requires_grad_(False)

    student_quant_scheme = str(args.student_quant_scheme).strip().lower()
    if student_quant_scheme == "q4":
        load_checkpoint_state(student, Path(args.student_checkpoint or args.checkpoint))
        q4_modules, q4_params = apply_q4_parametrizations(
            student,
            include=split_csv(args.q4_include),
            exclude=split_csv(args.q4_exclude),
            ste_include=split_csv(args.q4_ste_include),
        )
    elif student_quant_scheme in {"bitnet_qat", "qat_bitnet", "trainable_bitnet"}:
        from compress.quantization import quantize_linear_modules

        replacements = quantize_linear_modules(
            student,
            include=split_csv(args.q4_include),
            exclude=split_csv(args.q4_exclude),
            scheme="bitnet_qat",
            bitnet_qat_learned_scale=bool(args.bitnet_qat_learned_scale),
        )
        load_checkpoint_state(student, Path(args.student_checkpoint or args.checkpoint))
        q4_modules = len(replacements)
        q4_params = sum(int(module.weight.numel()) for module in replacements.values() if hasattr(module, "weight"))
    elif student_quant_scheme in {"none", "fp", "dense"}:
        load_checkpoint_state(student, Path(args.student_checkpoint or args.checkpoint))
        q4_modules = 0
        q4_params = 0
    else:
        raise ValueError(f"unsupported --student-quant-scheme: {args.student_quant_scheme}")
    train_tensors, train_params = configure_trainable_parameters(
        student,
        train_include=split_csv(args.train_include),
        train_exclude=split_csv(args.train_exclude),
    )
    if bool(args.train_in_eval_mode):
        student.eval()
    else:
        student.train()
    scale_lr_multiplier = float(args.bitnet_scale_lr_multiplier)
    trainable_named_parameters = [(name, parameter) for name, parameter in student.named_parameters() if parameter.requires_grad]
    trainable_parameters = [parameter for _, parameter in trainable_named_parameters]
    if not trainable_parameters:
        raise ValueError("no trainable parameters selected")
    if student_quant_scheme in {"bitnet_qat", "qat_bitnet", "trainable_bitnet"} and scale_lr_multiplier != 1.0:
        scale_parameters = [parameter for name, parameter in trainable_named_parameters if name.endswith(".weight_scale")]
        scale_parameter_ids = {id(parameter) for parameter in scale_parameters}
        base_parameters = [parameter for _, parameter in trainable_named_parameters if id(parameter) not in scale_parameter_ids]
        parameter_groups: list[dict[str, Any]] = []
        if base_parameters:
            parameter_groups.append({"params": base_parameters, "lr": float(args.lr)})
        if scale_parameters:
            parameter_groups.append({"params": scale_parameters, "lr": float(args.lr) * scale_lr_multiplier})
        optimizer_parameters: Any = parameter_groups
    else:
        scale_parameters = []
        optimizer_parameters = trainable_parameters
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    anchor_parameters = trainable_anchor_parameters(student) if float(args.anchor_weight_loss_weight) > 0.0 else {}

    local_rows = load_local_samples(args.local_samples)
    local_pair_rows = load_local_pairs(args.local_pairs)
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
        "student_quant_scheme": student_quant_scheme,
        "train_tensors": train_tensors,
        "train_params": train_params,
        "bitnet_scale_lr_multiplier": scale_lr_multiplier,
        "bitnet_scale_trainable_tensors": len(scale_parameters),
        "bitnet_scale_trainable_params": sum(int(parameter.numel()) for parameter in scale_parameters),
        "teacher_steps": args.teacher_steps,
        "student_steps": args.student_steps,
        "teacher_checkpoint": args.checkpoint,
        "student_checkpoint": args.student_checkpoint or args.checkpoint,
        "cfg_strength": args.cfg_strength,
        "teacher_cfg_strength": args.teacher_cfg_strength,
        "student_cfg_strength": args.student_cfg_strength,
        "trajectory_loss_weight": args.trajectory_loss_weight,
        "cond_delta_loss_weight": args.cond_delta_loss_weight,
        "segment_flow_loss_weight": args.segment_flow_loss_weight,
        "anchor_weight_loss_weight": args.anchor_weight_loss_weight,
        "temporal_delta_loss_weight": args.temporal_delta_loss_weight,
        "mel_energy_loss_weight": args.mel_energy_loss_weight,
        "energy_envelope_loss_weight": args.energy_envelope_loss_weight,
        "silence_envelope_loss_weight": args.silence_envelope_loss_weight,
        "high_mel_excess_loss_weight": args.high_mel_excess_loss_weight,
        "high_mel_match_loss_weight": args.high_mel_match_loss_weight,
        "high_mel_ratio_loss_weight": args.high_mel_ratio_loss_weight,
        "high_mel_temporal_loss_weight": args.high_mel_temporal_loss_weight,
        "low_mid_mel_body_loss_weight": args.low_mid_mel_body_loss_weight,
        "text_contrastive_loss_weight": args.text_contrastive_loss_weight,
        "text_flow_contrastive_loss_weight": args.text_flow_contrastive_loss_weight,
        "text_delta_loss_weight": args.text_delta_loss_weight,
        "text_contrastive_margin": args.text_contrastive_margin,
        "high_mel_start_bin": args.high_mel_start_bin,
        "detach_null_grad": bool(args.detach_null_grad),
        "train_in_eval_mode": bool(args.train_in_eval_mode),
        "local_profile_samples_available": len(local_rows),
        "local_pair_samples_available": len(local_pair_rows),
        "local_sample_prob": float(args.local_sample_prob),
        "local_profile_samples_enabled": (bool(local_rows) or bool(local_pair_rows)) and float(args.local_sample_prob) > 0.0,
    }
    print(json.dumps(setup_row, indent=2), flush=True)
    metrics_path = output_dir / "metrics.jsonl"
    append_jsonl(metrics_path, {"event": "setup", "time": time.time(), **setup_row})
    if bool(args.dry_run_init):
        return

    student_cfg_choices = parse_float_list(args.student_cfg_strengths)
    if not student_cfg_choices:
        student_cfg_choices = [float(args.student_cfg_strength)]

    pending: list[dict[str, Any]] = []
    best_loss = float("inf")
    step = 0
    while step < int(args.max_steps):
        use_local = (bool(local_rows) or bool(local_pair_rows)) and random.random() < float(args.local_sample_prob)
        if use_local:
            if local_pair_rows and (not local_rows or random.random() < 0.8):
                item = local_pair_to_item(random.choice(local_pair_rows), args=args, model=student, device=device)
            else:
                item = local_row_to_item(random.choice(local_rows), args=args, model=student, device=device)
        else:
            try:
                item = row_to_item(next(row_iter), args=args, model=student, device=device)
            except StopIteration:
                if use_hf_stream:
                    row_iter = iter(stream_hf_rows(args))
                    continue
                if (not local_rows and not local_pair_rows) or float(args.local_sample_prob) <= 0.0:
                    break
                if local_pair_rows and (not local_rows or random.random() < 0.8):
                    item = local_pair_to_item(random.choice(local_pair_rows), args=args, model=student, device=device)
                else:
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
        jitter_cond_frames = min(int(args.cond_frames), max(1, int(lens.min().item()) - int(args.min_gen_frames)))
        original_lens = lens.clone()
        mel, lens = apply_generated_frame_jitter(
            mel,
            lens,
            cond_frames=jitter_cond_frames,
            min_gen_frames=int(args.min_gen_frames),
            max_frames=int(args.max_frames),
            jitter_frames=int(args.generated_frame_jitter),
            probability=float(args.generated_frame_jitter_prob),
        )
        duration_jitter_frames = int((lens - original_lens.to(lens.device)).abs().max().item()) if lens.numel() else 0
        text_ids = text_to_ids(student, batch["text"], device, convert_to_pinyin=bool(args.convert_text_to_pinyin))
        cond_frames = min(int(args.cond_frames), max(1, int(lens.min().item()) - int(args.min_gen_frames)))
        cond = torch.zeros_like(mel)
        cond[:, :cond_frames, :] = mel[:, :cond_frames, :]
        noise = torch.randn_like(mel)
        loss_mask = build_loss_mask(lens, cond_frames, mel.shape[1], device)
        if not bool(loss_mask.any()):
            continue

        with torch.no_grad():
            teacher_rollout = rollout_sample(
                teacher,
                noise=noise,
                cond=cond,
                text_ids=text_ids,
                steps=int(args.teacher_steps),
                cfg_strength=float(args.teacher_cfg_strength),
                sway_sampling_coef=float(args.sway_sampling_coef),
                cond_frames=cond_frames,
                return_states=float(args.trajectory_loss_weight) > 0.0 or float(args.segment_flow_loss_weight) > 0.0,
            )
            if float(args.trajectory_loss_weight) > 0.0 or float(args.segment_flow_loss_weight) > 0.0:
                teacher_y, _, _, teacher_states, teacher_times = teacher_rollout
            else:
                teacher_y, _, _ = teacher_rollout
                teacher_states = []
                teacher_times = []
        current_student_cfg_strength = float(random.choice(student_cfg_choices))
        student_rollout = rollout_sample(
            student,
            noise=noise,
            cond=cond,
            text_ids=text_ids,
            steps=int(args.student_steps),
            cfg_strength=current_student_cfg_strength,
            sway_sampling_coef=float(args.sway_sampling_coef),
            cond_frames=cond_frames,
            collect_flow_loss=float(args.teacher_flow_loss_weight) > 0,
            teacher_model=teacher,
            teacher_cfg_strength=float(args.teacher_cfg_strength),
            loss_mask=loss_mask,
            detach_null_grad=bool(args.detach_null_grad),
            return_states=float(args.trajectory_loss_weight) > 0.0,
            collect_cond_delta_loss=float(args.cond_delta_loss_weight) > 0,
        )
        if float(args.trajectory_loss_weight) > 0.0:
            student_y, flow_loss, cond_delta_loss_value, student_states, student_times = student_rollout
            trajectory_loss = trajectory_state_loss(student_states, student_times, teacher_states, teacher_times, loss_mask)
        else:
            student_y, flow_loss, cond_delta_loss_value = student_rollout
            trajectory_loss = mel.new_zeros(())
        segment_loss = (
            segment_flow_loss(
                student,
                teacher_states,
                teacher_times,
                cond=cond,
                text_ids=text_ids,
                student_steps=int(args.student_steps),
                cfg_strength=current_student_cfg_strength,
                sway_sampling_coef=float(args.sway_sampling_coef),
                loss_mask=loss_mask,
                detach_null_grad=bool(args.detach_null_grad),
            )
            if float(args.segment_flow_loss_weight) > 0.0
            else mel.new_zeros(())
        )
        rollout_loss = F.mse_loss(student_y[loss_mask], teacher_y[loss_mask])
        real_mel_loss = F.l1_loss(student_y[loss_mask], mel[loss_mask])
        delta_loss = temporal_delta_loss(student_y, teacher_y, loss_mask)
        energy_loss = mel_energy_loss(student_y, teacher_y, loss_mask)
        energy_env_loss = energy_envelope_loss(student_y, teacher_y, loss_mask)
        silence_env_loss = silence_envelope_loss(student_y, teacher_y, loss_mask)
        high_mel_loss = high_mel_excess_loss(student_y, teacher_y, loss_mask, start_bin=int(args.high_mel_start_bin))
        high_mel_match = high_mel_match_loss(student_y, teacher_y, loss_mask, start_bin=int(args.high_mel_start_bin))
        high_mel_ratio = high_mel_ratio_loss(student_y, teacher_y, loss_mask, start_bin=int(args.high_mel_start_bin))
        high_mel_temporal = high_mel_temporal_loss(student_y, teacher_y, loss_mask, start_bin=int(args.high_mel_start_bin))
        low_mid_body = low_mid_mel_body_loss(student_y, teacher_y, loss_mask, end_bin=int(args.low_mid_mel_end_bin))
        text_contrastive_loss = mel.new_zeros(())
        text_flow_contrastive_loss = mel.new_zeros(())
        text_delta_loss = mel.new_zeros(())
        if float(args.text_contrastive_loss_weight) > 0.0:
            bad_text_ids = corrupt_text_ids(text_ids, mode=str(args.text_corruption_mode))
            bad_y, _, _ = rollout_sample(
                student,
                noise=noise,
                cond=cond,
                text_ids=bad_text_ids,
                steps=int(args.student_steps),
                cfg_strength=current_student_cfg_strength,
                sway_sampling_coef=float(args.sway_sampling_coef),
                cond_frames=cond_frames,
                detach_null_grad=bool(args.detach_null_grad),
            )
            bad_rollout_loss = F.mse_loss(bad_y[loss_mask], teacher_y[loss_mask])
            text_contrastive_loss = F.relu(float(args.text_contrastive_margin) + rollout_loss - bad_rollout_loss)
        if float(args.text_flow_contrastive_loss_weight) > 0.0:
            bad_text_ids = corrupt_text_ids(text_ids, mode=str(args.text_corruption_mode))
            t0 = torch.zeros((mel.shape[0],), device=device, dtype=mel.dtype)
            student_good_flow = cfg_flow(
                student,
                noise,
                cond,
                text_ids,
                t0,
                current_student_cfg_strength,
                detach_null_grad=bool(args.detach_null_grad),
            )
            student_bad_flow = cfg_flow(
                student,
                noise,
                cond,
                bad_text_ids,
                t0,
                current_student_cfg_strength,
                detach_null_grad=bool(args.detach_null_grad),
            )
            with torch.no_grad():
                teacher_good_flow = cfg_flow(
                    teacher,
                    noise,
                    cond,
                    text_ids,
                    t0,
                    float(args.teacher_cfg_strength),
                )
            good_flow_loss = F.mse_loss(student_good_flow[loss_mask], teacher_good_flow[loss_mask])
            bad_flow_loss = F.mse_loss(student_bad_flow[loss_mask], teacher_good_flow[loss_mask])
            text_flow_contrastive_loss = F.relu(
                float(args.text_flow_contrastive_margin) + good_flow_loss - bad_flow_loss
            )
        if float(args.text_delta_loss_weight) > 0.0:
            bad_text_ids = corrupt_text_ids(text_ids, mode=str(args.text_corruption_mode))
            t0 = torch.zeros((mel.shape[0],), device=device, dtype=mel.dtype)
            student_good_flow = cfg_flow(
                student,
                noise,
                cond,
                text_ids,
                t0,
                current_student_cfg_strength,
                detach_null_grad=bool(args.detach_null_grad),
            )
            student_bad_flow = cfg_flow(
                student,
                noise,
                cond,
                bad_text_ids,
                t0,
                current_student_cfg_strength,
                detach_null_grad=bool(args.detach_null_grad),
            )
            with torch.no_grad():
                teacher_good_flow = cfg_flow(
                    teacher,
                    noise,
                    cond,
                    text_ids,
                    t0,
                    float(args.teacher_cfg_strength),
                )
                teacher_bad_flow = cfg_flow(
                    teacher,
                    noise,
                    cond,
                    bad_text_ids,
                    t0,
                    float(args.teacher_cfg_strength),
                )
            student_text_delta = student_good_flow - student_bad_flow
            teacher_text_delta = teacher_good_flow - teacher_bad_flow
            text_delta_loss = F.smooth_l1_loss(student_text_delta[loss_mask], teacher_text_delta[loss_mask], beta=0.1)
        rollout_weight = scheduled_weight(float(args.rollout_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.rollout_loss_weight_final)
        teacher_flow_weight = scheduled_weight(float(args.teacher_flow_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.teacher_flow_loss_weight_final)
        cond_delta_weight = scheduled_weight(float(args.cond_delta_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.cond_delta_loss_weight_final)
        segment_flow_weight = scheduled_weight(float(args.segment_flow_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.segment_flow_loss_weight_final)
        trajectory_weight = scheduled_weight(float(args.trajectory_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.trajectory_loss_weight_final)
        real_mel_weight = scheduled_weight(float(args.real_mel_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.real_mel_loss_weight_final)
        anchor_weight = scheduled_weight(float(args.anchor_weight_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.anchor_weight_loss_weight_final)
        delta_weight = scheduled_weight(float(args.temporal_delta_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.temporal_delta_loss_weight_final)
        energy_weight = scheduled_weight(float(args.mel_energy_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.mel_energy_loss_weight_final)
        energy_env_weight = scheduled_weight(float(args.energy_envelope_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.energy_envelope_loss_weight_final)
        silence_env_weight = scheduled_weight(float(args.silence_envelope_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.silence_envelope_loss_weight_final)
        high_mel_weight = scheduled_weight(float(args.high_mel_excess_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.high_mel_excess_loss_weight_final)
        high_mel_match_weight = scheduled_weight(float(args.high_mel_match_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.high_mel_match_loss_weight_final)
        high_mel_ratio_weight = scheduled_weight(float(args.high_mel_ratio_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.high_mel_ratio_loss_weight_final)
        high_mel_temporal_weight = scheduled_weight(float(args.high_mel_temporal_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.high_mel_temporal_loss_weight_final)
        low_mid_body_weight = scheduled_weight(float(args.low_mid_mel_body_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.low_mid_mel_body_loss_weight_final)
        text_contrastive_weight = scheduled_weight(float(args.text_contrastive_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.text_contrastive_loss_weight_final)
        text_flow_contrastive_weight = scheduled_weight(float(args.text_flow_contrastive_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.text_flow_contrastive_loss_weight_final)
        text_delta_weight = scheduled_weight(float(args.text_delta_loss_weight), step=step, max_steps=int(args.max_steps), warmup_steps=int(args.loss_schedule_steps), final=args.text_delta_loss_weight_final)
        parameter_anchor_loss = anchor_weight_loss(student, anchor_parameters) if anchor_parameters else mel.new_zeros(())
        loss = (
            rollout_weight * rollout_loss
            + teacher_flow_weight * flow_loss
            + cond_delta_weight * cond_delta_loss_value
            + segment_flow_weight * segment_loss
            + trajectory_weight * trajectory_loss
            + real_mel_weight * real_mel_loss
            + delta_weight * delta_loss
            + energy_weight * energy_loss
            + energy_env_weight * energy_env_loss
            + silence_env_weight * silence_env_loss
            + high_mel_weight * high_mel_loss
            + high_mel_match_weight * high_mel_match
            + high_mel_ratio_weight * high_mel_ratio
            + high_mel_temporal_weight * high_mel_temporal
            + low_mid_body_weight * low_mid_body
            + text_contrastive_weight * text_contrastive_loss
            + text_flow_contrastive_weight * text_flow_contrastive_loss
            + text_delta_weight * text_delta_loss
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
                save_checkpoint(student, output_dir / checkpoint_name(args, "best"), step=step, args=args)
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss_value,
                            "rollout_loss": float(rollout_loss.detach().cpu()),
                            "teacher_flow_loss": float(flow_loss.detach().cpu()),
                            "cond_delta_loss": float(cond_delta_loss_value.detach().cpu()),
                            "segment_flow_loss": float(segment_loss.detach().cpu()),
                            "trajectory_loss": float(trajectory_loss.detach().cpu()),
                            "real_mel_loss": float(real_mel_loss.detach().cpu()),
                            "temporal_delta_loss": float(delta_loss.detach().cpu()),
                            "mel_energy_loss": float(energy_loss.detach().cpu()),
                            "energy_envelope_loss": float(energy_env_loss.detach().cpu()),
                            "silence_envelope_loss": float(silence_env_loss.detach().cpu()),
                            "high_mel_excess_loss": float(high_mel_loss.detach().cpu()),
                            "high_mel_match_loss": float(high_mel_match.detach().cpu()),
                            "high_mel_ratio_loss": float(high_mel_ratio.detach().cpu()),
                            "high_mel_temporal_loss": float(high_mel_temporal.detach().cpu()),
                            "low_mid_mel_body_loss": float(low_mid_body.detach().cpu()),
                            "text_contrastive_loss": float(text_contrastive_loss.detach().cpu()),
                            "text_flow_contrastive_loss": float(text_flow_contrastive_loss.detach().cpu()),
                            "text_delta_loss": float(text_delta_loss.detach().cpu()),
                            "student_cfg_strength": current_student_cfg_strength,
                            "rollout_weight": rollout_weight,
                            "teacher_flow_weight": teacher_flow_weight,
                            "cond_delta_weight": cond_delta_weight,
                            "segment_flow_weight": segment_flow_weight,
                            "trajectory_weight": trajectory_weight,
                            "real_mel_weight": real_mel_weight,
                            "temporal_delta_weight": delta_weight,
                            "mel_energy_weight": energy_weight,
                            "energy_envelope_weight": energy_env_weight,
                            "silence_envelope_weight": silence_env_weight,
                            "high_mel_excess_weight": high_mel_weight,
                            "high_mel_match_weight": high_mel_match_weight,
                            "high_mel_ratio_weight": high_mel_ratio_weight,
                            "high_mel_temporal_weight": high_mel_temporal_weight,
                            "low_mid_mel_body_weight": low_mid_body_weight,
                            "text_contrastive_weight": text_contrastive_weight,
                            "text_flow_contrastive_weight": text_flow_contrastive_weight,
                            "text_delta_weight": text_delta_weight,
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
                "cond_delta_loss": float(cond_delta_loss_value.detach().cpu()),
                "segment_flow_loss": float(segment_loss.detach().cpu()),
                "trajectory_loss": float(trajectory_loss.detach().cpu()),
                "real_mel_loss": float(real_mel_loss.detach().cpu()),
                "temporal_delta_loss": float(delta_loss.detach().cpu()),
                "mel_energy_loss": float(energy_loss.detach().cpu()),
                "energy_envelope_loss": float(energy_env_loss.detach().cpu()),
                "silence_envelope_loss": float(silence_env_loss.detach().cpu()),
                "high_mel_excess_loss": float(high_mel_loss.detach().cpu()),
                "high_mel_match_loss": float(high_mel_match.detach().cpu()),
                "high_mel_ratio_loss": float(high_mel_ratio.detach().cpu()),
                "high_mel_temporal_loss": float(high_mel_temporal.detach().cpu()),
                "low_mid_mel_body_loss": float(low_mid_body.detach().cpu()),
                "text_contrastive_loss": float(text_contrastive_loss.detach().cpu()),
                "text_flow_contrastive_loss": float(text_flow_contrastive_loss.detach().cpu()),
                "text_delta_loss": float(text_delta_loss.detach().cpu()),
                "student_cfg_strength": current_student_cfg_strength,
                "rollout_weight": rollout_weight,
                "teacher_flow_weight": teacher_flow_weight,
                "cond_delta_weight": cond_delta_weight,
                "segment_flow_weight": segment_flow_weight,
                "trajectory_weight": trajectory_weight,
                "real_mel_weight": real_mel_weight,
                "temporal_delta_weight": delta_weight,
                "mel_energy_weight": energy_weight,
                "energy_envelope_weight": energy_env_weight,
                "silence_envelope_weight": silence_env_weight,
                "high_mel_excess_weight": high_mel_weight,
                "high_mel_match_weight": high_mel_match_weight,
                "high_mel_ratio_weight": high_mel_ratio_weight,
                "high_mel_temporal_weight": high_mel_temporal_weight,
                "low_mid_mel_body_weight": low_mid_body_weight,
                "text_contrastive_weight": text_contrastive_weight,
                "text_flow_contrastive_weight": text_flow_contrastive_weight,
                "text_delta_weight": text_delta_weight,
                "parameter_anchor_loss": float(parameter_anchor_loss.detach().cpu()),
                "anchor_weight": anchor_weight,
                "best_loss": best_loss,
                "frames": int(mel.shape[1]),
                "cond_frames": int(cond_frames),
                "duration_jitter_frames": duration_jitter_frames,
                "convert_text_to_pinyin": bool(args.convert_text_to_pinyin),
            }
            print(json.dumps(metrics_row), flush=True)
            append_jsonl(metrics_path, {"event": "train", "time": time.time(), **metrics_row})
        if step % int(args.save_every) == 0:
            path = output_dir / f"model_q4_{int(args.teacher_steps)}to{int(args.student_steps)}_step_{step}.pt"
            save_checkpoint(student, path, step=step, args=args)
            print(f"saved={path}", flush=True)
            append_jsonl(metrics_path, {"event": "save", "time": time.time(), "step": step, "path": str(path)})
            if bool(args.prune_step_checkpoints):
                for old_path in output_dir.glob(f"model_q4_{int(args.teacher_steps)}to{int(args.student_steps)}_step_*.pt"):
                    if old_path == path:
                        continue
                    old_path.unlink(missing_ok=True)
                    append_jsonl(
                        metrics_path,
                        {"event": "prune_step_checkpoint", "time": time.time(), "step": step, "path": str(old_path)},
                    )

    final_path = output_dir / checkpoint_name(args, "last")
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
    parser.add_argument(
        "--split-probabilities",
        default="",
        help="Optional comma-separated sampling probabilities matching comma-separated streaming splits.",
    )
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=384)
    parser.add_argument("--cond-frames", type=int, default=256)
    parser.add_argument("--min-gen-frames", type=int, default=48)
    parser.add_argument("--generated-frame-jitter", type=int, default=0, help="Randomly shorten or extend generated mel frames during training for fixed-duration robustness.")
    parser.add_argument("--generated-frame-jitter-prob", type=float, default=0.0, help="Probability of applying generated-frame jitter to a batch.")
    parser.add_argument("--shuffle-buffer", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--student-steps", type=int, default=4)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--teacher-cfg-strength", type=float, default=None)
    parser.add_argument("--student-cfg-strength", type=float, default=None)
    parser.add_argument("--student-cfg-strengths", default="", help="Optional comma-separated deployment CFG values sampled per batch.")
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-flow-loss-weight", type=float, default=0.25)
    parser.add_argument("--cond-delta-loss-weight", type=float, default=0.0, help="Match the teacher conditional-minus-null CFG direction at student step states.")
    parser.add_argument("--segment-flow-loss-weight", type=float, default=0.0, help="Match average teacher flow over each coarse student interval.")
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.0, help="MSE against teacher intermediate rollout states at the student's coarse step times.")
    parser.add_argument("--real-mel-loss-weight", type=float, default=0.05)
    parser.add_argument("--anchor-weight-loss-weight", type=float, default=0.0)
    parser.add_argument("--temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--mel-energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--energy-envelope-loss-weight", type=float, default=0.0)
    parser.add_argument("--silence-envelope-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-temporal-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-mid-mel-body-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-mid-mel-end-bin", type=int, default=80)
    parser.add_argument("--text-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-contrastive-loss-weight-final", type=float, default=None)
    parser.add_argument("--text-contrastive-margin", type=float, default=0.08)
    parser.add_argument("--text-corruption-mode", default="reverse", choices=("reverse", "local", "adjacent", "hard-local", "roll", "shift"))
    parser.add_argument("--text-flow-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-flow-contrastive-loss-weight-final", type=float, default=None)
    parser.add_argument("--text-flow-contrastive-margin", type=float, default=0.04)
    parser.add_argument("--text-delta-loss-weight", type=float, default=0.0, help="Match teacher text sensitivity: flow(correct text) minus flow(corrupted text).")
    parser.add_argument("--text-delta-loss-weight-final", type=float, default=None)
    parser.add_argument("--high-mel-start-bin", type=int, default=80)
    parser.add_argument("--loss-schedule-steps", type=int, default=0)
    parser.add_argument("--rollout-loss-weight-final", type=float, default=None)
    parser.add_argument("--teacher-flow-loss-weight-final", type=float, default=None)
    parser.add_argument("--cond-delta-loss-weight-final", type=float, default=None)
    parser.add_argument("--segment-flow-loss-weight-final", type=float, default=None)
    parser.add_argument("--trajectory-loss-weight-final", type=float, default=None)
    parser.add_argument("--real-mel-loss-weight-final", type=float, default=None)
    parser.add_argument("--anchor-weight-loss-weight-final", type=float, default=None)
    parser.add_argument("--temporal-delta-loss-weight-final", type=float, default=None)
    parser.add_argument("--mel-energy-loss-weight-final", type=float, default=None)
    parser.add_argument("--energy-envelope-loss-weight-final", type=float, default=None)
    parser.add_argument("--silence-envelope-loss-weight-final", type=float, default=None)
    parser.add_argument("--high-mel-excess-loss-weight-final", type=float, default=None)
    parser.add_argument("--high-mel-match-loss-weight-final", type=float, default=None)
    parser.add_argument("--high-mel-ratio-loss-weight-final", type=float, default=None)
    parser.add_argument("--high-mel-temporal-loss-weight-final", type=float, default=None)
    parser.add_argument("--low-mid-mel-body-loss-weight-final", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--prune-step-checkpoints", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--q4-include", default="")
    parser.add_argument("--q4-exclude", default="text_embed.text_embed,mel_spec")
    parser.add_argument("--q4-ste-include", default="", help="Optional module-name filter for STE fake quant. Other Q4 modules are materialized once to reduce memory.")
    parser.add_argument(
        "--student-quant-scheme",
        choices=("q4", "bitnet_qat", "none"),
        default="q4",
        help="Student projection used during distillation. bitnet_qat keeps the F5 graph but replaces selected Linear layers with trainable BitNet STE layers.",
    )
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true", help="Train one runtime row scale per BitNet output channel during QAT.")
    parser.add_argument("--bitnet-scale-lr-multiplier", type=float, default=1.0, help="Optional LR multiplier for BitNet learned row-scale parameters.")
    parser.add_argument("--local-samples", default="/data/resumebot/voice_profiles/Peyton/samples.txt")
    parser.add_argument("--local-pairs", default="")
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
    parser.add_argument("--train-in-eval-mode", action="store_true", help="Disable dropout/stochastic training behavior while still computing gradients for distillation.")
    parser.add_argument("--convert-text-to-pinyin", action="store_true", help="Match F5TTS inference preprocessing by converting training text with convert_char_to_pinyin before tokenization.")
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
