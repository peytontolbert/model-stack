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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distill_f5tts_12_to_4_q4 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_VOCAB,
    apply_q4_parametrizations,
    anchor_weight_loss,
    build_model,
    cfg_flow,
    cfg_flow_with_delta,
    configure_trainable_parameters,
    corrupt_text_ids,
    load_checkpoint_state,
    make_time_grid,
    materialized_state_dict,
    parse_float_list,
    split_csv,
    trainable_anchor_parameters,
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def normalize_ctc_text(text: str) -> str:
    text = str(text or "").upper()
    text = re.sub(r"[^A-Z ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_ctc_text(*texts: str) -> str:
    joined = " ".join(str(text or "") for text in texts).lower()
    pieces: list[str] = []
    if "f5" in joined or "eff five" in joined or "f five" in joined:
        pieces.append("FIVE T T S")
    if "vocos" in joined or "voh cohs" in joined or "voh coes" in joined or "vo kose" in joined:
        pieces.extend(["VOH", "COES"])
    if "webgpu" in joined or "web g p u" in joined:
        pieces.append("WEB G P U")
    if "wasm" in joined or "webassembly" in joined or "web assembly" in joined:
        pieces.append("WEB ASSEMBLY")
    if not pieces:
        return ""
    return " ".join(pieces)


def domain_ctc_text(*texts: str) -> str:
    joined = " ".join(str(text or "") for text in texts).lower()
    if (
        ("f5" in joined or "eff five" in joined or "f five" in joined)
        and ("vocos" in joined or "voh cohs" in joined or "voh coes" in joined or "vo kose" in joined)
        and ("webgpu" in joined or "web g p u" in joined)
        and ("wasm" in joined or "webassembly" in joined or "web assembly" in joined)
    ):
        return "F FIVE T T S VOH COES WEB G P U AND WEB ASSEMBLY SHOULD ALL BE PRONOUNCED CLEARLY"
    return keyword_ctc_text(*texts)


def ctc_target_from_text(text: str, label_to_index: dict[str, int], device: torch.device) -> torch.Tensor:
    normalized = normalize_ctc_text(text)
    tokens: list[int] = []
    word_sep = "|" if "|" in label_to_index else " "
    for word_index, word in enumerate(normalized.split()):
        if word_index > 0 and word_sep in label_to_index:
            tokens.append(label_to_index[word_sep])
        for char in word:
            index = label_to_index.get(char)
            if index is not None:
                tokens.append(index)
    if not tokens:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(tokens, dtype=torch.long, device=device)


def decoded_waveform_ctc_loss(
    *,
    asr_model: nn.Module,
    asr_sample_rate: int,
    label_to_index: dict[str, int],
    blank_index: int,
    pred_wave: torch.Tensor,
    target_text: str,
    source_sample_rate: int,
) -> torch.Tensor:
    target = ctc_target_from_text(target_text, label_to_index, pred_wave.device)
    if int(target.numel()) == 0:
        return pred_wave.new_zeros(())
    wave = pred_wave
    if wave.ndim == 3:
        wave = wave.squeeze(1)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if int(source_sample_rate) != int(asr_sample_rate):
        target_size = max(1, int(round(wave.shape[-1] * float(asr_sample_rate) / float(source_sample_rate))))
        wave = F.interpolate(wave.unsqueeze(1), size=target_size, mode="linear", align_corners=False).squeeze(1)
    emissions, lengths = asr_model(wave)
    log_probs = F.log_softmax(emissions, dim=-1).transpose(0, 1)
    if lengths is None:
        input_lengths = torch.full(
            (int(emissions.shape[0]),),
            int(emissions.shape[1]),
            dtype=torch.long,
            device=pred_wave.device,
        )
    else:
        input_lengths = lengths.to(device=pred_wave.device, dtype=torch.long)
    target_lengths = torch.tensor([int(target.numel())], dtype=torch.long, device=pred_wave.device)
    return F.ctc_loss(
        log_probs,
        target.unsqueeze(0),
        input_lengths,
        target_lengths,
        blank=int(blank_index),
        reduction="mean",
        zero_infinity=True,
    )


def resample_waveform_linear(wave: torch.Tensor, source_sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if wave.ndim == 3:
        wave = wave.squeeze(1)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if int(source_sample_rate) == int(target_sample_rate):
        return wave
    target_size = max(1, int(round(wave.shape[-1] * float(target_sample_rate) / float(source_sample_rate))))
    return F.interpolate(wave.unsqueeze(1), size=target_size, mode="linear", align_corners=False).squeeze(1)


def compute_decoded_waveform_ssl_emission_match_loss(
    *,
    ssl_model: nn.Module,
    ssl_sample_rate: int,
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    source_sample_rate: int,
) -> torch.Tensor:
    pred = resample_waveform_linear(pred_wave, source_sample_rate, ssl_sample_rate)
    target = resample_waveform_linear(target_wave, source_sample_rate, ssl_sample_rate)
    pred_emissions, _ = ssl_model(pred)
    with torch.no_grad():
        target_emissions, _ = ssl_model(target)
    frames = min(int(pred_emissions.shape[1]), int(target_emissions.shape[1]))
    if frames <= 0:
        return pred_wave.new_zeros(())
    pred_features = F.layer_norm(pred_emissions[:, :frames, :], pred_emissions.shape[-1:])
    target_features = F.layer_norm(target_emissions[:, :frames, :], target_emissions.shape[-1:])
    return F.smooth_l1_loss(pred_features, target_features, beta=0.25)


def compute_decoded_waveform_wavlm_profile_loss(
    *,
    wavlm_model: nn.Module,
    wavlm_sample_rate: int,
    ref_embedding: torch.Tensor,
    pred_wave: torch.Tensor,
    source_sample_rate: int,
) -> torch.Tensor:
    pred = resample_waveform_linear(pred_wave, source_sample_rate, wavlm_sample_rate)
    features, _ = wavlm_model.extract_features(pred)
    pred_embedding = F.normalize(features[-1].mean(1).float(), dim=-1)
    target_embedding = F.normalize(ref_embedding.to(device=pred_embedding.device, dtype=pred_embedding.dtype), dim=-1)
    if target_embedding.ndim == 1:
        target_embedding = target_embedding.unsqueeze(0)
    return (1.0 - (pred_embedding * target_embedding).sum(dim=-1)).mean()


class ConditionalMelCritic(nn.Module):
    def __init__(self, mel_channels: int, hidden: int = 64) -> None:
        super().__init__()
        channels = int(mel_channels) * 4
        hidden = int(hidden)
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(channels, hidden, kernel_size=5, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden * 2, 1, kernel_size=1),
        )

    def forward(self, teacher_mel: torch.Tensor, candidate_mel: torch.Tensor) -> torch.Tensor:
        length = min(int(teacher_mel.shape[1]), int(candidate_mel.shape[1]))
        teacher = teacher_mel[:, :length, :].float()
        candidate = candidate_mel[:, :length, :].float()
        features = torch.cat(
            [
                teacher,
                candidate,
                candidate - teacher,
                (candidate - teacher).abs(),
            ],
            dim=-1,
        ).permute(0, 2, 1)
        return self.net(features).mean(dim=(1, 2))


def crop_sequence_frames(
    teacher_mel: torch.Tensor,
    candidate_mel: torch.Tensor,
    max_frames: int,
    *,
    random_crop: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(int(teacher_mel.shape[1]), int(candidate_mel.shape[1]))
    teacher_mel = teacher_mel[:, :length, :]
    candidate_mel = candidate_mel[:, :length, :]
    max_frames = int(max_frames)
    if max_frames <= 0 or length <= max_frames:
        return teacher_mel, candidate_mel
    start = random.randint(0, length - max_frames) if bool(random_crop) else (length - max_frames) // 2
    end = start + max_frames
    return teacher_mel[:, start:end, :], candidate_mel[:, start:end, :]


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def load_rows(cache_dir: Path, *, source_index: int = 0, sample_weight: float = 1.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata = cache_dir / "metadata.jsonl"
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "done":
            continue
        if "path" in row:
            row["_cache_dir"] = str(cache_dir)
            row["_source_index"] = int(source_index)
            row["_sample_weight"] = float(sample_weight)
            rows.append(row)
    if not rows:
        raise ValueError(f"no cached F5TTS teacher rows found in {metadata}")
    return rows


def parse_cache_dirs(value: str) -> list[Path]:
    cache_dirs = [Path(item).expanduser() for item in str(value).split(",") if item.strip()]
    if not cache_dirs:
        raise ValueError("--cache-dir did not contain any cache directories")
    return cache_dirs


def parse_cache_weights(value: str, count: int) -> list[float]:
    if not str(value).strip():
        return [1.0] * count
    weights = [float(item) for item in str(value).split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError(f"--cache-sampling-weights expected {count} values, got {len(weights)}")
    return weights


def row_payload_path(row: dict[str, Any], fallback_cache_dir: Path) -> Path:
    return Path(str(row.get("_cache_dir") or fallback_cache_dir)) / str(row["path"])


def load_focus_row_keys(path: str) -> set[tuple[str, str]]:
    if not str(path).strip():
        return set()
    focus_path = Path(path)
    if not focus_path.exists():
        raise FileNotFoundError(focus_path)
    if focus_path.suffix == ".json":
        data = json.loads(focus_path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("rows", [])
    else:
        rows = [json.loads(line) for line in focus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys: set[tuple[str, str]] = set()
    for row in rows:
        cache_dir = str(row.get("cache_dir") or row.get("_cache_dir") or "")
        sample_path = str(row.get("path") or "")
        if sample_path:
            keys.add((cache_dir, sample_path))
    return keys


def apply_focus_row_weights(
    rows: list[dict[str, Any]],
    *,
    focus_keys: set[tuple[str, str]],
    multiplier: float,
) -> int:
    if not focus_keys or float(multiplier) <= 0.0:
        return 0
    matched = 0
    path_only_keys = {path for cache_dir, path in focus_keys if not cache_dir}
    for row in rows:
        cache_dir = str(row.get("_cache_dir") or "")
        sample_path = str(row.get("path") or "")
        if (cache_dir, sample_path) in focus_keys or sample_path in path_only_keys:
            row["_sample_weight"] = float(row.get("_sample_weight", 1.0)) * float(multiplier)
            matched += 1
    return matched


def load_negative_text_ids(
    *,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    current_path: str,
    fallback: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if len(rows) <= 1:
        return corrupt_text_ids(fallback)
    for _ in range(8):
        row = random.choice(rows)
        if str(row.get("path")) == str(current_path):
            continue
        payload = torch.load(row_payload_path(row, cache_dir), map_location="cpu")
        return payload["text_ids"].to(device=device)
    return corrupt_text_ids(fallback)


def load_negative_condition(
    *,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    current_path: str,
    current_speaker: str,
    fallback: torch.Tensor,
    cond_frames: int,
    device: torch.device,
) -> torch.Tensor:
    if len(rows) <= 1 or int(cond_frames) <= 0:
        return fallback
    for _ in range(16):
        row = random.choice(rows)
        if str(row.get("path")) == str(current_path):
            continue
        if current_speaker and str(row.get("speaker_id", "")) == current_speaker:
            continue
        payload = torch.load(row_payload_path(row, cache_dir), map_location="cpu")
        source_cond = payload["cond"].to(device=device, dtype=fallback.dtype)
        source_frames = min(int(cond_frames), int(source_cond.shape[1]), int(fallback.shape[1]))
        if source_frames <= 0:
            continue
        bad_cond = fallback.clone()
        bad_cond[:, :source_frames, :] = source_cond[:, :source_frames, :]
        return bad_cond
    return fallback


def nearest_state_index(times: list[float], target: float) -> int:
    return min(range(len(times)), key=lambda idx: abs(float(times[idx]) - float(target)))


def direction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_flat = F.normalize(pred[mask].float().reshape(1, -1), dim=-1)
    target_flat = F.normalize(target[mask].float().reshape(1, -1), dim=-1)
    return 1.0 - (pred_flat * target_flat).sum()


def norm_ratio_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_norm = pred[mask].float().norm()
    target_norm = target[mask].float().norm().clamp_min(1e-6)
    return F.mse_loss(pred_norm / target_norm, torch.ones((), device=pred.device))


def orthogonal_flow_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_flat = pred[mask].float().reshape(-1)
    target_flat = target[mask].float().reshape(-1)
    target_energy = target_flat.pow(2).mean().clamp_min(1e-8)
    projection = target_flat * (pred_flat.dot(target_flat) / target_flat.dot(target_flat).clamp_min(1e-8))
    return (pred_flat - projection).pow(2).mean() / target_energy


def maybe_augment_conditioning(
    cond: torch.Tensor,
    *,
    cond_frames: int,
    prob: float,
    noise_std: float,
    high_band_noise_std: float,
    high_band_start_bin: int,
    gain_jitter: float,
    bias: float,
    high_band_gain: float,
    smooth_prob: float,
    smooth_kernel: int,
) -> torch.Tensor:
    if float(prob) <= 0.0 or random.random() >= float(prob):
        return cond
    out = cond.clone()
    prefix = out[:, : int(cond_frames), :]
    if float(bias) != 0.0:
        prefix.add_(float(bias))
    if float(gain_jitter) > 0.0:
        prefix.mul_(1.0 + random.uniform(-float(gain_jitter), float(gain_jitter)))
    if float(high_band_gain) != 1.0 and int(high_band_start_bin) < prefix.shape[-1]:
        prefix[:, :, int(high_band_start_bin) :].mul_(float(high_band_gain))
    if float(noise_std) > 0.0:
        prefix.add_(torch.randn_like(prefix) * float(noise_std))
    if float(high_band_noise_std) > 0.0 and int(high_band_start_bin) < prefix.shape[-1]:
        high = prefix[:, :, int(high_band_start_bin) :]
        high.add_(torch.randn_like(high) * float(high_band_noise_std))
    if float(smooth_prob) > 0.0 and random.random() < float(smooth_prob) and int(smooth_kernel) > 1:
        kernel = int(smooth_kernel)
        if kernel % 2 == 0:
            kernel += 1
        smoothed = F.avg_pool1d(
            prefix.transpose(1, 2),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
            count_include_pad=False,
        ).transpose(1, 2)
        prefix.copy_(smoothed)
    out[:, : int(cond_frames), :] = prefix
    return out


def temporal_delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if not bool(pair_mask.any()):
        return pred.new_zeros(())
    pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
    target_delta = target[:, 1:, :] - target[:, :-1, :]
    return F.smooth_l1_loss(pred_delta[pair_mask], target_delta[pair_mask], beta=0.2)


def frame_energy_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return pred.new_zeros(())
    pred_energy = pred.float().pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    target_energy = target.float().pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    return F.smooth_l1_loss(pred_energy[mask], target_energy[mask], beta=0.05)


def high_band_match_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, start_bin: int) -> torch.Tensor:
    if not bool(mask.any()) or int(start_bin) >= pred.shape[-1]:
        return pred.new_zeros(())
    pred_band = pred[..., int(start_bin) :]
    target_band = target[..., int(start_bin) :]
    return F.smooth_l1_loss(pred_band[mask], target_band[mask], beta=0.2)


def high_band_temporal_delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    start_bin: int,
) -> torch.Tensor:
    if int(start_bin) >= pred.shape[-1]:
        return pred.new_zeros(())
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if not bool(pair_mask.any()):
        return pred.new_zeros(())
    pred_band = pred[..., int(start_bin) :]
    target_band = target[..., int(start_bin) :]
    pred_delta = pred_band[:, 1:, :] - pred_band[:, :-1, :]
    target_delta = target_band[:, 1:, :] - target_band[:, :-1, :]
    return F.smooth_l1_loss(pred_delta[pair_mask], target_delta[pair_mask], beta=0.1)


def high_band_second_delta_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    start_bin: int,
) -> torch.Tensor:
    if int(start_bin) >= pred.shape[-1]:
        return pred.new_zeros(())
    triple_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    if not bool(triple_mask.any()):
        return pred.new_zeros(())
    pred_band = pred[..., int(start_bin) :]
    target_band = target[..., int(start_bin) :]
    pred_second = pred_band[:, 2:, :] - 2.0 * pred_band[:, 1:-1, :] + pred_band[:, :-2, :]
    target_second = target_band[:, 2:, :] - 2.0 * target_band[:, 1:-1, :] + target_band[:, :-2, :]
    return F.smooth_l1_loss(pred_second[triple_mask], target_second[triple_mask], beta=0.1)


def high_band_excess_ratio_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    start_bin: int,
    margin: float,
) -> torch.Tensor:
    if not bool(mask.any()) or int(start_bin) >= pred.shape[-1]:
        return pred.new_zeros(())
    pred_power = pred.float().pow(2)
    target_power = target.float().pow(2)
    pred_total = pred_power.mean(dim=-1).clamp_min(1e-6)
    target_total = target_power.mean(dim=-1).clamp_min(1e-6)
    pred_ratio = pred_power[..., int(start_bin) :].mean(dim=-1) / pred_total
    target_ratio = target_power[..., int(start_bin) :].mean(dim=-1) / target_total
    excess = F.relu(pred_ratio - target_ratio - float(margin))
    return F.smooth_l1_loss(excess[mask], torch.zeros_like(excess[mask]), beta=0.02)


def waveform_high_ratio_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    pred_power = pred_stft.abs().pow(2)
    target_power = target_stft.abs().pow(2)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_total = pred_power.mean(dim=1).clamp_min(1e-8)
    target_total = target_power.mean(dim=1).clamp_min(1e-8)
    pred_ratio = pred_power[:, high_mask, :].mean(dim=1) / pred_total
    target_ratio = target_power[:, high_mask, :].mean(dim=1) / target_total
    excess = F.relu(pred_ratio - target_ratio.detach() - float(margin))
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.02)


def waveform_high_ratio_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    pred_power = pred_stft.abs().pow(2)
    target_power = target_stft.abs().pow(2)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_total = pred_power.mean(dim=1).clamp_min(1e-8)
    target_total = target_power.mean(dim=1).clamp_min(1e-8)
    pred_ratio = pred_power[:, high_mask, :].mean(dim=1) / pred_total
    target_ratio = target_power[:, high_mask, :].mean(dim=1) / target_total
    return F.smooth_l1_loss(pred_ratio, target_ratio.detach(), beta=0.02)


def waveform_high_ratio_topk_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
    topk_fraction: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    pred_power = pred_stft.abs().pow(2)
    target_power = target_stft.abs().pow(2)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_total = pred_power.mean(dim=1).clamp_min(1e-8)
    target_total = target_power.mean(dim=1).clamp_min(1e-8)
    pred_ratio = pred_power[:, high_mask, :].mean(dim=1) / pred_total
    target_ratio = target_power[:, high_mask, :].mean(dim=1) / target_total
    excess = F.relu(pred_ratio - target_ratio.detach() - float(margin))
    return topk_mean(excess, float(topk_fraction))


def waveform_global_high_ratio(wave: torch.Tensor, sample_rate: int, cutoff_hz: float) -> torch.Tensor:
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if int(wave.shape[-1]) < 2048:
        return wave.new_zeros((wave.shape[0],), dtype=torch.float32)
    wave = wave.float()
    spectrum = torch.fft.rfft(wave, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(int(wave.shape[-1]), d=1.0 / float(sample_rate)).to(wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return wave.new_zeros((wave.shape[0],), dtype=torch.float32)
    return spectrum[:, high_mask].sum(dim=-1) / spectrum.sum(dim=-1).clamp_min(1e-8)


def waveform_global_high_ratio_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_ratio = waveform_global_high_ratio(pred_wave[..., :length], sample_rate, cutoff_hz)
    target_ratio = waveform_global_high_ratio(target_wave[..., :length], sample_rate, cutoff_hz)
    excess = F.relu(pred_ratio - target_ratio.detach() - float(margin))
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.005)


def waveform_global_high_ratio_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_ratio = waveform_global_high_ratio(pred_wave[..., :length], sample_rate, cutoff_hz)
    target_ratio = waveform_global_high_ratio(target_wave[..., :length], sample_rate, cutoff_hz)
    return F.smooth_l1_loss(pred_ratio, target_ratio.detach(), beta=0.005)


def waveform_spectral_centroid_values(
    wave: torch.Tensor,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if int(wave.shape[-1]) < int(n_fft):
        return wave.new_zeros((wave.shape[0], 0), dtype=torch.float32)
    wave = wave.float()
    window = torch.hann_window(int(n_fft), device=wave.device, dtype=wave.dtype)
    stft = torch.stft(
        wave,
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        win_length=int(n_fft),
        window=window,
        return_complex=True,
    )
    power = stft.abs().pow(2)
    freqs = torch.fft.rfftfreq(int(n_fft), d=1.0 / float(sample_rate)).to(wave.device)
    return (power * freqs.view(1, -1, 1)).sum(dim=1) / power.sum(dim=1).clamp_min(1e-8)


def waveform_spectral_centroid_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_centroid = waveform_spectral_centroid_values(pred_wave[..., :length], sample_rate)
    target_centroid = waveform_spectral_centroid_values(target_wave[..., :length], sample_rate)
    frame_count = min(int(pred_centroid.shape[-1]), int(target_centroid.shape[-1]))
    if frame_count <= 0:
        return pred_wave.new_zeros(())
    scale = float(sample_rate) * 0.5
    return F.smooth_l1_loss(
        pred_centroid[..., :frame_count] / scale,
        target_centroid[..., :frame_count].detach() / scale,
        beta=0.02,
    )


def waveform_spectral_centroid_topk_drift_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    topk_fraction: float,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_centroid = waveform_spectral_centroid_values(pred_wave[..., :length], sample_rate)
    target_centroid = waveform_spectral_centroid_values(target_wave[..., :length], sample_rate)
    frame_count = min(int(pred_centroid.shape[-1]), int(target_centroid.shape[-1]))
    if frame_count <= 0:
        return pred_wave.new_zeros(())
    drift = (pred_centroid[..., :frame_count] - target_centroid[..., :frame_count].detach()).abs()
    return topk_mean(drift / (float(sample_rate) * 0.5), float(topk_fraction))


def waveform_spectral_centroid_topk_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    margin_hz: float,
    topk_fraction: float,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_centroid = waveform_spectral_centroid_values(pred_wave[..., :length], sample_rate)
    target_centroid = waveform_spectral_centroid_values(target_wave[..., :length], sample_rate)
    frame_count = min(int(pred_centroid.shape[-1]), int(target_centroid.shape[-1]))
    if frame_count <= 0:
        return pred_wave.new_zeros(())
    excess = F.relu(pred_centroid[..., :frame_count] - target_centroid[..., :frame_count].detach() - float(margin_hz))
    return topk_mean(excess / (float(sample_rate) * 0.5), float(topk_fraction))


def waveform_teacher_conditioned_drift_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    ratio_margin: float,
    centroid_margin_hz: float,
    topk_fraction: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(
        target_wave,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        return_complex=True,
    )
    pred_power = pred_stft.abs().pow(2)
    target_power = target_stft.abs().pow(2)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_total = pred_power.mean(dim=1).clamp_min(1e-8)
    target_total = target_power.mean(dim=1).clamp_min(1e-8)
    pred_ratio = pred_power[:, high_mask, :].mean(dim=1) / pred_total
    target_ratio = target_power[:, high_mask, :].mean(dim=1) / target_total
    pred_centroid = (pred_power * freqs.view(1, -1, 1)).sum(dim=1) / pred_power.sum(dim=1).clamp_min(1e-8)
    target_centroid = (target_power * freqs.view(1, -1, 1)).sum(dim=1) / target_power.sum(dim=1).clamp_min(1e-8)
    frame_count = min(int(pred_ratio.shape[-1]), int(target_ratio.shape[-1]), int(pred_centroid.shape[-1]), int(target_centroid.shape[-1]))
    if frame_count <= 0:
        return pred_wave.new_zeros(())
    ratio_excess = F.relu(pred_ratio[..., :frame_count] - target_ratio[..., :frame_count].detach() - float(ratio_margin))
    centroid_excess = F.relu(
        pred_centroid[..., :frame_count] - target_centroid[..., :frame_count].detach() - float(centroid_margin_hz)
    ) / (float(sample_rate) * 0.5)
    joint_excess = torch.sqrt((ratio_excess * centroid_excess).clamp_min(0.0) + 1.0e-12)
    return topk_mean(joint_excess, float(topk_fraction))


def waveform_high_flatness_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_high = pred_stft[:, high_mask, :].abs().clamp_min(1e-10)
    target_high = target_stft[:, high_mask, :].abs().clamp_min(1e-10)
    pred_flatness = torch.exp(torch.log(pred_high).mean(dim=1)) / pred_high.mean(dim=1).clamp_min(1e-10)
    target_flatness = torch.exp(torch.log(target_high).mean(dim=1)) / target_high.mean(dim=1).clamp_min(1e-10)
    excess = F.relu(pred_flatness - target_flatness.detach() - float(margin))
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.02)


def waveform_high_flatness_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_high = pred_stft[:, high_mask, :].abs().clamp_min(1e-10)
    target_high = target_stft[:, high_mask, :].abs().clamp_min(1e-10)
    pred_flatness = torch.exp(torch.log(pred_high).mean(dim=1)) / pred_high.mean(dim=1).clamp_min(1e-10)
    target_flatness = torch.exp(torch.log(target_high).mean(dim=1)) / target_high.mean(dim=1).clamp_min(1e-10)
    return F.smooth_l1_loss(pred_flatness, target_flatness.detach(), beta=0.02)


def waveform_high_flatness_topk_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
    topk_fraction: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_high = pred_stft[:, high_mask, :].abs().clamp_min(1e-10)
    target_high = target_stft[:, high_mask, :].abs().clamp_min(1e-10)
    pred_flatness = torch.exp(torch.log(pred_high).mean(dim=1)) / pred_high.mean(dim=1).clamp_min(1e-10)
    target_flatness = torch.exp(torch.log(target_high).mean(dim=1)) / target_high.mean(dim=1).clamp_min(1e-10)
    excess = F.relu(pred_flatness - target_flatness.detach() - float(margin))
    return topk_mean(excess, float(topk_fraction))


def waveform_weighted_high_flatness_topk_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
    topk_fraction: float,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 2048:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(1024, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=1024, hop_length=256, win_length=1024, window=window, return_complex=True)
    pred_mag = pred_stft.abs().clamp_min(1e-10)
    target_mag = target_stft.abs().clamp_min(1e-10)
    pred_power = pred_mag.pow(2)
    target_power = target_mag.pow(2)
    freqs = torch.fft.rfftfreq(1024, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_total = pred_power.mean(dim=1).clamp_min(1e-8)
    target_total = target_power.mean(dim=1).clamp_min(1e-8)
    pred_high_ratio = pred_power[:, high_mask, :].mean(dim=1) / pred_total
    target_high_ratio = target_power[:, high_mask, :].mean(dim=1) / target_total
    pred_high = pred_mag[:, high_mask, :]
    target_high = target_mag[:, high_mask, :]
    pred_flatness = torch.exp(torch.log(pred_high).mean(dim=1)) / pred_high.mean(dim=1).clamp_min(1e-10)
    target_flatness = torch.exp(torch.log(target_high).mean(dim=1)) / target_high.mean(dim=1).clamp_min(1e-10)
    pred_weighted = pred_high_ratio * pred_flatness
    target_weighted = target_high_ratio * target_flatness
    excess = F.relu(pred_weighted - target_weighted.detach() - float(margin))
    return topk_mean(excess, float(topk_fraction))


def waveform_rms_match_loss(pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 512:
        return pred_wave.new_zeros(())
    pred = pred_wave[..., :length].float()
    target = target_wave[..., :length].float()
    pred_rms = pred.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    target_rms = target.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    return F.smooth_l1_loss(pred_rms, target_rms.detach(), beta=0.01)


def waveform_derivative_rms_match_loss(pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 513:
        return pred_wave.new_zeros(())
    pred = pred_wave[..., :length].float()
    target = target_wave[..., :length].float()
    pred_diff_rms = (pred[..., 1:] - pred[..., :-1]).pow(2).mean(dim=-1).clamp_min(1e-10).sqrt()
    target_diff_rms = (target[..., 1:] - target[..., :-1]).pow(2).mean(dim=-1).clamp_min(1e-10).sqrt()
    return F.smooth_l1_loss(pred_diff_rms, target_diff_rms.detach(), beta=0.005)


def waveform_derivative_rms_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 513:
        return pred_wave.new_zeros(())
    pred = pred_wave[..., :length].float()
    target = target_wave[..., :length].float()
    pred_diff_rms = (pred[..., 1:] - pred[..., :-1]).pow(2).mean(dim=-1).clamp_min(1e-10).sqrt()
    target_diff_rms = (target[..., 1:] - target[..., :-1]).pow(2).mean(dim=-1).clamp_min(1e-10).sqrt()
    return F.relu(pred_diff_rms - target_diff_rms.detach() - float(margin)).mean()


def waveform_soft_zcr(
    wave: torch.Tensor,
    *,
    gain: float = 80.0,
) -> torch.Tensor:
    if int(wave.shape[-1]) < 2:
        return wave.new_zeros((int(wave.shape[0]),))
    softened_sign = torch.tanh(wave.float() * float(gain))
    crossings = 0.5 * (1.0 - softened_sign[..., 1:] * softened_sign[..., :-1])
    return crossings.mean(dim=-1)


def waveform_soft_zcr_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    gain: float = 80.0,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 512:
        return pred_wave.new_zeros(())
    pred_rate = waveform_soft_zcr(pred_wave[..., :length], gain=float(gain))
    target_rate = waveform_soft_zcr(target_wave[..., :length], gain=float(gain))
    return F.smooth_l1_loss(pred_rate, target_rate.detach(), beta=0.005)


def waveform_soft_zcr_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    gain: float = 80.0,
    margin: float = 0.0,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 512:
        return pred_wave.new_zeros(())
    pred_rate = waveform_soft_zcr(pred_wave[..., :length], gain=float(gain))
    target_rate = waveform_soft_zcr(target_wave[..., :length], gain=float(gain))
    return F.relu(pred_rate - target_rate.detach() - float(margin)).mean()


def waveform_framed_soft_zcr_rates(
    wave: torch.Tensor,
    *,
    window_size: int = 2048,
    hop_size: int = 512,
    gain: float = 80.0,
) -> torch.Tensor:
    window_size = int(window_size)
    hop_size = int(hop_size)
    if int(wave.shape[-1]) < window_size or window_size < 2 or hop_size <= 0:
        return wave.new_zeros((int(wave.shape[0]), 0))
    frames = wave.float().unfold(-1, window_size, hop_size)
    softened_sign = torch.tanh(frames * float(gain))
    crossings = 0.5 * (1.0 - softened_sign[..., 1:] * softened_sign[..., :-1])
    return crossings.mean(dim=-1)


def waveform_framed_soft_zcr_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    window_size: int = 2048,
    hop_size: int = 512,
    gain: float = 80.0,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < int(window_size):
        return pred_wave.new_zeros(())
    pred_rate = waveform_framed_soft_zcr_rates(
        pred_wave[..., :length],
        window_size=int(window_size),
        hop_size=int(hop_size),
        gain=float(gain),
    )
    target_rate = waveform_framed_soft_zcr_rates(
        target_wave[..., :length],
        window_size=int(window_size),
        hop_size=int(hop_size),
        gain=float(gain),
    )
    if int(pred_rate.shape[-1]) == 0 or int(target_rate.shape[-1]) == 0:
        return pred_wave.new_zeros(())
    return F.smooth_l1_loss(pred_rate, target_rate.detach(), beta=0.005)


def waveform_framed_soft_zcr_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    window_size: int = 2048,
    hop_size: int = 512,
    gain: float = 80.0,
    margin: float = 0.0,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < int(window_size):
        return pred_wave.new_zeros(())
    pred_rate = waveform_framed_soft_zcr_rates(
        pred_wave[..., :length],
        window_size=int(window_size),
        hop_size=int(hop_size),
        gain=float(gain),
    )
    target_rate = waveform_framed_soft_zcr_rates(
        target_wave[..., :length],
        window_size=int(window_size),
        hop_size=int(hop_size),
        gain=float(gain),
    )
    if int(pred_rate.shape[-1]) == 0 or int(target_rate.shape[-1]) == 0:
        return pred_wave.new_zeros(())
    return F.relu(pred_rate - target_rate.detach() - float(margin)).mean()


def topk_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    flat = values.reshape(-1)
    if int(flat.numel()) == 0:
        return values.new_zeros(())
    k = max(1, int(round(float(flat.numel()) * float(fraction))))
    k = min(k, int(flat.numel()))
    return torch.topk(flat, k=k, largest=True).values.mean()


def waveform_framed_soft_zcr_topk_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    window_size: int = 2048,
    hop_size: int = 512,
    gain: float = 80.0,
    margin: float = 0.0,
    topk_fraction: float = 0.2,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < int(window_size):
        return pred_wave.new_zeros(())
    pred_rate = waveform_framed_soft_zcr_rates(
        pred_wave[..., :length],
        window_size=int(window_size),
        hop_size=int(hop_size),
        gain=float(gain),
    )
    target_rate = waveform_framed_soft_zcr_rates(
        target_wave[..., :length],
        window_size=int(window_size),
        hop_size=int(hop_size),
        gain=float(gain),
    )
    if int(pred_rate.shape[-1]) == 0 or int(target_rate.shape[-1]) == 0:
        return pred_wave.new_zeros(())
    excess = F.relu(pred_rate - target_rate.detach() - float(margin))
    return topk_mean(excess, float(topk_fraction))


def waveform_weighted_high_zcr_topk_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    window_size: int = 1024,
    hop_size: int = 256,
    gain: float = 80.0,
    margin: float = 0.0,
    topk_fraction: float = 0.2,
) -> torch.Tensor:
    if pred_wave.ndim == 1:
        pred_wave = pred_wave.unsqueeze(0)
    if target_wave.ndim == 1:
        target_wave = target_wave.unsqueeze(0)
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    window_size = int(window_size)
    hop_size = int(hop_size)
    if length < window_size or window_size < 2 or hop_size <= 0:
        return pred_wave.new_zeros(())
    pred_wave = pred_wave[..., :length].float()
    target_wave = target_wave[..., :length].float()
    window = torch.hann_window(window_size, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(
        pred_wave,
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        return_complex=True,
    )
    target_stft = torch.stft(
        target_wave,
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        return_complex=True,
    )
    pred_power = pred_stft.abs().pow(2)
    target_power = target_stft.abs().pow(2)
    freqs = torch.fft.rfftfreq(window_size, d=1.0 / float(sample_rate)).to(pred_wave.device)
    high_mask = freqs >= float(cutoff_hz)
    if not bool(high_mask.any()):
        return pred_wave.new_zeros(())
    pred_total = pred_power.mean(dim=1).clamp_min(1e-8)
    target_total = target_power.mean(dim=1).clamp_min(1e-8)
    pred_high_ratio = pred_power[:, high_mask, :].mean(dim=1) / pred_total
    target_high_ratio = target_power[:, high_mask, :].mean(dim=1) / target_total
    pred_zcr = waveform_framed_soft_zcr_rates(pred_wave, window_size=window_size, hop_size=hop_size, gain=float(gain))
    target_zcr = waveform_framed_soft_zcr_rates(target_wave, window_size=window_size, hop_size=hop_size, gain=float(gain))
    frame_count = min(int(pred_high_ratio.shape[-1]), int(target_high_ratio.shape[-1]), int(pred_zcr.shape[-1]), int(target_zcr.shape[-1]))
    if frame_count <= 0:
        return pred_wave.new_zeros(())
    pred_weighted = pred_high_ratio[..., :frame_count] * pred_zcr[..., :frame_count]
    target_weighted = target_high_ratio[..., :frame_count] * target_zcr[..., :frame_count]
    excess = F.relu(pred_weighted - target_weighted.detach() - float(margin))
    return topk_mean(excess, float(topk_fraction))


def waveform_rms_underfill_topk_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    window_size: int = 2048,
    hop_size: int = 512,
    topk_fraction: float = 0.2,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    window_size = int(window_size)
    hop_size = int(hop_size)
    if length < window_size or window_size <= 0 or hop_size <= 0:
        return pred_wave.new_zeros(())
    pred_frames = pred_wave[..., :length].float().unfold(-1, window_size, hop_size)
    target_frames = target_wave[..., :length].float().unfold(-1, window_size, hop_size)
    pred_rms = pred_frames.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    target_rms = target_frames.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    voiced = target_rms.detach() > 0.01
    underfill = F.relu((target_rms.detach() - pred_rms) / target_rms.detach().clamp_min(1e-4))
    if not bool(voiced.any()):
        return pred_wave.new_zeros(())
    return topk_mean(underfill[voiced], float(topk_fraction))


def waveform_rms_envelope_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    window_size: int = 2048,
    hop_size: int = 512,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    window_size = int(window_size)
    hop_size = int(hop_size)
    if length < window_size or window_size <= 0 or hop_size <= 0:
        return pred_wave.new_zeros(())
    pred = pred_wave[..., :length].float()
    target = target_wave[..., :length].float()
    pred_frames = pred.unfold(-1, window_size, hop_size)
    target_frames = target.unfold(-1, window_size, hop_size)
    pred_rms = pred_frames.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    target_rms = target_frames.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt()
    return F.smooth_l1_loss(pred_rms, target_rms.detach(), beta=0.01)


def waveform_rms_envelope_second_delta_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    window_size: int = 2048,
    hop_size: int = 512,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    window_size = int(window_size)
    hop_size = int(hop_size)
    if length < window_size or window_size <= 0 or hop_size <= 0:
        return pred_wave.new_zeros(())
    pred = pred_wave[..., :length].float()
    target = target_wave[..., :length].float()
    pred_frames = pred.unfold(-1, window_size, hop_size)
    target_frames = target.unfold(-1, window_size, hop_size)
    pred_rms = pred_frames.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt().log()
    target_rms = target_frames.pow(2).mean(dim=-1).clamp_min(1e-8).sqrt().log()
    if pred_rms.shape[-1] < 3 or target_rms.shape[-1] < 3:
        return pred_wave.new_zeros(())
    pred_second = pred_rms[..., 2:] - 2.0 * pred_rms[..., 1:-1] + pred_rms[..., :-2]
    target_second = target_rms[..., 2:] - 2.0 * target_rms[..., 1:-1] + target_rms[..., :-2]
    return F.smooth_l1_loss(pred_second, target_second.detach(), beta=0.02)


def waveform_multires_stft_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    fft_sizes: str = "512,1024,2048",
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 512:
        return pred_wave.new_zeros(())
    pred = pred_wave[..., :length].float()
    target = target_wave[..., :length].float()
    losses: list[torch.Tensor] = []
    for item in str(fft_sizes).split(","):
        if not item.strip():
            continue
        n_fft = int(item)
        if n_fft <= 0 or length < n_fft:
            continue
        hop = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        pred_mag = torch.stft(
            pred,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        target_mag = torch.stft(
            target.detach(),
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        log_loss = F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag.detach()))
        spectral_convergence = (pred_mag - target_mag.detach()).norm(p="fro") / target_mag.detach().norm(p="fro").clamp_min(1e-6)
        losses.append(log_loss + 0.1 * spectral_convergence)
    if not losses:
        return pred_wave.new_zeros(())
    return torch.stack(losses).mean()


def waveform_spectral_envelope_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    n_fft: int,
    hop: int,
    smooth_bins: int,
) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < int(n_fft):
        return pred_wave.new_zeros(())
    pred = pred_wave[..., :length].float().reshape(-1, length)
    target = target_wave[..., :length].float().reshape(-1, length)
    window = torch.hann_window(int(n_fft), device=pred.device, dtype=pred.dtype)
    pred_spec = torch.stft(pred, n_fft=int(n_fft), hop_length=int(hop), window=window, return_complex=True)
    target_spec = torch.stft(target, n_fft=int(n_fft), hop_length=int(hop), window=window, return_complex=True)
    pred_log = torch.log(pred_spec.abs().square().clamp_min(1.0e-8))
    target_log = torch.log(target_spec.abs().square().clamp_min(1.0e-8))
    kernel = max(1, int(smooth_bins))
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        pred_log = F.avg_pool1d(pred_log, kernel_size=kernel, stride=1, padding=pad)
        target_log = F.avg_pool1d(target_log, kernel_size=kernel, stride=1, padding=pad)
    pred_env = pred_log - pred_log.mean(dim=1, keepdim=True)
    target_env = target_log - target_log.mean(dim=1, keepdim=True)
    return F.smooth_l1_loss(pred_env, target_env, beta=0.25)


def waveform_peak_excess_loss(pred_wave: torch.Tensor, target_wave: torch.Tensor, margin: float) -> torch.Tensor:
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    if length < 512:
        return pred_wave.new_zeros(())
    pred_peak = pred_wave[..., :length].float().abs().amax(dim=-1)
    target_peak = target_wave[..., :length].float().abs().amax(dim=-1).detach()
    excess = F.relu(pred_peak - target_peak - float(margin))
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.01)


def waveform_peak_cap_loss(pred_wave: torch.Tensor, cap: float) -> torch.Tensor:
    if int(pred_wave.shape[-1]) < 512 or float(cap) <= 0.0:
        return pred_wave.new_zeros(())
    pred_peak = pred_wave.float().abs().amax(dim=-1)
    excess = F.relu(pred_peak - float(cap))
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.01)


def crop_wave_frames(wave: torch.Tensor, start_frame: int, end_frame: int, hop_length: int = 256) -> torch.Tensor:
    start = max(0, int(start_frame) * int(hop_length))
    end = max(start, int(end_frame) * int(hop_length))
    if end <= start or start >= int(wave.shape[-1]):
        return wave[..., :0]
    return wave[..., start : min(end, int(wave.shape[-1]))]


def crop_wave_ratio(wave: torch.Tensor, start_ratio: float, end_ratio: float) -> torch.Tensor:
    length = int(wave.shape[-1])
    if length <= 0:
        return wave
    start_ratio = min(0.98, max(0.0, float(start_ratio)))
    end_ratio = min(1.0, max(start_ratio + 0.01, float(end_ratio)))
    start = min(length - 1, max(0, int(length * start_ratio)))
    end = min(length, max(start + 1, int(length * end_ratio)))
    return wave[..., start:end]


def crop_mel_frames(
    pred_mel: torch.Tensor,
    target_mel: torch.Tensor,
    max_frames: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    length = min(int(pred_mel.shape[-1]), int(target_mel.shape[-1]))
    max_frames = int(max_frames)
    if max_frames <= 0 or length <= max_frames:
        return pred_mel[..., :length], target_mel[..., :length], False
    mode = str(mode).lower()
    if mode == "skip":
        return pred_mel, target_mel, False
    if mode == "tail":
        start = length - max_frames
    elif mode == "mid":
        start = max(0, (length - max_frames) // 2)
    elif mode == "random":
        start = random.randint(0, max(0, length - max_frames))
    else:
        raise ValueError(f"unsupported decoded waveform crop mode: {mode}")
    end = start + max_frames
    return pred_mel[..., start:end], target_mel[..., start:end], True


def tail_loss_mask(mask: torch.Tensor, start_ratio: float) -> torch.Tensor:
    if not bool(mask.any()):
        return mask
    start_ratio = min(0.95, max(0.0, float(start_ratio)))
    tail = torch.zeros_like(mask)
    for batch_index in range(mask.shape[0]):
        indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        offset = int(indices.numel() * start_ratio)
        offset = min(max(0, offset), int(indices.numel()) - 1)
        tail[batch_index, indices[offset:]] = True
    return tail


def window_loss_mask(mask: torch.Tensor, start_ratio: float, end_ratio: float) -> torch.Tensor:
    if not bool(mask.any()):
        return mask
    start_ratio = min(0.98, max(0.0, float(start_ratio)))
    end_ratio = min(1.0, max(start_ratio + 0.01, float(end_ratio)))
    window = torch.zeros_like(mask)
    for batch_index in range(mask.shape[0]):
        indices = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        start = int(indices.numel() * start_ratio)
        end = int(indices.numel() * end_ratio)
        start = min(max(0, start), int(indices.numel()) - 1)
        end = min(max(start + 1, end), int(indices.numel()))
        window[batch_index, indices[start:end]] = True
    return window


def force_non_reentrant_checkpointing() -> None:
    import torch.utils.checkpoint as torch_checkpoint

    original_checkpoint = torch_checkpoint.checkpoint
    if getattr(original_checkpoint, "_f5tts_non_reentrant_default", False):
        return

    def checkpoint_with_non_reentrant_default(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return original_checkpoint(function, *args, **kwargs)

    checkpoint_with_non_reentrant_default._f5tts_non_reentrant_default = True  # type: ignore[attr-defined]
    torch_checkpoint.checkpoint = checkpoint_with_non_reentrant_default


def save_checkpoint(model, path: Path, *, step: int, args: argparse.Namespace, loss: float) -> None:
    payload = {
        "model_state_dict": materialized_state_dict(model),
        "step": int(step),
        "loss": float(loss),
        "args": vars(args),
        "distillation": {
            "mode": "f5tts_cached_teacher_bridge_2step",
            "teacher_steps": int(args.teacher_steps),
            "student_steps": int(args.student_steps),
            "teacher_cfg_strength": float(args.teacher_cfg_strength),
            "student_cfg_strength": float(args.student_cfg_strength),
            "sway_sampling_coef": float(args.sway_sampling_coef),
            "cache_dir": str(args.cache_dir),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def checkpoint_name(args: argparse.Namespace, suffix: str) -> str:
    return f"model_q4_{int(args.teacher_steps)}to{int(args.student_steps)}_{suffix}.pt"


def load_functional_anchor(args: argparse.Namespace, device: torch.device):
    anchor_checkpoint = str(args.anchor_checkpoint or args.student_checkpoint or "").strip()
    if not anchor_checkpoint:
        return None
    if float(args.anchor_flow_loss_weight) <= 0.0 and float(args.anchor_rollout_loss_weight) <= 0.0:
        return None
    anchor = build_model(Path(args.vocab), device)
    load_checkpoint_state(anchor, Path(anchor_checkpoint))
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    return anchor


@torch.no_grad()
def validation_rollout_loss(
    *,
    student,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    device: torch.device,
    student_times: list[float],
    cfg_strength: float,
) -> float:
    if not rows:
        return 0.0
    student_was_training = bool(student.training)
    student.eval()
    total = 0.0
    count = 0
    for row in rows:
        payload = torch.load(row_payload_path(row, cache_dir), map_location="cpu")
        cond = payload["cond"].to(device=device, dtype=torch.float32)
        y = payload["noise"].to(device=device, dtype=torch.float32).clone()
        text_ids = payload["text_ids"].to(device=device)
        loss_mask = payload["loss_mask"].to(device=device).bool()
        teacher_y = payload["teacher_y"].to(device=device, dtype=torch.float32)
        cond_frames = int(payload["cond_frames"])
        if not bool(loss_mask.any()):
            continue
        for interval in range(len(student_times) - 1):
            t0 = student_times[interval]
            dt = float(student_times[interval + 1] - student_times[interval])
            time_tensor = torch.full((y.shape[0],), t0, device=device, dtype=y.dtype)
            y = y + dt * cfg_flow(student, y, cond, text_ids, time_tensor, float(cfg_strength))
        if cond_frames > 0:
            y[:, :cond_frames, :] = cond[:, :cond_frames, :]
        total += float(F.mse_loss(y[loss_mask], teacher_y[loss_mask]).detach().cpu())
        count += 1
    if student_was_training:
        student.train()
    return total / float(max(1, count))


@torch.no_grad()
def validation_decoded_quality(
    *,
    student,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    device: torch.device,
    student_times: list[float],
    cfg_strength: float,
    decoded_vocoder,
    args,
) -> dict[str, float]:
    max_rows = int(args.val_decoded_quality_rows)
    if decoded_vocoder is None or max_rows <= 0 or not rows:
        return {}
    student_was_training = bool(student.training)
    student.eval()
    totals: dict[str, float] = {
        "decoded_quality_loss": 0.0,
        "decoded_waveform_high_ratio_loss": 0.0,
        "decoded_waveform_high_ratio_match_loss": 0.0,
        "decoded_waveform_high_ratio_topk_excess_loss": 0.0,
        "decoded_waveform_global_high_ratio_loss": 0.0,
        "decoded_waveform_global_high_ratio_match_loss": 0.0,
        "decoded_waveform_spectral_centroid_match_loss": 0.0,
        "decoded_waveform_spectral_centroid_topk_drift_loss": 0.0,
        "decoded_waveform_spectral_centroid_topk_excess_loss": 0.0,
        "decoded_waveform_teacher_conditioned_drift_loss": 0.0,
        "decoded_waveform_high_flatness_loss": 0.0,
        "decoded_waveform_high_flatness_match_loss": 0.0,
        "decoded_waveform_high_flatness_topk_excess_loss": 0.0,
        "decoded_waveform_weighted_high_flatness_topk_excess_loss": 0.0,
        "decoded_waveform_weighted_high_zcr_topk_excess_loss": 0.0,
        "decoded_waveform_rms_match_loss": 0.0,
        "decoded_waveform_derivative_rms_match_loss": 0.0,
        "decoded_waveform_derivative_rms_excess_loss": 0.0,
        "decoded_waveform_soft_zcr_match_loss": 0.0,
        "decoded_waveform_soft_zcr_excess_loss": 0.0,
        "decoded_waveform_framed_soft_zcr_match_loss": 0.0,
        "decoded_waveform_framed_soft_zcr_excess_loss": 0.0,
        "decoded_waveform_framed_soft_zcr_topk_excess_loss": 0.0,
        "decoded_waveform_rms_underfill_topk_loss": 0.0,
        "decoded_waveform_rms_envelope_match_loss": 0.0,
        "decoded_waveform_rms_envelope_second_delta_match_loss": 0.0,
        "decoded_waveform_multires_stft_loss": 0.0,
        "decoded_waveform_peak_cap_loss": 0.0,
    }
    count = 0
    for row in rows[:max_rows]:
        payload = torch.load(row_payload_path(row, cache_dir), map_location="cpu")
        cond = payload["cond"].to(device=device, dtype=torch.float32)
        y = payload["noise"].to(device=device, dtype=torch.float32).clone()
        text_ids = payload["text_ids"].to(device=device)
        loss_mask = payload["loss_mask"].to(device=device).bool()
        teacher_y = payload["teacher_y"].to(device=device, dtype=torch.float32)
        cond_frames = int(payload["cond_frames"])
        if not bool(loss_mask.any()):
            continue
        for interval in range(len(student_times) - 1):
            t0 = student_times[interval]
            dt = float(student_times[interval + 1] - student_times[interval])
            time_tensor = torch.full((y.shape[0],), t0, device=device, dtype=y.dtype)
            y = y + dt * cfg_flow(student, y, cond, text_ids, time_tensor, float(cfg_strength))
        if cond_frames > 0:
            y[:, :cond_frames, :] = cond[:, :cond_frames, :]
        if bool(args.decoded_waveform_generated_only):
            pred_mel = y[:, cond_frames:, :].permute(0, 2, 1).float()
            target_mel = teacher_y[:, cond_frames:, :].permute(0, 2, 1).float()
        else:
            pred_mel = y.permute(0, 2, 1).float()
            target_mel = teacher_y.permute(0, 2, 1).float()
        pred_mel, target_mel, _ = crop_mel_frames(
            pred_mel,
            target_mel,
            int(args.decoded_waveform_max_frames),
            str(args.decoded_waveform_crop_mode),
        )
        if str(args.decoded_waveform_crop_mode) == "skip" and int(pred_mel.shape[-1]) > int(args.decoded_waveform_max_frames):
            continue
        pred_wave = decoded_vocoder.decode(pred_mel)
        target_wave = decoded_vocoder.decode(target_mel)
        if min(int(pred_wave.shape[-1]), int(target_wave.shape[-1])) < 2048:
            continue

        components = {
            "decoded_waveform_high_ratio_loss": waveform_high_ratio_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                float(args.decoded_waveform_margin),
            ),
            "decoded_waveform_high_ratio_match_loss": waveform_high_ratio_match_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
            ),
            "decoded_waveform_high_ratio_topk_excess_loss": waveform_high_ratio_topk_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                float(args.decoded_waveform_margin),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_global_high_ratio_loss": waveform_global_high_ratio_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                float(args.decoded_waveform_margin),
            ),
            "decoded_waveform_global_high_ratio_match_loss": waveform_global_high_ratio_match_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
            ),
            "decoded_waveform_spectral_centroid_match_loss": waveform_spectral_centroid_match_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
            ),
            "decoded_waveform_spectral_centroid_topk_drift_loss": waveform_spectral_centroid_topk_drift_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_spectral_centroid_topk_excess_loss": waveform_spectral_centroid_topk_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_spectral_centroid_margin_hz),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_teacher_conditioned_drift_loss": waveform_teacher_conditioned_drift_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                float(args.decoded_waveform_teacher_conditioned_ratio_margin),
                float(args.decoded_waveform_teacher_conditioned_centroid_margin_hz),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_high_flatness_loss": waveform_high_flatness_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                float(args.decoded_waveform_high_flatness_margin),
            ),
            "decoded_waveform_high_flatness_match_loss": waveform_high_flatness_match_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
            ),
            "decoded_waveform_high_flatness_topk_excess_loss": waveform_high_flatness_topk_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                float(args.decoded_waveform_high_flatness_margin),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_weighted_high_flatness_topk_excess_loss": waveform_weighted_high_flatness_topk_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                float(args.decoded_waveform_weighted_high_margin),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_rms_match_loss": waveform_rms_match_loss(pred_wave, target_wave),
            "decoded_waveform_derivative_rms_match_loss": waveform_derivative_rms_match_loss(pred_wave, target_wave),
            "decoded_waveform_derivative_rms_excess_loss": waveform_derivative_rms_excess_loss(
                pred_wave,
                target_wave,
                float(args.decoded_waveform_derivative_rms_margin),
            ),
            "decoded_waveform_soft_zcr_match_loss": waveform_soft_zcr_match_loss(
                pred_wave,
                target_wave,
                float(args.decoded_waveform_soft_zcr_gain),
            ),
            "decoded_waveform_soft_zcr_excess_loss": waveform_soft_zcr_excess_loss(
                pred_wave,
                target_wave,
                float(args.decoded_waveform_soft_zcr_gain),
                float(args.decoded_waveform_soft_zcr_margin),
            ),
            "decoded_waveform_framed_soft_zcr_match_loss": waveform_framed_soft_zcr_match_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_framed_soft_zcr_window),
                int(args.decoded_waveform_framed_soft_zcr_hop),
                float(args.decoded_waveform_soft_zcr_gain),
            ),
            "decoded_waveform_framed_soft_zcr_excess_loss": waveform_framed_soft_zcr_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_framed_soft_zcr_window),
                int(args.decoded_waveform_framed_soft_zcr_hop),
                float(args.decoded_waveform_soft_zcr_gain),
                float(args.decoded_waveform_framed_soft_zcr_margin),
            ),
            "decoded_waveform_framed_soft_zcr_topk_excess_loss": waveform_framed_soft_zcr_topk_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_framed_soft_zcr_window),
                int(args.decoded_waveform_framed_soft_zcr_hop),
                float(args.decoded_waveform_soft_zcr_gain),
                float(args.decoded_waveform_framed_soft_zcr_margin),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_weighted_high_zcr_topk_excess_loss": waveform_weighted_high_zcr_topk_excess_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_sample_rate),
                float(args.decoded_waveform_cutoff_hz),
                int(args.decoded_waveform_framed_soft_zcr_window),
                int(args.decoded_waveform_framed_soft_zcr_hop),
                float(args.decoded_waveform_soft_zcr_gain),
                float(args.decoded_waveform_weighted_high_margin),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_rms_underfill_topk_loss": waveform_rms_underfill_topk_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_rms_envelope_window),
                int(args.decoded_waveform_rms_envelope_hop),
                float(args.decoded_waveform_local_topk_fraction),
            ),
            "decoded_waveform_rms_envelope_match_loss": waveform_rms_envelope_match_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_rms_envelope_window),
                int(args.decoded_waveform_rms_envelope_hop),
            ),
            "decoded_waveform_rms_envelope_second_delta_match_loss": waveform_rms_envelope_second_delta_match_loss(
                pred_wave,
                target_wave,
                int(args.decoded_waveform_rms_envelope_window),
                int(args.decoded_waveform_rms_envelope_hop),
            ),
            "decoded_waveform_multires_stft_loss": waveform_multires_stft_loss(
                pred_wave,
                target_wave,
                str(args.decoded_waveform_multires_stft_ffts),
            ),
            "decoded_waveform_peak_cap_loss": waveform_peak_cap_loss(pred_wave, float(args.decoded_waveform_peak_cap)),
        }
        quality_loss = (
            float(args.decoded_waveform_high_ratio_loss_weight) * components["decoded_waveform_high_ratio_loss"]
            + float(args.decoded_waveform_high_ratio_match_loss_weight)
            * components["decoded_waveform_high_ratio_match_loss"]
            + float(args.decoded_waveform_high_ratio_topk_excess_loss_weight)
            * components["decoded_waveform_high_ratio_topk_excess_loss"]
            + float(args.decoded_waveform_global_high_ratio_loss_weight)
            * components["decoded_waveform_global_high_ratio_loss"]
            + float(args.decoded_waveform_global_high_ratio_match_loss_weight)
            * components["decoded_waveform_global_high_ratio_match_loss"]
            + float(args.decoded_waveform_spectral_centroid_match_loss_weight)
            * components["decoded_waveform_spectral_centroid_match_loss"]
            + float(args.decoded_waveform_spectral_centroid_topk_drift_loss_weight)
            * components["decoded_waveform_spectral_centroid_topk_drift_loss"]
            + float(args.decoded_waveform_spectral_centroid_topk_excess_loss_weight)
            * components["decoded_waveform_spectral_centroid_topk_excess_loss"]
            + float(args.decoded_waveform_teacher_conditioned_drift_loss_weight)
            * components["decoded_waveform_teacher_conditioned_drift_loss"]
            + float(args.decoded_waveform_high_flatness_excess_loss_weight)
            * components["decoded_waveform_high_flatness_loss"]
            + float(args.decoded_waveform_high_flatness_match_loss_weight)
            * components["decoded_waveform_high_flatness_match_loss"]
            + float(args.decoded_waveform_high_flatness_topk_excess_loss_weight)
            * components["decoded_waveform_high_flatness_topk_excess_loss"]
            + float(args.decoded_waveform_weighted_high_flatness_topk_excess_loss_weight)
            * components["decoded_waveform_weighted_high_flatness_topk_excess_loss"]
            + float(args.decoded_waveform_weighted_high_zcr_topk_excess_loss_weight)
            * components["decoded_waveform_weighted_high_zcr_topk_excess_loss"]
            + float(args.decoded_waveform_rms_match_loss_weight)
            * components["decoded_waveform_rms_match_loss"]
            + float(args.decoded_waveform_derivative_rms_match_loss_weight)
            * components["decoded_waveform_derivative_rms_match_loss"]
            + float(args.decoded_waveform_derivative_rms_excess_loss_weight)
            * components["decoded_waveform_derivative_rms_excess_loss"]
            + float(args.decoded_waveform_soft_zcr_match_loss_weight)
            * components["decoded_waveform_soft_zcr_match_loss"]
            + float(args.decoded_waveform_soft_zcr_excess_loss_weight)
            * components["decoded_waveform_soft_zcr_excess_loss"]
            + float(args.decoded_waveform_framed_soft_zcr_match_loss_weight)
            * components["decoded_waveform_framed_soft_zcr_match_loss"]
            + float(args.decoded_waveform_framed_soft_zcr_excess_loss_weight)
            * components["decoded_waveform_framed_soft_zcr_excess_loss"]
            + float(args.decoded_waveform_framed_soft_zcr_topk_excess_loss_weight)
            * components["decoded_waveform_framed_soft_zcr_topk_excess_loss"]
            + float(args.decoded_waveform_rms_underfill_topk_loss_weight)
            * components["decoded_waveform_rms_underfill_topk_loss"]
            + float(args.decoded_waveform_rms_envelope_match_loss_weight)
            * components["decoded_waveform_rms_envelope_match_loss"]
            + float(args.decoded_waveform_rms_envelope_second_delta_match_loss_weight)
            * components["decoded_waveform_rms_envelope_second_delta_match_loss"]
            + float(args.decoded_waveform_multires_stft_loss_weight) * components["decoded_waveform_multires_stft_loss"]
            + float(args.decoded_waveform_peak_cap_loss_weight) * components["decoded_waveform_peak_cap_loss"]
        )
        totals["decoded_quality_loss"] += float(quality_loss.detach().cpu())
        for key, value in components.items():
            totals[key] += float(value.detach().cpu())
        count += 1

    if student_was_training:
        student.train()
    if count <= 0:
        return {}
    averaged = {key: value / float(count) for key, value in totals.items()}
    averaged["decoded_quality_rows"] = float(count)
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F5TTS 2-step bridge from cached FP32 teacher trajectories.")
    parser.add_argument("--cache-dir", required=True, help="Teacher cache directory, or comma-separated directories.")
    parser.add_argument(
        "--cache-sampling-weights",
        default="",
        help="Optional comma-separated sampling weights matching --cache-dir directories.",
    )
    parser.add_argument("--focus-row-manifest", default="")
    parser.add_argument("--focus-row-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--max-cache-frames", type=int, default=0, help="Drop cached rows whose total frame count exceeds this value.")
    parser.add_argument("--min-cache-frames", type=int, default=0, help="Drop cached rows whose total frame count is below this value.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--student-checkpoint", default="")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--student-steps", type=int, default=2)
    parser.add_argument("--teacher-cfg-strength", type=float, default=2.0)
    parser.add_argument("--student-cfg-strength", type=float, default=2.0)
    parser.add_argument("--student-cfg-strengths", default="")
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument(
        "--path-target",
        choices=("teacher_trajectory", "linear_endpoint"),
        default="teacher_trajectory",
        help="Supervise either the cached teacher ODE path or a straight flow from noise to teacher endpoint.",
    )
    parser.add_argument("--student-quant-scheme", choices=("q4", "bitnet_qat", "none"), default="bitnet_qat")
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--bitnet-scale-lr-multiplier", type=float, default=4.0)
    parser.add_argument("--q4-include", default="")
    parser.add_argument("--q4-exclude", default="text_embed.text_embed,mel_spec")
    parser.add_argument("--q4-ste-include", default="__none__")
    parser.add_argument(
        "--q4-skip-initial-requant",
        action="store_true",
        help="Use when continuing from an already materialized q4 checkpoint; avoids a lossy second rowwise quantization on load.",
    )
    parser.add_argument("--train-include", default="transformer.transformer_blocks,transformer.norm_out,transformer.proj_out")
    parser.add_argument("--train-exclude", default="")
    parser.add_argument(
        "--checkpoint-activations",
        action="store_true",
        help="Use DiT block activation checkpointing to fit all-block adaptation in less GPU memory.",
    )
    parser.add_argument(
        "--train-in-eval-mode",
        action="store_true",
        help="Disable dropout/stochastic training behavior while still computing gradients for distillation.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--bridge-flow-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--student-interval-loss-weights",
        default="",
        help="Optional comma-separated weights for each student interval bridge loss.",
    )
    parser.add_argument(
        "--self-rollout-interval-loss-weights",
        default="",
        help="Optional comma-separated weights for each student interval self-rollout loss.",
    )
    parser.add_argument("--teacher-instant-flow-loss-weight", type=float, default=0.0)
    parser.add_argument("--bridge-direction-loss-weight", type=float, default=0.25)
    parser.add_argument("--bridge-norm-loss-weight", type=float, default=0.05)
    parser.add_argument("--bridge-orthogonal-loss-weight", type=float, default=0.0)
    parser.add_argument("--next-state-loss-weight", type=float, default=5.0)
    parser.add_argument("--final-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--final-rollout-loss-weight-final", type=float, default=1.0)
    parser.add_argument(
        "--self-rollout-bridge-loss-weight",
        type=float,
        default=0.0,
        help="Train each student step on states produced by the student's own rollout toward the teacher endpoint.",
    )
    parser.add_argument("--self-rollout-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--self-rollout-norm-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--rollout-trajectory-loss-weight",
        type=float,
        default=0.0,
        help="During student rollout, match each intermediate student state to the cached teacher trajectory state.",
    )
    parser.add_argument(
        "--rollout-trajectory-interval-loss-weights",
        default="",
        help="Optional comma-separated weights for intermediate rollout trajectory losses.",
    )
    parser.add_argument("--conditional-mel-critic-loss-weight", type=float, default=0.0)
    parser.add_argument("--conditional-mel-critic-lr", type=float, default=1.0e-4)
    parser.add_argument("--conditional-mel-critic-hidden", type=int, default=64)
    parser.add_argument("--conditional-mel-critic-update-every", type=int, default=1)
    parser.add_argument("--conditional-mel-critic-start-step", type=int, default=1)
    parser.add_argument("--conditional-mel-critic-max-frames", type=int, default=512)
    parser.add_argument("--conditional-mel-critic-grad-clip", type=float, default=1.0)
    parser.add_argument("--text-flow-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-flow-contrastive-margin", type=float, default=0.04)
    parser.add_argument("--text-rollout-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-rollout-contrastive-margin", type=float, default=0.08)
    parser.add_argument("--cond-flow-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--cond-flow-contrastive-margin", type=float, default=0.04)
    parser.add_argument("--cond-rollout-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--cond-rollout-contrastive-margin", type=float, default=0.08)
    parser.add_argument(
        "--hard-negative-text-prob",
        type=float,
        default=0.0,
        help="Use text ids from another cached sample as the contrastive negative instead of only reversing this text.",
    )
    parser.add_argument("--temporal-delta-loss-weight", type=float, default=0.1)
    parser.add_argument("--frame-energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-band-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-band-temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-band-second-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-band-excess-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-band-excess-ratio-margin", type=float, default=0.02)
    parser.add_argument("--anchor-high-band-excess-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-band-start-bin", type=int, default=80)
    parser.add_argument("--tail-start-ratio", type=float, default=0.62)
    parser.add_argument("--tail-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-high-band-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-high-band-temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-high-band-excess-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--mid-start-ratio", type=float, default=0.30)
    parser.add_argument("--mid-end-ratio", type=float, default=0.70)
    parser.add_argument("--mid-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--mid-temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--mid-high-band-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--mid-high-band-temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--mid-high-band-excess-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--focus-start-ratio", type=float, default=0.0)
    parser.add_argument("--focus-end-ratio", type=float, default=1.0)
    parser.add_argument("--focus-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--focus-temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--focus-high-band-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--focus-high-band-temporal-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--focus-high-band-excess-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-high-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-high-ratio-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-high-ratio-topk-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-global-high-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-global-high-ratio-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-spectral-centroid-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-spectral-centroid-topk-drift-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-spectral-centroid-topk-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-spectral-centroid-margin-hz", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-teacher-conditioned-drift-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-teacher-conditioned-ratio-margin", type=float, default=0.45)
    parser.add_argument("--decoded-waveform-teacher-conditioned-centroid-margin-hz", type=float, default=3000.0)
    parser.add_argument("--decoded-waveform-high-flatness-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-high-flatness-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-high-flatness-topk-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-high-flatness-margin", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-weighted-high-flatness-topk-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-weighted-high-zcr-topk-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-weighted-high-margin", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-rms-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-derivative-rms-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-derivative-rms-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-derivative-rms-margin", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-soft-zcr-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-soft-zcr-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-soft-zcr-gain", type=float, default=80.0)
    parser.add_argument("--decoded-waveform-soft-zcr-margin", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-framed-soft-zcr-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-framed-soft-zcr-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-framed-soft-zcr-window", type=int, default=2048)
    parser.add_argument("--decoded-waveform-framed-soft-zcr-hop", type=int, default=512)
    parser.add_argument("--decoded-waveform-framed-soft-zcr-margin", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-framed-soft-zcr-topk-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-local-topk-fraction", type=float, default=0.2)
    parser.add_argument("--decoded-waveform-rms-underfill-topk-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-rms-envelope-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-rms-envelope-second-delta-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-rms-envelope-window", type=int, default=2048)
    parser.add_argument("--decoded-waveform-rms-envelope-hop", type=int, default=512)
    parser.add_argument("--decoded-waveform-multires-stft-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-multires-stft-ffts", default="512,1024,2048")
    parser.add_argument("--decoded-waveform-spectral-envelope-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-spectral-envelope-n-fft", type=int, default=1024)
    parser.add_argument("--decoded-waveform-spectral-envelope-hop", type=int, default=256)
    parser.add_argument("--decoded-waveform-spectral-envelope-smooth-bins", type=int, default=17)
    parser.add_argument("--decoded-waveform-peak-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-peak-cap-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-peak-cap", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-ctc-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--decoded-waveform-ctc-target-mode",
        choices=("gen_text", "keywords", "domain"),
        default="gen_text",
    )
    parser.add_argument(
        "--decoded-waveform-ctc-bundle",
        default="WAV2VEC2_ASR_BASE_960H",
        help="torchaudio.pipelines bundle name for decoded waveform CTC content loss.",
    )
    parser.add_argument("--decoded-waveform-ssl-emission-match-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--decoded-waveform-ssl-bundle",
        default="WAV2VEC2_ASR_BASE_960H",
        help="torchaudio.pipelines bundle name for decoded waveform teacher/student emission matching.",
    )
    parser.add_argument("--decoded-waveform-wavlm-profile-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--decoded-waveform-wavlm-profile-bundle",
        default="WAVLM_BASE",
        help="torchaudio.pipelines bundle name for decoded waveform speaker-profile loss.",
    )
    parser.add_argument(
        "--decoded-waveform-wavlm-profile-ref-audio",
        default="/data/resumebot/voice_profiles/Peyton/sample_0.wav",
        help="Reference voice audio for decoded waveform WavLM profile loss.",
    )
    parser.add_argument("--decoded-waveform-mid-start-ratio", type=float, default=0.30)
    parser.add_argument("--decoded-waveform-mid-end-ratio", type=float, default=0.70)
    parser.add_argument("--decoded-waveform-mid-high-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-mid-high-ratio-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-tail-start-ratio", type=float, default=0.66)
    parser.add_argument("--decoded-waveform-tail-high-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-tail-high-ratio-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-waveform-loss-every", type=int, default=0)
    parser.add_argument("--val-decoded-quality-every", type=int, default=0)
    parser.add_argument("--val-decoded-quality-rows", type=int, default=16)
    parser.add_argument("--decoded-waveform-max-frames", type=int, default=560)
    parser.add_argument(
        "--decoded-waveform-crop-mode",
        choices=("skip", "random", "mid", "tail"),
        default="skip",
        help="For long samples, crop mel frames for decoded waveform losses instead of skipping them.",
    )
    parser.add_argument("--decoded-waveform-cutoff-hz", type=float, default=7000.0)
    parser.add_argument("--decoded-waveform-margin", type=float, default=0.01)
    parser.add_argument("--decoded-waveform-sample-rate", type=int, default=24000)
    parser.add_argument(
        "--decoded-vocoder-checkpoint",
        default="",
        help="Optional Vocos checkpoint for decoded waveform losses; use the same decoder as held-out evaluation.",
    )
    parser.add_argument("--decoded-waveform-generated-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cond-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--cond-augment-prob", type=float, default=0.0)
    parser.add_argument("--cond-noise-std", type=float, default=0.0)
    parser.add_argument("--cond-high-band-noise-std", type=float, default=0.0)
    parser.add_argument("--cond-gain-jitter", type=float, default=0.0)
    parser.add_argument("--cond-bias", type=float, default=0.0)
    parser.add_argument("--cond-high-band-gain", type=float, default=1.0)
    parser.add_argument("--cond-smooth-prob", type=float, default=0.0)
    parser.add_argument("--cond-smooth-kernel", type=int, default=5)
    parser.add_argument("--anchor-weight-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--anchor-checkpoint",
        default="",
        help="Optional known-good checkpoint for functional trust-region losses. Defaults to --student-checkpoint.",
    )
    parser.add_argument(
        "--anchor-flow-loss-weight",
        type=float,
        default=0.0,
        help="Keep each supervised bridge flow close to a frozen anchor checkpoint.",
    )
    parser.add_argument(
        "--anchor-rollout-loss-weight",
        type=float,
        default=0.0,
        help="Keep the final student rollout close to a frozen anchor checkpoint.",
    )
    parser.add_argument("--loss-schedule-steps", type=int, default=400)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-max-rows", type=int, default=32)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    cache_dirs = parse_cache_dirs(args.cache_dir)
    cache_weights = parse_cache_weights(args.cache_sampling_weights, len(cache_dirs))
    cache_dir = cache_dirs[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    source_counts: dict[str, int] = {}
    for source_index, (source_cache_dir, source_weight) in enumerate(zip(cache_dirs, cache_weights, strict=True)):
        source_rows = load_rows(source_cache_dir, source_index=source_index, sample_weight=source_weight)
        rows.extend(source_rows)
        source_counts[str(source_cache_dir)] = len(source_rows)
    random.shuffle(rows)
    if int(args.max_cache_frames) > 0:
        rows = [row for row in rows if int(row.get("frames", 0)) <= int(args.max_cache_frames)]
        if not rows:
            raise ValueError("--max-cache-frames removed every cached row")
    if int(args.min_cache_frames) > 0:
        rows = [row for row in rows if int(row.get("frames", 0)) >= int(args.min_cache_frames)]
        if not rows:
            raise ValueError("--min-cache-frames removed every cached row")
    val_count = min(int(args.val_max_rows), int(round(len(rows) * float(args.val_fraction))))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:] if val_count < len(rows) else rows
    focus_row_keys = load_focus_row_keys(str(args.focus_row_manifest))
    focus_row_matches = apply_focus_row_weights(
        train_rows,
        focus_keys=focus_row_keys,
        multiplier=float(args.focus_row_weight_multiplier),
    )
    device = torch.device(args.device)

    if bool(args.checkpoint_activations):
        force_non_reentrant_checkpointing()

    student = build_model(Path(args.vocab), device)
    if bool(args.checkpoint_activations) and hasattr(student, "transformer"):
        setattr(student.transformer, "checkpoint_activations", True)
    quant_scheme = str(args.student_quant_scheme).lower()
    if quant_scheme == "bitnet_qat":
        from compress.quantization import quantize_linear_modules

        replacements = quantize_linear_modules(
            student,
            include=split_csv(args.q4_include),
            exclude=split_csv(args.q4_exclude),
            scheme="bitnet_qat",
            bitnet_qat_learned_scale=bool(args.bitnet_qat_learned_scale),
        )
        load_checkpoint_state(student, Path(args.student_checkpoint or args.checkpoint))
        quant_modules = len(replacements)
        quant_params = sum(int(module.weight.numel()) for module in replacements.values() if hasattr(module, "weight"))
    elif quant_scheme == "q4":
        load_checkpoint_state(student, Path(args.student_checkpoint or args.checkpoint))
        if bool(args.q4_skip_initial_requant):
            quant_modules = 0
            quant_params = 0
        else:
            quant_modules, quant_params = apply_q4_parametrizations(
                student,
                include=split_csv(args.q4_include),
                exclude=split_csv(args.q4_exclude),
                ste_include=split_csv(args.q4_ste_include),
            )
    else:
        load_checkpoint_state(student, Path(args.student_checkpoint or args.checkpoint))
        quant_modules = 0
        quant_params = 0
    functional_anchor = load_functional_anchor(args, device)
    decoded_vocoder = None
    decoded_ctc_model = None
    decoded_ctc_sample_rate = 16000
    decoded_ctc_label_to_index: dict[str, int] = {}
    decoded_ctc_blank_index = 0
    decoded_ssl_model = None
    decoded_ssl_sample_rate = 16000
    decoded_wavlm_profile_model = None
    decoded_wavlm_profile_sample_rate = 16000
    decoded_wavlm_profile_ref_embedding = None
    if (
        (
            float(args.decoded_waveform_high_ratio_loss_weight) > 0.0
            or float(args.decoded_waveform_high_ratio_match_loss_weight) > 0.0
            or float(args.decoded_waveform_high_ratio_topk_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_global_high_ratio_loss_weight) > 0.0
            or float(args.decoded_waveform_global_high_ratio_match_loss_weight) > 0.0
            or float(args.decoded_waveform_spectral_centroid_match_loss_weight) > 0.0
            or float(args.decoded_waveform_spectral_centroid_topk_drift_loss_weight) > 0.0
            or float(args.decoded_waveform_spectral_centroid_topk_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_teacher_conditioned_drift_loss_weight) > 0.0
            or float(args.decoded_waveform_high_flatness_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_high_flatness_match_loss_weight) > 0.0
            or float(args.decoded_waveform_high_flatness_topk_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_weighted_high_flatness_topk_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_weighted_high_zcr_topk_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_rms_match_loss_weight) > 0.0
            or float(args.decoded_waveform_derivative_rms_match_loss_weight) > 0.0
            or float(args.decoded_waveform_derivative_rms_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_soft_zcr_match_loss_weight) > 0.0
            or float(args.decoded_waveform_soft_zcr_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_framed_soft_zcr_match_loss_weight) > 0.0
            or float(args.decoded_waveform_framed_soft_zcr_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_framed_soft_zcr_topk_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_rms_underfill_topk_loss_weight) > 0.0
            or float(args.decoded_waveform_rms_envelope_match_loss_weight) > 0.0
            or float(args.decoded_waveform_rms_envelope_second_delta_match_loss_weight) > 0.0
            or float(args.decoded_waveform_multires_stft_loss_weight) > 0.0
            or float(args.decoded_waveform_peak_excess_loss_weight) > 0.0
            or float(args.decoded_waveform_peak_cap_loss_weight) > 0.0
            or float(args.decoded_waveform_ctc_loss_weight) > 0.0
            or float(args.decoded_waveform_ssl_emission_match_loss_weight) > 0.0
            or float(args.decoded_waveform_wavlm_profile_loss_weight) > 0.0
            or float(args.decoded_waveform_mid_high_ratio_loss_weight) > 0.0
            or float(args.decoded_waveform_mid_high_ratio_match_loss_weight) > 0.0
            or float(args.decoded_waveform_tail_high_ratio_loss_weight) > 0.0
            or float(args.decoded_waveform_tail_high_ratio_match_loss_weight) > 0.0
        )
        and int(args.decoded_waveform_loss_every) > 0
    ):
        resumebot_root = Path("/data/resumebot")
        if str(resumebot_root) not in sys.path:
            sys.path.insert(0, str(resumebot_root))
        from f5_tts.infer.utils_infer import load_vocoder

        decoded_vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=str(device))
        if str(args.decoded_vocoder_checkpoint).strip():
            vocos_payload = torch.load(str(args.decoded_vocoder_checkpoint), map_location=device)
            vocos_state = vocos_payload.get("model_state_dict", vocos_payload)
            missing, unexpected = decoded_vocoder.load_state_dict(vocos_state, strict=False)
            actionable_unexpected = [name for name in unexpected if not str(name).endswith(".weight_scale")]
            if missing or actionable_unexpected:
                print(
                    json.dumps(
                        {
                            "event": "decoded_vocoder_checkpoint_load",
                            "checkpoint": str(args.decoded_vocoder_checkpoint),
                            "missing": missing,
                            "unexpected": actionable_unexpected,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        decoded_vocoder.eval().requires_grad_(False)
        if (
            float(args.decoded_waveform_ctc_loss_weight) > 0.0
            or float(args.decoded_waveform_ssl_emission_match_loss_weight) > 0.0
            or float(args.decoded_waveform_wavlm_profile_loss_weight) > 0.0
        ):
            import torchaudio
            import torchaudio.pipelines as torchaudio_pipelines

        if float(args.decoded_waveform_ctc_loss_weight) > 0.0:
            ctc_bundle = getattr(torchaudio_pipelines, str(args.decoded_waveform_ctc_bundle))
            decoded_ctc_model = ctc_bundle.get_model().to(device).eval()
            decoded_ctc_model.requires_grad_(False)
            decoded_ctc_sample_rate = int(ctc_bundle.sample_rate)
            ctc_labels = list(ctc_bundle.get_labels())
            decoded_ctc_label_to_index = {str(label): index for index, label in enumerate(ctc_labels)}
            decoded_ctc_blank_index = int(decoded_ctc_label_to_index.get("-", 0))
        if float(args.decoded_waveform_ssl_emission_match_loss_weight) > 0.0:
            ssl_bundle = getattr(torchaudio_pipelines, str(args.decoded_waveform_ssl_bundle))
            decoded_ssl_model = ssl_bundle.get_model().to(device).eval()
            decoded_ssl_model.requires_grad_(False)
            decoded_ssl_sample_rate = int(ssl_bundle.sample_rate)
        if float(args.decoded_waveform_wavlm_profile_loss_weight) > 0.0:
            wavlm_bundle = getattr(torchaudio_pipelines, str(args.decoded_waveform_wavlm_profile_bundle))
            decoded_wavlm_profile_model = wavlm_bundle.get_model().to(device).eval()
            decoded_wavlm_profile_model.requires_grad_(False)
            decoded_wavlm_profile_sample_rate = int(wavlm_bundle.sample_rate)
            import soundfile as sf

            ref_audio, ref_sample_rate = sf.read(str(args.decoded_waveform_wavlm_profile_ref_audio), always_2d=True)
            ref_wave = torch.from_numpy(ref_audio).float().mean(1, keepdim=True).T.to(device)
            ref_wave = resample_waveform_linear(ref_wave, int(ref_sample_rate), decoded_wavlm_profile_sample_rate)
            with torch.no_grad():
                ref_features, _ = decoded_wavlm_profile_model.extract_features(ref_wave)
                decoded_wavlm_profile_ref_embedding = F.normalize(ref_features[-1].mean(1).float(), dim=-1).detach()

    train_tensors, train_params = configure_trainable_parameters(
        student,
        train_include=split_csv(args.train_include),
        train_exclude=split_csv(args.train_exclude),
    )
    anchor_parameters = trainable_anchor_parameters(student) if float(args.anchor_weight_loss_weight) > 0.0 else {}
    if bool(args.train_in_eval_mode):
        student.eval()
    else:
        student.train()
    named_parameters = [(name, parameter) for name, parameter in student.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        raise ValueError("no trainable parameters selected")
    scale_parameters = [parameter for name, parameter in named_parameters if name.endswith(".weight_scale")]
    scale_ids = {id(parameter) for parameter in scale_parameters}
    base_parameters = [parameter for _, parameter in named_parameters if id(parameter) not in scale_ids]
    parameter_groups: list[dict[str, Any]] = []
    if base_parameters:
        parameter_groups.append({"params": base_parameters, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
    if scale_parameters:
        parameter_groups.append(
            {
                "params": scale_parameters,
                "lr": float(args.lr) * float(args.bitnet_scale_lr_multiplier),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
    conditional_mel_critic: ConditionalMelCritic | None = None
    conditional_mel_critic_optimizer: torch.optim.Optimizer | None = None
    if float(args.conditional_mel_critic_loss_weight) > 0.0:
        probe_payload = torch.load(row_payload_path(train_rows[0], cache_dir), map_location="cpu")
        mel_channels = int(probe_payload["teacher_y"].shape[-1])
        conditional_mel_critic = ConditionalMelCritic(
            mel_channels=mel_channels,
            hidden=int(args.conditional_mel_critic_hidden),
        ).to(device)
        conditional_mel_critic_optimizer = torch.optim.AdamW(
            conditional_mel_critic.parameters(),
            lr=float(args.conditional_mel_critic_lr),
            weight_decay=0.0,
        )

    student_times_tensor = make_time_grid(int(args.student_steps), float(args.sway_sampling_coef), device, torch.float32)
    student_times = [float(value) for value in student_times_tensor.cpu()]
    cfg_choices = parse_float_list(args.student_cfg_strengths) or [float(args.student_cfg_strength)]
    interval_loss_weights = parse_float_list(args.student_interval_loss_weights)
    self_rollout_interval_loss_weights = parse_float_list(args.self_rollout_interval_loss_weights)
    rollout_trajectory_interval_loss_weights = parse_float_list(args.rollout_trajectory_interval_loss_weights)
    if interval_loss_weights and len(interval_loss_weights) != int(args.student_steps):
        raise ValueError("--student-interval-loss-weights must match --student-steps")
    if self_rollout_interval_loss_weights and len(self_rollout_interval_loss_weights) != int(args.student_steps):
        raise ValueError("--self-rollout-interval-loss-weights must match --student-steps")
    if rollout_trajectory_interval_loss_weights and len(rollout_trajectory_interval_loss_weights) != int(args.student_steps):
        raise ValueError("--rollout-trajectory-interval-loss-weights must match --student-steps")
    setup = {
        "event": "setup",
        "cache_rows": len(rows),
        "cache_dirs": [str(path) for path in cache_dirs],
        "cache_source_counts": source_counts,
        "cache_sampling_weights": cache_weights,
        "focus_row_manifest": str(args.focus_row_manifest),
        "focus_row_matches": int(focus_row_matches),
        "focus_row_weight_multiplier": float(args.focus_row_weight_multiplier),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "student_times": student_times,
        "student_interval_loss_weights": interval_loss_weights,
        "self_rollout_interval_loss_weights": self_rollout_interval_loss_weights,
        "rollout_trajectory_interval_loss_weights": rollout_trajectory_interval_loss_weights,
        "conditional_mel_critic": bool(conditional_mel_critic is not None),
        "quant_scheme": quant_scheme,
        "quant_modules": quant_modules,
        "quant_params": quant_params,
        "train_tensors": train_tensors,
        "train_params": train_params,
        "anchor_params": sum(int(tensor.numel()) for tensor in anchor_parameters.values()),
        "functional_anchor": str(args.anchor_checkpoint or args.student_checkpoint or "") if functional_anchor is not None else "",
        "scale_params": sum(int(parameter.numel()) for parameter in scale_parameters),
        "student_checkpoint": str(args.student_checkpoint or args.checkpoint),
        "decoded_waveform_loss": bool(decoded_vocoder is not None),
        "decoded_waveform_loss_every": int(args.decoded_waveform_loss_every),
        "decoded_waveform_ctc_loss": bool(decoded_ctc_model is not None),
        "decoded_waveform_ssl_emission_match_loss": bool(decoded_ssl_model is not None),
        "decoded_waveform_wavlm_profile_loss": bool(decoded_wavlm_profile_model is not None),
    }
    ledger_path = output_dir / "cached_bridge_metrics.jsonl"
    append_jsonl(ledger_path, setup)
    print(json.dumps(setup, indent=2), flush=True)

    best_loss = float("inf")
    best_rollout_loss = float("inf")
    best_val_rollout_loss = float("inf")
    best_val_quality_loss = float("inf")
    started = time.time()
    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        bridge_flow_accum = 0.0
        next_state_accum = 0.0
        rollout_accum = 0.0
        direction_accum = 0.0
        norm_accum = 0.0
        orthogonal_accum = 0.0
        cond_delta_accum = 0.0
        temporal_accum = 0.0
        energy_accum = 0.0
        high_band_accum = 0.0
        high_band_temporal_accum = 0.0
        high_band_second_accum = 0.0
        high_band_excess_ratio_accum = 0.0
        anchor_high_band_excess_ratio_accum = 0.0
        decoded_waveform_high_ratio_accum = 0.0
        decoded_waveform_high_ratio_match_accum = 0.0
        decoded_waveform_high_ratio_topk_excess_accum = 0.0
        decoded_waveform_global_high_ratio_accum = 0.0
        decoded_waveform_global_high_ratio_match_accum = 0.0
        decoded_waveform_spectral_centroid_match_accum = 0.0
        decoded_waveform_spectral_centroid_topk_drift_accum = 0.0
        decoded_waveform_spectral_centroid_topk_excess_accum = 0.0
        decoded_waveform_teacher_conditioned_drift_accum = 0.0
        decoded_waveform_high_flatness_accum = 0.0
        decoded_waveform_high_flatness_match_accum = 0.0
        decoded_waveform_high_flatness_topk_excess_accum = 0.0
        decoded_waveform_weighted_high_flatness_topk_excess_accum = 0.0
        decoded_waveform_weighted_high_zcr_topk_excess_accum = 0.0
        decoded_waveform_rms_match_accum = 0.0
        decoded_waveform_derivative_rms_match_accum = 0.0
        decoded_waveform_derivative_rms_excess_accum = 0.0
        decoded_waveform_soft_zcr_match_accum = 0.0
        decoded_waveform_soft_zcr_excess_accum = 0.0
        decoded_waveform_framed_soft_zcr_match_accum = 0.0
        decoded_waveform_framed_soft_zcr_excess_accum = 0.0
        decoded_waveform_framed_soft_zcr_topk_excess_accum = 0.0
        decoded_waveform_rms_underfill_topk_accum = 0.0
        decoded_waveform_rms_envelope_match_accum = 0.0
        decoded_waveform_rms_envelope_second_delta_match_accum = 0.0
        decoded_waveform_multires_stft_accum = 0.0
        decoded_waveform_spectral_envelope_match_accum = 0.0
        decoded_waveform_peak_excess_accum = 0.0
        decoded_waveform_peak_cap_accum = 0.0
        decoded_waveform_ctc_accum = 0.0
        decoded_waveform_ssl_emission_match_accum = 0.0
        decoded_waveform_wavlm_profile_accum = 0.0
        decoded_waveform_mid_high_ratio_accum = 0.0
        decoded_waveform_mid_high_ratio_match_accum = 0.0
        decoded_waveform_tail_high_ratio_accum = 0.0
        decoded_waveform_tail_high_ratio_match_accum = 0.0
        decoded_waveform_updates = 0
        decoded_waveform_crops = 0
        decoded_waveform_skip_frames = 0
        decoded_waveform_skip_short = 0
        tail_rollout_accum = 0.0
        tail_temporal_accum = 0.0
        tail_high_band_accum = 0.0
        tail_high_band_temporal_accum = 0.0
        tail_high_band_excess_ratio_accum = 0.0
        mid_rollout_accum = 0.0
        mid_temporal_accum = 0.0
        mid_high_band_accum = 0.0
        mid_high_band_temporal_accum = 0.0
        mid_high_band_excess_ratio_accum = 0.0
        focus_rollout_accum = 0.0
        focus_temporal_accum = 0.0
        focus_high_band_accum = 0.0
        focus_high_band_temporal_accum = 0.0
        focus_high_band_excess_ratio_accum = 0.0
        self_rollout_bridge_accum = 0.0
        self_rollout_direction_accum = 0.0
        self_rollout_norm_accum = 0.0
        rollout_trajectory_accum = 0.0
        conditional_mel_critic_accum = 0.0
        conditional_mel_critic_discriminator_accum = 0.0
        text_flow_contrastive_accum = 0.0
        text_rollout_contrastive_accum = 0.0
        cond_flow_contrastive_accum = 0.0
        cond_rollout_contrastive_accum = 0.0
        anchor_flow_accum = 0.0
        anchor_rollout_accum = 0.0
        anchor_accum = 0.0
        for _ in range(max(1, int(args.grad_accum_steps))):
            batch_weights = [float(row.get("_sample_weight", 1.0)) for row in train_rows]
            batch_rows = random.choices(train_rows, weights=batch_weights, k=max(1, int(args.batch_size)))
            micro_loss = None
            for row in batch_rows:
                payload = torch.load(row_payload_path(row, cache_dir), map_location="cpu")
                cond = payload["cond"].to(device=device, dtype=torch.float32)
                noise = payload["noise"].to(device=device, dtype=torch.float32)
                text_ids = payload["text_ids"].to(device=device)
                loss_mask = payload["loss_mask"].to(device=device).bool()
                states = [state.to(device=device, dtype=torch.float32) for state in payload["teacher_states"]]
                state_times = [float(value) for value in payload["teacher_times"]]
                teacher_y = payload["teacher_y"].to(device=device, dtype=torch.float32)
                teacher_grid_flows = [
                    tensor.to(device=device, dtype=torch.float32)
                    for tensor in payload.get("teacher_grid_flows", [])
                ]
                teacher_grid_cond_deltas = [
                    tensor.to(device=device, dtype=torch.float32)
                    for tensor in payload.get("teacher_grid_cond_deltas", [])
                ]
                cond_frames = int(payload["cond_frames"])
                cond = maybe_augment_conditioning(
                    cond,
                    cond_frames=cond_frames,
                    prob=float(args.cond_augment_prob),
                    noise_std=float(args.cond_noise_std),
                    high_band_noise_std=float(args.cond_high_band_noise_std),
                    high_band_start_bin=int(args.high_band_start_bin),
                    gain_jitter=float(args.cond_gain_jitter),
                    bias=float(args.cond_bias),
                    high_band_gain=float(args.cond_high_band_gain),
                    smooth_prob=float(args.cond_smooth_prob),
                    smooth_kernel=int(args.cond_smooth_kernel),
                )
                if not bool(loss_mask.any()):
                    continue
                current_cfg = float(random.choice(cfg_choices))
                if random.random() < float(args.hard_negative_text_prob):
                    bad_text_ids = load_negative_text_ids(
                        rows=train_rows,
                        cache_dir=cache_dir,
                        current_path=str(row["path"]),
                        fallback=text_ids,
                        device=device,
                    )
                else:
                    bad_text_ids = corrupt_text_ids(text_ids)
                bad_cond = None
                if (
                    float(args.cond_flow_contrastive_loss_weight) > 0.0
                    or float(args.cond_rollout_contrastive_loss_weight) > 0.0
                ):
                    bad_cond = load_negative_condition(
                        rows=train_rows,
                        cache_dir=cache_dir,
                        current_path=str(row["path"]),
                        current_speaker=str(row.get("speaker_id", "")),
                        fallback=cond,
                        cond_frames=cond_frames,
                        device=device,
                    )
                sample_loss = cond.new_zeros(())
                sample_bridge = cond.new_zeros(())
                sample_next = cond.new_zeros(())
                sample_direction = cond.new_zeros(())
                sample_norm = cond.new_zeros(())
                sample_orthogonal = cond.new_zeros(())
                sample_cond_delta = cond.new_zeros(())
                sample_text_flow_contrastive = cond.new_zeros(())
                sample_cond_flow_contrastive = cond.new_zeros(())
                sample_anchor_flow = cond.new_zeros(())
                intervals = 0
                interval_weight_sum = 0.0
                for interval in range(int(args.student_steps)):
                    interval_weight = float(interval_loss_weights[interval]) if interval_loss_weights else 1.0
                    t0 = student_times[interval]
                    t1 = student_times[interval + 1]
                    dt = float(t1 - t0)
                    if str(args.path_target) == "linear_endpoint":
                        y0 = ((1.0 - float(t0)) * noise + float(t0) * teacher_y).detach()
                        y1 = ((1.0 - float(t1)) * noise + float(t1) * teacher_y).detach()
                        target_flow = (teacher_y - noise).detach()
                    else:
                        idx0 = nearest_state_index(state_times, t0)
                        idx1 = nearest_state_index(state_times, t1)
                        y0 = states[idx0].detach()
                        y1 = states[idx1].detach()
                        target_flow = (y1 - y0) / dt
                    time_tensor = torch.full((y0.shape[0],), t0, device=device, dtype=y0.dtype)
                    pred_flow, pred_cond_delta = cfg_flow_with_delta(
                        student,
                        y0,
                        cond,
                        text_ids,
                        time_tensor,
                        current_cfg,
                        detach_null_grad=False,
                    )
                    next_state = y0 + dt * pred_flow
                    bridge_loss = F.mse_loss(pred_flow[loss_mask], target_flow[loss_mask])
                    text_flow_contrastive_loss = y0.new_zeros(())
                    if float(args.text_flow_contrastive_loss_weight) > 0.0:
                        bad_pred_flow = cfg_flow(student, y0, cond, bad_text_ids, time_tensor, current_cfg)
                        bad_bridge_loss = F.mse_loss(bad_pred_flow[loss_mask], target_flow[loss_mask])
                        text_flow_contrastive_loss = F.relu(
                            float(args.text_flow_contrastive_margin) + bridge_loss - bad_bridge_loss
                        )
                        sample_text_flow_contrastive = sample_text_flow_contrastive + text_flow_contrastive_loss
                    cond_flow_contrastive_loss = y0.new_zeros(())
                    if bad_cond is not None and float(args.cond_flow_contrastive_loss_weight) > 0.0:
                        bad_cond_pred_flow = cfg_flow(student, y0, bad_cond, text_ids, time_tensor, current_cfg)
                        bad_cond_bridge_loss = F.mse_loss(bad_cond_pred_flow[loss_mask], target_flow[loss_mask])
                        cond_flow_contrastive_loss = F.relu(
                            float(args.cond_flow_contrastive_margin) + bridge_loss - bad_cond_bridge_loss
                        )
                        sample_cond_flow_contrastive = sample_cond_flow_contrastive + cond_flow_contrastive_loss
                    anchor_flow_loss = y0.new_zeros(())
                    if functional_anchor is not None and float(args.anchor_flow_loss_weight) > 0.0:
                        with torch.no_grad():
                            anchor_pred_flow = cfg_flow(
                                functional_anchor,
                                y0,
                                cond,
                                text_ids,
                                time_tensor,
                                current_cfg,
                            )
                        anchor_flow_loss = F.smooth_l1_loss(
                            pred_flow[loss_mask],
                            anchor_pred_flow[loss_mask],
                            beta=0.2,
                        )
                        sample_anchor_flow = sample_anchor_flow + anchor_flow_loss
                    instant_flow_loss = y0.new_zeros(())
                    if interval < len(teacher_grid_flows) and float(args.teacher_instant_flow_loss_weight) > 0:
                        teacher_instant_flow = teacher_grid_flows[interval].detach()
                        instant_flow_loss = F.smooth_l1_loss(
                            pred_flow[loss_mask],
                            teacher_instant_flow[loss_mask],
                            beta=0.2,
                        )
                    next_loss = F.mse_loss(next_state[loss_mask], y1[loss_mask])
                    dir_loss = direction_loss(pred_flow, target_flow, loss_mask)
                    nr_loss = norm_ratio_loss(pred_flow, target_flow, loss_mask)
                    ortho_loss = orthogonal_flow_loss(pred_flow, target_flow, loss_mask)
                    if float(args.cond_delta_loss_weight) > 0 and interval < len(teacher_grid_cond_deltas):
                        teacher_cond_delta = teacher_grid_cond_deltas[interval].detach()
                        cond_delta_loss = F.smooth_l1_loss(pred_cond_delta[loss_mask], teacher_cond_delta[loss_mask], beta=0.2)
                    elif float(args.cond_delta_loss_weight) > 0:
                        with torch.no_grad():
                            teacher_cond_delta = target_flow - cfg_flow(
                                student,
                                y0,
                                cond,
                                text_ids,
                                time_tensor,
                                0.0,
                            ).detach()
                        cond_delta_loss = F.smooth_l1_loss(pred_cond_delta[loss_mask], teacher_cond_delta[loss_mask], beta=0.2)
                    else:
                        cond_delta_loss = y0.new_zeros(())
                    sample_loss = (
                        sample_loss
                        + interval_weight
                        * (
                            float(args.bridge_flow_loss_weight) * bridge_loss
                            + float(args.teacher_instant_flow_loss_weight) * instant_flow_loss
                            + float(args.next_state_loss_weight) * next_loss
                            + float(args.bridge_direction_loss_weight) * dir_loss
                            + float(args.bridge_norm_loss_weight) * nr_loss
                            + float(args.bridge_orthogonal_loss_weight) * ortho_loss
                            + float(args.cond_delta_loss_weight) * cond_delta_loss
                            + float(args.text_flow_contrastive_loss_weight) * text_flow_contrastive_loss
                            + float(args.cond_flow_contrastive_loss_weight) * cond_flow_contrastive_loss
                            + float(args.anchor_flow_loss_weight) * anchor_flow_loss
                        )
                    )
                    sample_bridge = sample_bridge + interval_weight * bridge_loss.detach()
                    sample_next = sample_next + interval_weight * next_loss.detach()
                    sample_direction = sample_direction + interval_weight * dir_loss.detach()
                    sample_norm = sample_norm + interval_weight * nr_loss.detach()
                    sample_orthogonal = sample_orthogonal + interval_weight * ortho_loss.detach()
                    sample_cond_delta = sample_cond_delta + interval_weight * cond_delta_loss.detach()
                    interval_weight_sum += interval_weight
                    intervals += 1
                if intervals:
                    interval_denom = float(interval_weight_sum or intervals)
                    sample_loss = sample_loss / interval_denom
                    sample_bridge = sample_bridge / interval_denom
                    sample_next = sample_next / interval_denom
                    sample_direction = sample_direction / interval_denom
                    sample_norm = sample_norm / interval_denom
                    sample_orthogonal = sample_orthogonal / interval_denom
                    sample_cond_delta = sample_cond_delta / interval_denom
                    sample_text_flow_contrastive = sample_text_flow_contrastive / float(intervals)
                    sample_cond_flow_contrastive = sample_cond_flow_contrastive / float(intervals)
                    sample_anchor_flow = sample_anchor_flow / float(intervals)

                progress = min(1.0, step / float(max(1, int(args.loss_schedule_steps))))
                rollout_weight = float(args.final_rollout_loss_weight) + (
                    float(args.final_rollout_loss_weight_final) - float(args.final_rollout_loss_weight)
                ) * progress
                rollout_loss = cond.new_zeros(())
                temporal_loss = cond.new_zeros(())
                energy_loss = cond.new_zeros(())
                high_band_loss = cond.new_zeros(())
                high_band_temporal_loss = cond.new_zeros(())
                high_band_second_loss = cond.new_zeros(())
                high_band_excess_ratio = cond.new_zeros(())
                anchor_high_band_excess_ratio = cond.new_zeros(())
                decoded_waveform_high_ratio_loss = cond.new_zeros(())
                decoded_waveform_high_ratio_match_loss = cond.new_zeros(())
                decoded_waveform_high_ratio_topk_excess_loss = cond.new_zeros(())
                decoded_waveform_global_high_ratio_loss = cond.new_zeros(())
                decoded_waveform_global_high_ratio_match_loss = cond.new_zeros(())
                decoded_waveform_spectral_centroid_match_loss = cond.new_zeros(())
                decoded_waveform_spectral_centroid_topk_drift_loss = cond.new_zeros(())
                decoded_waveform_spectral_centroid_topk_excess_loss = cond.new_zeros(())
                decoded_waveform_teacher_conditioned_drift_loss = cond.new_zeros(())
                decoded_waveform_high_flatness_loss = cond.new_zeros(())
                decoded_waveform_high_flatness_match_loss = cond.new_zeros(())
                decoded_waveform_high_flatness_topk_excess_loss = cond.new_zeros(())
                decoded_waveform_weighted_high_flatness_topk_excess_loss = cond.new_zeros(())
                decoded_waveform_weighted_high_zcr_topk_excess_loss = cond.new_zeros(())
                decoded_waveform_rms_match_loss = cond.new_zeros(())
                decoded_waveform_derivative_rms_match_loss = cond.new_zeros(())
                decoded_waveform_derivative_rms_excess_loss = cond.new_zeros(())
                decoded_waveform_soft_zcr_match_loss = cond.new_zeros(())
                decoded_waveform_soft_zcr_excess_loss = cond.new_zeros(())
                decoded_waveform_framed_soft_zcr_match_loss = cond.new_zeros(())
                decoded_waveform_framed_soft_zcr_excess_loss = cond.new_zeros(())
                decoded_waveform_framed_soft_zcr_topk_excess_loss = cond.new_zeros(())
                decoded_waveform_rms_underfill_topk_loss = cond.new_zeros(())
                decoded_waveform_rms_envelope_match_loss = cond.new_zeros(())
                decoded_waveform_rms_envelope_second_delta_match_loss = cond.new_zeros(())
                decoded_waveform_multires_stft_loss = cond.new_zeros(())
                decoded_waveform_spectral_envelope_match_loss = cond.new_zeros(())
                decoded_waveform_peak_excess_loss = cond.new_zeros(())
                decoded_waveform_peak_cap_loss = cond.new_zeros(())
                decoded_waveform_ctc_loss_value = cond.new_zeros(())
                decoded_waveform_ssl_emission_match_loss = cond.new_zeros(())
                decoded_waveform_wavlm_profile_loss = cond.new_zeros(())
                decoded_waveform_mid_high_ratio_loss = cond.new_zeros(())
                decoded_waveform_mid_high_ratio_match_loss = cond.new_zeros(())
                decoded_waveform_tail_high_ratio_loss = cond.new_zeros(())
                decoded_waveform_tail_high_ratio_match_loss = cond.new_zeros(())
                tail_rollout_loss = cond.new_zeros(())
                tail_temporal_loss = cond.new_zeros(())
                tail_high_band_loss = cond.new_zeros(())
                tail_high_band_temporal_loss = cond.new_zeros(())
                tail_high_band_excess_ratio = cond.new_zeros(())
                mid_rollout_loss = cond.new_zeros(())
                mid_temporal_loss = cond.new_zeros(())
                mid_high_band_loss = cond.new_zeros(())
                mid_high_band_temporal_loss = cond.new_zeros(())
                mid_high_band_excess_ratio = cond.new_zeros(())
                focus_rollout_loss = cond.new_zeros(())
                focus_temporal_loss = cond.new_zeros(())
                focus_high_band_loss = cond.new_zeros(())
                focus_high_band_temporal_loss = cond.new_zeros(())
                focus_high_band_excess_ratio = cond.new_zeros(())
                self_rollout_bridge_loss = cond.new_zeros(())
                self_rollout_direction_loss = cond.new_zeros(())
                self_rollout_norm_loss = cond.new_zeros(())
                rollout_trajectory_loss = cond.new_zeros(())
                conditional_mel_critic_loss = cond.new_zeros(())
                conditional_mel_critic_discriminator_loss = cond.new_zeros(())
                text_rollout_contrastive_loss = cond.new_zeros(())
                cond_rollout_contrastive_loss = cond.new_zeros(())
                anchor_rollout_loss = cond.new_zeros(())
                if rollout_weight > 0:
                    y = noise.clone()
                    anchor_y = noise.clone() if functional_anchor is not None and float(args.anchor_rollout_loss_weight) > 0.0 else None
                    bad_y = noise.clone() if float(args.text_rollout_contrastive_loss_weight) > 0.0 else None
                    bad_cond_y = noise.clone() if bad_cond is not None and float(args.cond_rollout_contrastive_loss_weight) > 0.0 else None
                    self_weight_sum = 0.0
                    for interval in range(int(args.student_steps)):
                        self_interval_weight = (
                            float(self_rollout_interval_loss_weights[interval])
                            if self_rollout_interval_loss_weights
                            else 1.0
                        )
                        t0 = student_times[interval]
                        dt = float(student_times[interval + 1] - student_times[interval])
                        time_tensor = torch.full((y.shape[0],), t0, device=device, dtype=y.dtype)
                        pred_flow = cfg_flow(student, y, cond, text_ids, time_tensor, current_cfg)
                        if float(args.self_rollout_bridge_loss_weight) > 0.0:
                            remaining = max(1e-6, float(1.0 - t0))
                            self_target_flow = ((teacher_y - y) / remaining).detach()
                            self_rollout_bridge_loss = self_rollout_bridge_loss + self_interval_weight * F.mse_loss(
                                pred_flow[loss_mask],
                                self_target_flow[loss_mask],
                            )
                            if float(args.self_rollout_direction_loss_weight) > 0.0:
                                self_rollout_direction_loss = (
                                    self_rollout_direction_loss
                                    + self_interval_weight * direction_loss(pred_flow, self_target_flow, loss_mask)
                                )
                            if float(args.self_rollout_norm_loss_weight) > 0.0:
                                self_rollout_norm_loss = (
                                    self_rollout_norm_loss
                                    + self_interval_weight * norm_ratio_loss(pred_flow, self_target_flow, loss_mask)
                                )
                            self_weight_sum += self_interval_weight
                        y = y + dt * pred_flow
                        if float(args.rollout_trajectory_loss_weight) > 0.0:
                            trajectory_interval_weight = (
                                float(rollout_trajectory_interval_loss_weights[interval])
                                if rollout_trajectory_interval_loss_weights
                                else 1.0
                            )
                            target_state = states[nearest_state_index(state_times, student_times[interval + 1])].detach()
                            rollout_trajectory_loss = rollout_trajectory_loss + trajectory_interval_weight * F.mse_loss(
                                y[loss_mask],
                                target_state[loss_mask],
                            )
                        if anchor_y is not None:
                            with torch.no_grad():
                                anchor_pred_flow = cfg_flow(
                                    functional_anchor,
                                    anchor_y,
                                    cond,
                                    text_ids,
                                    time_tensor,
                                    current_cfg,
                                )
                            anchor_y = anchor_y + dt * anchor_pred_flow
                        if bad_y is not None:
                            bad_pred_flow = cfg_flow(student, bad_y, cond, bad_text_ids, time_tensor, current_cfg)
                            bad_y = bad_y + dt * bad_pred_flow
                        if bad_cond_y is not None:
                            bad_cond_pred_flow = cfg_flow(student, bad_cond_y, bad_cond, text_ids, time_tensor, current_cfg)
                            bad_cond_y = bad_cond_y + dt * bad_cond_pred_flow
                    if float(args.self_rollout_bridge_loss_weight) > 0.0:
                        self_denom = float(self_weight_sum or max(1, int(args.student_steps)))
                        self_rollout_bridge_loss = self_rollout_bridge_loss / self_denom
                        self_rollout_direction_loss = self_rollout_direction_loss / self_denom
                        self_rollout_norm_loss = self_rollout_norm_loss / self_denom
                    if float(args.rollout_trajectory_loss_weight) > 0.0:
                        trajectory_denom = (
                            sum(float(value) for value in rollout_trajectory_interval_loss_weights)
                            if rollout_trajectory_interval_loss_weights
                            else float(max(1, int(args.student_steps)))
                        )
                        rollout_trajectory_loss = rollout_trajectory_loss / float(max(1e-6, trajectory_denom))
                    if cond_frames > 0:
                        y[:, :cond_frames, :] = cond[:, :cond_frames, :]
                    if (
                        conditional_mel_critic is not None
                        and conditional_mel_critic_optimizer is not None
                        and step >= int(args.conditional_mel_critic_start_step)
                    ):
                        teacher_for_critic = teacher_y[:, int(cond_frames) :, :].detach()
                        student_for_critic = y[:, int(cond_frames) :, :]
                        teacher_for_critic, student_for_critic = crop_sequence_frames(
                            teacher_for_critic,
                            student_for_critic,
                            int(args.conditional_mel_critic_max_frames),
                            random_crop=True,
                        )
                        if int(teacher_for_critic.shape[1]) > 0:
                            if step % max(1, int(args.conditional_mel_critic_update_every)) == 0:
                                conditional_mel_critic_optimizer.zero_grad(set_to_none=True)
                                real_logits = conditional_mel_critic(
                                    teacher_for_critic,
                                    teacher_for_critic,
                                )
                                fake_logits = conditional_mel_critic(
                                    teacher_for_critic,
                                    student_for_critic.detach(),
                                )
                                conditional_mel_critic_discriminator_loss = (
                                    F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
                                )
                                conditional_mel_critic_discriminator_loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    conditional_mel_critic.parameters(),
                                    float(args.conditional_mel_critic_grad_clip),
                                )
                                conditional_mel_critic_optimizer.step()
                            set_requires_grad(conditional_mel_critic, False)
                            conditional_mel_critic_loss = F.softplus(
                                -conditional_mel_critic(teacher_for_critic, student_for_critic)
                            ).mean()
                            set_requires_grad(conditional_mel_critic, True)
                    rollout_loss = F.mse_loss(y[loss_mask], teacher_y[loss_mask])
                    if bad_y is not None:
                        if cond_frames > 0:
                            bad_y[:, :cond_frames, :] = cond[:, :cond_frames, :]
                        bad_rollout_loss = F.mse_loss(bad_y[loss_mask], teacher_y[loss_mask])
                        text_rollout_contrastive_loss = F.relu(
                            float(args.text_rollout_contrastive_margin) + rollout_loss - bad_rollout_loss
                        )
                    if bad_cond_y is not None:
                        if cond_frames > 0:
                            bad_cond_y[:, :cond_frames, :] = bad_cond[:, :cond_frames, :]
                        bad_cond_rollout_loss = F.mse_loss(bad_cond_y[loss_mask], teacher_y[loss_mask])
                        cond_rollout_contrastive_loss = F.relu(
                            float(args.cond_rollout_contrastive_margin) + rollout_loss - bad_cond_rollout_loss
                        )
                    if anchor_y is not None:
                        if cond_frames > 0:
                            anchor_y[:, :cond_frames, :] = cond[:, :cond_frames, :]
                        anchor_rollout_loss = F.smooth_l1_loss(
                            y[loss_mask],
                            anchor_y[loss_mask],
                            beta=0.2,
                        )
                        if float(args.anchor_high_band_excess_ratio_loss_weight) > 0.0:
                            anchor_high_band_excess_ratio = high_band_excess_ratio_loss(
                                y,
                                anchor_y,
                                loss_mask,
                                int(args.high_band_start_bin),
                                float(args.high_band_excess_ratio_margin),
                            )
                    temporal_loss = temporal_delta_loss(y, teacher_y, loss_mask)
                    energy_loss = frame_energy_loss(y, teacher_y, loss_mask)
                    high_band_loss = high_band_match_loss(y, teacher_y, loss_mask, int(args.high_band_start_bin))
                    high_band_temporal_loss = high_band_temporal_delta_loss(
                        y,
                        teacher_y,
                        loss_mask,
                        int(args.high_band_start_bin),
                    )
                    high_band_second_loss = high_band_second_delta_loss(
                        y,
                        teacher_y,
                        loss_mask,
                        int(args.high_band_start_bin),
                    )
                    high_band_excess_ratio = high_band_excess_ratio_loss(
                        y,
                        teacher_y,
                        loss_mask,
                        int(args.high_band_start_bin),
                        float(args.high_band_excess_ratio_margin),
                    )
                    if (
                        float(args.tail_rollout_loss_weight) > 0.0
                        or float(args.tail_temporal_delta_loss_weight) > 0.0
                        or float(args.tail_high_band_match_loss_weight) > 0.0
                        or float(args.tail_high_band_temporal_delta_loss_weight) > 0.0
                        or float(args.tail_high_band_excess_ratio_loss_weight) > 0.0
                    ):
                        tail_mask = tail_loss_mask(loss_mask, float(args.tail_start_ratio))
                        if bool(tail_mask.any()):
                            tail_rollout_loss = F.mse_loss(y[tail_mask], teacher_y[tail_mask])
                            tail_temporal_loss = temporal_delta_loss(y, teacher_y, tail_mask)
                            tail_high_band_loss = high_band_match_loss(
                                y,
                                teacher_y,
                                tail_mask,
                                int(args.high_band_start_bin),
                            )
                            tail_high_band_temporal_loss = high_band_temporal_delta_loss(
                                y,
                                teacher_y,
                                tail_mask,
                                int(args.high_band_start_bin),
                            )
                            tail_high_band_excess_ratio = high_band_excess_ratio_loss(
                                y,
                                teacher_y,
                                tail_mask,
                                int(args.high_band_start_bin),
                                float(args.high_band_excess_ratio_margin),
                            )
                    if (
                        float(args.mid_rollout_loss_weight) > 0.0
                        or float(args.mid_temporal_delta_loss_weight) > 0.0
                        or float(args.mid_high_band_match_loss_weight) > 0.0
                        or float(args.mid_high_band_temporal_delta_loss_weight) > 0.0
                        or float(args.mid_high_band_excess_ratio_loss_weight) > 0.0
                    ):
                        mid_mask = window_loss_mask(
                            loss_mask,
                            float(args.mid_start_ratio),
                            float(args.mid_end_ratio),
                        )
                        if bool(mid_mask.any()):
                            mid_rollout_loss = F.mse_loss(y[mid_mask], teacher_y[mid_mask])
                            mid_temporal_loss = temporal_delta_loss(y, teacher_y, mid_mask)
                            mid_high_band_loss = high_band_match_loss(
                                y,
                                teacher_y,
                                mid_mask,
                                int(args.high_band_start_bin),
                            )
                            mid_high_band_temporal_loss = high_band_temporal_delta_loss(
                                y,
                                teacher_y,
                                mid_mask,
                                int(args.high_band_start_bin),
                            )
                            mid_high_band_excess_ratio = high_band_excess_ratio_loss(
                                y,
                                teacher_y,
                                mid_mask,
                                int(args.high_band_start_bin),
                                float(args.high_band_excess_ratio_margin),
                            )
                    if (
                        float(args.focus_rollout_loss_weight) > 0.0
                        or float(args.focus_temporal_delta_loss_weight) > 0.0
                        or float(args.focus_high_band_match_loss_weight) > 0.0
                        or float(args.focus_high_band_temporal_delta_loss_weight) > 0.0
                        or float(args.focus_high_band_excess_ratio_loss_weight) > 0.0
                    ):
                        focus_mask = window_loss_mask(
                            loss_mask,
                            float(args.focus_start_ratio),
                            float(args.focus_end_ratio),
                        )
                        if bool(focus_mask.any()):
                            focus_rollout_loss = F.mse_loss(y[focus_mask], teacher_y[focus_mask])
                            focus_temporal_loss = temporal_delta_loss(y, teacher_y, focus_mask)
                            focus_high_band_loss = high_band_match_loss(
                                y,
                                teacher_y,
                                focus_mask,
                                int(args.high_band_start_bin),
                            )
                            focus_high_band_temporal_loss = high_band_temporal_delta_loss(
                                y,
                                teacher_y,
                                focus_mask,
                                int(args.high_band_start_bin),
                            )
                            focus_high_band_excess_ratio = high_band_excess_ratio_loss(
                                y,
                                teacher_y,
                                focus_mask,
                                int(args.high_band_start_bin),
                                float(args.high_band_excess_ratio_margin),
                            )
                    if decoded_vocoder is not None and int(args.decoded_waveform_loss_every) > 0 and step % int(args.decoded_waveform_loss_every) == 0:
                        if bool(args.decoded_waveform_generated_only):
                            # Match F5TTS inference: the vocoder only sees generated frames.
                            pred_mel = y[:, int(cond_frames) :, :].permute(0, 2, 1).float()
                            target_mel = teacher_y[:, int(cond_frames) :, :].permute(0, 2, 1).float()
                        else:
                            pred_mel = y.permute(0, 2, 1).float()
                            target_mel = teacher_y.permute(0, 2, 1).float()
                        pred_mel, target_mel, did_crop_waveform = crop_mel_frames(
                            pred_mel,
                            target_mel,
                            int(args.decoded_waveform_max_frames),
                            str(args.decoded_waveform_crop_mode),
                        )
                        if did_crop_waveform:
                            decoded_waveform_crops += 1
                        if (
                            str(args.decoded_waveform_crop_mode) != "skip"
                            or int(pred_mel.shape[-1]) <= int(args.decoded_waveform_max_frames)
                        ):
                            with torch.no_grad():
                                target_wave = decoded_vocoder.decode(target_mel).detach()
                            pred_wave = decoded_vocoder.decode(pred_mel)
                            if min(int(pred_wave.shape[-1]), int(target_wave.shape[-1])) >= 2048:
                                decoded_waveform_updates += 1
                                decoded_waveform_high_ratio_loss = waveform_high_ratio_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_sample_rate),
                                    float(args.decoded_waveform_cutoff_hz),
                                    float(args.decoded_waveform_margin),
                                )
                                decoded_waveform_high_ratio_match_loss = waveform_high_ratio_match_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_sample_rate),
                                    float(args.decoded_waveform_cutoff_hz),
                                )
                                decoded_waveform_high_ratio_topk_excess_loss = waveform_high_ratio_topk_excess_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_sample_rate),
                                    float(args.decoded_waveform_cutoff_hz),
                                    float(args.decoded_waveform_margin),
                                    float(args.decoded_waveform_local_topk_fraction),
                                )
                                decoded_waveform_global_high_ratio_loss = waveform_global_high_ratio_excess_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_sample_rate),
                                    float(args.decoded_waveform_cutoff_hz),
                                    float(args.decoded_waveform_margin),
                                )
                                decoded_waveform_global_high_ratio_match_loss = waveform_global_high_ratio_match_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_sample_rate),
                                    float(args.decoded_waveform_cutoff_hz),
                                )
                                decoded_waveform_spectral_centroid_match_loss = (
                                    waveform_spectral_centroid_match_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                    )
                                )
                                decoded_waveform_spectral_centroid_topk_drift_loss = (
                                    waveform_spectral_centroid_topk_drift_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_local_topk_fraction),
                                    )
                                )
                                decoded_waveform_spectral_centroid_topk_excess_loss = (
                                    waveform_spectral_centroid_topk_excess_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_spectral_centroid_margin_hz),
                                        float(args.decoded_waveform_local_topk_fraction),
                                    )
                                )
                                decoded_waveform_teacher_conditioned_drift_loss = (
                                    waveform_teacher_conditioned_drift_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                        float(args.decoded_waveform_teacher_conditioned_ratio_margin),
                                        float(args.decoded_waveform_teacher_conditioned_centroid_margin_hz),
                                        float(args.decoded_waveform_local_topk_fraction),
                                    )
                                )
                                decoded_waveform_high_flatness_loss = waveform_high_flatness_excess_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_sample_rate),
                                    float(args.decoded_waveform_cutoff_hz),
                                    float(args.decoded_waveform_high_flatness_margin),
                                )
                                decoded_waveform_high_flatness_match_loss = waveform_high_flatness_match_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_sample_rate),
                                    float(args.decoded_waveform_cutoff_hz),
                                )
                                decoded_waveform_high_flatness_topk_excess_loss = (
                                    waveform_high_flatness_topk_excess_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                        float(args.decoded_waveform_high_flatness_margin),
                                        float(args.decoded_waveform_local_topk_fraction),
                                    )
                                )
                                decoded_waveform_weighted_high_flatness_topk_excess_loss = (
                                    waveform_weighted_high_flatness_topk_excess_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                        float(args.decoded_waveform_weighted_high_margin),
                                        float(args.decoded_waveform_local_topk_fraction),
                                    )
                                )
                                decoded_waveform_rms_match_loss = waveform_rms_match_loss(pred_wave, target_wave)
                                decoded_waveform_derivative_rms_match_loss = waveform_derivative_rms_match_loss(
                                    pred_wave,
                                    target_wave,
                                )
                                decoded_waveform_derivative_rms_excess_loss = waveform_derivative_rms_excess_loss(
                                    pred_wave,
                                    target_wave,
                                    float(args.decoded_waveform_derivative_rms_margin),
                                )
                                decoded_waveform_soft_zcr_match_loss = waveform_soft_zcr_match_loss(
                                    pred_wave,
                                    target_wave,
                                    float(args.decoded_waveform_soft_zcr_gain),
                                )
                                decoded_waveform_soft_zcr_excess_loss = waveform_soft_zcr_excess_loss(
                                    pred_wave,
                                    target_wave,
                                    float(args.decoded_waveform_soft_zcr_gain),
                                    float(args.decoded_waveform_soft_zcr_margin),
                                )
                                decoded_waveform_framed_soft_zcr_match_loss = waveform_framed_soft_zcr_match_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_framed_soft_zcr_window),
                                    int(args.decoded_waveform_framed_soft_zcr_hop),
                                    float(args.decoded_waveform_soft_zcr_gain),
                                )
                                decoded_waveform_framed_soft_zcr_excess_loss = waveform_framed_soft_zcr_excess_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_framed_soft_zcr_window),
                                    int(args.decoded_waveform_framed_soft_zcr_hop),
                                    float(args.decoded_waveform_soft_zcr_gain),
                                    float(args.decoded_waveform_framed_soft_zcr_margin),
                                )
                                decoded_waveform_framed_soft_zcr_topk_excess_loss = (
                                    waveform_framed_soft_zcr_topk_excess_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_framed_soft_zcr_window),
                                        int(args.decoded_waveform_framed_soft_zcr_hop),
                                        float(args.decoded_waveform_soft_zcr_gain),
                                        float(args.decoded_waveform_framed_soft_zcr_margin),
                                        float(args.decoded_waveform_local_topk_fraction),
                                    )
                                )
                                decoded_waveform_weighted_high_zcr_topk_excess_loss = (
                                    waveform_weighted_high_zcr_topk_excess_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                        int(args.decoded_waveform_framed_soft_zcr_window),
                                        int(args.decoded_waveform_framed_soft_zcr_hop),
                                        float(args.decoded_waveform_soft_zcr_gain),
                                        float(args.decoded_waveform_weighted_high_margin),
                                        float(args.decoded_waveform_local_topk_fraction),
                                    )
                                )
                                decoded_waveform_rms_underfill_topk_loss = waveform_rms_underfill_topk_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_rms_envelope_window),
                                    int(args.decoded_waveform_rms_envelope_hop),
                                    float(args.decoded_waveform_local_topk_fraction),
                                )
                                decoded_waveform_rms_envelope_match_loss = waveform_rms_envelope_match_loss(
                                    pred_wave,
                                    target_wave,
                                    int(args.decoded_waveform_rms_envelope_window),
                                    int(args.decoded_waveform_rms_envelope_hop),
                                )
                                decoded_waveform_rms_envelope_second_delta_match_loss = (
                                    waveform_rms_envelope_second_delta_match_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_rms_envelope_window),
                                        int(args.decoded_waveform_rms_envelope_hop),
                                    )
                                )
                                decoded_waveform_multires_stft_loss = waveform_multires_stft_loss(
                                    pred_wave,
                                    target_wave,
                                    str(args.decoded_waveform_multires_stft_ffts),
                                )
                                decoded_waveform_spectral_envelope_match_loss = (
                                    waveform_spectral_envelope_match_loss(
                                        pred_wave,
                                        target_wave,
                                        int(args.decoded_waveform_spectral_envelope_n_fft),
                                        int(args.decoded_waveform_spectral_envelope_hop),
                                        int(args.decoded_waveform_spectral_envelope_smooth_bins),
                                    )
                                )
                                decoded_waveform_peak_excess_loss = waveform_peak_excess_loss(
                                    pred_wave,
                                    target_wave,
                                    float(args.decoded_waveform_margin),
                                )
                                decoded_waveform_peak_cap_loss = waveform_peak_cap_loss(
                                    pred_wave,
                                    float(args.decoded_waveform_peak_cap),
                                )
                                if decoded_ctc_model is not None and float(args.decoded_waveform_ctc_loss_weight) > 0.0:
                                    ctc_target_text = str(payload.get("gen_text") or payload.get("original_gen_text") or "")
                                    if str(args.decoded_waveform_ctc_target_mode) == "keywords":
                                        ctc_target_text = keyword_ctc_text(
                                            str(payload.get("gen_text") or ""),
                                            str(payload.get("original_gen_text") or ""),
                                        )
                                    elif str(args.decoded_waveform_ctc_target_mode) == "domain":
                                        ctc_target_text = domain_ctc_text(
                                            str(payload.get("gen_text") or ""),
                                            str(payload.get("original_gen_text") or ""),
                                        )
                                    decoded_waveform_ctc_loss_value = decoded_waveform_ctc_loss(
                                        asr_model=decoded_ctc_model,
                                        asr_sample_rate=int(decoded_ctc_sample_rate),
                                        label_to_index=decoded_ctc_label_to_index,
                                        blank_index=int(decoded_ctc_blank_index),
                                        pred_wave=pred_wave,
                                        target_text=ctc_target_text,
                                        source_sample_rate=int(args.decoded_waveform_sample_rate),
                                    )
                                if (
                                    decoded_ssl_model is not None
                                    and float(args.decoded_waveform_ssl_emission_match_loss_weight) > 0.0
                                ):
                                    decoded_waveform_ssl_emission_match_loss = (
                                        compute_decoded_waveform_ssl_emission_match_loss(
                                            ssl_model=decoded_ssl_model,
                                            ssl_sample_rate=int(decoded_ssl_sample_rate),
                                            pred_wave=pred_wave,
                                            target_wave=target_wave,
                                            source_sample_rate=int(args.decoded_waveform_sample_rate),
                                        )
                                    )
                                if (
                                    decoded_wavlm_profile_model is not None
                                    and decoded_wavlm_profile_ref_embedding is not None
                                    and float(args.decoded_waveform_wavlm_profile_loss_weight) > 0.0
                                ):
                                    decoded_waveform_wavlm_profile_loss = (
                                        compute_decoded_waveform_wavlm_profile_loss(
                                            wavlm_model=decoded_wavlm_profile_model,
                                            wavlm_sample_rate=int(decoded_wavlm_profile_sample_rate),
                                            ref_embedding=decoded_wavlm_profile_ref_embedding,
                                            pred_wave=pred_wave,
                                            source_sample_rate=int(args.decoded_waveform_sample_rate),
                                        )
                                    )
                                mid_pred_wave = crop_wave_ratio(
                                    pred_wave,
                                    float(args.decoded_waveform_mid_start_ratio),
                                    float(args.decoded_waveform_mid_end_ratio),
                                )
                                mid_target_wave = crop_wave_ratio(
                                    target_wave,
                                    float(args.decoded_waveform_mid_start_ratio),
                                    float(args.decoded_waveform_mid_end_ratio),
                                )
                                tail_pred_wave = crop_wave_ratio(
                                    pred_wave,
                                    float(args.decoded_waveform_tail_start_ratio),
                                    1.0,
                                )
                                tail_target_wave = crop_wave_ratio(
                                    target_wave,
                                    float(args.decoded_waveform_tail_start_ratio),
                                    1.0,
                                )
                                if min(int(mid_pred_wave.shape[-1]), int(mid_target_wave.shape[-1])) >= 2048:
                                    decoded_waveform_mid_high_ratio_loss = waveform_high_ratio_loss(
                                        mid_pred_wave,
                                        mid_target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                        float(args.decoded_waveform_margin),
                                    )
                                    decoded_waveform_mid_high_ratio_match_loss = waveform_high_ratio_match_loss(
                                        mid_pred_wave,
                                        mid_target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                    )
                                if min(int(tail_pred_wave.shape[-1]), int(tail_target_wave.shape[-1])) >= 2048:
                                    decoded_waveform_tail_high_ratio_loss = waveform_high_ratio_loss(
                                        tail_pred_wave,
                                        tail_target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                        float(args.decoded_waveform_margin),
                                    )
                                    decoded_waveform_tail_high_ratio_match_loss = waveform_high_ratio_match_loss(
                                        tail_pred_wave,
                                        tail_target_wave,
                                        int(args.decoded_waveform_sample_rate),
                                        float(args.decoded_waveform_cutoff_hz),
                                    )
                            else:
                                decoded_waveform_skip_short += 1
                        else:
                            decoded_waveform_skip_frames += 1
                    sample_loss = (
                        sample_loss
                        + rollout_weight * rollout_loss
                        + float(args.temporal_delta_loss_weight) * temporal_loss
                        + float(args.frame_energy_loss_weight) * energy_loss
                        + float(args.high_band_match_loss_weight) * high_band_loss
                        + float(args.high_band_temporal_delta_loss_weight) * high_band_temporal_loss
                        + float(args.high_band_second_delta_loss_weight) * high_band_second_loss
                        + float(args.high_band_excess_ratio_loss_weight) * high_band_excess_ratio
                        + float(args.anchor_high_band_excess_ratio_loss_weight) * anchor_high_band_excess_ratio
                        + float(args.tail_rollout_loss_weight) * tail_rollout_loss
                        + float(args.tail_temporal_delta_loss_weight) * tail_temporal_loss
                        + float(args.tail_high_band_match_loss_weight) * tail_high_band_loss
                        + float(args.tail_high_band_temporal_delta_loss_weight) * tail_high_band_temporal_loss
                        + float(args.tail_high_band_excess_ratio_loss_weight) * tail_high_band_excess_ratio
                        + float(args.mid_rollout_loss_weight) * mid_rollout_loss
                        + float(args.mid_temporal_delta_loss_weight) * mid_temporal_loss
                        + float(args.mid_high_band_match_loss_weight) * mid_high_band_loss
                        + float(args.mid_high_band_temporal_delta_loss_weight) * mid_high_band_temporal_loss
                        + float(args.mid_high_band_excess_ratio_loss_weight) * mid_high_band_excess_ratio
                        + float(args.focus_rollout_loss_weight) * focus_rollout_loss
                        + float(args.focus_temporal_delta_loss_weight) * focus_temporal_loss
                        + float(args.focus_high_band_match_loss_weight) * focus_high_band_loss
                        + float(args.focus_high_band_temporal_delta_loss_weight) * focus_high_band_temporal_loss
                        + float(args.focus_high_band_excess_ratio_loss_weight) * focus_high_band_excess_ratio
                        + float(args.decoded_waveform_high_ratio_loss_weight) * decoded_waveform_high_ratio_loss
                        + float(args.decoded_waveform_high_ratio_match_loss_weight) * decoded_waveform_high_ratio_match_loss
                        + float(args.decoded_waveform_high_ratio_topk_excess_loss_weight)
                        * decoded_waveform_high_ratio_topk_excess_loss
                        + float(args.decoded_waveform_global_high_ratio_loss_weight) * decoded_waveform_global_high_ratio_loss
                        + float(args.decoded_waveform_global_high_ratio_match_loss_weight)
                        * decoded_waveform_global_high_ratio_match_loss
                        + float(args.decoded_waveform_spectral_centroid_match_loss_weight)
                        * decoded_waveform_spectral_centroid_match_loss
                        + float(args.decoded_waveform_spectral_centroid_topk_drift_loss_weight)
                        * decoded_waveform_spectral_centroid_topk_drift_loss
                        + float(args.decoded_waveform_spectral_centroid_topk_excess_loss_weight)
                        * decoded_waveform_spectral_centroid_topk_excess_loss
                        + float(args.decoded_waveform_teacher_conditioned_drift_loss_weight)
                        * decoded_waveform_teacher_conditioned_drift_loss
                        + float(args.decoded_waveform_high_flatness_excess_loss_weight)
                        * decoded_waveform_high_flatness_loss
                        + float(args.decoded_waveform_high_flatness_match_loss_weight)
                        * decoded_waveform_high_flatness_match_loss
                        + float(args.decoded_waveform_high_flatness_topk_excess_loss_weight)
                        * decoded_waveform_high_flatness_topk_excess_loss
                        + float(args.decoded_waveform_weighted_high_flatness_topk_excess_loss_weight)
                        * decoded_waveform_weighted_high_flatness_topk_excess_loss
                        + float(args.decoded_waveform_weighted_high_zcr_topk_excess_loss_weight)
                        * decoded_waveform_weighted_high_zcr_topk_excess_loss
                        + float(args.decoded_waveform_rms_match_loss_weight) * decoded_waveform_rms_match_loss
                        + float(args.decoded_waveform_derivative_rms_match_loss_weight)
                        * decoded_waveform_derivative_rms_match_loss
                        + float(args.decoded_waveform_derivative_rms_excess_loss_weight)
                        * decoded_waveform_derivative_rms_excess_loss
                        + float(args.decoded_waveform_soft_zcr_match_loss_weight)
                        * decoded_waveform_soft_zcr_match_loss
                        + float(args.decoded_waveform_soft_zcr_excess_loss_weight)
                        * decoded_waveform_soft_zcr_excess_loss
                        + float(args.decoded_waveform_framed_soft_zcr_match_loss_weight)
                        * decoded_waveform_framed_soft_zcr_match_loss
                        + float(args.decoded_waveform_framed_soft_zcr_excess_loss_weight)
                        * decoded_waveform_framed_soft_zcr_excess_loss
                        + float(args.decoded_waveform_framed_soft_zcr_topk_excess_loss_weight)
                        * decoded_waveform_framed_soft_zcr_topk_excess_loss
                        + float(args.decoded_waveform_rms_underfill_topk_loss_weight)
                        * decoded_waveform_rms_underfill_topk_loss
                        + float(args.decoded_waveform_rms_envelope_match_loss_weight)
                        * decoded_waveform_rms_envelope_match_loss
                        + float(args.decoded_waveform_rms_envelope_second_delta_match_loss_weight)
                        * decoded_waveform_rms_envelope_second_delta_match_loss
                        + float(args.decoded_waveform_multires_stft_loss_weight)
                        * decoded_waveform_multires_stft_loss
                        + float(args.decoded_waveform_spectral_envelope_match_loss_weight)
                        * decoded_waveform_spectral_envelope_match_loss
                        + float(args.decoded_waveform_peak_excess_loss_weight) * decoded_waveform_peak_excess_loss
                        + float(args.decoded_waveform_peak_cap_loss_weight) * decoded_waveform_peak_cap_loss
                        + float(args.decoded_waveform_ctc_loss_weight) * decoded_waveform_ctc_loss_value
                        + float(args.decoded_waveform_ssl_emission_match_loss_weight)
                        * decoded_waveform_ssl_emission_match_loss
                        + float(args.decoded_waveform_wavlm_profile_loss_weight)
                        * decoded_waveform_wavlm_profile_loss
                        + float(args.decoded_waveform_mid_high_ratio_loss_weight)
                        * decoded_waveform_mid_high_ratio_loss
                        + float(args.decoded_waveform_mid_high_ratio_match_loss_weight)
                        * decoded_waveform_mid_high_ratio_match_loss
                        + float(args.decoded_waveform_tail_high_ratio_loss_weight)
                        * decoded_waveform_tail_high_ratio_loss
                        + float(args.decoded_waveform_tail_high_ratio_match_loss_weight)
                        * decoded_waveform_tail_high_ratio_match_loss
                        + float(args.self_rollout_bridge_loss_weight) * self_rollout_bridge_loss
                        + float(args.self_rollout_direction_loss_weight) * self_rollout_direction_loss
                        + float(args.self_rollout_norm_loss_weight) * self_rollout_norm_loss
                        + float(args.rollout_trajectory_loss_weight) * rollout_trajectory_loss
                        + float(args.conditional_mel_critic_loss_weight) * conditional_mel_critic_loss
                        + float(args.text_rollout_contrastive_loss_weight) * text_rollout_contrastive_loss
                        + float(args.cond_rollout_contrastive_loss_weight) * cond_rollout_contrastive_loss
                        + float(args.anchor_rollout_loss_weight) * anchor_rollout_loss
                    )
                parameter_anchor_loss = anchor_weight_loss(student, anchor_parameters) if anchor_parameters else cond.new_zeros(())
                if float(args.anchor_weight_loss_weight) > 0.0:
                    sample_loss = sample_loss + float(args.anchor_weight_loss_weight) * parameter_anchor_loss
                micro_loss = sample_loss if micro_loss is None else micro_loss + sample_loss
                bridge_flow_accum += float(sample_bridge.cpu())
                next_state_accum += float(sample_next.cpu())
                direction_accum += float(sample_direction.cpu())
                norm_accum += float(sample_norm.cpu())
                orthogonal_accum += float(sample_orthogonal.cpu())
                cond_delta_accum += float(sample_cond_delta.cpu())
                rollout_accum += float(rollout_loss.detach().cpu())
                temporal_accum += float(temporal_loss.detach().cpu())
                energy_accum += float(energy_loss.detach().cpu())
                high_band_accum += float(high_band_loss.detach().cpu())
                high_band_temporal_accum += float(high_band_temporal_loss.detach().cpu())
                high_band_second_accum += float(high_band_second_loss.detach().cpu())
                high_band_excess_ratio_accum += float(high_band_excess_ratio.detach().cpu())
                anchor_high_band_excess_ratio_accum += float(anchor_high_band_excess_ratio.detach().cpu())
                decoded_waveform_high_ratio_accum += float(decoded_waveform_high_ratio_loss.detach().cpu())
                decoded_waveform_high_ratio_match_accum += float(decoded_waveform_high_ratio_match_loss.detach().cpu())
                decoded_waveform_high_ratio_topk_excess_accum += float(
                    decoded_waveform_high_ratio_topk_excess_loss.detach().cpu()
                )
                decoded_waveform_global_high_ratio_accum += float(decoded_waveform_global_high_ratio_loss.detach().cpu())
                decoded_waveform_global_high_ratio_match_accum += float(
                    decoded_waveform_global_high_ratio_match_loss.detach().cpu()
                )
                decoded_waveform_spectral_centroid_match_accum += float(
                    decoded_waveform_spectral_centroid_match_loss.detach().cpu()
                )
                decoded_waveform_spectral_centroid_topk_drift_accum += float(
                    decoded_waveform_spectral_centroid_topk_drift_loss.detach().cpu()
                )
                decoded_waveform_spectral_centroid_topk_excess_accum += float(
                    decoded_waveform_spectral_centroid_topk_excess_loss.detach().cpu()
                )
                decoded_waveform_teacher_conditioned_drift_accum += float(
                    decoded_waveform_teacher_conditioned_drift_loss.detach().cpu()
                )
                decoded_waveform_high_flatness_accum += float(decoded_waveform_high_flatness_loss.detach().cpu())
                decoded_waveform_high_flatness_match_accum += float(
                    decoded_waveform_high_flatness_match_loss.detach().cpu()
                )
                decoded_waveform_high_flatness_topk_excess_accum += float(
                    decoded_waveform_high_flatness_topk_excess_loss.detach().cpu()
                )
                decoded_waveform_weighted_high_flatness_topk_excess_accum += float(
                    decoded_waveform_weighted_high_flatness_topk_excess_loss.detach().cpu()
                )
                decoded_waveform_weighted_high_zcr_topk_excess_accum += float(
                    decoded_waveform_weighted_high_zcr_topk_excess_loss.detach().cpu()
                )
                decoded_waveform_rms_match_accum += float(decoded_waveform_rms_match_loss.detach().cpu())
                decoded_waveform_derivative_rms_match_accum += float(
                    decoded_waveform_derivative_rms_match_loss.detach().cpu()
                )
                decoded_waveform_derivative_rms_excess_accum += float(
                    decoded_waveform_derivative_rms_excess_loss.detach().cpu()
                )
                decoded_waveform_soft_zcr_match_accum += float(decoded_waveform_soft_zcr_match_loss.detach().cpu())
                decoded_waveform_soft_zcr_excess_accum += float(decoded_waveform_soft_zcr_excess_loss.detach().cpu())
                decoded_waveform_framed_soft_zcr_match_accum += float(
                    decoded_waveform_framed_soft_zcr_match_loss.detach().cpu()
                )
                decoded_waveform_framed_soft_zcr_excess_accum += float(
                    decoded_waveform_framed_soft_zcr_excess_loss.detach().cpu()
                )
                decoded_waveform_framed_soft_zcr_topk_excess_accum += float(
                    decoded_waveform_framed_soft_zcr_topk_excess_loss.detach().cpu()
                )
                decoded_waveform_rms_underfill_topk_accum += float(
                    decoded_waveform_rms_underfill_topk_loss.detach().cpu()
                )
                decoded_waveform_rms_envelope_match_accum += float(
                    decoded_waveform_rms_envelope_match_loss.detach().cpu()
                )
                decoded_waveform_rms_envelope_second_delta_match_accum += float(
                    decoded_waveform_rms_envelope_second_delta_match_loss.detach().cpu()
                )
                decoded_waveform_multires_stft_accum += float(decoded_waveform_multires_stft_loss.detach().cpu())
                decoded_waveform_spectral_envelope_match_accum += float(
                    decoded_waveform_spectral_envelope_match_loss.detach().cpu()
                )
                decoded_waveform_peak_excess_accum += float(decoded_waveform_peak_excess_loss.detach().cpu())
                decoded_waveform_peak_cap_accum += float(decoded_waveform_peak_cap_loss.detach().cpu())
                decoded_waveform_ctc_accum += float(decoded_waveform_ctc_loss_value.detach().cpu())
                decoded_waveform_ssl_emission_match_accum += float(
                    decoded_waveform_ssl_emission_match_loss.detach().cpu()
                )
                decoded_waveform_wavlm_profile_accum += float(decoded_waveform_wavlm_profile_loss.detach().cpu())
                decoded_waveform_mid_high_ratio_accum += float(decoded_waveform_mid_high_ratio_loss.detach().cpu())
                decoded_waveform_mid_high_ratio_match_accum += float(
                    decoded_waveform_mid_high_ratio_match_loss.detach().cpu()
                )
                decoded_waveform_tail_high_ratio_accum += float(decoded_waveform_tail_high_ratio_loss.detach().cpu())
                decoded_waveform_tail_high_ratio_match_accum += float(
                    decoded_waveform_tail_high_ratio_match_loss.detach().cpu()
                )
                tail_rollout_accum += float(tail_rollout_loss.detach().cpu())
                tail_temporal_accum += float(tail_temporal_loss.detach().cpu())
                tail_high_band_accum += float(tail_high_band_loss.detach().cpu())
                tail_high_band_temporal_accum += float(tail_high_band_temporal_loss.detach().cpu())
                tail_high_band_excess_ratio_accum += float(tail_high_band_excess_ratio.detach().cpu())
                mid_rollout_accum += float(mid_rollout_loss.detach().cpu())
                mid_temporal_accum += float(mid_temporal_loss.detach().cpu())
                mid_high_band_accum += float(mid_high_band_loss.detach().cpu())
                mid_high_band_temporal_accum += float(mid_high_band_temporal_loss.detach().cpu())
                mid_high_band_excess_ratio_accum += float(mid_high_band_excess_ratio.detach().cpu())
                focus_rollout_accum += float(focus_rollout_loss.detach().cpu())
                focus_temporal_accum += float(focus_temporal_loss.detach().cpu())
                focus_high_band_accum += float(focus_high_band_loss.detach().cpu())
                focus_high_band_temporal_accum += float(focus_high_band_temporal_loss.detach().cpu())
                focus_high_band_excess_ratio_accum += float(focus_high_band_excess_ratio.detach().cpu())
                self_rollout_bridge_accum += float(self_rollout_bridge_loss.detach().cpu())
                self_rollout_direction_accum += float(self_rollout_direction_loss.detach().cpu())
                self_rollout_norm_accum += float(self_rollout_norm_loss.detach().cpu())
                rollout_trajectory_accum += float(rollout_trajectory_loss.detach().cpu())
                conditional_mel_critic_accum += float(conditional_mel_critic_loss.detach().cpu())
                conditional_mel_critic_discriminator_accum += float(
                    conditional_mel_critic_discriminator_loss.detach().cpu()
                )
                text_flow_contrastive_accum += float(sample_text_flow_contrastive.detach().cpu())
                text_rollout_contrastive_accum += float(text_rollout_contrastive_loss.detach().cpu())
                cond_flow_contrastive_accum += float(sample_cond_flow_contrastive.detach().cpu())
                cond_rollout_contrastive_accum += float(cond_rollout_contrastive_loss.detach().cpu())
                anchor_flow_accum += float(sample_anchor_flow.detach().cpu())
                anchor_rollout_accum += float(anchor_rollout_loss.detach().cpu())
                anchor_accum += float(parameter_anchor_loss.detach().cpu())
            if micro_loss is None:
                continue
            micro_loss = micro_loss / float(max(1, len(batch_rows)))
            (micro_loss / float(max(1, int(args.grad_accum_steps)))).backward()
            loss_accum += float(micro_loss.detach().cpu())

        torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.max_grad_norm))
        optimizer.step()
        loss_value = loss_accum / float(max(1, int(args.grad_accum_steps)))
        denom = float(max(1, int(args.grad_accum_steps) * int(args.batch_size)))
        rollout_value = rollout_accum / denom
        if loss_value < best_loss:
            best_loss = loss_value
            save_checkpoint(student, output_dir / checkpoint_name(args, "best"), step=step, args=args, loss=loss_value)
        if rollout_value > 0.0 and rollout_value < best_rollout_loss:
            best_rollout_loss = rollout_value
            save_checkpoint(
                student,
                output_dir / checkpoint_name(args, "best_rollout"),
                step=step,
                args=args,
                loss=loss_value,
            )
        if step % int(args.log_every) == 0 or step == 1:
            metrics = {
                "event": "train",
                "step": step,
                "loss": loss_value,
                "best_loss": best_loss,
                "best_rollout_loss": best_rollout_loss,
                "bridge_flow_loss": bridge_flow_accum / denom,
                "next_state_loss": next_state_accum / denom,
                "direction_loss": direction_accum / denom,
                "norm_loss": norm_accum / denom,
                "orthogonal_loss": orthogonal_accum / denom,
                "cond_delta_loss": cond_delta_accum / denom,
                "rollout_loss": rollout_value,
                "temporal_delta_loss": temporal_accum / denom,
                "frame_energy_loss": energy_accum / denom,
                "high_band_match_loss": high_band_accum / denom,
                "high_band_temporal_delta_loss": high_band_temporal_accum / denom,
                "high_band_second_delta_loss": high_band_second_accum / denom,
                "high_band_excess_ratio_loss": high_band_excess_ratio_accum / denom,
                "anchor_high_band_excess_ratio_loss": anchor_high_band_excess_ratio_accum / denom,
                "decoded_waveform_high_ratio_loss": decoded_waveform_high_ratio_accum / denom,
                "decoded_waveform_high_ratio_match_loss": decoded_waveform_high_ratio_match_accum / denom,
                "decoded_waveform_high_ratio_topk_excess_loss": (
                    decoded_waveform_high_ratio_topk_excess_accum / denom
                ),
                "decoded_waveform_global_high_ratio_loss": decoded_waveform_global_high_ratio_accum / denom,
                "decoded_waveform_global_high_ratio_match_loss": decoded_waveform_global_high_ratio_match_accum / denom,
                "decoded_waveform_spectral_centroid_match_loss": (
                    decoded_waveform_spectral_centroid_match_accum / denom
                ),
                "decoded_waveform_spectral_centroid_topk_drift_loss": (
                    decoded_waveform_spectral_centroid_topk_drift_accum / denom
                ),
                "decoded_waveform_spectral_centroid_topk_excess_loss": (
                    decoded_waveform_spectral_centroid_topk_excess_accum / denom
                ),
                "decoded_waveform_teacher_conditioned_drift_loss": (
                    decoded_waveform_teacher_conditioned_drift_accum / denom
                ),
                "decoded_waveform_high_flatness_loss": decoded_waveform_high_flatness_accum / denom,
                "decoded_waveform_high_flatness_match_loss": decoded_waveform_high_flatness_match_accum / denom,
                "decoded_waveform_high_flatness_topk_excess_loss": (
                    decoded_waveform_high_flatness_topk_excess_accum / denom
                ),
                "decoded_waveform_weighted_high_flatness_topk_excess_loss": (
                    decoded_waveform_weighted_high_flatness_topk_excess_accum / denom
                ),
                "decoded_waveform_weighted_high_zcr_topk_excess_loss": (
                    decoded_waveform_weighted_high_zcr_topk_excess_accum / denom
                ),
                "decoded_waveform_rms_match_loss": decoded_waveform_rms_match_accum / denom,
                "decoded_waveform_derivative_rms_match_loss": decoded_waveform_derivative_rms_match_accum / denom,
                "decoded_waveform_derivative_rms_excess_loss": decoded_waveform_derivative_rms_excess_accum / denom,
                "decoded_waveform_soft_zcr_match_loss": decoded_waveform_soft_zcr_match_accum / denom,
                "decoded_waveform_soft_zcr_excess_loss": decoded_waveform_soft_zcr_excess_accum / denom,
                "decoded_waveform_framed_soft_zcr_match_loss": (
                    decoded_waveform_framed_soft_zcr_match_accum / denom
                ),
                "decoded_waveform_framed_soft_zcr_excess_loss": (
                    decoded_waveform_framed_soft_zcr_excess_accum / denom
                ),
                "decoded_waveform_framed_soft_zcr_topk_excess_loss": (
                    decoded_waveform_framed_soft_zcr_topk_excess_accum / denom
                ),
                "decoded_waveform_rms_underfill_topk_loss": decoded_waveform_rms_underfill_topk_accum / denom,
                "decoded_waveform_rms_envelope_match_loss": decoded_waveform_rms_envelope_match_accum / denom,
                "decoded_waveform_rms_envelope_second_delta_match_loss": (
                    decoded_waveform_rms_envelope_second_delta_match_accum / denom
                ),
                "decoded_waveform_multires_stft_loss": decoded_waveform_multires_stft_accum / denom,
                "decoded_waveform_spectral_envelope_match_loss": (
                    decoded_waveform_spectral_envelope_match_accum / denom
                ),
                "decoded_waveform_peak_excess_loss": decoded_waveform_peak_excess_accum / denom,
                "decoded_waveform_peak_cap_loss": decoded_waveform_peak_cap_accum / denom,
                "decoded_waveform_ctc_loss": decoded_waveform_ctc_accum / denom,
                "decoded_waveform_ssl_emission_match_loss": decoded_waveform_ssl_emission_match_accum / denom,
                "decoded_waveform_wavlm_profile_loss": decoded_waveform_wavlm_profile_accum / denom,
                "decoded_waveform_mid_high_ratio_loss": decoded_waveform_mid_high_ratio_accum / denom,
                "decoded_waveform_mid_high_ratio_match_loss": decoded_waveform_mid_high_ratio_match_accum / denom,
                "decoded_waveform_tail_high_ratio_loss": decoded_waveform_tail_high_ratio_accum / denom,
                "decoded_waveform_tail_high_ratio_match_loss": decoded_waveform_tail_high_ratio_match_accum / denom,
                "decoded_waveform_updates": decoded_waveform_updates,
                "decoded_waveform_crops": decoded_waveform_crops,
                "decoded_waveform_skip_frames": decoded_waveform_skip_frames,
                "decoded_waveform_skip_short": decoded_waveform_skip_short,
                "tail_rollout_loss": tail_rollout_accum / denom,
                "tail_temporal_delta_loss": tail_temporal_accum / denom,
                "tail_high_band_match_loss": tail_high_band_accum / denom,
                "tail_high_band_temporal_delta_loss": tail_high_band_temporal_accum / denom,
                "tail_high_band_excess_ratio_loss": tail_high_band_excess_ratio_accum / denom,
                "mid_rollout_loss": mid_rollout_accum / denom,
                "mid_temporal_delta_loss": mid_temporal_accum / denom,
                "mid_high_band_match_loss": mid_high_band_accum / denom,
                "mid_high_band_temporal_delta_loss": mid_high_band_temporal_accum / denom,
                "mid_high_band_excess_ratio_loss": mid_high_band_excess_ratio_accum / denom,
                "focus_rollout_loss": focus_rollout_accum / denom,
                "focus_temporal_delta_loss": focus_temporal_accum / denom,
                "focus_high_band_match_loss": focus_high_band_accum / denom,
                "focus_high_band_temporal_delta_loss": focus_high_band_temporal_accum / denom,
                "focus_high_band_excess_ratio_loss": focus_high_band_excess_ratio_accum / denom,
                "self_rollout_bridge_loss": self_rollout_bridge_accum / denom,
                "self_rollout_direction_loss": self_rollout_direction_accum / denom,
                "self_rollout_norm_loss": self_rollout_norm_accum / denom,
                "rollout_trajectory_loss": rollout_trajectory_accum / denom,
                "conditional_mel_critic_loss": conditional_mel_critic_accum / denom,
                "conditional_mel_critic_discriminator_loss": conditional_mel_critic_discriminator_accum / denom,
                "text_flow_contrastive_loss": text_flow_contrastive_accum / denom,
                "text_rollout_contrastive_loss": text_rollout_contrastive_accum / denom,
                "cond_flow_contrastive_loss": cond_flow_contrastive_accum / denom,
                "cond_rollout_contrastive_loss": cond_rollout_contrastive_accum / denom,
                "anchor_flow_loss": anchor_flow_accum / denom,
                "anchor_rollout_loss": anchor_rollout_accum / denom,
                "parameter_anchor_loss": anchor_accum / denom,
                "seconds": round(time.time() - started, 3),
            }
            append_jsonl(ledger_path, metrics)
            print(json.dumps(metrics), flush=True)
        if val_rows and (step % int(args.val_every) == 0 or step == 1):
            val_rollout_value = validation_rollout_loss(
                student=student,
                rows=val_rows,
                cache_dir=cache_dir,
                device=device,
                student_times=student_times,
                cfg_strength=float(args.student_cfg_strength),
            )
            val_metrics = {
                "event": "validation",
                "step": step,
                "rollout_loss": val_rollout_value,
                "best_val_rollout_loss": min(best_val_rollout_loss, val_rollout_value),
                "cfg_strength": float(args.student_cfg_strength),
                "rows": len(val_rows),
                "seconds": round(time.time() - started, 3),
            }
            append_jsonl(ledger_path, val_metrics)
            print(json.dumps(val_metrics), flush=True)
            if val_rollout_value > 0.0 and val_rollout_value < best_val_rollout_loss:
                best_val_rollout_loss = val_rollout_value
                save_checkpoint(
                    student,
                    output_dir / checkpoint_name(args, "best_val_rollout"),
                    step=step,
                    args=args,
                    loss=val_rollout_value,
                )
            if (
                decoded_vocoder is not None
                and int(args.val_decoded_quality_every) > 0
                and (step % int(args.val_decoded_quality_every) == 0 or step == 1)
            ):
                val_quality = validation_decoded_quality(
                    student=student,
                    rows=val_rows,
                    cache_dir=cache_dir,
                    device=device,
                    student_times=student_times,
                    cfg_strength=float(args.student_cfg_strength),
                    decoded_vocoder=decoded_vocoder,
                    args=args,
                )
                if val_quality:
                    val_quality_metrics = {
                        "event": "validation_decoded_quality",
                        "step": step,
                        "best_val_quality_loss": min(
                            best_val_quality_loss,
                            float(val_quality["decoded_quality_loss"]),
                        ),
                        "cfg_strength": float(args.student_cfg_strength),
                        "seconds": round(time.time() - started, 3),
                        **val_quality,
                    }
                    append_jsonl(ledger_path, val_quality_metrics)
                    print(json.dumps(val_quality_metrics), flush=True)
                    if float(val_quality["decoded_quality_loss"]) < best_val_quality_loss:
                        best_val_quality_loss = float(val_quality["decoded_quality_loss"])
                        save_checkpoint(
                            student,
                            output_dir / checkpoint_name(args, "best_val_quality"),
                            step=step,
                            args=args,
                            loss=best_val_quality_loss,
                        )
        if step % int(args.save_every) == 0:
            save_checkpoint(
                student,
                output_dir / checkpoint_name(args, f"step_{step}"),
                step=step,
                args=args,
                loss=loss_value,
            )

    save_checkpoint(student, output_dir / checkpoint_name(args, "last"), step=int(args.steps), args=args, loss=loss_value)


if __name__ == "__main__":
    main()
