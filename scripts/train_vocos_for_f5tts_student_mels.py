#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DISTILL_PATH = ROOT / "scripts" / "distill_f5tts_12_to_4_q4.py"
_DISTILL_SPEC = importlib.util.spec_from_file_location("_f5tts_distill_helpers", _DISTILL_PATH)
if _DISTILL_SPEC is None or _DISTILL_SPEC.loader is None:
    raise RuntimeError(f"could not load distill helpers from {_DISTILL_PATH}")
_DISTILL = importlib.util.module_from_spec(_DISTILL_SPEC)
sys.modules["_f5tts_distill_helpers"] = _DISTILL
_DISTILL_SPEC.loader.exec_module(_DISTILL)

apply_q4_parametrizations = _DISTILL.apply_q4_parametrizations
build_model = _DISTILL.build_model
cfg_flow = _DISTILL.cfg_flow
load_checkpoint_state = _DISTILL.load_checkpoint_state
make_time_grid = _DISTILL.make_time_grid
split_csv = _DISTILL.split_csv


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def parse_cache_dirs(value: str) -> list[Path]:
    dirs = [Path(item).expanduser() for item in str(value).split(",") if item.strip()]
    if not dirs:
        raise ValueError("--cache-dir did not contain any cache directories")
    return dirs


def load_rows(cache_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cache_dir in cache_dirs:
        metadata = cache_dir / "metadata.jsonl"
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_cache_dir"] = str(cache_dir)
            rows.append(row)
    if not rows:
        raise ValueError("no cache rows loaded")
    return rows


def row_payload_path(row: dict[str, Any]) -> Path:
    return Path(str(row["_cache_dir"])) / str(row["path"])


def configure_vocos_trainable(vocoder: torch.nn.Module, include: tuple[str, ...]) -> tuple[int, int]:
    tensors = 0
    params = 0
    for name, parameter in vocoder.named_parameters():
        trainable = any(item in name for item in include) if include else True
        parameter.requires_grad_(trainable)
        if trainable:
            tensors += 1
            params += int(parameter.numel())
    return tensors, params


def trainable_anchor(vocoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in vocoder.named_parameters() if p.requires_grad}


def anchor_loss(vocoder: torch.nn.Module, anchors: dict[str, torch.Tensor]) -> torch.Tensor:
    loss: torch.Tensor | None = None
    terms = 0
    for name, parameter in vocoder.named_parameters():
        if not parameter.requires_grad or name not in anchors:
            continue
        term = F.mse_loss(parameter, anchors[name].to(device=parameter.device, dtype=parameter.dtype))
        loss = term if loss is None else loss + term
        terms += 1
    if loss is None:
        return next(vocoder.parameters()).new_zeros(())
    return loss / max(1, terms)


def rollout_student_mel(
    model: torch.nn.Module,
    payload: dict[str, Any],
    *,
    device: torch.device,
    student_steps: int,
    cfg_strength: float,
    sway_sampling_coef: float,
) -> torch.Tensor:
    cond = payload["cond"].to(device=device, dtype=torch.float32)
    noise = payload["noise"].to(device=device, dtype=torch.float32)
    text_ids = payload["text_ids"].to(device=device)
    cond_frames = int(payload["cond_frames"])
    times = make_time_grid(int(student_steps), float(sway_sampling_coef), device, torch.float32)
    y = noise.clone()
    y[:, :cond_frames, :] = cond[:, :cond_frames, :]
    for interval in range(int(student_steps)):
        t0 = times[interval]
        dt = float(times[interval + 1] - times[interval])
        time_tensor = torch.full((y.shape[0],), float(t0), device=device, dtype=y.dtype)
        pred_flow = cfg_flow(model, y, cond, text_ids, time_tensor, float(cfg_strength))
        y = y + dt * pred_flow
        y[:, :cond_frames, :] = cond[:, :cond_frames, :]
    return y.detach()


def stft_logmag_loss(pred: torch.Tensor, target: torch.Tensor, fft_sizes: tuple[int, ...]) -> torch.Tensor:
    loss = pred.new_zeros(())
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        pred_stft = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        target_stft = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        loss = loss + F.l1_loss(torch.log1p(pred_stft.abs()), torch.log1p(target_stft.abs()))
    return loss / float(max(1, len(fft_sizes)))


def waveform_high_ratio_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
) -> torch.Tensor:
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / float(sample_rate)).to(pred_wave.device)
    mask = freqs >= float(cutoff_hz)
    pred_power = pred_stft.abs().pow(2)
    target_power = target_stft.abs().pow(2)
    pred_ratio = pred_power[:, mask, :].sum(dim=(1, 2)) / pred_power.sum(dim=(1, 2)).clamp_min(1e-8)
    target_ratio = target_power[:, mask, :].sum(dim=(1, 2)) / target_power.sum(dim=(1, 2)).clamp_min(1e-8)
    return F.relu(pred_ratio - target_ratio - float(margin)).mean()


def waveform_high_ratio_match_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
) -> torch.Tensor:
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / float(sample_rate)).to(pred_wave.device)
    mask = freqs >= float(cutoff_hz)
    pred_power = pred_stft.abs().pow(2)
    target_power = target_stft.abs().pow(2)
    pred_ratio = pred_power[:, mask, :].sum(dim=(1, 2)) / pred_power.sum(dim=(1, 2)).clamp_min(1e-8)
    target_ratio = target_power[:, mask, :].sum(dim=(1, 2)) / target_power.sum(dim=(1, 2)).clamp_min(1e-8)
    return F.l1_loss(pred_ratio, target_ratio)


def waveform_global_high_ratio(wave: torch.Tensor, sample_rate: int, cutoff_hz: float) -> torch.Tensor:
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if int(wave.shape[-1]) < 2048:
        return wave.new_zeros((wave.shape[0],), dtype=torch.float32)
    wave = wave.float()
    spectrum = torch.fft.rfft(wave, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(int(wave.shape[-1]), d=1.0 / float(sample_rate)).to(wave.device)
    mask = freqs >= float(cutoff_hz)
    if not bool(mask.any()):
        return wave.new_zeros((wave.shape[0],), dtype=torch.float32)
    return spectrum[:, mask].sum(dim=-1) / spectrum.sum(dim=-1).clamp_min(1e-8)


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


def waveform_peak_cap_loss(pred_wave: torch.Tensor, cap: float) -> torch.Tensor:
    if int(pred_wave.shape[-1]) < 512 or float(cap) <= 0.0:
        return pred_wave.new_zeros(())
    pred_peak = pred_wave.float().abs().amax(dim=-1)
    excess = F.relu(pred_peak - float(cap))
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.01)


def waveform_rms_match_loss(pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
    if int(pred_wave.shape[-1]) < 512 or int(target_wave.shape[-1]) < 512:
        return pred_wave.new_zeros(())
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    pred_rms = pred_wave[..., :length].float().pow(2).mean(dim=-1).sqrt()
    target_rms = target_wave[..., :length].float().pow(2).mean(dim=-1).sqrt()
    return F.smooth_l1_loss(pred_rms, target_rms.detach(), beta=0.002)


def waveform_rms_underfill_loss(pred_wave: torch.Tensor, target_wave: torch.Tensor, margin: float) -> torch.Tensor:
    if int(pred_wave.shape[-1]) < 512 or int(target_wave.shape[-1]) < 512:
        return pred_wave.new_zeros(())
    length = min(int(pred_wave.shape[-1]), int(target_wave.shape[-1]))
    pred_rms = pred_wave[..., :length].float().pow(2).mean(dim=-1).sqrt()
    target_rms = target_wave[..., :length].float().pow(2).mean(dim=-1).sqrt()
    floor = target_rms.detach() * float(margin)
    return F.smooth_l1_loss(F.relu(floor - pred_rms), torch.zeros_like(pred_rms), beta=0.002)


def waveform_high_flatness_excess_loss(
    pred_wave: torch.Tensor,
    target_wave: torch.Tensor,
    sample_rate: int,
    cutoff_hz: float,
    margin: float,
) -> torch.Tensor:
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft, device=pred_wave.device, dtype=pred_wave.dtype)
    pred_stft = torch.stft(pred_wave, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    target_stft = torch.stft(target_wave, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / float(sample_rate)).to(pred_wave.device)
    mask = freqs >= float(cutoff_hz)
    if not bool(mask.any()):
        return pred_wave.new_zeros(())
    pred_power = pred_stft[:, mask, :].abs().pow(2).clamp_min(1e-10)
    target_power = target_stft[:, mask, :].abs().pow(2).clamp_min(1e-10)
    pred_flatness = torch.exp(torch.log(pred_power).mean(dim=1)) / pred_power.mean(dim=1).clamp_min(1e-10)
    target_flatness = torch.exp(torch.log(target_power).mean(dim=1)) / target_power.mean(dim=1).clamp_min(1e-10)
    excess = F.relu(pred_flatness - target_flatness.detach() - float(margin))
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.02)


def waveform_high_ratio(wave: torch.Tensor, sample_rate: int, cutoff_hz: float) -> torch.Tensor:
    n_fft = 1024
    hop = 256
    window = torch.hann_window(n_fft, device=wave.device, dtype=wave.dtype)
    stft = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / float(sample_rate)).to(wave.device)
    mask = freqs >= float(cutoff_hz)
    power = stft.abs().pow(2)
    return power[:, mask, :].sum(dim=(1, 2)) / power.sum(dim=(1, 2)).clamp_min(1e-8)


def crop_generated(wave: torch.Tensor, cond_frames: int, total_frames: int, hop_length: int = 256) -> torch.Tensor:
    start = int(cond_frames) * int(hop_length)
    end = int(total_frames) * int(hop_length)
    return wave[:, start:end]


def decode_vocos_trainable(vocoder: torch.nn.Module, mel_channels_first: torch.Tensor) -> torch.Tensor:
    features = vocoder.backbone(mel_channels_first)
    return vocoder.head(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt Vocos to F5TTS student mel rollouts without changing F5 architecture.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--f5-checkpoint", required=True)
    parser.add_argument("--vocab", default="/data/resumebot/checkpoints/F5TTS_Base_vocab.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocos-checkpoint", default="", help="Optional Vocos checkpoint to continue adapting from.")
    parser.add_argument("--student-steps", type=int, default=6)
    parser.add_argument("--cfg-strength", type=float, default=1.25)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--q4-include", default="")
    parser.add_argument("--q4-exclude", default="text_embed.text_embed,mel_spec")
    parser.add_argument(
        "--q4-skip-initial-requant",
        action="store_true",
        help="Use when --f5-checkpoint is already a materialized q4 checkpoint; avoids lossy second quantization before mel rollout.",
    )
    parser.add_argument("--train-include", default="head.out,backbone.final_layer_norm,backbone.convnext.7")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max-frames", type=int, default=760)
    parser.add_argument(
        "--decode-generated-only",
        action="store_true",
        help="Match F5TTS inference: remove reference frames before Vocos decode instead of decoding full mel then cropping.",
    )
    parser.add_argument("--stft-loss-weight", type=float, default=1.0)
    parser.add_argument("--wave-l1-loss-weight", type=float, default=0.15)
    parser.add_argument("--high-ratio-loss-weight", type=float, default=0.25)
    parser.add_argument("--high-ratio-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--global-high-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--global-high-ratio-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-flatness-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-flatness-margin", type=float, default=0.01)
    parser.add_argument("--rms-match-loss-weight", type=float, default=0.0)
    parser.add_argument("--rms-underfill-loss-weight", type=float, default=0.0)
    parser.add_argument("--rms-underfill-margin", type=float, default=0.90)
    parser.add_argument("--peak-cap-loss-weight", type=float, default=0.0)
    parser.add_argument("--peak-cap", type=float, default=0.0)
    parser.add_argument(
        "--anchor-wave-high-ratio-loss-weight",
        type=float,
        default=0.0,
        help="Penalize extra generated high-band energy against frozen base Vocos on the same student mel.",
    )
    parser.add_argument(
        "--anchor-wave-high-ratio-match-loss-weight",
        type=float,
        default=0.0,
        help="Match generated high-band energy ratio to frozen base Vocos on the same student mel.",
    )
    parser.add_argument(
        "--anchor-wave-high-flatness-excess-loss-weight",
        type=float,
        default=0.0,
        help="Penalize extra high-band flatness against frozen base Vocos on the same student mel.",
    )
    parser.add_argument("--anchor-wave-high-ratio-margin", type=float, default=0.0)
    parser.add_argument("--anchor-wave-high-flatness-margin", type=float, default=0.0)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.02)
    parser.add_argument("--cutoff-hz", type=float, default=7000.0)
    parser.add_argument("--high-ratio-margin", type=float, default=0.005)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in load_rows(parse_cache_dirs(args.cache_dir))
        if int(row.get("cond_frames", 0)) < int(row.get("frames", 0))
        and int(row.get("frames", 0)) <= int(args.max_frames)
    ]
    if not rows:
        raise ValueError("no eligible cache rows after max-frame filtering")
    random.shuffle(rows)

    f5 = build_model(Path(args.vocab), device)
    load_checkpoint_state(f5, Path(args.f5_checkpoint))
    if not bool(args.q4_skip_initial_requant):
        apply_q4_parametrizations(
            f5,
            include=split_csv(args.q4_include),
            exclude=split_csv(args.q4_exclude),
            ste_include=("__none__",),
        )
    f5.eval().requires_grad_(False)

    resumebot_root = Path("/data/resumebot")
    if str(resumebot_root) not in sys.path:
        sys.path.insert(0, str(resumebot_root))
    from f5_tts.infer.utils_infer import load_vocoder

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=str(device))
    if str(args.vocos_checkpoint).strip():
        payload = torch.load(str(args.vocos_checkpoint), map_location=device)
        state = payload.get("model_state_dict", payload)
        missing, unexpected = vocoder.load_state_dict(state, strict=False)
        actionable_unexpected = [name for name in unexpected if not str(name).endswith(".weight_scale")]
        if missing or actionable_unexpected:
            print(
                json.dumps(
                    {
                        "vocos_checkpoint_load": str(args.vocos_checkpoint),
                        "missing": missing,
                        "unexpected": actionable_unexpected,
                    },
                    indent=2,
                )
            )
    teacher_vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=str(device))
    teacher_vocoder.eval().requires_grad_(False)
    train_tensors, train_params = configure_vocos_trainable(vocoder, split_csv(args.train_include))
    anchors = trainable_anchor(vocoder)
    optimizer = torch.optim.AdamW([p for p in vocoder.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=0.0)
    ledger = output_dir / "vocos_student_mel_metrics.jsonl"
    setup = {
        "event": "setup",
        "rows": len(rows),
        "f5_checkpoint": str(args.f5_checkpoint),
        "vocos_checkpoint": str(args.vocos_checkpoint),
        "train_tensors": train_tensors,
        "train_params": train_params,
        "train_include": split_csv(args.train_include),
    }
    append_jsonl(ledger, setup)
    print(json.dumps(setup, indent=2), flush=True)

    best_loss = float("inf")
    started = time.time()
    for step in range(1, int(args.steps) + 1):
        batch_rows = random.sample(rows, k=max(1, int(args.batch_size)))
        optimizer.zero_grad(set_to_none=True)
        loss_accum = pred_high_accum = target_high_accum = 0.0
        stft_accum = wave_l1_accum = high_accum = high_match_accum = 0.0
        global_high_accum = global_high_match_accum = high_flatness_accum = 0.0
        rms_match_accum = rms_underfill_accum = peak_cap_accum = 0.0
        anchor_high_accum = anchor_high_match_accum = anchor_high_flatness_accum = parameter_anchor_accum = 0.0
        for row in batch_rows:
            payload = torch.load(row_payload_path(row), map_location="cpu")
            with torch.no_grad():
                student_mel = rollout_student_mel(
                    f5,
                    payload,
                    device=device,
                    student_steps=int(args.student_steps),
                    cfg_strength=float(args.cfg_strength),
                    sway_sampling_coef=float(args.sway_sampling_coef),
                )
                teacher_mel = payload["teacher_y"].to(device=device, dtype=torch.float32)
            cond_frames = int(payload["cond_frames"])
            total_frames = int(payload["teacher_y"].shape[1])
            if bool(args.decode_generated_only):
                gen_student_mel = student_mel[:, cond_frames:total_frames, :]
                gen_teacher_mel = teacher_mel[:, cond_frames:total_frames, :]
                with torch.no_grad():
                    target_gen = teacher_vocoder.decode(gen_teacher_mel.permute(0, 2, 1).float()).detach()
                    anchor_gen = teacher_vocoder.decode(gen_student_mel.permute(0, 2, 1).float()).detach()
                pred_gen = decode_vocos_trainable(vocoder, gen_student_mel.permute(0, 2, 1).float())
            else:
                with torch.no_grad():
                    target_wave = teacher_vocoder.decode(teacher_mel.permute(0, 2, 1).float()).detach()
                    anchor_wave = teacher_vocoder.decode(student_mel.permute(0, 2, 1).float()).detach()
                pred_wave = decode_vocos_trainable(vocoder, student_mel.permute(0, 2, 1).float())
                pred_gen = crop_generated(pred_wave, cond_frames, total_frames)
                target_gen = crop_generated(target_wave, cond_frames, total_frames)
                anchor_gen = crop_generated(anchor_wave, cond_frames, total_frames)
            length = min(pred_gen.shape[-1], target_gen.shape[-1], anchor_gen.shape[-1])
            pred_gen = pred_gen[..., :length]
            target_gen = target_gen[..., :length]
            anchor_gen = anchor_gen[..., :length]
            stft_loss = stft_logmag_loss(pred_gen, target_gen, (512, 1024, 2048))
            wave_l1 = F.l1_loss(pred_gen, target_gen)
            high_loss = waveform_high_ratio_loss(
                pred_gen,
                target_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
                float(args.high_ratio_margin),
            )
            high_match_loss = waveform_high_ratio_match_loss(
                pred_gen,
                target_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
            )
            global_high_loss = waveform_global_high_ratio_excess_loss(
                pred_gen,
                target_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
                float(args.high_ratio_margin),
            )
            global_high_match_loss = waveform_global_high_ratio_match_loss(
                pred_gen,
                target_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
            )
            high_flatness_loss = waveform_high_flatness_excess_loss(
                pred_gen,
                target_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
                float(args.high_flatness_margin),
            )
            rms_match_loss = waveform_rms_match_loss(pred_gen, target_gen)
            rms_underfill_loss = waveform_rms_underfill_loss(
                pred_gen,
                target_gen,
                float(args.rms_underfill_margin),
            )
            peak_cap_loss = waveform_peak_cap_loss(pred_gen, float(args.peak_cap))
            anchor_high_loss = waveform_high_ratio_loss(
                pred_gen,
                anchor_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
                float(args.anchor_wave_high_ratio_margin),
            )
            anchor_high_match_loss = waveform_high_ratio_match_loss(
                pred_gen,
                anchor_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
            )
            anchor_high_flatness_loss = waveform_high_flatness_excess_loss(
                pred_gen,
                anchor_gen,
                int(args.sample_rate),
                float(args.cutoff_hz),
                float(args.anchor_wave_high_flatness_margin),
            )
            parameter_anchor = anchor_loss(vocoder, anchors)
            loss = (
                float(args.stft_loss_weight) * stft_loss
                + float(args.wave_l1_loss_weight) * wave_l1
                + float(args.high_ratio_loss_weight) * high_loss
                + float(args.high_ratio_match_loss_weight) * high_match_loss
                + float(args.global_high_ratio_loss_weight) * global_high_loss
                + float(args.global_high_ratio_match_loss_weight) * global_high_match_loss
                + float(args.high_flatness_excess_loss_weight) * high_flatness_loss
                + float(args.rms_match_loss_weight) * rms_match_loss
                + float(args.rms_underfill_loss_weight) * rms_underfill_loss
                + float(args.peak_cap_loss_weight) * peak_cap_loss
                + float(args.anchor_wave_high_ratio_loss_weight) * anchor_high_loss
                + float(args.anchor_wave_high_ratio_match_loss_weight) * anchor_high_match_loss
                + float(args.anchor_wave_high_flatness_excess_loss_weight) * anchor_high_flatness_loss
                + float(args.anchor_loss_weight) * parameter_anchor
            )
            (loss / float(len(batch_rows))).backward()
            loss_accum += float(loss.detach().cpu())
            stft_accum += float(stft_loss.detach().cpu())
            wave_l1_accum += float(wave_l1.detach().cpu())
            high_accum += float(high_loss.detach().cpu())
            high_match_accum += float(high_match_loss.detach().cpu())
            global_high_accum += float(global_high_loss.detach().cpu())
            global_high_match_accum += float(global_high_match_loss.detach().cpu())
            high_flatness_accum += float(high_flatness_loss.detach().cpu())
            rms_match_accum += float(rms_match_loss.detach().cpu())
            rms_underfill_accum += float(rms_underfill_loss.detach().cpu())
            peak_cap_accum += float(peak_cap_loss.detach().cpu())
            anchor_high_accum += float(anchor_high_loss.detach().cpu())
            anchor_high_match_accum += float(anchor_high_match_loss.detach().cpu())
            anchor_high_flatness_accum += float(anchor_high_flatness_loss.detach().cpu())
            parameter_anchor_accum += float(parameter_anchor.detach().cpu())
            with torch.no_grad():
                pred_high_accum += float(waveform_high_ratio(pred_gen, int(args.sample_rate), float(args.cutoff_hz)).mean().detach().cpu())
                target_high_accum += float(waveform_high_ratio(target_gen, int(args.sample_rate), float(args.cutoff_hz)).mean().detach().cpu())
        torch.nn.utils.clip_grad_norm_([p for p in vocoder.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        loss_value = loss_accum / float(max(1, len(batch_rows)))
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(
                {"model_state_dict": vocoder.state_dict(), "step": step, "args": vars(args), "loss": loss_value},
                output_dir / "vocos_student_mel_best.pt",
            )
        if step % int(args.log_every) == 0 or step == 1:
            row = {
                "event": "train",
                "step": step,
                "loss": loss_value,
                "best_loss": best_loss,
                "stft_loss": stft_accum / float(max(1, len(batch_rows))),
                "wave_l1_loss": wave_l1_accum / float(max(1, len(batch_rows))),
                "high_ratio_loss": high_accum / float(max(1, len(batch_rows))),
                "high_ratio_match_loss": high_match_accum / float(max(1, len(batch_rows))),
                "global_high_ratio_loss": global_high_accum / float(max(1, len(batch_rows))),
                "global_high_ratio_match_loss": global_high_match_accum / float(max(1, len(batch_rows))),
                "high_flatness_loss": high_flatness_accum / float(max(1, len(batch_rows))),
                "rms_match_loss": rms_match_accum / float(max(1, len(batch_rows))),
                "rms_underfill_loss": rms_underfill_accum / float(max(1, len(batch_rows))),
                "peak_cap_loss": peak_cap_accum / float(max(1, len(batch_rows))),
                "anchor_high_ratio_loss": anchor_high_accum / float(max(1, len(batch_rows))),
                "anchor_high_ratio_match_loss": anchor_high_match_accum / float(max(1, len(batch_rows))),
                "anchor_high_flatness_loss": anchor_high_flatness_accum / float(max(1, len(batch_rows))),
                "parameter_anchor_loss": parameter_anchor_accum / float(max(1, len(batch_rows))),
                "pred_high_proxy": pred_high_accum / float(max(1, len(batch_rows))),
                "target_high_proxy": target_high_accum / float(max(1, len(batch_rows))),
                "seconds": round(time.time() - started, 3),
            }
            append_jsonl(ledger, row)
            print(json.dumps(row), flush=True)
        if step % int(args.save_every) == 0:
            torch.save(
                {"model_state_dict": vocoder.state_dict(), "step": step, "args": vars(args), "loss": loss_value},
                output_dir / f"vocos_student_mel_step_{step}.pt",
            )
    torch.save(
        {"model_state_dict": vocoder.state_dict(), "step": int(args.steps), "args": vars(args), "loss": loss_value},
        output_dir / "vocos_student_mel_last.pt",
    )


if __name__ == "__main__":
    main()
