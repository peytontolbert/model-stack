#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath

import numpy as np
import soundfile as sf
import torch


DEFAULT_CHECKPOINT = "/data/transformer_10/checkpoints/f5tts_q4_12to4_distill/model_q4_12to4_best.pt"
DEFAULT_VOCAB = "/data/resumebot/checkpoints/F5TTS_Base_vocab.txt"
DEFAULT_PROFILE = "/data/resumebot/voice_profiles/Peyton"
DEFAULT_OUT_DIR = "/data/transformer_10/evals/f5tts_voice_profiles"


def runtime_metadata(device: torch.device) -> dict:
    metadata = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "cuda_index": int(index),
                "cuda_name": str(props.name),
                "cuda_total_memory_gb": float(props.total_memory) / (1024.0**3),
                "cuda_capability": f"{props.major}.{props.minor}",
            }
        )
    return metadata


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


def load_checkpoint(model, checkpoint_path: Path, device: torch.device) -> int | None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"]
    model.load_state_dict(state, strict=False)
    step = checkpoint.get("step")
    return int(step) if step is not None else None


def should_q4_simulate(name: str, tensor: torch.Tensor) -> bool:
    if not name.endswith(".weight") or tensor.ndim < 2:
        return False
    return not any(part in name for part in ("text_embed.text_embed", "mel_spec"))


def apply_q4_simulation(model) -> tuple[int, int]:
    tensors = 0
    params = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not should_q4_simulate(name, parameter):
                continue
            original_dtype = parameter.dtype
            flat = parameter.detach().float().reshape(parameter.shape[0], -1)
            scale = flat.abs().amax(dim=1).clamp_min(1e-8) / 7.0
            quantized = torch.round(flat / scale[:, None]).clamp(-8, 7)
            dequantized = (quantized * scale[:, None]).reshape_as(parameter).to(original_dtype)
            parameter.copy_(dequantized)
            tensors += 1
            params += int(parameter.numel())
    return tensors, params


def first_profile_sample(profile_dir: Path) -> tuple[str, str]:
    samples_path = profile_dir / "samples.txt"
    if not samples_path.exists():
        raise FileNotFoundError(f"missing profile samples file: {samples_path}")
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "|" not in line:
            continue
        audio_ref, text = line.split("|", 1)
        audio_path = Path(audio_ref)
        if not audio_path.exists():
            basename = PureWindowsPath(audio_ref).name if "\\" in audio_ref else audio_path.name
            audio_path = profile_dir / basename
        if audio_path.exists():
            return str(audio_path), text.strip()
    raise FileNotFoundError(f"no usable profile sample found in {samples_path}")


def patch_torchaudio_load() -> None:
    import torchaudio

    def load_with_soundfile(path, *_, **__):
        array, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        audio = torch.from_numpy(np.asarray(array, dtype=np.float32))
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.transpose(0, 1).contiguous()
        return audio, int(sample_rate)

    torchaudio.load = load_with_soundfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an F5TTS checkpoint with a voice-cloning profile.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--voice-profile-dir", default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-path", default="", help="Optional explicit WAV output path.")
    parser.add_argument("--text", default="This is a four step distilled F5 TTS voice cloning test from Agent Kernel Lite.")
    parser.add_argument("--nfe-step", type=int, default=4)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--ode-method", default="", help="Optional torchdiffeq ODE method override, e.g. euler or midpoint.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--simulate-q4", action="store_true")
    parser.add_argument("--save-mel", action="store_true", help="Save generated mel spectrogram next to the WAV output.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--min-cuda-total-gb",
        type=float,
        default=0.0,
        help="Abort CUDA rendering if the selected GPU has less memory. Use for promotion/parity gates.",
    )
    args = parser.parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    repo_root = Path("/data/resumebot")
    sys.path.insert(0, str(repo_root))
    from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text

    patch_torchaudio_load()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    runtime = runtime_metadata(device)
    if device.type == "cuda" and float(args.min_cuda_total_gb) > 0.0:
        total_gb = float(runtime.get("cuda_total_memory_gb", 0.0))
        if total_gb < float(args.min_cuda_total_gb):
            raise RuntimeError(
                f"selected CUDA device is below promotion/parity memory floor: "
                f"{runtime.get('cuda_name')} has {total_gb:.2f} GB, "
                f"required {float(args.min_cuda_total_gb):.2f} GB"
            )

    checkpoint_path = Path(args.checkpoint)
    profile_dir = Path(args.voice_profile_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(Path(args.vocab), device)
    step = load_checkpoint(model, checkpoint_path, device)
    if str(args.ode_method).strip():
        model.odeint_kwargs = {**getattr(model, "odeint_kwargs", {}), "method": str(args.ode_method).strip()}
    q4_tensors = 0
    q4_params = 0
    if args.simulate_q4:
        q4_tensors, q4_params = apply_q4_simulation(model)
    model.eval()
    infer_device = str(device)
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=infer_device)
    ref_audio_path, ref_text = first_profile_sample(profile_dir)
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

    with torch.no_grad():
        audio, sample_rate, mel = infer_process(
            ref_audio,
            ref_text,
            str(args.text),
            model,
            vocoder,
            mel_spec_type="vocos",
            speed=float(args.speed),
            nfe_step=int(args.nfe_step),
            cfg_strength=float(args.cfg_strength),
            sway_sampling_coef=float(args.sway_sampling_coef),
            device=infer_device,
        )

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"{profile_dir.name.lower()}_f5tts_step{step or 0}_nfe{int(args.nfe_step)}_{timestamp}.wav"
    sf.write(output_path, audio, int(sample_rate), format="WAV", subtype="PCM_16")
    mel_path = None
    if bool(args.save_mel) and mel is not None:
        mel_path = output_path.with_suffix(".mel.npy")
        np.save(mel_path, np.asarray(mel, dtype=np.float32))
    print(json.dumps({
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": step,
        "voice_profile_dir": str(profile_dir),
        "ref_audio": ref_audio_path,
        "nfe_step": int(args.nfe_step),
        "cfg_strength": float(args.cfg_strength),
        "simulate_q4": bool(args.simulate_q4),
        "q4_tensors": int(q4_tensors),
        "q4_params": int(q4_params),
        "runtime": runtime,
        "samples": int(audio.size),
        "sample_rate": int(sample_rate),
        "bytes": output_path.stat().st_size,
        **({"mel_path": str(mel_path)} if mel_path is not None else {}),
    }, indent=2))


if __name__ == "__main__":
    main()
