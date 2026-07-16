#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from eval_f5tts_voice_profile import (
    DEFAULT_PROFILE,
    DEFAULT_VOCAB,
    build_model,
    first_profile_sample,
    load_checkpoint,
    patch_torchaudio_load,
)
from tts_text_normalizer import normalize_f5tts_speech_text


DEFAULT_TEXTS = (
    "This is Peyton speaking from Agent Kernel Lite with the clean LibriTTS two step distilled F5 TTS model.",
    "Agent Kernel Lite can generate my voice on iPhone with the custom int4 WebAssembly runtime.",
    "The model should speak clearly without static, clipping, or missing words.",
    "Today we are testing longer voice cloning quality across several different sentences.",
)


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


def parse_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    for text in args.text:
        text = text.strip()
        if text:
            texts.append(text)
    if args.text_file:
        for line in Path(args.text_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                texts.append(line)
    return texts or list(DEFAULT_TEXTS)


def load_fixed_durations(manifest_path: str) -> list[float | None]:
    if not str(manifest_path).strip():
        return []
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    durations: list[float | None] = []
    for item in manifest.get("outputs", []):
        sample_rate = int(item.get("sample_rate") or 0)
        samples = int(item.get("samples") or 0)
        if sample_rate > 0 and samples > 0:
            durations.append(float(samples) / float(sample_rate))
        else:
            durations.append(None)
    return durations


def safe_slug(index: int, text: str) -> str:
    words = "".join(char.lower() if char.isalnum() else "_" for char in text).strip("_")
    return f"{index:02d}_{words[:48].strip('_') or 'sample'}"


def load_runtime(checkpoint: Path, vocab: Path, device: torch.device):
    model = build_model(vocab, device)
    step = load_checkpoint(model, checkpoint, device)
    model.eval()
    return model, step


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Peyton F5TTS validation suite from one loaded checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--voice-profile-dir", default=DEFAULT_PROFILE)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--text-file", default="")
    parser.add_argument(
        "--normalize-f5tts-text",
        action="store_true",
        help="Render with app-compatible F5TTS speech text normalization while preserving original text in the manifest.",
    )
    parser.add_argument("--nfe-step", type=int, default=8)
    parser.add_argument("--cfg-strength", type=float, default=1.25)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--ode-method", default="", help="Optional torchdiffeq ODE method override, e.g. euler or midpoint.")
    parser.add_argument("--vocos-checkpoint", default="", help="Optional adapted Vocos checkpoint to load before decoding.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration-manifest",
        default="",
        help="Optional manifest whose per-index sample counts set F5TTS fix_duration for parity rendering.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--save-mel", action="store_true", help="Save generated mel spectrogram arrays next to WAV outputs.")
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

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    model, step = load_runtime(checkpoint, Path(args.vocab), device)
    if str(args.ode_method).strip():
        model.odeint_kwargs = {**getattr(model, "odeint_kwargs", {}), "method": str(args.ode_method).strip()}
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=str(device))
    if str(args.vocos_checkpoint).strip():
        vocos_payload = torch.load(str(args.vocos_checkpoint), map_location=device)
        vocos_state = vocos_payload.get("model_state_dict", vocos_payload)
        missing, unexpected = vocoder.load_state_dict(vocos_state, strict=False)
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
    ref_audio_path, ref_text = first_profile_sample(Path(args.voice_profile_dir))
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
    fixed_durations = load_fixed_durations(args.duration_manifest)
    ref_info = sf.info(ref_audio)
    ref_duration = float(ref_info.frames) / float(ref_info.samplerate) if ref_info.samplerate else 0.0

    outputs = []
    with torch.no_grad():
        for index, text in enumerate(parse_texts(args)):
            speech_text = normalize_f5tts_speech_text(text) if args.normalize_f5tts_text else text
            fixed_generated_duration = fixed_durations[index] if index < len(fixed_durations) else None
            fix_duration = (
                float(ref_duration) + float(fixed_generated_duration)
                if fixed_generated_duration is not None
                else None
            )
            audio, sample_rate, mel = infer_process(
                ref_audio,
                ref_text,
                speech_text,
                model,
                vocoder,
                mel_spec_type="vocos",
                speed=float(args.speed),
                nfe_step=int(args.nfe_step),
                cfg_strength=float(args.cfg_strength),
                sway_sampling_coef=float(args.sway_sampling_coef),
                fix_duration=fix_duration,
                device=str(device),
            )
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32)
            raw_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if raw_peak > 0.98:
                audio = audio * (0.98 / raw_peak)
            path = out_dir / f"{safe_slug(index, text)}.wav"
            sf.write(path, audio, int(sample_rate), format="WAV", subtype="PCM_16")
            mel_path = None
            if bool(args.save_mel) and mel is not None:
                mel_path = out_dir / f"{safe_slug(index, text)}.mel.npy"
                np.save(mel_path, np.asarray(mel, dtype=np.float32))
            outputs.append(
                {
                    "index": index,
                    "text": text,
                    "speech_text": speech_text,
                    "normalized_f5tts_text": bool(args.normalize_f5tts_text and speech_text != text),
                    "path": str(path),
                    "raw_peak": raw_peak,
                    "samples": int(audio.size),
                    "sample_rate": int(sample_rate),
                    "fix_duration": fix_duration,
                    "fixed_generated_duration": fixed_generated_duration,
                    **({"mel_path": str(mel_path)} if mel_path is not None else {}),
                }
            )

    manifest = {
        "label": args.label,
        "checkpoint": str(checkpoint),
        "vocos_checkpoint": str(args.vocos_checkpoint).strip() or None,
        "checkpoint_step": step,
        "voice_profile_dir": str(Path(args.voice_profile_dir)),
        "ref_audio": ref_audio_path,
        "nfe_step": int(args.nfe_step),
        "cfg_strength": float(args.cfg_strength),
        "duration_manifest": str(args.duration_manifest).strip() or None,
        "ode_method": str(args.ode_method).strip() or None,
        "seed": int(args.seed),
        "runtime": runtime,
        "outputs": outputs,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
