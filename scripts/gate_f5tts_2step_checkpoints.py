#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def checkpoint_candidates(checkpoint_dir: Path) -> list[Path]:
    preferred = [
        checkpoint_dir / "model_q4_12to2_best_rollout.pt",
        checkpoint_dir / "model_q4_12to2_best.pt",
        checkpoint_dir / "model_q4_12to2_last.pt",
    ]
    steps = sorted(checkpoint_dir.glob("model_q4_12to2_step_*.pt"))
    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in [*steps, *preferred]:
        if path.exists() and path not in seen:
            candidates.append(path)
            seen.add(path)
    if not candidates:
        raise FileNotFoundError(f"no 2-step checkpoints found in {checkpoint_dir}")
    return candidates


def run_json(command: list[str]) -> dict:
    proc = subprocess.run(command, check=True, text=True, capture_output=True)
    start = proc.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"command did not emit JSON: {' '.join(command)}\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout[start:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and ASR-gate F5TTS 2-step checkpoints.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--teacher-wav", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice-profile-dir", default="/data/resumebot/voice_profiles/Peyton")
    parser.add_argument("--vocab", default="/data/resumebot/checkpoints/F5TTS_Base_vocab.txt")
    parser.add_argument("--out-dir", default="/data/transformer_10/evals/f5tts_checkpoint_gate")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nfe-step", type=int, default=2)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir) / checkpoint_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    render_script = str(ROOT_DIR / "scripts" / "eval_f5tts_voice_profile.py")
    for checkpoint in checkpoint_candidates(checkpoint_dir):
        stem = checkpoint.stem.replace("model_q4_12to2_", "")
        output = out_dir / f"{stem}_seed{int(args.seed)}.wav"
        command = [
            sys.executable,
            render_script,
            "--checkpoint",
            str(checkpoint),
            "--vocab",
            str(args.vocab),
            "--voice-profile-dir",
            str(args.voice_profile_dir),
            "--output-path",
            str(output),
            "--text",
            str(args.text),
            "--nfe-step",
            str(int(args.nfe_step)),
            "--cfg-strength",
            str(float(args.cfg_strength)),
            "--sway-sampling-coef",
            str(float(args.sway_sampling_coef)),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
        ]
        run_json(command)
        rendered.append(output)

    gate_script = str(ROOT_DIR / "scripts" / "eval_f5tts_audio_gate.py")
    command = [
        sys.executable,
        gate_script,
        "--teacher",
        str(args.teacher_wav),
        "--text",
        str(args.text),
        "--asr-model",
        str(args.asr_model),
        "--asr-device",
        str(args.device),
        "--asr-local-files-only",
    ]
    for output in rendered:
        command.extend(["--candidate", str(output)])
    result = run_json(command)
    rows = result["results"]
    best = min(
        rows,
        key=lambda row: (
            float(row.get("asr_phonetic_wer", row.get("asr_wer", 999.0))),
            float(row.get("asr_wer", 999.0)),
            float(row.get("logmel_dtw_to_teacher", 999.0)),
            float(row.get("mfcc_dtw_to_teacher", 999.0)),
        ),
    )
    result["best"] = best
    result_path = out_dir / "gate_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "best": best}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
