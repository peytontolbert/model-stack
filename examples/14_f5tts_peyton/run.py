#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


RESUMEBOT_DIR = Path("/data/resumebot")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Peyton-voice F5TTS smoke-test WAV.")
    parser.add_argument(
        "--text",
        default="This is a quick local test of my F5 TTS voice running from transformer ten.",
        help="Text to synthesize.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory to copy the generated WAV into.")
    parser.add_argument("--model-dir", default=str(RESUMEBOT_DIR / "checkpoints"), help="F5TTS checkpoint directory.")
    parser.add_argument("--voice-profile", default="Peyton", help="Voice profile name.")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the original temp file under /data/resumebot/temp_audio in addition to the copied WAV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not RESUMEBOT_DIR.exists():
        raise SystemExit(f"missing resumebot directory: {RESUMEBOT_DIR}")

    sys.path.insert(0, str(RESUMEBOT_DIR))
    from tts_service import F5TTSService  # noqa: PLC0415

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    service = F5TTSService(model_dir=str(Path(args.model_dir)), voice_profile=str(args.voice_profile))
    generated = service.synthesize(str(args.text))
    if not generated:
        raise SystemExit("F5TTS synthesis failed")

    generated_path = Path(generated)
    target_path = out_dir / "peyton_f5tts_smoke.wav"
    shutil.copy2(generated_path, target_path)
    if not args.keep_temp:
        generated_path.unlink(missing_ok=True)

    print(f"generated_wav={target_path}")
    print(f"bytes={target_path.stat().st_size}")


if __name__ == "__main__":
    main()
