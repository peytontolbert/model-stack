#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = "/data/resumebot/checkpoints/F5TTS_Base_vocab.txt"
DEFAULT_PROFILE = "/data/resumebot/voice_profiles/Peyton"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def checkpoint_candidates(checkpoint_dir: Path, patterns: list[str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(checkpoint_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                candidates.append(path)
                seen.add(path)
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir} for patterns {patterns}")
    return candidates


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def json_from_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeError(f"command did not emit JSON:\n{stdout[-4000:]}")
    return json.loads(stdout[start:])


def load_gate_summary(path: str) -> dict[str, Any]:
    if not str(path).strip():
        return {}
    payload_path = Path(path)
    if not payload_path.exists():
        return {}
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    return dict(((payload.get("candidate") or {}).get("summary")) or {})


def candidate_score(summary: dict[str, Any]) -> tuple[float, float, int, float, float]:
    return (
        float(summary.get("mean_asr_phonetic_wer", 999.0)),
        float(summary.get("mean_asr_wer", 999.0)),
        int(summary.get("raw_peak_over_1_outputs", 999)),
        float(summary.get("max_raw_peak", 999.0)),
        float(summary.get("mean_logmel_dtw_to_teacher", 999.0)),
    )


def beats_baseline(summary: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> bool:
    if not baseline:
        return False
    if int(summary.get("raw_peak_over_1_outputs", 0)) > int(baseline.get("raw_peak_over_1_outputs", 0)):
        return False
    candidate_pwer = float(summary.get("mean_asr_phonetic_wer", 999.0))
    baseline_pwer = float(baseline.get("mean_asr_phonetic_wer", 999.0))
    candidate_wer = float(summary.get("mean_asr_wer", 999.0))
    baseline_wer = float(baseline.get("mean_asr_wer", 999.0))
    if candidate_pwer < baseline_pwer - float(tolerance):
        return True
    if abs(candidate_pwer - baseline_pwer) <= float(tolerance) and candidate_wer < baseline_wer - float(tolerance):
        return True
    return False


def label_for(checkpoint: Path, nfe_step: int, cfg_strength: float) -> str:
    stem = checkpoint.stem
    for prefix in ("model_q4_", "model_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    cfg_label = str(float(cfg_strength)).replace(".", "p")
    return f"{stem}_nfe{int(nfe_step)}_cfg{cfg_label}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and suite-gate F5TTS checkpoints.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--checkpoint-patterns",
        default="model_q4_*best_val_rollout.pt,model_q4_*best_rollout.pt,model_q4_*best.pt",
        help="Comma-separated glob patterns inside --checkpoint-dir.",
    )
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--baseline-gate", default="")
    parser.add_argument("--voice-profile-dir", default=DEFAULT_PROFILE)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nfe-step", type=int, default=6)
    parser.add_argument("--cfg-strength", type=float, default=1.25)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--min-cuda-total-gb", type=float, default=0.0)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--asr-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--asr-reference-field", choices=("text", "speech_text"), default="speech_text")
    parser.add_argument("--improvement-tolerance", type=float, default=1e-6)
    parser.add_argument("--normalize-f5tts-text", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if str(args.cuda_visible_devices).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()

    render_script = ROOT_DIR / "scripts" / "eval_f5tts_voice_profile_suite.py"
    gate_script = ROOT_DIR / "scripts" / "eval_f5tts_suite_gate.py"
    baseline_summary = load_gate_summary(str(args.baseline_gate))
    results: list[dict[str, Any]] = []

    for checkpoint in checkpoint_candidates(checkpoint_dir, parse_csv(args.checkpoint_patterns)):
        label = label_for(checkpoint, int(args.nfe_step), float(args.cfg_strength))
        render_command = [
            sys.executable,
            str(render_script),
            "--checkpoint",
            str(checkpoint),
            "--vocab",
            str(args.vocab),
            "--voice-profile-dir",
            str(args.voice_profile_dir),
            "--out-dir",
            str(out_dir),
            "--label",
            label,
            "--text-file",
            str(args.text_file),
            "--nfe-step",
            str(int(args.nfe_step)),
            "--cfg-strength",
            str(float(args.cfg_strength)),
            "--speed",
            str(float(args.speed)),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
        ]
        if bool(args.normalize_f5tts_text):
            render_command.append("--normalize-f5tts-text")
        if float(args.min_cuda_total_gb) > 0.0:
            render_command.extend(["--min-cuda-total-gb", str(float(args.min_cuda_total_gb))])
        render_payload = json_from_stdout(run_command(render_command, env=env))
        manifest_path = Path(render_payload["outputs"][0]["path"]).parent / "manifest.json"
        gate_path = manifest_path.parent / "gate_vs_teacher.json"
        gate_command = [
            sys.executable,
            str(gate_script),
            "--candidate-manifest",
            str(manifest_path),
            "--teacher-manifest",
            str(args.teacher_manifest),
            "--asr-model",
            str(args.asr_model),
            "--asr-device",
            str(args.asr_device),
            "--asr-local-files-only",
            "--asr-reference-field",
            str(args.asr_reference_field),
        ]
        gate_payload = json_from_stdout(run_command(gate_command, env=env))
        gate_path.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = dict(((gate_payload.get("candidate") or {}).get("summary")) or {})
        results.append(
            {
                "checkpoint": str(checkpoint),
                "label": label,
                "manifest": str(manifest_path),
                "gate": str(gate_path),
                "summary": summary,
                "score": candidate_score(summary),
                "beats_baseline": beats_baseline(
                    summary,
                    baseline_summary,
                    float(args.improvement_tolerance),
                ),
            }
        )

    results.sort(key=lambda row: tuple(row["score"]))
    payload = {
        "checkpoint_dir": str(checkpoint_dir),
        "teacher_manifest": str(args.teacher_manifest),
        "baseline_gate": str(args.baseline_gate),
        "baseline_summary": baseline_summary,
        "best": results[0],
        "results": results,
    }
    result_path = out_dir / "suite_checkpoint_gate.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "best": payload["best"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
