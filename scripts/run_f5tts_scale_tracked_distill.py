#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_CHECKPOINT = "/data/resumebot/checkpoints/final_finetuned_model.pt"
DEFAULT_VOCAB = "/data/resumebot/checkpoints/F5TTS_Base_vocab.txt"
DEFAULT_START_CHECKPOINT = (
    "/data/transformer_10/checkpoints/"
    "f5tts_q4_4step_cfg2_original_teacher_progressive_v0_longctx_phoneme_repair_v0/"
    "model_q4_12to4_best.pt"
)
DEFAULT_TEACHER_MANIFEST = (
    "/data/transformer_10/evals/"
    "f5tts_voice_profile_suite_2step_textwin_20260523/"
    "teacher_fp32_12step/manifest.json"
)
DEFAULT_BEST_GATE = (
    "/data/transformer_10/evals/"
    "f5tts_voice_profile_suite_4step_longctx_gate_20260524/"
    "longctx_best_nfe4_cfg085/gate_vs_teacher_tiny.json"
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise RuntimeError(f"command did not emit JSON:\n{stdout[-4000:]}")
    return json.loads(stdout[start:])


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return int(completed.returncode), completed.stdout


def load_best_score(path: str, fallback: float) -> float:
    if not str(path).strip():
        return float(fallback)
    payload_path = Path(path)
    if not payload_path.exists():
        return float(fallback)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    summary = (((payload.get("candidate") or {}).get("summary")) or {})
    return float(summary.get("mean_asr_phonetic_wer", fallback))


def last_train_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    last: dict[str, Any] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("event") == "train":
                last = row
    return last


def train_phase(args: argparse.Namespace, phase: int, phase_dir: Path, student_checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "distill_f5tts_12_to_4_q4.py"),
        "--checkpoint",
        str(args.teacher_checkpoint),
        "--student-checkpoint",
        str(student_checkpoint),
        "--vocab",
        str(args.vocab),
        "--output-dir",
        str(phase_dir),
        "--dataset",
        str(args.dataset),
        "--config",
        str(args.config),
        "--split",
        str(args.split),
        "--split-probabilities",
        str(args.split_probabilities),
        "--audio-column",
        str(args.audio_column),
        "--text-column",
        str(args.text_column),
        "--sample-rate",
        str(int(args.sample_rate)),
        "--min-duration",
        str(float(args.min_duration)),
        "--max-duration",
        str(float(args.max_duration)),
        "--max-frames",
        str(int(args.max_frames)),
        "--cond-frames",
        str(int(args.cond_frames)),
        "--min-gen-frames",
        str(int(args.min_gen_frames)),
        "--shuffle-buffer",
        str(int(args.shuffle_buffer)),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-steps",
        str(int(args.steps_per_phase)),
        "--teacher-steps",
        str(int(args.teacher_steps)),
        "--student-steps",
        str(int(args.student_steps)),
        "--teacher-cfg-strength",
        str(float(args.teacher_cfg_strength)),
        "--student-cfg-strength",
        str(float(args.student_cfg_strength)),
        "--student-cfg-strengths",
        str(args.student_cfg_strengths),
        "--sway-sampling-coef",
        str(float(args.sway_sampling_coef)),
        "--rollout-loss-weight",
        str(float(args.rollout_loss_weight)),
        "--teacher-flow-loss-weight",
        str(float(args.teacher_flow_loss_weight)),
        "--cond-delta-loss-weight",
        str(float(args.cond_delta_loss_weight)),
        "--segment-flow-loss-weight",
        str(float(args.segment_flow_loss_weight)),
        "--trajectory-loss-weight",
        str(float(args.trajectory_loss_weight)),
        "--real-mel-loss-weight",
        str(float(args.real_mel_loss_weight)),
        "--anchor-weight-loss-weight",
        str(float(args.anchor_weight_loss_weight)),
        "--temporal-delta-loss-weight",
        str(float(args.temporal_delta_loss_weight)),
        "--mel-energy-loss-weight",
        str(float(args.mel_energy_loss_weight)),
        "--energy-envelope-loss-weight",
        str(float(args.energy_envelope_loss_weight)),
        "--silence-envelope-loss-weight",
        str(float(args.silence_envelope_loss_weight)),
        "--high-mel-excess-loss-weight",
        str(float(args.high_mel_excess_loss_weight)),
        "--high-mel-match-loss-weight",
        str(float(args.high_mel_match_loss_weight)),
        "--high-mel-ratio-loss-weight",
        str(float(args.high_mel_ratio_loss_weight)),
        "--high-mel-temporal-loss-weight",
        str(float(args.high_mel_temporal_loss_weight)),
        "--low-mid-mel-body-loss-weight",
        str(float(args.low_mid_mel_body_loss_weight)),
        "--low-mid-mel-end-bin",
        str(int(args.low_mid_mel_end_bin)),
        "--high-mel-start-bin",
        str(int(args.high_mel_start_bin)),
        "--text-contrastive-loss-weight",
        str(float(args.text_contrastive_loss_weight)),
        "--text-corruption-mode",
        str(args.text_corruption_mode),
        "--text-flow-contrastive-loss-weight",
        str(float(args.text_flow_contrastive_loss_weight)),
        "--text-delta-loss-weight",
        str(float(args.text_delta_loss_weight)),
        "--loss-schedule-steps",
        str(int(args.loss_schedule_steps)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--max-grad-norm",
        str(float(args.max_grad_norm)),
        "--log-every",
        str(int(args.log_every)),
        "--save-every",
        str(int(args.save_every)),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed) + int(phase) - 1),
        "--q4-include",
        str(args.q4_include),
        "--q4-exclude",
        str(args.q4_exclude),
        "--q4-ste-include",
        str(args.q4_ste_include),
        "--student-quant-scheme",
        str(args.student_quant_scheme),
        "--local-sample-prob",
        str(float(args.local_sample_prob)),
        "--local-samples",
        str(args.local_samples),
        "--local-pairs",
        str(args.local_pairs),
        "--train-include",
        str(args.train_include),
        "--train-exclude",
        str(args.train_exclude),
    ]
    if args.cuda_visible_devices:
        command.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])
    if args.detach_null_grad:
        command.append("--detach-null-grad")
    if args.train_in_eval_mode:
        command.append("--train-in-eval-mode")
    if args.convert_text_to_pinyin:
        command.append("--convert-text-to-pinyin")
    if args.bitnet_qat_learned_scale:
        command.append("--bitnet-qat-learned-scale")
    if float(args.bitnet_scale_lr_multiplier) != 1.0:
        command.extend(["--bitnet-scale-lr-multiplier", str(float(args.bitnet_scale_lr_multiplier))])
    command.append("--save-best" if args.save_best else "--no-save-best")

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    start = time.time()
    returncode, stdout = run_command(command, env=env)
    log_path = phase_dir / "train_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"train phase failed with {returncode}; see {log_path}")

    checkpoint_prefix = f"model_q4_{int(args.teacher_steps)}to{int(args.student_steps)}"
    best_checkpoint = phase_dir / f"{checkpoint_prefix}_best.pt"
    last_checkpoint = phase_dir / f"{checkpoint_prefix}_last.pt"
    checkpoint = best_checkpoint if bool(args.save_best) and best_checkpoint.exists() else last_checkpoint
    if not checkpoint.exists():
        checkpoint = best_checkpoint
    if not checkpoint.exists():
        raise RuntimeError(f"train phase did not produce a checkpoint in {phase_dir}")
    best_metrics_path = phase_dir / "best_metrics.json"
    best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8")) if best_metrics_path.exists() else {}
    if not best_metrics:
        best_metrics = last_train_metrics(phase_dir / "metrics.jsonl")
    best_metrics["wall_seconds"] = time.time() - start
    best_metrics["stdout_log"] = str(log_path)
    return checkpoint, best_metrics


def command_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    return env


def render_suite(args: argparse.Namespace, checkpoint: Path, phase_eval_dir: Path, label: str) -> Path:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "eval_f5tts_voice_profile_suite.py"),
        "--checkpoint",
        str(checkpoint),
        "--vocab",
        str(args.vocab),
        "--voice-profile-dir",
        str(args.voice_profile_dir),
        "--out-dir",
        str(phase_eval_dir),
        "--label",
        label,
        "--nfe-step",
        str(int(args.eval_nfe_step)),
        "--cfg-strength",
        str(float(args.eval_cfg_strength)),
        "--sway-sampling-coef",
        str(float(args.eval_sway_sampling_coef)),
        "--speed",
        str(float(args.eval_speed)),
        "--seed",
        str(int(args.eval_seed)),
        "--device",
        str(args.device),
    ]
    if str(args.eval_duration_manifest).strip():
        command.extend(["--duration-manifest", str(args.eval_duration_manifest)])
    if args.eval_text_file:
        command.extend(["--text-file", str(args.eval_text_file)])
    if args.normalize_f5tts_text:
        command.append("--normalize-f5tts-text")
    start = time.time()
    returncode, stdout = run_command(command, env=command_env(args))
    log_path = phase_eval_dir / label / "render_stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"render failed with {returncode}; see {log_path}")
    manifest = phase_eval_dir / label / "manifest.json"
    if not manifest.exists():
        parse_json_from_stdout(stdout)
        if not manifest.exists():
            raise RuntimeError(f"render did not produce manifest: {manifest}")
    append_jsonl(args.ledger_path, {"event": "render", "time": time.time(), "label": label, "manifest": str(manifest), "wall_seconds": time.time() - start})
    return manifest


def gate_suite(args: argparse.Namespace, candidate_manifest: Path, gate_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "eval_f5tts_suite_gate.py"),
        "--teacher-manifest",
        str(args.teacher_manifest),
        "--candidate-manifest",
        str(candidate_manifest),
        "--asr-model",
        str(args.asr_model),
        "--asr-device",
        str(args.device),
        "--asr-reference-field",
        str(args.asr_reference_field),
    ]
    if args.asr_local_files_only:
        command.append("--asr-local-files-only")
    else:
        command.append("--no-asr-local-files-only")
    start = time.time()
    returncode, stdout = run_command(command, env=command_env(args))
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.with_suffix(".stdout.log").write_text(stdout, encoding="utf-8")
    if returncode != 0:
        raise RuntimeError(f"gate failed with {returncode}; see {gate_path.with_suffix('.stdout.log')}")
    payload = parse_json_from_stdout(stdout)
    payload["wall_seconds"] = time.time() - start
    gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def score_gate(gate: dict[str, Any]) -> dict[str, Any]:
    summary = ((gate.get("candidate") or {}).get("summary")) or {}
    teacher_summary = ((gate.get("teacher") or {}).get("summary")) or {}
    return {
        "mean_asr_phonetic_wer": float(summary.get("mean_asr_phonetic_wer", 999.0)),
        "mean_asr_wer": float(summary.get("mean_asr_wer", 999.0)),
        "total_clipped": int(summary.get("total_clipped", 999999)),
        "mean_peak": float(summary.get("mean_peak", 999.0)),
        "raw_peak_over_1_outputs": int(summary.get("raw_peak_over_1_outputs", 999999)),
        "max_raw_peak": float(summary.get("max_raw_peak", 999.0)),
        "repetition_flagged_outputs": int(summary.get("repetition_flagged_outputs", 999999)),
        "max_asr_consecutive_word_repeat": int(summary.get("max_asr_consecutive_word_repeat", 999999)),
        "total_asr_repeated_phrase_hits": int(summary.get("total_asr_repeated_phrase_hits", 999999)),
        "max_asr_repeated_phrase_count": int(summary.get("max_asr_repeated_phrase_count", 999999)),
        "total_asr_repeated_syllables": int(summary.get("total_asr_repeated_syllables", 999999)),
        "min_duration_ratio_to_teacher": float(summary.get("min_duration_ratio_to_teacher", 0.0)),
        "max_duration_shortfall_seconds": float(summary.get("max_duration_shortfall_seconds", 999.0)),
        "teacher_mean_asr_phonetic_wer": float(teacher_summary.get("mean_asr_phonetic_wer", 999.0)),
        "teacher_mean_asr_wer": float(teacher_summary.get("mean_asr_wer", 999.0)),
    }


def promotion_reasons(args: argparse.Namespace, metrics: dict[str, Any], best_score: float) -> list[str]:
    reasons: list[str] = []
    if metrics["mean_asr_phonetic_wer"] + float(args.promote_margin) >= best_score:
        reasons.append("phonetic WER did not improve previous best by margin")
    if metrics["mean_asr_wer"] > float(args.max_promote_mean_wer):
        reasons.append("mean WER above promotion limit")
    if metrics["mean_asr_phonetic_wer"] > float(args.max_promote_mean_phonetic_wer):
        reasons.append("mean phonetic WER above promotion limit")
    if metrics["total_clipped"] > int(args.max_promote_clipped_samples):
        reasons.append("too many clipped samples")
    if metrics["raw_peak_over_1_outputs"] > int(args.max_promote_raw_clip_outputs):
        reasons.append("raw audio exceeded peak limit")
    if metrics["repetition_flagged_outputs"] > int(args.max_promote_repetition_outputs):
        reasons.append("ASR detected repeated/stuttered outputs")
    if metrics["max_duration_shortfall_seconds"] > float(args.max_promote_duration_shortfall):
        reasons.append("candidate cuts off too much audio versus teacher")
    if metrics["min_duration_ratio_to_teacher"] < float(args.min_promote_duration_ratio):
        reasons.append("candidate duration ratio below teacher-relative floor")
    if bool(args.must_beat_teacher):
        if metrics["mean_asr_phonetic_wer"] >= metrics["teacher_mean_asr_phonetic_wer"]:
            reasons.append("candidate does not beat teacher mean phonetic WER")
        if metrics["mean_asr_wer"] >= metrics["teacher_mean_asr_wer"]:
            reasons.append("candidate does not beat teacher mean WER")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale-tracked F5TTS distillation with held-out promotion gates.")
    parser.add_argument("--run-name", default="f5tts_q4_4step_scale_tracked_clean360_v0")
    parser.add_argument("--output-root", type=Path, default=Path("/data/transformer_10/checkpoints"))
    parser.add_argument("--eval-root", type=Path, default=Path("/data/transformer_10/evals"))
    parser.add_argument("--ledger-path", type=Path, default=Path("/data/transformer_10/logs/f5tts_scale_tracked_distill_ledger.jsonl"))
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--start-checkpoint", default=DEFAULT_START_CHECKPOINT)
    parser.add_argument("--teacher-manifest", default=DEFAULT_TEACHER_MANIFEST)
    parser.add_argument("--best-gate-json", default=DEFAULT_BEST_GATE)
    parser.add_argument("--baseline-phonetic-wer", type=float, default=0.5829545454545455)
    parser.add_argument("--regate-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--voice-profile-dir", default="/data/resumebot/voice_profiles/Peyton")
    parser.add_argument("--dataset", default="blabble-io/libritts_r")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--split", default="train.clean.360")
    parser.add_argument("--split-probabilities", default="")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text_normalized")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=7.0)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--cond-frames", type=int, default=256)
    parser.add_argument("--min-gen-frames", type=int, default=48)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--phases", type=int, default=8)
    parser.add_argument("--steps-per-phase", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=1000000000)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-steps", type=int, default=24)
    parser.add_argument("--student-steps", type=int, default=4)
    parser.add_argument("--teacher-cfg-strength", type=float, default=2.0)
    parser.add_argument("--student-cfg-strength", type=float, default=0.85)
    parser.add_argument("--student-cfg-strengths", default="0.8,0.85,0.9")
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument("--teacher-flow-loss-weight", type=float, default=0.0)
    parser.add_argument("--cond-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--segment-flow-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.0)
    parser.add_argument("--real-mel-loss-weight", type=float, default=0.14)
    parser.add_argument("--anchor-weight-loss-weight", type=float, default=0.14)
    parser.add_argument("--temporal-delta-loss-weight", type=float, default=0.3)
    parser.add_argument("--mel-energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--energy-envelope-loss-weight", type=float, default=0.0)
    parser.add_argument("--silence-envelope-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-match-loss-weight", type=float, default=0.04)
    parser.add_argument("--high-mel-ratio-loss-weight", type=float, default=0.0)
    parser.add_argument("--high-mel-temporal-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-mid-mel-body-loss-weight", type=float, default=0.0)
    parser.add_argument("--low-mid-mel-end-bin", type=int, default=80)
    parser.add_argument("--high-mel-start-bin", type=int, default=80)
    parser.add_argument("--text-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-corruption-mode", default="reverse")
    parser.add_argument("--text-flow-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--loss-schedule-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--student-quant-scheme", choices=("q4", "bitnet_qat", "none"), default="q4")
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--bitnet-scale-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--q4-include", default="")
    parser.add_argument("--q4-exclude", default="text_embed.text_embed,mel_spec")
    parser.add_argument("--q4-ste-include", default="transformer.transformer_blocks.14,transformer.transformer_blocks.15,transformer.transformer_blocks.16,transformer.transformer_blocks.17,transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out")
    parser.add_argument("--train-include", default="transformer.transformer_blocks.14,transformer.transformer_blocks.15,transformer.transformer_blocks.16,transformer.transformer_blocks.17,transformer.transformer_blocks.18,transformer.transformer_blocks.19,transformer.transformer_blocks.20,transformer.transformer_blocks.21,transformer.norm_out,transformer.proj_out")
    parser.add_argument("--train-exclude", default="")
    parser.add_argument("--local-samples", default="/data/resumebot/voice_profiles/Peyton/samples.txt")
    parser.add_argument("--local-pairs", default="")
    parser.add_argument("--local-sample-prob", type=float, default=0.0)
    parser.add_argument("--detach-null-grad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-in-eval-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--convert-text-to-pinyin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-nfe-step", type=int, default=4)
    parser.add_argument("--eval-cfg-strength", type=float, default=0.85)
    parser.add_argument("--eval-sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--eval-speed", type=float, default=1.0)
    parser.add_argument("--eval-duration-manifest", default="")
    parser.add_argument("--eval-seed", type=int, default=1337)
    parser.add_argument("--eval-text-file", default="")
    parser.add_argument("--normalize-f5tts-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--asr-reference-field", choices=("text", "speech_text"), default="text")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--asr-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--promote-margin", type=float, default=0.01)
    parser.add_argument("--must-beat-teacher", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-promote-mean-wer", type=float, default=0.25)
    parser.add_argument("--max-promote-mean-phonetic-wer", type=float, default=0.25)
    parser.add_argument("--max-promote-clipped-samples", type=int, default=64)
    parser.add_argument("--max-promote-raw-clip-outputs", type=int, default=0)
    parser.add_argument("--max-promote-repetition-outputs", type=int, default=0)
    parser.add_argument("--max-promote-duration-shortfall", type=float, default=0.35)
    parser.add_argument("--min-promote-duration-ratio", type=float, default=0.88)
    parser.add_argument(
        "--chain-phases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue the next training phase from the latest candidate even when it does not pass the held-out promotion gate.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_root / args.run_name
    eval_dir = args.eval_root / f"{args.run_name}_gate"
    args.ledger_path = Path(args.ledger_path)
    best_score = load_best_score(args.best_gate_json, args.baseline_phonetic_wer)
    best_checkpoint = Path(args.start_checkpoint)
    current_checkpoint = best_checkpoint
    append_jsonl(
        args.ledger_path,
        {
            "event": "setup",
            "time": time.time(),
            "run_name": args.run_name,
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "split_probabilities": args.split_probabilities,
            "student_quant_scheme": args.student_quant_scheme,
            "student_steps": int(args.student_steps),
            "teacher_steps": int(args.teacher_steps),
            "start_checkpoint": str(best_checkpoint),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "teacher_manifest": str(args.teacher_manifest),
            "heldout_voice_profile": str(args.voice_profile_dir),
            "local_sample_prob": float(args.local_sample_prob),
            "chain_phases": bool(args.chain_phases),
            "best_start_mean_asr_phonetic_wer": best_score,
        },
    )
    if args.dry_run:
        print(json.dumps({"ok": True, "run_dir": str(run_dir), "ledger": str(args.ledger_path), "best_score": best_score}, indent=2))
        return

    if args.regate_start and str(args.asr_model).strip():
        baseline_label = f"start_baseline_nfe{int(args.eval_nfe_step)}_cfg{str(args.eval_cfg_strength).replace('.', '')}"
        baseline_manifest = render_suite(args, current_checkpoint, eval_dir, baseline_label)
        baseline_gate_path = eval_dir / baseline_label / "gate_vs_teacher.json"
        baseline_gate = gate_suite(args, baseline_manifest, baseline_gate_path)
        baseline_metrics = score_gate(baseline_gate)
        best_score = float(baseline_metrics["mean_asr_phonetic_wer"])
        append_jsonl(
            args.ledger_path,
            {
                "event": "regate_start",
                "time": time.time(),
                "checkpoint": str(current_checkpoint),
                "manifest": str(baseline_manifest),
                "gate": str(baseline_gate_path),
                **baseline_metrics,
            },
        )

    for phase in range(1, int(args.phases) + 1):
        phase_dir = run_dir / f"phase_{phase:04d}"
        append_jsonl(args.ledger_path, {"event": "phase_start", "time": time.time(), "phase": phase, "student_checkpoint": str(current_checkpoint)})
        candidate_checkpoint, train_metrics = train_phase(args, phase, phase_dir, current_checkpoint)
        append_jsonl(
            args.ledger_path,
            {
                "event": "phase_train_done",
                "time": time.time(),
                "phase": phase,
                "checkpoint": str(candidate_checkpoint),
                "train_metrics": train_metrics,
            },
        )

        label = f"phase_{phase:04d}_nfe{int(args.eval_nfe_step)}_cfg{str(args.eval_cfg_strength).replace('.', '')}"
        candidate_manifest = render_suite(args, candidate_checkpoint, eval_dir, label)
        gate_path = eval_dir / label / "gate_vs_teacher.json"
        gate = gate_suite(args, candidate_manifest, gate_path)
        metrics = score_gate(gate)
        blocked_reasons = promotion_reasons(args, metrics, best_score)
        promoted = not blocked_reasons
        row = {
            "event": "phase_gate_done",
            "time": time.time(),
            "phase": phase,
            "checkpoint": str(candidate_checkpoint),
            "manifest": str(candidate_manifest),
            "gate": str(gate_path),
            **metrics,
            "previous_best_mean_asr_phonetic_wer": best_score,
            "max_promote_clipped_samples": int(args.max_promote_clipped_samples),
            "max_promote_raw_clip_outputs": int(args.max_promote_raw_clip_outputs),
            "max_promote_repetition_outputs": int(args.max_promote_repetition_outputs),
            "max_promote_duration_shortfall": float(args.max_promote_duration_shortfall),
            "min_promote_duration_ratio": float(args.min_promote_duration_ratio),
            "must_beat_teacher": bool(args.must_beat_teacher),
            "blocked_reasons": blocked_reasons,
            "promoted": promoted,
        }
        append_jsonl(args.ledger_path, row)
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
        if promoted:
            best_score = float(metrics["mean_asr_phonetic_wer"])
            best_checkpoint = candidate_checkpoint
            append_jsonl(args.ledger_path, {"event": "promote", "time": time.time(), "phase": phase, "best_checkpoint": str(best_checkpoint), "best_score": best_score})
        if bool(args.chain_phases):
            current_checkpoint = candidate_checkpoint
            append_jsonl(
                args.ledger_path,
                {
                    "event": "chain_candidate",
                    "time": time.time(),
                    "phase": phase,
                    "next_student_checkpoint": str(current_checkpoint),
                    "promoted": promoted,
                    "best_checkpoint": str(best_checkpoint),
                    "best_score": best_score,
                },
            )

    print(
        json.dumps(
            {
                "done": True,
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(current_checkpoint),
                "best_mean_asr_phonetic_wer": best_score,
                "ledger": str(args.ledger_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
