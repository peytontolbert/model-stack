#!/usr/bin/env python3
"""Render F5TTS Parquet jobs into utterance WAVs.

This is a thin bridge to the existing AgentKernel Lite F5TTS Q4 JS/WASM runtime.
It reads the Parquet jobs produced by `build_synthetic_meeting_asr_dataset.py
plan-f5`, renders each row, copies the generated WAV to `output_wav_path`, and
writes a compact rendered-utterance Parquet for the synthetic meeting mixer.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_AGENT_KERNEL_LITE_ROOT = Path("/data/agent_kernel_lite")
DEFAULT_F5_BUNDLE = Path("/data/agent_kernel_lite/artifacts/hf_releases/f5tts-4bit-distill")


def _extract_voice_tar(f5_bundle: Path, extract_dir: Path) -> Path:
    vocos_dir = extract_dir / "models" / "vocos_mel_24khz_q4_v0"
    if (vocos_dir / "manifest.json").exists():
        return vocos_dir
    tar_path = f5_bundle / "peyton_voice_q4.tar"
    if not tar_path.exists():
        raise FileNotFoundError(f"missing voice tar for Vocos bundle: {tar_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as archive:
        archive.extractall(extract_dir)
    if not (vocos_dir / "manifest.json").exists():
        raise FileNotFoundError(f"extracted Vocos bundle not found at {vocos_dir}")
    return vocos_dir


def _parse_json_stdout(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"renderer did not emit JSON: {stdout[-500:]}")
    return json.loads(stdout[start : end + 1])


def _limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def _run_renderer(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr or exc.stdout or str(exc)
        match = re.search(r"reference wav only yielded (\d+) mel frames", message)
        if not match:
            raise RuntimeError(f"F5TTS renderer failed:\n{message}") from exc
        yielded = max(32, int(match.group(1)) - 1)
        retry_command = list(command)
        retry_command[6] = str(yielded)
        try:
            return subprocess.run(
                retry_command,
                cwd=str(cwd),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as retry_exc:
            retry_message = retry_exc.stderr or retry_exc.stdout or str(retry_exc)
            raise RuntimeError(f"F5TTS renderer failed after cond-seq retry:\n{retry_message}") from retry_exc


def render_jobs(args: argparse.Namespace) -> None:
    job_table = pq.read_table(args.jobs)
    jobs = _limit_rows(job_table.to_pylist(), args.limit)
    if not jobs:
        raise ValueError(f"no render jobs found in {args.jobs}")

    agent_root = Path(args.agent_kernel_lite_root)
    runner = agent_root / "examples" / "18_f5tts_q4_peyton_ref_smoke" / "run.mjs"
    if not runner.exists():
        raise FileNotFoundError(f"AgentKernel Lite F5 runner not found: {runner}")

    default_f5_bundle = Path(args.f5_bundle)
    vocos_bundle = Path(args.vocos_bundle) if args.vocos_bundle else _extract_voice_tar(
        default_f5_bundle, Path(args.voice_extract_dir)
    )

    rendered_rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        text = str(job.get("text") or "").strip()
        reference_text = str(job.get("speaker_reference_text") or "").strip()
        reference = Path(str(job.get("speaker_reference_path") or ""))
        output_wav = Path(str(job.get("output_wav_path") or ""))
        f5_bundle = Path(str(job.get("f5_bundle_path") or default_f5_bundle))
        if not text:
            raise ValueError(f"job {index} has empty text")
        if not reference.exists():
            raise FileNotFoundError(f"job {index} reference wav not found: {reference}")
        if not output_wav:
            raise ValueError(f"job {index} has empty output_wav_path")

        gen_frames = int(args.gen_frames)
        if gen_frames <= 0:
            gen_frames = max(24, int(round(len(text.encode("utf-8")) * float(args.frames_per_text_byte))))
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "node",
            str(runner),
            str(f5_bundle),
            str(vocos_bundle),
            str(reference),
            text,
            str(args.cond_seq_len),
            str(gen_frames),
            str(args.steps),
            str(args.cfg_strength),
        ]
        if reference_text:
            command.append(reference_text)
        result = _run_renderer(command, cwd=agent_root)
        metadata = _parse_json_stdout(result.stdout)
        rendered_wav = Path(metadata["wavPath"])
        if not rendered_wav.exists():
            raise FileNotFoundError(f"renderer reported missing wav: {rendered_wav}")
        shutil.copyfile(rendered_wav, output_wav)
        rendered_rows.append(
            {
                "job_id": str(job.get("job_id") or f"f5_{index:08d}"),
                "audio_path": str(output_wav),
                "text": text,
                "speaker_id": str(job.get("speaker_id") or reference.stem),
                "voice_id": str(job.get("voice_id") or reference.stem),
                "speaker_reference_path": str(reference),
                "speaker_reference_text": reference_text,
                "f5_bundle_path": str(f5_bundle),
                "vocos_bundle_path": str(vocos_bundle),
                "renderer_metadata_json": json.dumps(metadata, ensure_ascii=True),
            }
        )
        print(f"rendered {index + 1}/{len(jobs)}: {output_wav}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rendered_rows), args.output, compression="zstd")
    print(f"wrote {len(rendered_rows)} rendered utterance rows to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, help="Parquet render jobs from plan-f5")
    parser.add_argument("--output", required=True, help="Rendered utterance Parquet")
    parser.add_argument("--agent-kernel-lite-root", default=str(DEFAULT_AGENT_KERNEL_LITE_ROOT))
    parser.add_argument("--f5-bundle", default=str(DEFAULT_F5_BUNDLE))
    parser.add_argument("--vocos-bundle", default="")
    parser.add_argument("--voice-extract-dir", default="/tmp/bddy-f5tts-voice-bundle")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cond-seq-len", type=int, default=256)
    parser.add_argument("--gen-frames", type=int, default=0, help="0 auto-sizes frames from text length")
    parser.add_argument("--frames-per-text-byte", type=float, default=6.1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    render_jobs(args)


if __name__ == "__main__":
    main()
