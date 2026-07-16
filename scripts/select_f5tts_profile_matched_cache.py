#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distill_f5tts_12_to_4_q4 import DEFAULT_VOCAB, audio_array_to_item, build_model  # noqa: E402


def mel_features(mel: torch.Tensor, high_start_bin: int) -> dict[str, float]:
    x = mel.float()
    if x.ndim == 3:
        x = x[0]
    high = x[:, high_start_bin:]
    low = x[:, :high_start_bin]
    temporal = torch.mean(torch.abs(x[1:] - x[:-1])).item() if x.shape[0] > 1 else 0.0
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p99": float(torch.quantile(x.flatten(), 0.99).item()),
        "high_to_low": float((high.abs().mean() / low.abs().mean().clamp_min(1e-6)).item()),
        "temporal_abs": float(temporal),
    }


def feature_distance(features: dict[str, float], target: dict[str, float], scales: dict[str, float]) -> float:
    total = 0.0
    for key, target_value in target.items():
        scale = max(float(scales.get(key, 1.0)), 1e-6)
        delta = (float(features[key]) - float(target_value)) / scale
        total += delta * delta
    return float(total)


def iter_cache_rows(cache_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cache_dir in cache_dirs:
        metadata_path = cache_dir / "metadata.jsonl"
        if not metadata_path.exists():
            continue
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "done" or not row.get("path"):
                continue
            row["_cache_dir"] = str(cache_dir)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select F5TTS teacher-cache rows whose conditioning mel matches a held-out profile statistically.")
    parser.add_argument("--source-cache-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile-audio", required=True)
    parser.add_argument("--profile-text", required=True)
    parser.add_argument("--preprocess-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration", type=float, default=0.25)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--min-gen-frames", type=int, default=1)
    parser.add_argument("--cond-frames", type=int, default=256)
    parser.add_argument("--high-start-bin", type=int, default=46)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(Path(args.vocab), device)
    profile_audio = str(args.profile_audio)
    profile_text = str(args.profile_text)
    if bool(args.preprocess_profile):
        resumebot_root = Path("/data/resumebot")
        if str(resumebot_root) not in sys.path:
            sys.path.insert(0, str(resumebot_root))
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        profile_audio, profile_text = preprocess_ref_audio_text(profile_audio, profile_text)
    audio, sample_rate = sf.read(profile_audio, always_2d=False, dtype="float32")
    item = audio_array_to_item(audio, profile_text, int(sample_rate), args=args, model=model, device=device)
    if item is None:
        raise RuntimeError("Profile audio did not produce a mel item")
    profile_mel = item["mel_spec"].transpose(0, 1).unsqueeze(0).cpu()
    profile_mel = profile_mel[:, : int(args.cond_frames), :]
    target = mel_features(profile_mel, int(args.high_start_bin))

    candidates: list[dict[str, Any]] = []
    all_features: list[dict[str, float]] = []
    for row in iter_cache_rows([Path(value) for value in args.source_cache_dir]):
        payload = torch.load(Path(row["_cache_dir"]) / str(row["path"]), map_location="cpu")
        cond_frames = int(payload.get("cond_frames", row.get("cond_frames", int(args.cond_frames))))
        cond = payload["cond"][:, : min(cond_frames, int(args.cond_frames)), :].cpu()
        feats = mel_features(cond, int(args.high_start_bin))
        all_features.append(feats)
        candidates.append({"row": row, "features": feats})
    if not candidates:
        raise RuntimeError("No cache rows found")

    scales = {}
    for key in target:
        values = np.asarray([entry[key] for entry in all_features], dtype=np.float64)
        scales[key] = float(max(values.std(), 1e-3))
    for candidate in candidates:
        candidate["distance"] = feature_distance(candidate["features"], target, scales)
    candidates.sort(key=lambda entry: entry["distance"])
    selected = candidates[: int(args.top_k)]

    out_dir = Path(args.output_dir)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"
    metadata_path.unlink(missing_ok=True)
    (out_dir / "cache_config.json").write_text(
        json.dumps({"args": vars(args), "target": target, "scales": scales}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_rows = []
    for index, entry in enumerate(selected):
        row = dict(entry["row"])
        source_path = Path(row["_cache_dir"]) / str(row["path"])
        dest_path = sample_dir / f"sample_{index:06d}.pt"
        dest_path.unlink(missing_ok=True)
        os.symlink(source_path, dest_path)
        row.pop("_cache_dir", None)
        row["path"] = str(dest_path.relative_to(out_dir))
        row["profile_match_distance"] = float(entry["distance"])
        row["profile_match_features"] = entry["features"]
        row["profile_match_target"] = target
        with metadata_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary_rows.append(row)
    done = {"event": "done", "saved": len(selected), "source_rows": len(candidates), "target": target, "best_distance": float(selected[0]["distance"])}
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(done, sort_keys=True) + "\n")
    print(json.dumps(done, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
