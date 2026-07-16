#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _gpu_memory() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append({"index": int(parts[0]), "name": parts[1], "memory_used_mb": int(parts[2]), "memory_total_mb": int(parts[3])})
    return rows


def _memory_sample() -> dict[str, Any]:
    return {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}


def _resolve_dtype(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name.lower()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Smoke Cosmos Embed1 local world-model embedders.")
    parser.add_argument("model_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--text", action="append", default=["a robot arm picks up a block"])
    parser.add_argument("--video", action="store_true", help="Also run synthetic video embedding.")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    model_path = Path(args.model_path)
    payload: dict[str, Any] = {
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "device": args.device,
        "dtype": args.dtype,
        "status": "started",
        "memory": {"start": _memory_sample()},
    }
    print(f"model_path={model_path}")
    print(f"model_exists={model_path.exists()}")

    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
    except Exception as exc:
        payload["status"] = "import_failed"
        payload["error"] = f"{type(exc).__name__}:{exc}"
        raise

    dtype = _resolve_dtype(args.dtype)
    config_started = time.perf_counter()
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    payload["config_seconds"] = time.perf_counter() - config_started
    payload["config_class"] = type(config).__name__
    payload["model_type"] = getattr(config, "model_type", None)
    payload["architectures"] = getattr(config, "architectures", None)
    print(f"config_class={type(config).__name__}")

    processor_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    payload["processor_seconds"] = time.perf_counter() - processor_started
    payload["processor_class"] = type(processor).__name__
    print(f"processor_class={type(processor).__name__}")

    tokenizer_started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    except Exception as exc:
        payload["tokenizer_seconds"] = time.perf_counter() - tokenizer_started
        payload["tokenizer_status"] = "failed"
        payload["tokenizer_error"] = f"{type(exc).__name__}:{exc}"
        print(f"tokenizer_status=failed")
        print(f"tokenizer_error={payload['tokenizer_error']}")
    else:
        payload["tokenizer_seconds"] = time.perf_counter() - tokenizer_started
        payload["tokenizer_status"] = "ok"
        payload["tokenizer_length"] = len(tokenizer)
        print(f"tokenizer_length={len(tokenizer)}")

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True, torch_dtype=dtype)
    payload["load_seconds"] = time.perf_counter() - load_started
    payload["model_class"] = type(model).__name__
    payload["memory"]["after_load"] = _memory_sample()
    print(f"model_class={type(model).__name__}")
    print(f"load_seconds={payload['load_seconds']:.3f}")

    move_started = time.perf_counter()
    model = model.to(args.device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    payload["move_seconds"] = time.perf_counter() - move_started
    payload["memory"]["after_move"] = _memory_sample()
    print(f"move_seconds={payload['move_seconds']:.3f}")

    with torch.no_grad():
        text_started = time.perf_counter()
        text_inputs = processor(text=args.text).to(args.device, dtype=dtype)
        text_out = model.get_text_embeddings(**text_inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        payload["text_seconds"] = time.perf_counter() - text_started
        payload["text_proj_shape"] = tuple(text_out.text_proj.shape)
        print(f"text_seconds={payload['text_seconds']:.3f}")
        print(f"text_proj_shape={payload['text_proj_shape']}")

        if args.video:
            height = args.height or (448 if "448" in model_path.name else 224)
            width = args.width or height
            video_started = time.perf_counter()
            video = torch.randint(0, 255, size=(1, args.frames, 3, height, width), device="cpu")
            video_inputs = processor(videos=video).to(args.device, dtype=dtype)
            video_out = model.get_video_embeddings(**video_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            payload["video_seconds"] = time.perf_counter() - video_started
            payload["video_proj_shape"] = tuple(video_out.visual_proj.shape)
            print(f"video_seconds={payload['video_seconds']:.3f}")
            print(f"video_proj_shape={payload['video_proj_shape']}")

    payload["status"] = "ok"
    payload["total_seconds"] = time.perf_counter() - started
    payload["memory"]["end"] = _memory_sample()
    print(f"status={payload['status']}")
    print(f"total_seconds={payload['total_seconds']:.3f}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"json_out={out}")


if __name__ == "__main__":
    main()
