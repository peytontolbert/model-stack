#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_PATH = _REPO_ROOT / "scripts" / "smoke_nemo_asr_bridge.py"
_SPEC = importlib.util.spec_from_file_location("smoke_nemo_asr_bridge", _SMOKE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smoke_nemo_asr_bridge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_nemo_asr_bridge)


def _rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    # ru_maxrss is KiB on Linux, bytes on macOS. This repo target is Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _gpu_memory() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
            }
        )
    return rows


def _gpu_process_memory(pid: int | None = None) -> list[dict[str, Any]]:
    selected_pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            row_pid = int(parts[1])
        except ValueError:
            continue
        if row_pid != selected_pid:
            continue
        used = parts[3].replace(" MiB", "").strip()
        try:
            used_mb: int | str = int(used)
        except ValueError:
            used_mb = used
        rows.append({"gpu_uuid": parts[0], "pid": row_pid, "process_name": parts[2], "used_memory_mb": used_mb})
    return rows


def _memory_sample() -> dict[str, Any]:
    return {"rss_mb": _rss_mb(), "gpus": _gpu_memory(), "process_gpus": _gpu_process_memory()}


def _torch_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _output_text(item: Any) -> str:
    return str(getattr(item, "text", item))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Benchmark warm NeMo ASR inference latency and memory.")
    parser.add_argument("model_id", nargs="?")
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--audio", required=True, help="Audio file passed to model.transcribe.")
    parser.add_argument("--index-path", default=smoke_nemo_asr_bridge.DEFAULT_MODEL_INDEX_PATH)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--restore-map-location", default="cpu")
    parser.add_argument("--device", default="cuda:0", help="Device to move the restored model to; use cpu to keep CPU inference.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--force-restore", action="store_true")
    args = parser.parse_args()

    target_args = argparse.Namespace(
        archive_path=args.archive_path,
        model_id=args.model_id,
        index_path=args.index_path,
        model_root=args.model_root,
    )
    model_id, lane, backend, model_path, nemo_path = smoke_nemo_asr_bridge._resolve_target(target_args)
    versions = smoke_nemo_asr_bridge._installed_versions()
    env_ok, env_problems = smoke_nemo_asr_bridge._env_status(versions)
    target = smoke_nemo_asr_bridge._nemo_model_target(nemo_path) if nemo_path else None
    target_importable, target_import_detail = smoke_nemo_asr_bridge._target_import_status(target)

    payload: dict[str, Any] = {
        "model_id": model_id,
        "lane": lane,
        "backend": backend,
        "model_path": str(model_path),
        "nemo_archive": str(nemo_path) if nemo_path else None,
        "nemo_model_target": target,
        "nemo_model_target_importable": target_importable,
        "nemo_model_target_import_detail": target_import_detail,
        "versions": versions,
        "env_ok": env_ok,
        "env_problems": env_problems,
        "restore_map_location": args.restore_map_location,
        "device": args.device,
        "audio": str(Path(args.audio)),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "status": "started",
        "memory": {"start": _memory_sample()},
    }

    print(f"model_id={model_id}")
    print(f"nemo_archive={payload['nemo_archive']}")
    print(f"nemo_model_target={target}")
    print(f"nemo_model_target_importable={target_importable}")
    print(f"device={args.device}")

    if nemo_path is None:
        payload["status"] = "missing_nemo_archive"
        raise FileNotFoundError(f"No .nemo archive found under {model_path}")
    if not env_ok:
        payload["status"] = "env_not_ready"
        raise RuntimeError(f"NeMo ASR environment is not ready: {env_problems}")
    if target_importable is False and not args.force_restore:
        payload["status"] = "needs_nemo_model_specific_env"
        payload["error"] = f"target_not_importable:{target}:{target_import_detail}"
        print(f"status={payload['status']}")
        print(f"error={payload['error']}")
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        raise ImportError(payload["error"])

    restore_started = time.perf_counter()
    model = smoke_nemo_asr_bridge._restore_model(nemo_path, args.restore_map_location)
    restore_seconds = time.perf_counter() - restore_started
    payload["restore_seconds"] = restore_seconds
    payload["restored_class"] = type(model).__name__
    payload["memory"]["after_restore"] = _memory_sample()
    print(f"restore_seconds={restore_seconds:.3f}")
    print(f"restored_class={type(model).__name__}")

    move_seconds = 0.0
    if args.device and args.device != "none":
        move_started = time.perf_counter()
        if args.device != "cpu":
            model = model.to(args.device)
        else:
            model = model.cpu()
        try:
            model.eval()
        except Exception:
            pass
        _torch_sync()
        move_seconds = time.perf_counter() - move_started
    payload["move_seconds"] = move_seconds
    payload["memory"]["after_move"] = _memory_sample()
    print(f"move_seconds={move_seconds:.3f}")

    audio_files = [str(Path(args.audio))]
    warmup_latencies: list[float] = []
    latencies: list[float] = []
    outputs: list[str] = []

    for idx in range(args.warmup):
        _torch_sync()
        started = time.perf_counter()
        result = model.transcribe(audio_files)
        _torch_sync()
        elapsed = time.perf_counter() - started
        warmup_latencies.append(elapsed)
        print(f"warmup_{idx}_seconds={elapsed:.3f}")

    payload["memory"]["after_warmup"] = _memory_sample()

    for idx in range(args.repeats):
        _torch_sync()
        started = time.perf_counter()
        result = model.transcribe(audio_files)
        _torch_sync()
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        outputs = [_output_text(item) for item in result]
        print(f"repeat_{idx}_seconds={elapsed:.3f}")

    payload["status"] = "ok"
    payload["warmup_latencies_seconds"] = warmup_latencies
    payload["latencies_seconds"] = latencies
    payload["latency_summary_seconds"] = _summarize(latencies)
    payload["outputs"] = outputs
    payload["memory"]["end"] = _memory_sample()
    print(f"latency_mean_seconds={payload['latency_summary_seconds']['mean']}")
    print(f"latency_median_seconds={payload['latency_summary_seconds']['median']}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"json_out={out_path}")


if __name__ == "__main__":
    main()
