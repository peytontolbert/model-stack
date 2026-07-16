#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from runtime.world_model_bridge import WorldModelBridgeOptions, WorldModelBridgeUnsupported, load_world_model, world_model_status


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


def _rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke local world/video model bridge status and optional no-generation load.")
    parser.add_argument("model_path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--load", action="store_true", help="Attempt no-generation bridge load.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    status = world_model_status(args.model_path, model_id=args.model_id)
    payload: dict[str, Any] = {
        "status": asdict(status),
        "load_requested": bool(args.load),
        "load_status": "not_requested",
        "memory": {"start": {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}},
    }
    print(json.dumps(payload["status"], indent=2, sort_keys=True))

    if args.load:
        load_started = time.perf_counter()
        try:
            artifacts = load_world_model(
                args.model_path,
                model_id=args.model_id,
                options=WorldModelBridgeOptions(device=args.device, dtype=args.dtype or status.recommended_dtype),
            )
        except WorldModelBridgeUnsupported as exc:
            payload["load_status"] = "unsupported"
            payload["load_seconds"] = time.perf_counter() - load_started
            payload["error"] = str(exc)
            payload["unsupported_status"] = asdict(exc.status)
            print(f"load_status=unsupported error={exc}")
        except Exception as exc:
            payload["load_status"] = "failed"
            payload["load_seconds"] = time.perf_counter() - load_started
            payload["error"] = f"{type(exc).__name__}:{exc}"
            print(f"load_status=failed error={payload['error']}")
        else:
            payload["load_status"] = "ok"
            payload["load_seconds"] = time.perf_counter() - load_started
            payload["artifact_class"] = type(artifacts).__name__
            pipeline = getattr(artifacts, "pipeline", None)
            if pipeline is not None:
                payload["pipeline_class"] = type(pipeline).__name__
                payload["enabled_optimizations"] = tuple(getattr(artifacts, "enabled_optimizations", ()))
            model = getattr(artifacts, "model", None)
            if model is not None:
                payload["model_class"] = type(model).__name__
            print(f"load_status=ok load_seconds={payload['load_seconds']:.3f}")

    payload["total_seconds"] = time.perf_counter() - started
    payload["memory"]["end"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
    _write_json(args.json_out, payload)
    if args.json_out:
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
