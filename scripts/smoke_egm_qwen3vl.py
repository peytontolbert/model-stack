#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def _parse_max_memory(items: list[str] | None) -> dict[int | str, str] | None:
    if not items:
        return None
    out: dict[int | str, str] = {}
    for item in items:
        key, value = item.split("=", 1)
        out[int(key) if key.isdigit() else key] = value
    return out


def _jsonable_max_memory(max_memory: dict[int | str, str] | None) -> dict[str, str] | None:
    if max_memory is None:
        return None
    return {str(key): value for key, value in max_memory.items()}


def _gpu_memory() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    rows: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        rows.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_bytes": int(total),
                "free_bytes": int(free),
                "used_bytes": int(total - free),
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    return rows


def _first_execution_device(model: Any) -> torch.device:
    hf_map = getattr(model, "hf_device_map", None) or {}
    for device in hf_map.values():
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        text = str(device)
        if text.startswith("cuda"):
            return torch.device(text)
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded EGM/Qwen3-VL placement and text-generation smoke.")
    parser.add_argument("model_path")
    parser.add_argument("--prompt", default="Say OK.")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--max-memory", action="append")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    max_memory = _parse_max_memory(args.max_memory)
    report: dict[str, Any] = {
        "model_path": str(model_path),
        "model_id": model_path.name,
        "env": "ai",
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory": _jsonable_max_memory(max_memory),
        "status": "started",
        "gpu_memory_before": _gpu_memory(),
    }
    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    started = time.monotonic()
    try:
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        report["processor_class"] = type(processor).__name__
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
            device_map=args.device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        model.eval()
        report["load_seconds"] = round(time.monotonic() - started, 3)
        report["model_class"] = type(model).__name__
        report["hf_device_map"] = {str(k): str(v) for k, v in (getattr(model, "hf_device_map", {}) or {}).items()}
        device = _first_execution_device(model)
        encode_started = time.monotonic()
        inputs = processor(text=args.prompt, return_tensors="pt")
        inputs = {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}
        report["encode_seconds"] = round(time.monotonic() - encode_started, 3)
        gen_started = time.monotonic()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        report["generation_seconds"] = round(time.monotonic() - gen_started, 3)
        try:
            decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        except Exception:
            decoded = []
        report["output_shape"] = list(output_ids.shape)
        report["decoded"] = decoded[:2]
        report["status"] = "generation_ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    finally:
        report["total_seconds"] = round(time.monotonic() - started, 3)
        report["gpu_memory_after"] = _gpu_memory()
        report_path = Path(args.report) if args.report else Path("reports/world-model-smokes") / f"{model_path.name}.qwen3vl_full_text_generate.ai.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
