#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded FLUX.2 full pipeline placement/generation smoke.")
    parser.add_argument("model_path")
    parser.add_argument("--prompt", default="a small red cube on a white table")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--max-memory", action="append")
    parser.add_argument("--output-type", default="latent")
    parser.add_argument("--skip-component", action="append", default=[], help="Load this Diffusers component as None; may repeat.")
    parser.add_argument("--dummy-prompt-embeds", action="store_true", help="Bypass text encoder with zero prompt embeddings sized from transformer config.")
    parser.add_argument("--load-only", action="store_true", help="Only construct the pipeline and report placement; do not generate.")
    parser.add_argument("--prompt-embeds-path", default=None, help="Load prompt embeddings from a torch .pt file and pass prompt=None.")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    report: dict[str, Any] = {
        "model_path": str(model_path),
        "model_id": model_path.name,
        "env": "ai",
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory": _jsonable_max_memory(_parse_max_memory(args.max_memory)),
        "output_type": args.output_type,
        "skip_components": list(args.skip_component),
        "dummy_prompt_embeds": bool(args.dummy_prompt_embeds),
        "prompt_embeds_path": args.prompt_embeds_path,
        "max_sequence_length": args.max_sequence_length,
        "status": "started",
        "gpu_memory_before": _gpu_memory(),
    }
    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
    started = time.monotonic()
    try:
        load_kwargs = {name: None for name in args.skip_component}
        pipe = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
            device_map=args.device_map,
            max_memory=_parse_max_memory(args.max_memory),
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        report["load_seconds"] = round(time.monotonic() - started, 3)
        report["pipeline_class"] = type(pipe).__name__
        report["components"] = {
            name: type(getattr(pipe, name)).__name__
            for name in getattr(pipe, "components", {})
            if hasattr(pipe, name)
        }
        report["hf_device_map"] = {}
        for component_name in ("transformer", "text_encoder", "vae"):
            component = getattr(pipe, component_name, None)
            device_map = getattr(component, "hf_device_map", None)
            if device_map:
                report["hf_device_map"][component_name] = {str(k): str(v) for k, v in device_map.items()}
        if args.load_only:
            report["status"] = "load_ok"
            return
        prompt = args.prompt
        prompt_embeds = None
        if args.prompt_embeds_path:
            prompt_embeds = torch.load(args.prompt_embeds_path, map_location="cpu")
            if isinstance(prompt_embeds, dict):
                prompt_embeds = prompt_embeds.get("prompt_embeds")
            if not isinstance(prompt_embeds, torch.Tensor):
                raise TypeError("prompt embeds file must contain a Tensor or {'prompt_embeds': Tensor}")
            prompt = None
        elif args.dummy_prompt_embeds:
            joint_attention_dim = int(getattr(pipe.transformer.config, "joint_attention_dim"))
            prompt_embeds = torch.zeros((1, int(args.max_sequence_length), joint_attention_dim), dtype=dtype)
            prompt = None
        if prompt_embeds is not None:
            device = getattr(pipe, "_execution_device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            report["prompt_embeds"] = {"shape": list(prompt_embeds.shape), "dtype": str(prompt_embeds.dtype), "device": str(prompt_embeds.device)}

        gen_started = time.monotonic()
        generator = torch.Generator(device="cpu").manual_seed(1234)
        result = pipe(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type=args.output_type,
            max_sequence_length=args.max_sequence_length,
        )
        report["generation_seconds"] = round(time.monotonic() - gen_started, 3)
        images = getattr(result, "images", result[0] if isinstance(result, tuple) and result else None)
        if isinstance(images, torch.Tensor):
            report["output"] = {"type": "tensor", "shape": list(images.shape), "dtype": str(images.dtype), "device": str(images.device)}
        elif isinstance(images, list):
            report["output"] = {"type": "list", "length": len(images), "first_type": type(images[0]).__name__ if images else None}
        else:
            report["output"] = {"type": type(images).__name__}
        report["status"] = "generation_ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    finally:
        report["total_seconds"] = round(time.monotonic() - started, 3)
        report["gpu_memory_after"] = _gpu_memory()
        report_path = Path(args.report) if args.report else Path("reports/world-model-smokes") / f"{model_path.name}.flux2_full_{args.width}x{args.height}_{args.steps}step.{args.output_type}.ai.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
