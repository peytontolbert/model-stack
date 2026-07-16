#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from runtime.compatibility import compatibility_report, report_as_dict


def _gpu_memory() -> list[dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                }
            )
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


def _torch_dtype(name: str | None) -> Any:
    import torch

    if name is None or name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]


def _jsonable_config(config: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", []) or []),
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
        "auto_map": getattr(config, "auto_map", None),
    }
    for attr in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size"):
        value = getattr(config, attr, None)
        if value is not None:
            payload[attr] = int(value)
    return payload


def _model_stats(model: Any) -> dict[str, Any]:
    try:
        params = list(model.parameters())
    except Exception:
        params = []
    trainable = 0
    total = 0
    first_device = None
    first_dtype = None
    for param in params:
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        if first_device is None:
            first_device = str(param.device)
            first_dtype = str(param.dtype)
    return {
        "class": type(model).__name__,
        "parameters": total,
        "trainable_parameters": trainable,
        "first_parameter_device": first_device,
        "first_parameter_dtype": first_dtype,
    }


def _load_tokenizer(model_path: Path, *, trust_remote_code: bool, local_files_only: bool) -> tuple[Any | None, dict[str, Any]]:
    from transformers import AutoTokenizer, LlamaTokenizer

    attempts: list[dict[str, Any]] = []
    for loader_name, loader, kwargs in (
        ("AutoTokenizer", AutoTokenizer.from_pretrained, {"use_fast": True}),
        ("LlamaTokenizer", LlamaTokenizer.from_pretrained, {}),
    ):
        try:
            tokenizer = loader(
                str(model_path),
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                **kwargs,
            )
        except Exception as exc:
            attempts.append({"loader": loader_name, "status": "failed", "error": f"{type(exc).__name__}:{exc}"})
            continue
        if callable(tokenizer) and hasattr(tokenizer, "decode"):
            attempts.append({"loader": loader_name, "status": "ok", "class": type(tokenizer).__name__})
            return tokenizer, {
                "tokenizer_status": "ok",
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_loader": loader_name,
                "tokenizer_attempts": attempts,
                "compatibility_patches_applied": ["transformers_mobilellm_slow_tokenizer"] if loader_name == "LlamaTokenizer" else [],
            }
        attempts.append({"loader": loader_name, "status": "invalid_return", "class": type(tokenizer).__name__, "repr": repr(tokenizer)})
    return None, {"tokenizer_status": "failed", "tokenizer_attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke a local Transformers causal LM bridge snapshot.")
    parser.add_argument("model_path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--load", action="store_true", help="Attempt AutoModelForCausalLM.from_pretrained.")
    parser.add_argument("--generate", action="store_true", help="Run a minimal model.generate call after loading.")
    parser.add_argument("--device", default=None, help="Device for loaded model, e.g. cuda:0 or cpu.")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float16, or float32.")
    parser.add_argument("--prompt", default="Hello from model-stack")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    model_path = Path(args.model_path)
    payload: dict[str, Any] = {
        "model_id": args.model_id or model_path.name,
        "model_path": str(model_path),
        "load_requested": bool(args.load or args.generate),
        "generate_requested": bool(args.generate),
        "use_cache": bool(args.use_cache),
        "status": "started",
        "memory": {"start": {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}},
    }
    payload["compatibility"] = report_as_dict(compatibility_report(model_path, model_id=str(payload["model_id"])))

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:
        payload["status"] = "failed_imports"
        payload["error"] = f"{type(exc).__name__}:{exc}"
        payload["total_seconds"] = time.perf_counter() - started
        payload["memory"]["end"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
        _write_json(args.json_out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1)

    config_started = time.perf_counter()
    try:
        config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        payload["status"] = "failed_config"
        payload["config_seconds"] = time.perf_counter() - config_started
        payload["error"] = f"{type(exc).__name__}:{exc}"
        payload["total_seconds"] = time.perf_counter() - started
        payload["memory"]["end"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
        _write_json(args.json_out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1)

    payload["config"] = _jsonable_config(config)
    payload["config_seconds"] = time.perf_counter() - config_started

    tokenizer_started = time.perf_counter()
    tokenizer, tokenizer_payload = _load_tokenizer(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    payload.update(tokenizer_payload)
    payload["tokenizer_seconds"] = time.perf_counter() - tokenizer_started

    model = None
    if args.load or args.generate:
        load_started = time.perf_counter()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                config=config,
                dtype=_torch_dtype(args.dtype),
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
            )
            if args.device:
                model = model.to(args.device)
            model.eval()
        except Exception as exc:
            payload["status"] = "failed_load"
            payload["load_seconds"] = time.perf_counter() - load_started
            payload["error"] = f"{type(exc).__name__}:{exc}"
            payload["memory"]["after_load_failure"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
            payload["total_seconds"] = time.perf_counter() - started
            payload["memory"]["end"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
            _write_json(args.json_out, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            raise SystemExit(1)
        payload["load_seconds"] = time.perf_counter() - load_started
        payload["model"] = _model_stats(model)
        payload["memory"]["after_load"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}

    if args.generate:
        if tokenizer is None:
            payload["status"] = "failed_generate"
            payload["error"] = "tokenizer unavailable; cannot encode prompt"
            payload["total_seconds"] = time.perf_counter() - started
            payload["memory"]["end"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
            _write_json(args.json_out, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            raise SystemExit(1)
        generate_started = time.perf_counter()
        if not args.use_cache:
            payload.setdefault("compatibility_patches_applied", [])
            if any(patch["id"] == "transformers_mobilellm_legacy_cache" for patch in payload["compatibility"].get("patches", [])):
                if "transformers_mobilellm_legacy_cache" not in payload["compatibility_patches_applied"]:
                    payload["compatibility_patches_applied"].append("transformers_mobilellm_legacy_cache")
        try:
            inputs = tokenizer(args.prompt, return_tensors="pt")
            if args.device:
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    use_cache=args.use_cache,
                )
            payload["generated_text"] = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            payload["generated_tokens"] = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
            payload["generate_status"] = "ok"
        except Exception as exc:
            payload["status"] = "failed_generate"
            payload["generate_seconds"] = time.perf_counter() - generate_started
            payload["error"] = f"{type(exc).__name__}:{exc}"
            payload["memory"]["after_generate_failure"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
            payload["total_seconds"] = time.perf_counter() - started
            payload["memory"]["end"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
            _write_json(args.json_out, payload)
            print(json.dumps(payload, indent=2, sort_keys=True))
            raise SystemExit(1)
        payload["generate_seconds"] = time.perf_counter() - generate_started
        payload["memory"]["after_generate"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}

    payload["status"] = "ok"
    payload["total_seconds"] = time.perf_counter() - started
    payload["memory"]["end"] = {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}
    _write_json(args.json_out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
