#!/usr/bin/env python3
"""Smoke a local Transformers encoder/classifier snapshot for model-stack."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"}
    for name in ("torch", "transformers"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[name] = "missing"
    return versions


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _gpu_memory() -> list[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        rows = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            rows.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_mb": int(total / 1024 / 1024),
                    "memory_used_mb": int((total - free) / 1024 / 1024),
                }
            )
        return rows
    except Exception:
        return []


def _infer_classifier_num_labels(model_path: Path) -> int | None:
    try:
        from safetensors.torch import safe_open

        safetensors_path = model_path / "model.safetensors"
        if not safetensors_path.exists():
            return None
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            for key in ("classifier.weight", "score.weight"):
                if key in handle.keys():
                    return int(handle.get_tensor(key).shape[0])
    except Exception:
        return None
    return None


def _snap() -> dict[str, Any]:
    return {"rss_mb": _rss_mb(), "gpus": _gpu_memory()}


def _shape(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return list(shape)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--text", default="model-stack encoder classifier bridge smoke")
    parser.add_argument("--json-out")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    started = time.perf_counter()
    model_path = Path(args.model_path)
    result: dict[str, Any] = {
        "model_id": args.model_id or model_path.name,
        "model_path": str(model_path),
        "installed_versions": _versions(),
        "memory": {"start": _snap()},
    }

    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
        t0 = time.perf_counter()
        config = AutoConfig.from_pretrained(model_path, local_files_only=args.local_files_only, trust_remote_code=True)
        result["config_seconds"] = time.perf_counter() - t0
        result["config"] = {
            "class": type(config).__name__,
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_labels": getattr(config, "num_labels", None),
        }

        architectures = tuple(getattr(config, "architectures", None) or ())
        is_classifier = any(str(a).endswith("ForSequenceClassification") for a in architectures)
        inferred_num_labels = _infer_classifier_num_labels(model_path) if is_classifier else None
        if inferred_num_labels and inferred_num_labels != getattr(config, "num_labels", None):
            result.setdefault("compatibility_patches_applied", []).append("transformers_classifier_head_num_labels_from_checkpoint")
            result["config_num_labels_original"] = getattr(config, "num_labels", None)
            config.num_labels = inferred_num_labels
            config.id2label = {i: f"LABEL_{i}" for i in range(inferred_num_labels)}
            config.label2id = {v: k for k, v in config.id2label.items()}
            result["config"]["num_labels"] = inferred_num_labels

        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=args.local_files_only, trust_remote_code=True)
        result["tokenizer_seconds"] = time.perf_counter() - t0
        result["tokenizer_class"] = type(tokenizer).__name__

        loader_name = "AutoModelForSequenceClassification" if is_classifier else "AutoModel"
        loader = AutoModelForSequenceClassification if is_classifier else AutoModel

        t0 = time.perf_counter()
        model = loader.from_pretrained(
            model_path,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
            torch_dtype=dtype,
            config=config,
        )
        model.eval()
        model.to(args.device)
        result["load_seconds"] = time.perf_counter() - t0
        result["memory"]["after_load"] = _snap()
        result["model"] = {
            "class": type(model).__name__,
            "loader": loader_name,
            "first_parameter_device": str(next(model.parameters()).device),
            "first_parameter_dtype": str(next(model.parameters()).dtype),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }

        encoded = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=64)
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(**encoded)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["forward_seconds"] = time.perf_counter() - t0
        result["memory"]["after_forward"] = _snap()
        result["output"] = {
            "logits_shape": _shape(getattr(output, "logits", None)),
            "last_hidden_state_shape": _shape(getattr(output, "last_hidden_state", None)),
            "pooler_output_shape": _shape(getattr(output, "pooler_output", None)),
        }
        result["status"] = "ok"
        result["total_seconds"] = time.perf_counter() - started
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}:{exc}"
        result["total_seconds"] = time.perf_counter() - started
        result["memory"]["end"] = _snap()
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1

    result["memory"]["end"] = _snap()
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
