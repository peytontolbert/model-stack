#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from runtime.cosmos3_omni_diffusers_pipeline import (
    load_cosmos3_upstream_diffusers_pipeline,
    patch_diffusers_cosmos3,
    prepare_cosmos3_upstream_diffusers_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke Cosmos3 through upstream Diffusers Cosmos3OmniPipeline.")
    parser.add_argument("model_path")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--load", action="store_true", help="Run DiffusionPipeline.from_pretrained after preparing metadata.")
    parser.add_argument("--skip-transformer", action="store_true", help="Pass transformer=None for a bounded construction probe.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    started = time.monotonic()
    cls = patch_diffusers_cosmos3()
    adapter = prepare_cosmos3_upstream_diffusers_snapshot(args.model_path, args.adapter_dir)
    model_index = json.loads((adapter / "model_index.json").read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "model_path": str(Path(args.model_path)),
        "adapter_dir": str(adapter),
        "pipeline_class": f"{cls.__module__}.{cls.__name__}",
        "model_index_class": model_index.get("_class_name"),
        "original_class": model_index.get("_model_stack_original_class_name"),
        "removed_components": model_index.get("_model_stack_removed_components", []),
        "loaded": False,
    }

    if args.load:
        kwargs = {
            "local_files_only": args.local_files_only,
            "safety_checker": None,
            "enable_safety_checker": False,
            "low_cpu_mem_usage": True,
        }
        if args.skip_transformer:
            kwargs["transformer"] = None
        pipe = load_cosmos3_upstream_diffusers_pipeline(args.model_path, adapter_dir=adapter, **kwargs)
        payload.update(
            {
                "loaded": True,
                "loaded_pipeline_class": f"{type(pipe).__module__}.{type(pipe).__name__}",
                "transformer_loaded": pipe.transformer is not None,
                "vae_loaded": pipe.vae is not None,
                "sound_tokenizer_loaded": pipe.sound_tokenizer is not None,
                "text_tokenizer_class": type(pipe.text_tokenizer).__name__,
            }
        )
    payload["elapsed_sec"] = round(time.monotonic() - started, 3)

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
