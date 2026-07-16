#!/usr/bin/env python
from __future__ import annotations

import argparse

from runtime.model_catalog import (
    DEFAULT_MODEL_INDEX_PATH,
    diffusers_catalog_adapter_status,
    diffusers_catalog_status,
    load_catalog_diffusers_component,
    load_catalog_diffusers_pipeline,
)
from runtime.diffusers_bridge import DiffusersBridgeOptions, diffusers_component_report


def _parse_max_memory(items: list[str] | None) -> dict[int | str, str] | None:
    if not items:
        return None
    out: dict[int | str, str] = {}
    for item in items:
        key, value = item.split("=", 1)
        out[int(key) if key.isdigit() else key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke Diffusers catalog entries through model-stack.")
    parser.add_argument("model_id")
    parser.add_argument("--index-path", default=DEFAULT_MODEL_INDEX_PATH)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--component", default=None, help="Load only one component, e.g. vae or transformer.")
    parser.add_argument("--adapter-status", action="store_true", help="Validate a LoRA/adapter-style Diffusers snapshot.")
    parser.add_argument("--load-pipeline", action="store_true", help="Attempt full no-generation pipeline construction.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-memory", action="append", help="Placement cap like 0=20GiB or cpu=110GiB; may repeat.")
    parser.add_argument("--skip-component", action="append", default=[], help="Pass a Diffusers component as None during pipeline load; may repeat.")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    if args.adapter_status:
        status = diffusers_catalog_adapter_status(args.model_id, index_path=args.index_path, model_root=args.model_root)
        print(status)
        print(f"complete={status.complete}")
        print(f"weights={status.weight_files}")
        print(f"configs={status.config_files}")
        return

    status = diffusers_catalog_status(args.model_id, index_path=args.index_path, model_root=args.model_root)
    print(status)
    print(f"complete={status.complete}")
    if not args.component and not args.load_pipeline:
        return

    options = DiffusersBridgeOptions(
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        max_memory=_parse_max_memory(args.max_memory),
        validate_snapshot=not args.no_validate,
        channels_last=args.device_map is None,
        skip_components=tuple(args.skip_component),
    )
    if args.component:
        artifacts = load_catalog_diffusers_component(
            args.model_id,
            args.component,
            index_path=args.index_path,
            model_root=args.model_root,
            options=options,
        )
        print(
            f"component={artifacts.name} class={artifacts.class_name} "
            f"params={artifacts.parameter_count} device={artifacts.device} dtype={artifacts.dtype}"
        )
        return

    artifacts = load_catalog_diffusers_pipeline(
        args.model_id,
        index_path=args.index_path,
        model_root=args.model_root,
        options=options,
    )
    print(f"pipeline={type(artifacts.pipeline).__name__} device={artifacts.device} dtype={artifacts.dtype}")
    print(f"optimizations={artifacts.enabled_optimizations}")
    print(f"components={diffusers_component_report(artifacts.pipeline)}")


if __name__ == "__main__":
    main()
