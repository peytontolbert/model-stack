from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from runtime.cosmos25_bridge import (
    build_cosmos25_predict_launch_plan,
    build_cosmos25_transfer_launch_plan,
    cosmos25_status,
)


PREDICT_ROOT = Path("/data/clone/third_party/cosmos-predict2.5")
TRANSFER_ROOT = Path("/data/clone/third_party/cosmos-transfer2.5")


def main() -> int:
    parser = argparse.ArgumentParser(description="Status/import smoke for local Cosmos 2.5 runtimes.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--print-launch", action="store_true")
    args = parser.parse_args()

    model_id = args.model_id or args.model_path.name
    runtime_root = args.runtime_root or _default_runtime_root(model_id)
    _register_runtime(runtime_root)

    modules = _modules_for_model(model_id)
    imports: dict[str, bool] = {}
    errors: dict[str, str] = {}
    timings: dict[str, float] = {}
    for module_name in modules:
        started = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
            imports[module_name] = True
            errors[module_name] = ""
            timings[module_name] = time.perf_counter() - started
            print(f"OK {module_name} {getattr(module, '__file__', '')}")
        except Exception as exc:
            imports[module_name] = False
            errors[module_name] = f"{type(exc).__name__}:{exc}"
            timings[module_name] = time.perf_counter() - started
            print(f"FAIL {module_name} {type(exc).__name__}:{exc}")
            traceback.print_exc(limit=4)

    report: dict[str, Any] = {
        "model_id": model_id,
        "model_path": str(args.model_path),
        "runtime_root": str(runtime_root),
        "status": asdict(cosmos25_status(args.model_path, model_id=model_id)),
        "python": sys.version,
        "sys_path_prefix": sys.path[:6],
        "imports": imports,
        "errors": errors,
        "timings_sec": timings,
    }
    if args.print_launch:
        if "transfer" in model_id.lower():
            plan = build_cosmos25_transfer_launch_plan(args.model_path, runtime_root=runtime_root)
        else:
            plan = build_cosmos25_predict_launch_plan(args.model_path, runtime_root=runtime_root)
        report["launch_plan"] = asdict(plan)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if all(imports.values()) else 2


def _default_runtime_root(model_id: str) -> Path:
    if "transfer" in model_id.lower():
        return TRANSFER_ROOT
    return PREDICT_ROOT


def _register_runtime(runtime_root: Path) -> None:
    for path in (
        runtime_root,
        runtime_root / "packages" / "cosmos-oss",
        runtime_root / "packages" / "cosmos-cuda",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _modules_for_model(model_id: str) -> tuple[str, ...]:
    if "transfer" in model_id.lower():
        return (
            "cosmos_cuda",
            "cosmos_oss",
            "cosmos_transfer2",
            "cosmos_transfer2.config",
            "cosmos_transfer2.inference",
        )
    return (
        "cosmos_cuda",
        "cosmos_oss",
        "cosmos_predict2",
        "cosmos_predict2.config",
        "cosmos_predict2.inference",
    )


if __name__ == "__main__":
    raise SystemExit(main())
