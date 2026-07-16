from __future__ import annotations

import argparse
from pathlib import Path

from runtime.hunyuan3d_lightx2v_bridge import (
    Hunyuan3DLightX2VPaths,
    probe_hunyuan3d_lightx2v_runtime,
    probe_to_json,
    status_to_json,
    hunyuan3d_lightx2v_status,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke Hunyuan3D LightX2V bridge assets/imports.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--lightx2v-root", type=Path, default=Path("/data/clone/third_party/LightX2V"))
    parser.add_argument("--imports", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    paths = Hunyuan3DLightX2VPaths(model_path=args.model_path, lightx2v_root=args.lightx2v_root)
    if args.imports:
        result = probe_hunyuan3d_lightx2v_runtime(paths, model_id=args.model_id)
        payload = probe_to_json(result)
        ok = result.status.runnable and all(result.imports.values())
    else:
        result = hunyuan3d_lightx2v_status(paths, model_id=args.model_id)
        payload = status_to_json(result)
        ok = result.runnable

    print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
