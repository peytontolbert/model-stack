#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict
from pathlib import Path

from runtime.hunyuan_avatar_bridge import (
    HunyuanAvatarPaths,
    build_hunyuan_avatar_launch_plan,
    hunyuan_avatar_status,
    probe_hunyuan_avatar_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke HunyuanVideo-Avatar model-stack bridge.")
    parser.add_argument("--model-path", default="/arxiv/models/HunyuanVideo-Avatar")
    parser.add_argument("--avatar-root", default="/data/clone/hunyuanvideo-avatar")
    parser.add_argument("--bf16-shard-dir", default="/data/transformer_10/checkpoints/hunyuan_avatar_bf16_fsdp2")
    parser.add_argument("--fp8-shard-dir", default="/data/transformer_10/checkpoints/hunyuan_avatar_fp8_fsdp2")
    parser.add_argument("--probe-imports", action="store_true")
    parser.add_argument("--print-launch", action="store_true")
    parser.add_argument("--mode", choices=("fp8_fsdp2", "bf16_fsdp"), default="fp8_fsdp2")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    paths = HunyuanAvatarPaths(
        model_path=Path(args.model_path),
        avatar_root=Path(args.avatar_root),
        bf16_shard_dir=Path(args.bf16_shard_dir),
        fp8_shard_dir=Path(args.fp8_shard_dir),
    )
    payload: dict[str, object] = {"status": asdict(hunyuan_avatar_status(paths))}
    if args.probe_imports:
        payload["runtime_probe"] = asdict(probe_hunyuan_avatar_runtime(paths))
    if args.print_launch:
        plan = build_hunyuan_avatar_launch_plan(paths=paths, mode=args.mode)
        payload["launch_plan"] = asdict(plan)
        env = " ".join(f"{key}={shlex.quote(value)}" for key, value in plan.env.items())
        payload["launch_shell"] = f"cd {shlex.quote(plan.cwd)} && {env} {' '.join(shlex.quote(part) for part in plan.command)}"
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
