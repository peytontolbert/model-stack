#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict
from pathlib import Path

from runtime.lingbot_world_bridge import (
    LingBotWorldPaths,
    build_lingbot_world_launch_plan,
    lingbot_world_status,
    probe_lingbot_world_runtime,
    write_lingbot_world_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke LingBot World LightX2V bridge status/imports and launch config.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--lightx2v-root", type=Path, default=Path("/data/clone/third_party/LightX2V"))
    parser.add_argument("--t5-checkpoint", type=Path, default=Path("/arxiv/models/models_t5_umt5-xxl-enc-bf16.pth"))
    parser.add_argument("--probe-imports", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--print-launch", action="store_true")
    parser.add_argument("--prompt", default="A calm lake with mountains and drifting clouds.")
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--action-path", default=None)
    parser.add_argument("--save-result-path", default=None)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--target-width", type=int, default=448)
    parser.add_argument("--target-video-length", type=int, default=13)
    parser.add_argument("--infer-steps", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, default=Path("/data/tmp/model-stack-smokes/lingbot-world"))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    paths = LingBotWorldPaths(model_path=args.model_path, lightx2v_root=args.lightx2v_root, t5_checkpoint=args.t5_checkpoint)
    model_id = args.model_id or (f"robbyant/{args.model_path.name}" if args.model_path.parent.name == "robbyant" else args.model_path.name)
    payload: dict[str, object] = {"status": asdict(lingbot_world_status(paths, model_id=model_id))}

    if args.probe_imports:
        payload["runtime_probe"] = asdict(probe_lingbot_world_runtime(paths, model_id=model_id))

    config_path = None
    if args.write_config or args.print_launch:
        safe_name = model_id.replace("/", "--")
        config_path = args.work_dir / safe_name / "lingbot_world_fast_bounded.json"
        write_lingbot_world_config(
            paths,
            config_path,
            target_height=args.target_height,
            target_width=args.target_width,
            target_video_length=args.target_video_length,
            infer_steps=args.infer_steps,
        )
        payload["bounded_config_path"] = str(config_path)

    if args.print_launch:
        image_path = args.image_path or str(args.model_path / "assets" / "teaser.png")
        plan = build_lingbot_world_launch_plan(
            paths,
            config_json=config_path,
            prompt=args.prompt,
            image_path=image_path,
            action_path=args.action_path,
            save_result_path=args.save_result_path or (args.work_dir / model_id.replace("/", "--") / "lingbot_world_fast_smoke.mp4"),
        )
        payload["launch_plan"] = asdict(plan)
        env = " ".join(f"{key}={shlex.quote(value)}" for key, value in plan.env.items())
        payload["launch_shell"] = f"cd {shlex.quote(plan.cwd)} && {env} {' '.join(shlex.quote(part) for part in plan.command)}"

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
