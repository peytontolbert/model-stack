#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from runtime.cosmos3_lightx2v_bridge import (
    Cosmos3LightX2VPaths,
    build_cosmos3_lightx2v_launch_plan,
    cosmos3_lightx2v_status,
    probe_cosmos3_lightx2v_runtime,
)


def _tail(text: str, *, lines: int = 80) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def _make_bounded_config(
    paths: Cosmos3LightX2VPaths,
    *,
    base_config_name: str,
    output_dir: Path,
    infer_steps: int,
    target_height: int,
    target_width: int,
    target_video_length: int,
    cpu_offload: bool,
    vae_cpu_offload: bool,
    self_attn_type: str | None,
    causal_self_attn_type: str | None,
) -> Path:
    base = paths.lightx2v_root / "configs" / "cosmos3" / base_config_name
    data = json.loads(base.read_text(encoding="utf-8"))
    data.update(
        {
            "infer_steps": infer_steps,
            "target_height": target_height,
            "target_width": target_width,
            "target_video_length": target_video_length,
            "target_fps": 24.0,
            "feature_caching": "NoCaching",
            "cpu_offload": cpu_offload,
            "vae_cpu_offload": vae_cpu_offload,
            "offload_granularity": "block",
        }
    )
    if self_attn_type:
        data["self_attn_type"] = self_attn_type
    if causal_self_attn_type:
        data["causal_self_attn_type"] = causal_self_attn_type
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"bounded_{base_config_name}"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _execute_plan(plan, *, timeout_sec: int) -> dict[str, object]:
    env = os.environ.copy()
    env.update(plan.env)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            plan.command,
            cwd=plan.cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        elapsed = time.monotonic() - started
        return {
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "returncode": None,
            "elapsed_sec": round(elapsed, 3),
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
            "timed_out": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke Cosmos3 LightX2V bridge.")
    parser.add_argument("model_path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--lightx2v-root", default="/data/clone/third_party/LightX2V")
    parser.add_argument("--probe-imports", action="store_true")
    parser.add_argument("--print-launch", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task", default="t2i")
    parser.add_argument("--config-name", default="cosmos3_super_t2i.json")
    parser.add_argument("--prompt", default="A small red cube on a table.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--infer-steps", type=int, default=1)
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--target-width", type=int, default=256)
    parser.add_argument("--target-video-length", type=int, default=1)
    parser.add_argument("--self-attn-type", default=None)
    parser.add_argument("--causal-self-attn-type", default=None)
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--no-vae-cpu-offload", action="store_true")
    parser.add_argument("--lazy-artifact-dir", default=None)
    parser.add_argument("--use-lazy-wrapper", action="store_true")
    parser.add_argument("--save-result-path", default=None)
    parser.add_argument("--work-dir", default="/data/tmp/model-stack-smokes/cosmos3")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    paths = Cosmos3LightX2VPaths(model_path=Path(args.model_path), lightx2v_root=Path(args.lightx2v_root))
    model_id = args.model_id or Path(args.model_path).name
    payload: dict[str, object] = {"status": asdict(cosmos3_lightx2v_status(paths, model_id=model_id))}
    if args.probe_imports:
        payload["runtime_probe"] = asdict(probe_cosmos3_lightx2v_runtime(paths, model_id=model_id))

    save_result_path = args.save_result_path
    bounded_config_path = None
    if args.execute:
        work_dir = Path(args.work_dir) / model_id.replace("/", "--")
        bounded_config_path = _make_bounded_config(
            paths,
            base_config_name=args.config_name,
            output_dir=work_dir,
            infer_steps=args.infer_steps,
            target_height=args.target_height,
            target_width=args.target_width,
            target_video_length=args.target_video_length,
            cpu_offload=not args.no_cpu_offload,
            vae_cpu_offload=not args.no_vae_cpu_offload,
            self_attn_type=args.self_attn_type,
            causal_self_attn_type=args.causal_self_attn_type,
        )
        if args.lazy_artifact_dir:
            config_data = json.loads(bounded_config_path.read_text(encoding="utf-8"))
            config_data["dit_original_ckpt"] = args.lazy_artifact_dir
            config_data["lazy_load"] = True
            config_data["num_disk_workers"] = 2
            bounded_config_path.write_text(json.dumps(config_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        suffix = "mp4" if args.task.endswith("v") or args.target_video_length > 1 else "png"
        save_result_path = save_result_path or str(work_dir / f"{model_id.replace('/', '--')}_{args.task}_{args.infer_steps}step.{suffix}")

    plan = build_cosmos3_lightx2v_launch_plan(
        paths,
        task=args.task,
        config_name=args.config_name,
        config_json=str(bounded_config_path) if bounded_config_path else None,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        save_result_path=save_result_path or "results/cosmos3_lightx2v_bridge.mp4",
        seed=args.seed,
        use_lazy_wrapper=args.use_lazy_wrapper or bool(args.lazy_artifact_dir),
    )
    if args.print_launch or args.execute:
        payload["launch_plan"] = asdict(plan)
        env = " ".join(f"{key}={shlex.quote(value)}" for key, value in plan.env.items())
        payload["launch_shell"] = f"cd {shlex.quote(plan.cwd)} && {env} {' '.join(shlex.quote(part) for part in plan.command)}"
    if bounded_config_path:
        payload["bounded_config_path"] = str(bounded_config_path)
    if args.execute:
        result = _execute_plan(plan, timeout_sec=args.timeout_sec)
        output_path = Path(save_result_path or "")
        result["save_result_path"] = str(output_path)
        result["output_exists"] = output_path.exists()
        result["output_size_bytes"] = output_path.stat().st_size if output_path.exists() else 0
        payload["execution"] = result

    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
