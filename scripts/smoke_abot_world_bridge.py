#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from runtime.abot_world_bridge import load_abot_generator_cuda, probe_as_dict, write_probe_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke ABot-World model-stack CUDA bridge.")
    parser.add_argument("--repo-path", default="/data/repositories/ABot-World")
    parser.add_argument("--model-path", default="/arxiv/models/acvlab--ABot-World-0-5B-LF")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--json-out", default="reports/world-model-smokes/acvlab--ABot-World-0-5B-LF.generator_cuda_bf16.json")
    args = parser.parse_args()

    try:
        _generator, probe = load_abot_generator_cuda(
            repo_path=args.repo_path,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
        )
        write_probe_json(probe, args.json_out)
        print(json.dumps(probe_as_dict(probe), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
