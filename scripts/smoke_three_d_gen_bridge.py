#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.three_d_gen_bridge import ThreeDGenRequest, compare_trellis_hunyuan3d, generate_3d, hunyuan3d_status, to_json, trellis2_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TRELLIS/Hunyuan3D 3D-generation bridge compatibility.")
    parser.add_argument("--backend", choices=["compare", "trellis", "hunyuan3d"], default="compare")
    parser.add_argument("--json-out")
    parser.add_argument("--generate", action="store_true", help="Build or run a 3D generation request instead of status-only probing.")
    parser.add_argument("--image-path", help="Input image for --generate.")
    parser.add_argument("--output-dir", help="Output directory for --generate.")
    parser.add_argument("--model-path", help="Optional backend model path for --generate.")
    parser.add_argument("--model-id", help="Optional backend model id for --generate.")
    parser.add_argument("--variant", help="Optional backend variant/subfolder for --generate.")
    parser.add_argument("--execute", action="store_true", help="Actually run the env worker; default is dry-run.")
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--load-only", action="store_true", help="For executable workers, load the backend pipeline without generation when supported.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--octree-resolution", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    args = parser.parse_args()

    if args.generate:
        if args.backend == "compare":
            raise SystemExit("--generate requires --backend trellis or --backend hunyuan3d")
        if not args.image_path or not args.output_dir:
            raise SystemExit("--generate requires --image-path and --output-dir")
        extra_args = {}
        if args.load_only:
            extra_args["load_only"] = True
        if args.num_inference_steps is not None:
            extra_args["num_inference_steps"] = args.num_inference_steps
        if args.octree_resolution is not None:
            extra_args["octree_resolution"] = args.octree_resolution
        if args.num_chunks is not None:
            extra_args["num_chunks"] = args.num_chunks
        result = generate_3d(
            ThreeDGenRequest(
                backend=args.backend,
                image_path=args.image_path,
                output_dir=args.output_dir,
                model_id=args.model_id,
                model_path=args.model_path,
                variant=args.variant,
                extra_args=extra_args or None,
            ),
            dry_run=not args.execute,
            timeout_sec=args.timeout_sec,
        )
    elif args.backend == "trellis":
        result = trellis2_status()
    elif args.backend == "hunyuan3d":
        result = hunyuan3d_status()
    else:
        result = compare_trellis_hunyuan3d()
    payload = to_json(result)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
