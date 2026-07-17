#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time

import torch

from runtime.sapiens2_pose_bridge import load_sapiens2_pose_model, sapiens2_pose_status, status_to_json


def _memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    return {
        'allocated_mb': int(torch.cuda.memory_allocated(index) / 1024 / 1024),
        'reserved_mb': int(torch.cuda.memory_reserved(index) / 1024 / 1024),
        'free_mb': int(free / 1024 / 1024),
        'total_mb': int(total / 1024 / 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Smoke Sapiens2 pose through the model-stack custom bridge.')
    parser.add_argument('--model-path', default='/arxiv/models/facebook/sapiens2-pose-1b')
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--load', action='store_true', help='Load checkpoint weights through the custom bridge.')
    parser.add_argument('--forward', action='store_true', help='Run one synthetic forward after loading.')
    args = parser.parse_args()

    status = sapiens2_pose_status(args.model_path)
    print(status_to_json(status))
    if not args.load:
        return

    started = time.perf_counter()
    artifacts = load_sapiens2_pose_model(args.model_path, device=args.device, dtype=args.dtype)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    report = {
        'status': status.status,
        'load_seconds': time.perf_counter() - started,
        'device': str(artifacts.device),
        'dtype': str(artifacts.dtype),
        'memory_after_load': _memory(),
    }
    if args.forward:
        height, width = artifacts.processor.size
        pixel_values = torch.zeros(1, 3, height, width, device=artifacts.device, dtype=artifacts.dtype)
        forward_started = time.perf_counter()
        with torch.inference_mode():
            out = artifacts.model(pixel_values)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        report['forward_seconds'] = time.perf_counter() - forward_started
        report['heatmaps_shape'] = list(out['heatmaps'].shape)
        report['memory_after_forward'] = _memory()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
