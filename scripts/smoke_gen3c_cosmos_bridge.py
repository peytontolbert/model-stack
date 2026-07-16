#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.gen3c_cosmos_bridge import gen3c_cosmos_status, status_to_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GEN3C-Cosmos local assets without loading model.pt tensors.")
    parser.add_argument("model_path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    status = gen3c_cosmos_status(args.model_path, model_id=args.model_id)
    payload = status_to_json(status)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status.runnable else 2


if __name__ == "__main__":
    raise SystemExit(main())
