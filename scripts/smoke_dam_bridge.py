#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from runtime.dam_bridge import DEFAULT_RUNTIME_SOURCE, probe_dam_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke DAM bridge status/imports and optional bounded component loads.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--runtime-source", type=Path, default=DEFAULT_RUNTIME_SOURCE)
    parser.add_argument("--load-components", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    probe = probe_dam_components(
        args.model_path,
        model_id=args.model_id or args.model_path.name,
        runtime_source=args.runtime_source,
        load_components=args.load_components,
    )
    text = json.dumps(asdict(probe), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
