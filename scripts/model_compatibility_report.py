#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.compatibility import compatibility_report, report_as_dict


def _markdown(payload: dict) -> str:
    lines = [
        f"# Compatibility Report: {payload['model_id']}",
        "",
        f"Model path: `{payload['model_path']}`",
        "",
        "## Installed Versions",
        "",
        "| Package | Version |",
        "| --- | --- |",
    ]
    for package, version in sorted(payload["installed_versions"].items()):
        lines.append(f"| `{package}` | `{version}` |")
    lines.extend(["", "## Narrow API Drift Patches", "", "| Patch | Status | Expected API | Probe | Runtime Patch |", "| --- | --- | --- | --- | --- |"])
    if payload["patches"]:
        for patch in payload["patches"]:
            lines.append(
                f"| `{patch['id']}` | `{patch['status']}` | `{patch['expected_api']}` | {patch['current_probe']} | {patch['patch']} |"
            )
    else:
        lines.append("| none | none | no known narrow drift rule matched | - | - |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report model-stack narrow API drift patches for a local model.")
    parser.add_argument("model_path")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    report = compatibility_report(args.model_path, model_id=args.model_id)
    payload = report_as_dict(report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
