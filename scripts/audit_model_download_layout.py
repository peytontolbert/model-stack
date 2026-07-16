#!/usr/bin/env python3
"""Audit Hugging Face downloads that were flattened into a shared model root."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


GENERIC_ROOT_NAMES = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "model_index.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "requirements.txt",
    "tokenizer.json",
    "tokenizer_config.json",
}


def read_metadata(path: Path) -> tuple[str | None, str | None]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None, None
    revision = lines[0].strip() if lines else None
    etag = lines[1].strip() if len(lines) > 1 else None
    return revision, etag


def file_info(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


def audit_model_root(model_root: Path) -> dict[str, Any]:
    download_cache = model_root / ".cache" / "huggingface" / "download"
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root_collisions: list[dict[str, Any]] = []

    if not download_cache.exists():
        return {
            "model_root": str(model_root),
            "error": f"missing Hugging Face local-dir cache: {download_cache}",
        }

    for metadata in sorted(download_cache.rglob("*.metadata")):
        revision, etag = read_metadata(metadata)
        rel_metadata = metadata.relative_to(download_cache)
        rel_file = Path(str(rel_metadata)[: -len(".metadata")])
        actual = model_root / rel_file
        top = rel_file.parts[0] if rel_file.parts else ""
        entry = {
            "path": str(rel_file),
            "etag": etag,
            "top_level": top,
            **file_info(actual),
        }
        groups[revision or "unknown"].append(entry)
        if len(rel_file.parts) == 1 and rel_file.name in GENERIC_ROOT_NAMES:
            root_collisions.append({"revision": revision, **entry})

    grouped = []
    for revision, entries in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        existing_entries = [entry for entry in entries if entry.get("exists")]
        total_size = sum(int(entry.get("size") or 0) for entry in existing_entries)
        top_levels = sorted({entry["top_level"] for entry in entries if entry.get("top_level")})
        root_files = sorted(entry["path"] for entry in entries if "/" not in entry["path"])
        grouped.append(
            {
                "revision": revision,
                "file_count": len(entries),
                "existing_file_count": len(existing_entries),
                "existing_total_size": total_size,
                "top_levels": top_levels,
                "root_files": root_files,
                "files": entries,
            }
        )

    return {
        "model_root": str(model_root),
        "cache": str(download_cache),
        "revision_group_count": len(grouped),
        "root_collision_count": len(root_collisions),
        "root_collisions": sorted(root_collisions, key=lambda item: (item.get("revision") or "", item["path"])),
        "revision_groups": grouped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", nargs="?", default="/arxiv/models")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--summary", action="store_true", help="Print a compact text summary.")
    args = parser.parse_args()

    report = audit_model_root(Path(args.model_root))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.summary:
        print(f"model_root: {report.get('model_root')}")
        print(f"revision_groups: {report.get('revision_group_count')}")
        print(f"root_collisions: {report.get('root_collision_count')}")
        for group in report.get("revision_groups", [])[:12]:
            total_gb = group["existing_total_size"] / 1_000_000_000
            roots = ", ".join(group["root_files"][:8])
            if len(group["root_files"]) > 8:
                roots += ", ..."
            print(
                f"- {group['revision'][:12]} files={group['file_count']} "
                f"existing={group['existing_file_count']} size={total_gb:.2f}GB "
                f"top={','.join(group['top_levels'][:6])} roots=[{roots}]"
            )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
