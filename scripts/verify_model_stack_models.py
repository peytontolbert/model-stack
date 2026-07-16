#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runtime.model_catalog import DEFAULT_MODEL_INDEX_PATH, load_model_catalog, plan_model_integration, primary_lane_records
from runtime.diffusers_bridge import diffusers_adapter_status, diffusers_snapshot_status


PACKAGE_NAMES = ("torch", "diffusers", "transformers", "accelerate", "peft", "huggingface_hub", "safetensors", "nemo_toolkit")


@dataclass(frozen=True)
class VerificationRow:
    id: str
    lane: str
    backend: str
    preferred_env: str
    status: str
    env_status: str
    local_path: str
    local_exists: bool
    detail: str


def _package_versions() -> dict[str, str]:
    installed = {dist.metadata["Name"].lower().replace("-", "_"): dist.version for dist in metadata.distributions()}
    versions = {name: installed.get(name, "missing") for name in PACKAGE_NAMES}
    try:
        import torch

        versions["torch_import"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = str(torch.cuda.is_available())
        versions["torch_cuda_version"] = str(getattr(torch.version, "cuda", None))
    except Exception as exc:  # pragma: no cover - env diagnostic path
        versions["torch_import"] = f"failed:{type(exc).__name__}:{exc}"
        versions["torch_cuda"] = "False"
        versions["torch_cuda_version"] = "None"
    return versions


def _env_name(explicit: str | None) -> str:
    if explicit:
        return explicit
    return os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name or "unknown"


def _preferred_env(lane: str, backend: str, versions: dict[str, str]) -> tuple[str, str]:
    if backend == "nemo_or_transformers_asr" or lane == "nemo_asr_bridge":
        if versions.get("nemo_toolkit") == "missing":
            return "nemo_speech", "nemo_toolkit missing; use/create nemo_speech"
        return "current", "nemo_toolkit present"
    if backend == "diffusers":
        if versions.get("diffusers") == "missing":
            return "needs_diffusers_env", "diffusers missing"
        return "ai", "diffusers present"
    if backend == "transformers":
        if versions.get("transformers") == "missing":
            return "needs_transformers_env", "transformers missing"
        return "ai", "transformers present"
    if backend == "peft":
        if versions.get("peft") == "missing":
            return "match_base_model_env", "peft missing in current env"
        return "match_base_model_env", "adapter follows base model env"
    return "manual", "manual lane"


def _classify_record(record: Any, versions: dict[str, str]) -> VerificationRow:
    plan = plan_model_integration(record)
    path = Path(plan.local_path)
    exists = path.exists()
    preferred_env, env_status = _preferred_env(record.integration_lane, plan.backend, versions)
    status = "missing_local_path"
    detail = "download or resolver alias needed before runtime verification"

    if exists:
        if plan.backend == "diffusers":
            diffusers_status = diffusers_snapshot_status(str(path))
            adapter_status = diffusers_adapter_status(str(path))
            if diffusers_status.complete:
                status = "works_snapshot_status"
                detail = f"class={diffusers_status.class_name}; components={','.join(diffusers_status.present_components)}"
            elif adapter_status.complete:
                status = "works_adapter_status"
                detail = f"weights={','.join(adapter_status.weight_files)}"
            elif diffusers_status.missing_components == ("model_index.json",):
                status = "needs_custom_bridge_or_env"
                detail = "local path exists but no Diffusers model_index.json; verify model-specific repo/env"
            else:
                status = "incomplete_diffusers_snapshot"
                detail = "missing=" + ",".join(diffusers_status.missing_components[:8])
        elif plan.backend == "transformers":
            if (path / "config.json").is_file():
                status = "candidate_transformers_snapshot"
                detail = "config.json present; full load smoke still needed"
            else:
                status = "needs_custom_bridge_or_env"
                detail = "local path exists but no config.json"
        elif plan.backend == "peft":
            if (path / "adapter_config.json").is_file() or any(path.glob("*.safetensors")):
                status = "adapter_needs_base_model"
                detail = "adapter files present; verify with explicit base model env"
            else:
                status = "needs_custom_bridge_or_env"
                detail = "adapter path exists but adapter files not found"
        elif plan.backend == "nemo_or_transformers_asr":
            status = "needs_nemo_speech_env" if versions.get("nemo_toolkit") == "missing" else "candidate_nemo_snapshot"
            detail = env_status
        else:
            status = "manual_triage"
            detail = ";".join(plan.notes)

    return VerificationRow(
        id=record.id,
        lane=record.integration_lane,
        backend=plan.backend,
        preferred_env=preferred_env,
        status=status,
        env_status=env_status,
        local_path=str(path),
        local_exists=exists,
        detail=detail,
    )


def _markdown(env_name: str, versions: dict[str, str], rows: list[VerificationRow]) -> str:
    status_counts = Counter(row.status for row in rows)
    lane_counts = Counter(row.lane for row in rows)
    lines = [
        "# Model Stack Model Verification",
        "",
        "Lightweight verification of first-wave catalog entries. This checks env packages, local path/cache resolution, and bridge-compatible snapshot layout. It does not load every full model by default.",
        "",
        "Last verified: 2026-07-13.",
        f"Verifier env: `{env_name}`.",
        "",
        "## Environment",
        "",
        "| Package | Version / Status |",
        "| --- | --- |",
    ]
    for key in sorted(versions):
        lines.append(f"| `{key}` | `{versions[key]}` |")
    lines.extend(["", "## Summary", "", "| Status | Count |", "| --- | ---: |"])
    for status, count in status_counts.most_common():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "| Lane | Count |", "| --- | ---: |"])
    for lane, count in lane_counts.most_common():
        lines.append(f"| `{lane}` | {count} |")

    by_lane: dict[str, list[VerificationRow]] = defaultdict(list)
    for row in rows:
        by_lane[row.lane].append(row)
    for lane in sorted(by_lane):
        lines.extend(["", f"## {lane}", "", "| Model | Status | Preferred Env | Local | Detail |", "| --- | --- | --- | --- | --- |"])
        for row in sorted(by_lane[lane], key=lambda item: item.id):
            local = "yes" if row.local_exists else "no"
            detail = row.detail.replace("|", "\\|")
            lines.append(f"| `{row.id}` | `{row.status}` | `{row.preferred_env}` | {local} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify model-stack catalog entries without loading every full model.")
    parser.add_argument("--index-path", default=DEFAULT_MODEL_INDEX_PATH)
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    env_name = _env_name(args.env_name)
    versions = _package_versions()
    records = primary_lane_records(load_model_catalog(args.index_path))
    rows = [_classify_record(record, versions) for record in records]
    payload: dict[str, Any] = {
        "env_name": env_name,
        "versions": versions,
        "status_counts": dict(Counter(row.status for row in rows)),
        "lane_counts": dict(Counter(row.lane for row in rows)),
        "rows": [asdict(row) for row in rows],
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(_markdown(env_name, versions, rows), encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
