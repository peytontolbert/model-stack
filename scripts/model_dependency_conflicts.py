#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import model_dependency_diagnostics as diag  # noqa: E402
from runtime.model_catalog import DEFAULT_MODEL_INDEX_PATH, find_catalog_record, plan_model_integration


@dataclass(frozen=True)
class ModelRequirement:
    model_id: str
    backend: str
    local_path: str
    package: str
    required: str
    source: str
    installed: str
    status: str
    detail: str
    import_usages: tuple[str, ...]


def _env_name(explicit: str | None) -> str:
    return explicit or os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name or "unknown"


def _diagnose_model(model_id: str, index_path: str) -> tuple[dict[str, Any], list[ModelRequirement]]:
    record = find_catalog_record(model_id, index_path=index_path)
    plan = plan_model_integration(record)
    model_path = Path(plan.local_path)
    requirements = diag._collect_requirements(plan.backend, model_path)
    installed = diag._installed_versions(set(diag.BASE_PACKAGES) | set(requirements))
    checks = diag._checks(requirements, installed)
    usages_by_package: dict[str, list[str]] = defaultdict(list)
    for usage in diag._scan_python_import_usages(model_path):
        location = f"{usage['file']}:{usage['line']}:{usage['scope']} imports {usage['module']}"
        usages_by_package[str(usage["package"])].append(location)
    layout_status, layout_detail = diag._local_layout_status(plan.backend, model_path)
    payload = {
        "model_id": record.id,
        "backend": plan.backend,
        "lane": record.integration_lane,
        "local_path": str(model_path),
        "layout_status": layout_status,
        "layout_detail": layout_detail,
    }
    rows = [
        ModelRequirement(
            model_id=record.id,
            backend=plan.backend,
            local_path=str(model_path),
            package=check.package,
            required=check.required,
            source=check.source,
            installed=check.installed,
            status=check.status,
            detail=check.detail,
            import_usages=tuple(usages_by_package.get(check.package, ())),
        )
        for check in checks
    ]
    return payload, rows


def _is_conflict(rows: list[ModelRequirement]) -> bool:
    required_versions = {row.required for row in rows if row.required not in {"present", "review"}}
    statuses = {row.status for row in rows}
    return len(required_versions) > 1 or "blocked" in statuses


def _payload(env_name: str, models: list[dict[str, Any]], rows: list[ModelRequirement]) -> dict[str, Any]:
    by_package: dict[str, list[ModelRequirement]] = defaultdict(list)
    for row in rows:
        by_package[row.package].append(row)
    conflicts = []
    for package, package_rows in sorted(by_package.items()):
        if not _is_conflict(package_rows):
            continue
        conflicts.append(
            {
                "package": package,
                "installed_values": sorted({row.installed for row in package_rows}),
                "required_values": sorted({row.required for row in package_rows}),
                "models": [asdict(row) for row in package_rows],
            }
        )
    return {
        "env_name": env_name,
        "models": models,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Model Dependency Conflict Report",
        "",
        f"Env: `{data['env_name']}`",
        f"Conflicts/blockers: `{data['conflict_count']}`",
        "",
        "## Models",
        "",
        "| Model | Backend | Layout | Local Path |",
        "| --- | --- | --- | --- |",
    ]
    for model in data["models"]:
        lines.append(f"| `{model['model_id']}` | `{model['backend']}` | `{model['layout_status']}` | `{model['local_path']}` |")
    for conflict in data["conflicts"]:
        lines.extend(["", f"## `{conflict['package']}`", "", f"Installed: `{', '.join(conflict['installed_values'])}`", f"Required: `{', '.join(conflict['required_values'])}`", "", "| Model | Required | Installed | Status | Source | Import Usages |", "| --- | --- | --- | --- | --- | --- |"])
        for row in conflict["models"]:
            usages = "<br>".join(row["import_usages"]) if row["import_usages"] else ""
            source = str(row["source"]).replace("|", "\\|")
            lines.append(f"| `{row['model_id']}` | `{row['required']}` | `{row['installed']}` | `{row['status']}` | `{source}` | {usages} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dependency requirements across multiple catalog models in one env.")
    parser.add_argument("model_ids", nargs="+")
    parser.add_argument("--index-path", default=DEFAULT_MODEL_INDEX_PATH)
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    env_name = _env_name(args.env_name)
    models: list[dict[str, Any]] = []
    rows: list[ModelRequirement] = []
    for model_id in args.model_ids:
        model_payload, model_rows = _diagnose_model(model_id, args.index_path)
        models.append(model_payload)
        rows.extend(model_rows)
    data = _payload(env_name, models, rows)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(_markdown(data), encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
