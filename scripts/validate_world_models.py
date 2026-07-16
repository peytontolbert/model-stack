#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runtime.diffusers_bridge import diffusers_snapshot_status
from runtime.model_catalog import DEFAULT_MODEL_INDEX_PATH, load_model_catalog, plan_model_integration


WORLD_IDS = {
    "Cosmos-Embed1-224p",
    "Cosmos-Embed1-448p",
    "Cosmos-Embed1-448p-anomaly-detection",
    "Cosmos3-Nano",
    "Cosmos3-Nano-Policy-DROID",
    "nvidia/Cosmos3-Nano",
    "GEN3C-Cosmos-7B",
    "HunyuanWorld-Voyager",
    "repository_library/world-planner-adapter",
    "robbyant/lingbot-world-base-act-preview",
    "robbyant/lingbot-world-v2-14b-causal-fast",
}


@dataclass(frozen=True)
class WorldModelValidationRow:
    id: str
    lane: str
    backend: str
    local_path: str
    local_exists: bool
    status: str
    preferred_env: str
    detail: str
    config_status: str | None = None
    diffusers_class: str | None = None
    diffusers_missing: tuple[str, ...] = ()


def _is_world_record(record: Any) -> bool:
    tags = tuple(str(item).lower() for item in (record.raw.get("tags") or ()))
    tasks = tuple(str(item).lower() for item in record.tasks)
    return (
        record.id in WORLD_IDS
        or "world-modeling" in tags
        or "world-modeling" in tasks
        or record.integration_lane == "world_model_bridge"
    )


def _config_status(path: Path) -> str:
    config_path = path / "config.json"
    if not config_path.is_file():
        return "missing_config_json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid_config_json:{type(exc).__name__}:{exc}"
    model_type = data.get("model_type")
    architectures = data.get("architectures")
    auto_map = data.get("auto_map")
    return f"ok:model_type={model_type}:architectures={architectures}:auto_map={auto_map}"


def _diffusers_pipeline_available(class_name: str | None) -> bool:
    if not class_name:
        return False
    spec = importlib.util.find_spec("diffusers")
    if spec is None:
        return False
    try:
        import diffusers

        return hasattr(diffusers, class_name)
    except Exception:
        return False


def _classify(record: Any) -> WorldModelValidationRow:
    plan = plan_model_integration(record)
    path = Path(plan.local_path)
    exists = path.exists()
    config_status = _config_status(path) if exists else None
    status = "missing_local_path"
    preferred_env = "manual"
    detail = "local path missing"
    diffusers_class = None
    diffusers_missing: tuple[str, ...] = ()

    if exists:
        if record.id == "repository_library/world-planner-adapter" or (path / "adapter_config.json").is_file():
            status = "adapter_needs_base_model"
            preferred_env = "match_base_model_env"
            detail = "PEFT adapter files present; needs explicit world/planner base model"
        elif (path / "model_index.json").is_file():
            snapshot = diffusers_snapshot_status(str(path))
            diffusers_class = snapshot.class_name
            diffusers_missing = snapshot.missing_components
            if snapshot.complete and _diffusers_pipeline_available(snapshot.class_name):
                status = "candidate_diffusers_world_snapshot"
                preferred_env = "ai"
                detail = f"Diffusers snapshot complete and pipeline class importable: {snapshot.class_name}"
            elif snapshot.complete:
                status = "needs_diffusers_pipeline_implementation"
                preferred_env = "ai_or_cosmos_env"
                detail = f"Diffusers layout complete, but installed diffusers lacks pipeline class {snapshot.class_name}"
            else:
                status = "incomplete_diffusers_world_snapshot"
                preferred_env = "ai_or_cosmos_env"
                detail = "missing=" + ",".join(snapshot.missing_components)
        elif record.library_name == "cosmos" or "Cosmos-Embed1" in record.id:
            if config_status and config_status.startswith("ok:"):
                status = "candidate_transformers_remote_code"
                preferred_env = "ai"
                detail = "AutoConfig/trust_remote_code path expected; full AutoModel smoke still needed"
            else:
                status = "needs_custom_cosmos_env"
                preferred_env = "cosmos_env"
                detail = config_status or "custom Cosmos layout"
        elif record.id == "HunyuanWorld-Voyager":
            status = "needs_hunyuanworld_custom_bridge"
            preferred_env = "py311build_or_custom"
            detail = f"custom HunyuanWorld layout; top config status: {config_status}"
        elif "lingbot-world" in record.id:
            status = "needs_lingbot_world_bridge"
            preferred_env = "ai_or_custom_wan_env"
            detail = "custom Wan/LingBot layout with no model_index.json; component configs present"
        elif "GEN3C-Cosmos" in record.id:
            status = "needs_gen3c_custom_bridge"
            preferred_env = "cosmos_env"
            detail = "custom GEN3C checkpoint layout with model.pt and non-Transformers config"
        else:
            status = "manual_world_triage"
            preferred_env = "manual"
            detail = config_status or "custom world-model layout"

    return WorldModelValidationRow(
        id=record.id,
        lane=record.integration_lane,
        backend=plan.backend,
        local_path=str(path),
        local_exists=exists,
        status=status,
        preferred_env=preferred_env,
        detail=detail,
        config_status=config_status,
        diffusers_class=diffusers_class,
        diffusers_missing=diffusers_missing,
    )



def _local_extra_rows(existing_ids: set[str]) -> list[WorldModelValidationRow]:
    rows: list[WorldModelValidationRow] = []
    path = Path("/arxiv/models/HunyuanWorld-Voyager")
    if "HunyuanWorld-Voyager" not in existing_ids and path.exists():
        rows.append(
            WorldModelValidationRow(
                id="HunyuanWorld-Voyager",
                lane="world_model_bridge",
                backend="manual",
                local_path=str(path),
                local_exists=True,
                status="needs_hunyuanworld_custom_bridge",
                preferred_env="py311build_or_custom",
                detail=f"custom HunyuanWorld layout; top config status: {_config_status(path)}; Voyager subconfig status: {_config_status(path / 'Voyager')}",
                config_status=_config_status(path),
            )
        )
    return rows

def _markdown(rows: list[WorldModelValidationRow]) -> str:
    lines = [
        "# World Model Validation",
        "",
        "Lightweight validation of locally downloaded world-model candidates. This checks local layout, config readability, Diffusers snapshot completeness, and whether the installed Diffusers package exposes the required pipeline class. It does not run heavy generation.",
        "",
        "Last verified: 2026-07-13.",
        "",
        "| Model | Status | Preferred Env | Local | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda item: item.id):
        detail = row.detail.replace("|", "\\|")
        lines.append(f"| `{row.id}` | `{row.status}` | `{row.preferred_env}` | {'yes' if row.local_exists else 'no'} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local world-model layouts for model-stack integration.")
    parser.add_argument("--index-path", default=DEFAULT_MODEL_INDEX_PATH)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    rows = [_classify(record) for record in load_model_catalog(args.index_path) if _is_world_record(record)]
    rows.extend(_local_extra_rows({row.id for row in rows}))
    payload = {"rows": [asdict(row) for row in rows], "status_counts": {}}
    for row in rows:
        payload["status_counts"][row.status] = payload["status_counts"].get(row.status, 0) + 1

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(_markdown(rows), encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
