from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_directory, now_utc_iso, read_git_info


def _render_markdown(metadata: Dict[str, Any]) -> str:
    name = metadata.get("model", {}).get("name", metadata.get("name", "Model"))
    version = metadata.get("model", {}).get("version", metadata.get("version", ""))
    license_id = metadata.get("license", "")
    datasets = metadata.get("datasets", [])
    training = metadata.get("training", {})
    evaluation = metadata.get("evaluation", {})
    limitations = metadata.get("limitations", "")
    ethical = metadata.get("ethical_considerations", "")
    intended = metadata.get("intended_use", "")
    sources = metadata.get("sources", [])

    lines: list[str] = []
    title = f"# {name}"
    if version:
        title += f" (v{version})"
    lines.append(title)
    lines.append("")
    if license_id:
        lines.append(f"License: `{license_id}`")
        lines.append("")

    if intended:
        lines.append("## Intended Use")
        lines.append(intended)
        lines.append("")

    if datasets:
        lines.append("## Datasets")
        for ds in datasets:
            if isinstance(ds, dict):
                nm = ds.get("name", "dataset")
                src = ds.get("source", "")
                lines.append(f"- {nm}{f' ({src})' if src else ''}")
            else:
                lines.append(f"- {ds}")
        lines.append("")

    if training:
        lines.append("## Training")
        for k, v in training.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if evaluation:
        lines.append("## Evaluation")
        for k, v in evaluation.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if limitations:
        lines.append("## Limitations")
        lines.append(str(limitations))
        lines.append("")

    if ethical:
        lines.append("## Ethical Considerations")
        lines.append(str(ethical))
        lines.append("")

    if sources:
        lines.append("## Sources")
        for s in sources:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)


def write_model_card(
    artifact_path: str | os.PathLike[str],
    metadata: Dict[str, Any],
    sbom_paths: Optional[list[str]] = None,
    output_path: Optional[str | os.PathLike[str]] = None,
) -> Path:
    art = Path(artifact_path)
    if output_path is None:
        output_path = art.parent / "MODEL_CARD.md"
    outp = Path(output_path)
    ensure_directory(outp)

    meta = dict(metadata or {})
    meta.setdefault("generated_at", now_utc_iso())
    meta.setdefault("artifact", str(art.resolve()))
    meta["git"] = read_git_info(art.parent)
    if sbom_paths:
        meta["sbom"] = [str(Path(p).resolve()) for p in sbom_paths]

    md = _render_markdown(meta)
    with open(outp, "w", encoding="utf-8") as f:
        f.write(md)

    # Emit a machine-readable sidecar for CI/debugging
    sidecar = outp.with_suffix(".json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return outp

