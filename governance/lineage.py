from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils import ensure_directory, try_import


def _dot_escape(value: str) -> str:
    return value.replace("\"", "'")


def write_lineage_graph(
    output_dot_path: str | os.PathLike[str],
    nodes: Iterable[Dict[str, Any]],
    edges: Iterable[Tuple[str, str, str]] | None = None,
    render_image: bool = True,
) -> Path:
    outp = Path(output_dot_path)
    ensure_directory(outp)

    node_lines: List[str] = []
    id_set: set[str] = set()
    for n in nodes:
        node_id = str(n.get("id") or n.get("name"))
        if not node_id:
            continue
        id_set.add(node_id)
        label = n.get("label") or node_id
        kind = n.get("type", "node")
        node_lines.append(f'  "{_dot_escape(node_id)}" [label="{_dot_escape(label)}\n({kind})"];')

    edge_lines: List[str] = []
    if edges:
        for s, t, lbl in edges:
            if s in id_set and t in id_set:
                edge_lines.append(f'  "{_dot_escape(s)}" -> "{_dot_escape(t)}" [label="{_dot_escape(lbl)}"];')

    content = [
        "digraph lineage {",
        "  rankdir=LR;",
        *node_lines,
        *edge_lines,
        "}",
    ]

    with open(outp, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    if render_image:
        graphviz = try_import("graphviz")
        if graphviz is not None:
            from graphviz import Source  # type: ignore
            src = Source("\n".join(content))
            # Renders alongside .dot as .png
            png_path = str(outp.with_suffix(".png"))
            src.render(filename=str(outp.with_suffix("")), format="png", cleanup=True)

    return outp


def lineage_from_training_metadata(meta: Dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[tuple[str, str, str]] = []

    model = meta.get("model", {})
    model_id = model.get("name", "model")
    nodes.append({"id": model_id, "label": model.get("name", model_id), "type": "model"})

    for ds in meta.get("datasets", []) or []:
        if isinstance(ds, dict):
            ds_id = ds.get("name", "dataset")
            nodes.append({"id": ds_id, "label": ds_id, "type": "dataset"})
            edges.append((ds_id, model_id, "trained_on"))
        else:
            ds_id = str(ds)
            nodes.append({"id": ds_id, "label": ds_id, "type": "dataset"})
            edges.append((ds_id, model_id, "trained_on"))

    base = meta.get("model", {}).get("base", None)
    if base:
        nodes.append({"id": base, "label": base, "type": "checkpoint"})
        edges.append((base, model_id, "fine_tuned_from"))

    return nodes, edges


