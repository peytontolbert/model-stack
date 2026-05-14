from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import torch


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() <= 256:
            return value.detach().cpu().tolist()
        return {"shape": tuple(value.shape), "mean": float(value.float().mean().item())}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def render_interpretability_html_report(
    *,
    title: str,
    sections: Iterable[tuple[str, Any]],
) -> str:
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:32px;line-height:1.45}"
        "section{margin:0 0 28px}pre{background:#f6f8fa;padding:12px;overflow:auto}"
        "h1,h2{line-height:1.15}</style>",
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]
    for heading, payload in sections:
        body.append("<section>")
        body.append(f"<h2>{html.escape(heading)}</h2>")
        body.append(f"<pre>{html.escape(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))}</pre>")
        body.append("</section>")
    body.append("</body></html>")
    return "\n".join(body)


def save_interpretability_html_report(path: str | Path, *, title: str, sections: Iterable[tuple[str, Any]]) -> str:
    rendered = render_interpretability_html_report(title=title, sections=sections)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    return str(target)
