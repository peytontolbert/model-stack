from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _read_scalars(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    data: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return {}
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            try:
                step = int(row[0])
                key = str(row[1])
                val = float(row[2])
            except Exception:
                continue
            data[key].append((step, val))
    # sort by step for deterministic plots
    for k in list(data.keys()):
        data[k] = sorted(data[k], key=lambda x: x[0])
    return data


def render_index(log_dir: str | Path, *, title: str | None = None) -> Path:
    """Generate a static HTML dashboard at <log_dir>/index.html.

    Currently renders line charts for scalar series per key.
    """
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore

    log_dir = Path(log_dir)
    scalars = _read_scalars(log_dir / "scalars.csv")

    figs: List[go.Figure] = []
    for key, series in sorted(scalars.items()):
        if not series:
            continue
        xs = [s for s, _ in series]
        ys = [v for _, v in series]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=key))
        fig.update_layout(title=key, xaxis_title="step", yaxis_title="value", template="plotly_white")
        figs.append(fig)

    # Combine figures into a simple HTML page
    html_parts: List[str] = [
        "<html>",
        "<head><meta charset='utf-8'><title>{}</title></head>".format(title or "Viz"),
        "<body>",
        "<h1>{}</h1>".format(title or "Viz Dashboard"),
    ]
    for fig in figs:
        html_parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))
    html_parts.append("</body></html>")

    out = log_dir / "index.html"
    out.write_text("\n".join(html_parts))
    return out


