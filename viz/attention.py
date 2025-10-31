from __future__ import annotations

from typing import Optional

import torch


def attention_heatmap_figure(attn: torch.Tensor, *, head: Optional[int] = None):
    """Create a Plotly heatmap for an attention matrix.

    attn: Tensor of shape [H, T, T] or [T, T].
    """
    import plotly.graph_objects as go  # type: ignore

    if attn.ndim == 3:
        if head is None:
            head = 0
        mat = attn[head]
    elif attn.ndim == 2:
        mat = attn
    else:
        raise ValueError("attention tensor must be [H, T, T] or [T, T]")
    mat = mat.detach().float().cpu().numpy()
    fig = go.Figure(data=go.Heatmap(z=mat, colorscale="Viridis"))
    fig.update_layout(
        title=f"Attention Heatmap{'' if head is None else f' (head {head})'}",
        xaxis_title="Key position",
        yaxis_title="Query position",
        template="plotly_white",
    )
    return fig


