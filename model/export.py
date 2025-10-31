from __future__ import annotations

import os
from dataclasses import asdict
import torch

from specs.export import ExportConfig
from specs.config import ModelConfig
from tensor.masking import build_causal_mask


def _dummy_inputs(cfg: ModelConfig, seq_len: int = 16, device: str | torch.device = "cpu"):
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)
    # Boolean causal mask (T,T); broadcastable by blocks
    mask = build_causal_mask(seq_len, device=device)
    return ids, mask


def export_onnx(model: torch.nn.Module, cfg: ModelConfig, out_path: str, *, export_cfg: ExportConfig | None = None) -> str:
    ec = export_cfg or ExportConfig()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    ids, mask = _dummy_inputs(cfg, min(ec.max_seq_len, 16), device)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids):
            return self.m(input_ids, None)

    wrapped = Wrapper(model)
    dynamic_axes = None
    if ec.dynamic_axes:
        dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}}
    torch.onnx.export(
        wrapped,
        (ids,),
        out_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(ec.opset),
    )
    return out_path


def export_torchscript(model: torch.nn.Module, cfg: ModelConfig, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.eval()
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids: torch.Tensor):
            return self.m(input_ids, None)
    wrapped = Wrapper(model)
    traced = torch.jit.trace(wrapped, (torch.randint(0, cfg.vocab_size, (1, 4))))
    torch.jit.save(traced, out_path)
    return out_path


