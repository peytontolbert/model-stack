from __future__ import annotations

from pathlib import Path

import torch

from specs.config import ModelConfig
from export.exporter import export_from_dir
from specs.export import ExportConfig
from model.lm import TransformerLM
from model.checkpoint import save_pretrained


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _repo_root()
    artifacts = root / "examples" / "02_int8_export" / "artifacts"
    ckpt_dir = artifacts / "ckpt"
    export_dir = artifacts / "export"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl="flash")
    model = TransformerLM(cfg).eval()
    _ = save_pretrained(model, cfg, str(ckpt_dir))

    ecfg = ExportConfig(target="onnx", opset=19, quantize="int8", dynamic_axes=True, outdir=str(export_dir))
    out = export_from_dir(str(ckpt_dir), ecfg)
    print("exported:", str(out))


if __name__ == "__main__":
    main()


