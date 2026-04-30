from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not hasattr(torch, "nn"):
    pytest.skip("PyTorch with torch.nn is required for browser BitNet export tests", allow_module_level=True)

from compress.quantization import QuantizedLinearBitNet
from export.exporter import export_model
from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig
from specs.export import ExportConfig


class TinyBitNetModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dense = torch.nn.Linear(4, 3, bias=True)
        with torch.no_grad():
            dense.weight.copy_(
                torch.tensor(
                    [
                        [-1.0, 0.0, 1.0, 0.5],
                        [0.25, -0.5, 0.0, 1.0],
                        [1.0, 1.0, -1.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            )
            dense.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))
        self.proj = QuantizedLinearBitNet(4, 3, bias=True).from_float(dense, spin=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TinyTiedEncoderDecoderBitNetModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_embed = torch.nn.Embedding(8, 4)
        self.dec_embed = torch.nn.Embedding(8, 4)
        self.dec_embed.weight = self.enc_embed.weight
        self.encoder = torch.nn.ModuleList()
        self.decoder = torch.nn.ModuleList()
        self.enc_norm = torch.nn.LayerNorm(4)
        self.dec_norm = torch.nn.LayerNorm(4)
        self.lm_head = QuantizedLinearBitNet(4, 8, bias=False).from_float(torch.nn.Linear(4, 8, bias=False), spin=False)


def test_browser_bitnet_export_writes_manifest_runtime_and_layer_bins(tmp_path: Path) -> None:
    out = export_model(
        TinyBitNetModel(),
        ExportConfig(target="browser-bitnet", outdir=str(tmp_path)),
        model_cfg=ModelConfig(d_model=4, n_heads=1, n_layers=1, d_ff=8, vocab_size=16),
    )

    assert out == tmp_path / "manifest.json"
    manifest = json.loads(out.read_text())

    assert manifest["format"] == "model-stack-browser-bitnet"
    assert manifest["runtime"]["primary"] == "webgpu"
    assert manifest["runtime"]["fallback"] == "wasm"
    assert manifest["runtime"]["safari"]["https_required"] is True
    assert manifest["runtime"]["safari"]["webgpu_feature_detection"] == "navigator.gpu"
    assert manifest["runtime"]["safari"]["threads_require_cross_origin_isolation"] is True
    assert manifest["runtime"]["files"]["webgpu_js"] == "runtime/bitnet_webgpu.js"
    assert manifest["runtime"]["files"]["wgsl"] == "runtime/bitnet_linear.wgsl"
    assert manifest["runtime"]["files"]["encdec_js"] == "runtime/encdec_runtime.js"
    assert manifest["runtime"]["files"]["wasm_runtime_js"] == "runtime/bitnet_wasm_runtime.js"
    assert manifest["runtime"]["files"]["wasm_js"] == "runtime/model_stack_bitnet_wasm.js"
    assert manifest["runtime"]["files"]["wasm_binary"] == "runtime/model_stack_bitnet_wasm_bg.wasm"
    assert (tmp_path / "runtime" / "bitnet_webgpu.js").exists()
    assert (tmp_path / "runtime" / "bitnet_linear.wgsl").exists()
    assert (tmp_path / "runtime" / "encdec_runtime.js").exists()
    assert (tmp_path / "runtime" / "bitnet_wasm_runtime.js").exists()
    assert (tmp_path / "runtime" / "model_stack_bitnet_wasm.js").exists()
    assert (tmp_path / "runtime" / "model_stack_bitnet_wasm_bg.wasm").exists()

    assert len(manifest["layers"]) == 1
    layer = manifest["layers"][0]
    assert layer["name"] == "proj"
    assert layer["format"] == "bitnet_w2a8"
    assert layer["layout_header"][:3] == [1, 16, 32]
    assert layer["layout_header"][9] == 1

    for tensor in layer["tensors"].values():
        path = tmp_path / "layers" / tensor["path"]
        assert path.exists()
        assert path.stat().st_size == tensor["bytes"]


def test_browser_bitnet_export_rejects_dynamic_activation_quant(tmp_path: Path) -> None:
    model = TinyBitNetModel()
    model.proj.act_quant_mode = "dynamic_int8"

    try:
        export_model(model, ExportConfig(target="browser-bitnet", outdir=str(tmp_path)))
    except ValueError as exc:
        assert "dynamic_int8" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dynamic_int8 browser export to fail")


def test_browser_bitnet_export_preserves_tied_embedding_aliases(tmp_path: Path) -> None:
    out = export_model(
        TinyTiedEncoderDecoderBitNetModel(),
        ExportConfig(target="browser-bitnet", outdir=str(tmp_path)),
        model_cfg=ModelConfig(d_model=4, n_heads=1, n_layers=0, d_ff=8, vocab_size=8),
    )
    manifest = json.loads(out.read_text())

    assert manifest["graph"]["architecture"] == "encoder_decoder"
    assert manifest["graph"]["embeddings"]["encoder"] == "enc_embed.weight"
    assert manifest["graph"]["embeddings"]["decoder"] == "dec_embed.weight"
    assert "enc_embed.weight" in manifest["dense_tensors"]
    assert "dec_embed.weight" in manifest["dense_tensors"]


def test_browser_bitnet_export_covers_encoder_decoder_graph(tmp_path: Path) -> None:
    cfg = ModelConfig(
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        vocab_size=32,
        activation="silu",
    )
    model = EncoderDecoderLM(cfg)

    out = export_model(
        model,
        ExportConfig(target="browser-bitnet", quantize="bitnet", outdir=str(tmp_path)),
        model_cfg=cfg,
    )
    manifest = json.loads(out.read_text())

    assert manifest["graph"]["architecture"] == "encoder_decoder"
    assert manifest["graph"]["supports"]["encode"] is True
    assert manifest["graph"]["supports"]["decode"] is True
    assert manifest["graph"]["supports"]["cross_attention"] is True

    layer_names = {layer["name"] for layer in manifest["layers"]}
    assert "encoder.0.attn.w_q" in layer_names
    assert "encoder.0.attn.w_k" in layer_names
    assert "encoder.0.attn.w_v" in layer_names
    assert "encoder.0.attn.w_o" in layer_names
    assert "encoder.0.mlp.w_in" in layer_names
    assert "encoder.0.mlp.w_out" in layer_names
    assert "decoder.0.self_attn_block.attn.w_q" in layer_names
    assert "decoder.0.self_attn_block.attn.w_k" in layer_names
    assert "decoder.0.self_attn_block.attn.w_v" in layer_names
    assert "decoder.0.self_attn_block.attn.w_o" in layer_names
    assert "decoder.0.self_attn_block.mlp.w_in" in layer_names
    assert "decoder.0.self_attn_block.mlp.w_out" in layer_names
    assert "decoder.0.cross_block.cross.w_q" in layer_names
    assert "decoder.0.cross_block.cross.w_k" in layer_names
    assert "decoder.0.cross_block.cross.w_v" in layer_names
    assert "decoder.0.cross_block.cross.w_o" in layer_names
    assert "decoder.0.cross_block.mlp.w_in" in layer_names
    assert "decoder.0.cross_block.mlp.w_out" in layer_names
    assert "lm_head" in layer_names

    dense = manifest["dense_tensors"]
    assert "enc_embed.weight" in dense
    assert "dec_embed.weight" in dense
    assert "enc_norm.weight" in dense
    assert "enc_norm.bias" in dense
    assert "dec_norm.weight" in dense
    assert "dec_norm.bias" in dense

    roles = {(role["name"], role.get("block"), role.get("projection")) for role in manifest["graph"]["linear_roles"]}
    assert ("decoder.0.cross_block.cross.w_q", "cross_attention", "q") in roles
    assert ("decoder.0.self_attn_block.attn.w_q", "self_attention", "q") in roles
    assert ("encoder.0.mlp.w_in", "mlp", "mlp_in") in roles
