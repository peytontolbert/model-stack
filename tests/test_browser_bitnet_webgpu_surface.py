from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WGSL = ROOT / "browser" / "bitnet" / "bitnet_linear.wgsl"
JS = ROOT / "browser" / "bitnet" / "bitnet_webgpu.js"
ENCDEC_JS = ROOT / "browser" / "bitnet" / "encdec_runtime.js"


def test_bitnet_wgsl_documents_model_stack_ternary_decode() -> None:
    source = WGSL.read_text()

    assert "fn decode_signed_ternary" in source
    assert "code == 0u" in source
    assert "return -1.0" in source
    assert "code == 2u" in source
    assert "return 1.0" in source


def test_bitnet_wgsl_uses_model_stack_packed_row_stride() -> None:
    source = WGSL.read_text()

    assert "row_stride_bytes = params.padded_in_features / 4u" in source
    assert "out_idx * row_stride_bytes + (in_idx / 4u)" in source
    assert "byte_offset / 4u" in source
    assert "byte_lane = byte_offset & 3u" in source


def test_bitnet_webgpu_wrapper_rejects_non_v1_layouts() -> None:
    source = JS.read_text()

    assert "layoutHeader" in source
    assert "header[0] !== 1" in source
    assert "header[1] !== 16" in source
    assert "header[2] !== 32" in source
    assert "header[9] !== 1" in source


def test_bitnet_webgpu_wrapper_is_plain_browser_module() -> None:
    source = JS.read_text()

    assert "?raw" not in source
    assert "fromManifestUrl" in source
    assert "fetchText" in source


def test_encoder_decoder_browser_runtime_has_required_execution_surface() -> None:
    source = ENCDEC_JS.read_text()

    assert "class BitNetEncoderDecoderWebGPU" in source
    assert "async encode" in source
    assert "async decode" in source
    assert "async forward" in source
    assert "cross_block.cross" in source
    assert "self_attn_block.attn" in source
    assert "w_in" in source
    assert "w_out" in source
    assert "gatedActivation" in source
