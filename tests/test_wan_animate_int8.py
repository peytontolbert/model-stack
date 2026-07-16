import torch
from torch import nn

from runtime.wan_animate_int8 import (
    WanInt8Config,
    attach_wan_block_offload,
    block_state_dict,
    convert_wan_linears_to_int8,
    write_int8_block_manifest,
)


def test_selective_int8_conversion_preserves_small_and_norm_paths():
    model = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.Linear(128, 4))
    inventory = convert_wan_linears_to_int8(
        model, WanInt8Config(min_weight_elements=1_000, activation_quant="none")
    )
    assert inventory.quantized == ("0",)
    assert inventory.skipped == ("2",)
    output = model(torch.randn(2, 128))
    assert output.shape == (2, 4)


def test_block_state_and_cpu_offload_hook():
    class TinyWan(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = TinyWan().bfloat16()
    convert_wan_linears_to_int8(model, WanInt8Config(min_weight_elements=1, activation_quant="none"))
    state = block_state_dict(model, 0)
    assert "qweight" in state
    offloader = attach_wan_block_offload(model, device="cpu", group_size=1, prefetch=False)
    assert model(torch.randn(1, 8, dtype=torch.bfloat16)).shape == (1, 8)
    offloader.close()


def test_block_manifest_records_quantized_modules_by_block(tmp_path):
    inventory = type("Inventory", (), {
        "quantized": ("blocks.0.self_attn.q", "blocks.1.ffn.0", "head.head"),
        "skipped": (),
    })()

    manifest = write_int8_block_manifest(
        tmp_path,
        source_checkpoint=tmp_path / "ckpt",
        config=WanInt8Config(),
        inventory=inventory,
        num_blocks=2,
    )

    import json
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["quantized_modules_by_block"] == {"0": ["self_attn.q"], "1": ["ffn.0"]}
