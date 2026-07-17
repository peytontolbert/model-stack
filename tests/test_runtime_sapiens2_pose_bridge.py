from __future__ import annotations

import json

import torch

from runtime.sapiens2_pose_bridge import (
    ModelStackSapiens2ForPoseEstimation,
    Sapiens2PoseImageProcessor,
    sapiens2_pose_status,
    _namespace,
)


def _tiny_config():
    return _namespace(
        {
            'architectures': ['Sapiens2ForPoseEstimation'],
            'model_type': 'sapiens2',
            'hidden_size': 16,
            'intermediate_size': 32,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'num_key_value_heads_per_layer': [4, 2],
            'num_register_tokens': 2,
            'num_channels': 3,
            'image_size': [32, 24],
            'patch_size': 8,
            'rope_theta': 100.0,
            'query_bias': True,
            'key_bias': True,
            'value_bias': True,
            'proj_bias': True,
            'mlp_bias': True,
            'use_qk_norm': True,
            'rms_norm_eps': 1e-6,
            'layerscale_value': 1.0,
            'num_labels': 5,
            'head_config': {
                'upsample_out_channels': [12, 8],
                'upsample_kernel_sizes': [4, 4],
                'conv_out_channels': [7, 6, 4],
                'conv_kernel_sizes': [1, 1, 1],
            },
        }
    )


def test_sapiens2_pose_status_detects_complete_snapshot(tmp_path):
    model_dir = tmp_path / 'sapiens2-pose-1b'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(
        json.dumps({'model_type': 'sapiens2', 'architectures': ['Sapiens2ForPoseEstimation']}),
        encoding='utf-8',
    )
    (model_dir / 'preprocessor_config.json').write_text('{}', encoding='utf-8')
    (model_dir / 'model.safetensors').write_text('', encoding='utf-8')

    status = sapiens2_pose_status(model_dir)

    assert status.status == 'verified_sapiens2_pose_load_bridge'
    assert status.runnable is True
    assert status.loader == 'runtime.sapiens2_pose_bridge.load_sapiens2_pose_model'
    assert 'Transformers 5.x' in status.detail


def test_sapiens2_pose_processor_resizes_and_normalizes_hwc_tensor():
    processor = Sapiens2PoseImageProcessor(
        _namespace(
            {
                'size': {'height': 32, 'width': 24},
                'image_mean': [0.5, 0.5, 0.5],
                'image_std': [0.5, 0.5, 0.5],
                'rescale_factor': 1 / 255,
                'do_rescale': True,
                'do_resize': True,
                'do_normalize': True,
            }
        )
    )

    batch = processor(torch.full((16, 12, 3), 255, dtype=torch.uint8))

    assert batch['pixel_values'].shape == (1, 3, 32, 24)
    assert torch.allclose(batch['pixel_values'], torch.ones(1, 3, 32, 24), atol=1e-5)


def test_sapiens2_pose_tiny_forward_shape_and_checkpoint_schema_keys():
    model = ModelStackSapiens2ForPoseEstimation(_tiny_config()).eval()
    pixel_values = torch.randn(1, 3, 32, 24)

    with torch.no_grad():
        out = model(pixel_values)

    assert out['heatmaps'].shape == (1, 5, 16, 12)
    keys = set(model.state_dict())
    assert 'backbone.blocks.0.attn.wq.weight' in keys
    assert 'backbone.blocks.1.attn.wk.weight' in keys
    assert 'backbone.patch_embed.projection.weight' in keys
    assert 'backbone.rope_embed.periods' in keys
    assert 'decode_head.deconv_layers.0.weight' in keys
    assert 'decode_head.deconv_layers.3.weight' in keys
    assert 'decode_head.conv_layers.6.weight' in keys
    assert 'decode_head.conv_pose.weight' in keys
