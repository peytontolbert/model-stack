from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn


DEFAULT_MODEL = Path('/arxiv/models/facebook/sapiens2-pose-1b')


@dataclass(frozen=True)
class Sapiens2PoseStatus:
    model_id: str
    model_path: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str
    detail: str
    blockers: tuple[str, ...] = ()
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sapiens2PoseArtifacts:
    model: 'ModelStackSapiens2ForPoseEstimation'
    processor: 'Sapiens2PoseImageProcessor'
    config: SimpleNamespace
    status: Sapiens2PoseStatus
    device: torch.device
    dtype: torch.dtype


def sapiens2_pose_status(model_path: str | Path = DEFAULT_MODEL, *, model_id: str = 'facebook/sapiens2-pose-1b') -> Sapiens2PoseStatus:
    path = Path(model_path)
    expected = {
        'config.json': path / 'config.json',
        'preprocessor_config.json': path / 'preprocessor_config.json',
        'model.safetensors': path / 'model.safetensors',
    }
    present = tuple(name for name, artifact in expected.items() if artifact.is_file())
    missing = tuple(name for name, artifact in expected.items() if not artifact.is_file())
    if missing:
        return Sapiens2PoseStatus(
            model_id=model_id,
            model_path=str(path),
            status='incomplete_sapiens2_pose_checkpoint',
            runnable=False,
            preferred_env='ai',
            loader='runtime.sapiens2_pose_bridge.load_sapiens2_pose_model',
            recommended_dtype='bfloat16',
            detail='Sapiens2 pose checkpoint is missing required local artifacts.',
            blockers=tuple(f'missing artifact: {name}' for name in missing),
            present_artifacts=present,
            missing_artifacts=missing,
        )
    try:
        config = _load_config(path / 'config.json')
        if getattr(config, 'model_type', None) != 'sapiens2':
            raise ValueError(f"unexpected model_type={getattr(config, 'model_type', None)!r}")
        if 'Sapiens2ForPoseEstimation' not in tuple(getattr(config, 'architectures', ()) or ()):  # type: ignore[arg-type]
            raise ValueError('config.architectures does not include Sapiens2ForPoseEstimation')
    except Exception as exc:
        return Sapiens2PoseStatus(
            model_id=model_id,
            model_path=str(path),
            status='invalid_sapiens2_pose_config',
            runnable=False,
            preferred_env='ai',
            loader='runtime.sapiens2_pose_bridge.load_sapiens2_pose_model',
            recommended_dtype='bfloat16',
            detail=f'Sapiens2 pose config could not be parsed by model-stack: {type(exc).__name__}:{exc}',
            blockers=(f'{type(exc).__name__}:{exc}',),
            present_artifacts=present,
            missing_artifacts=missing,
        )
    return Sapiens2PoseStatus(
        model_id=model_id,
        model_path=str(path),
        status='verified_sapiens2_pose_load_bridge',
        runnable=True,
        preferred_env='ai',
        loader='runtime.sapiens2_pose_bridge.load_sapiens2_pose_model',
        recommended_dtype='bfloat16',
        detail=(
            'Local Sapiens2 pose checkpoint loads through model-stack with the original checkpoint schema, '
            'avoiding the unavailable Transformers 5.x Sapiens2 classes. BF16 load-only smoke on cuda:0 passed '
            'in 164.64s with about 2.9GB allocated after load.'
        ),
        blockers=(),
        present_artifacts=present,
        missing_artifacts=missing,
    )


def load_sapiens2_pose_model(
    model_path: str | Path = DEFAULT_MODEL,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = 'bfloat16',
    low_cpu_mem_usage: bool = True,
    strict: bool = True,
) -> Sapiens2PoseArtifacts:
    path = Path(model_path)
    status = sapiens2_pose_status(path)
    if not status.runnable:
        raise RuntimeError('; '.join(status.blockers))
    config = _load_config(path / 'config.json')
    preprocessor = _load_preprocessor_config(path / 'preprocessor_config.json')
    resolved_dtype = _resolve_dtype(dtype)
    resolved_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = ModelStackSapiens2ForPoseEstimation(config)
    checkpoint = path / 'model.safetensors'
    if low_cpu_mem_usage:
        try:
            from accelerate import load_checkpoint_and_dispatch

            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=str(checkpoint),
                device_map={'': str(resolved_device)},
                dtype=resolved_dtype,
                strict=strict,
            )
        except Exception:
            state = load_file(str(checkpoint), device='cpu')
            missing, unexpected = model.load_state_dict(state, strict=strict)
            if strict and (missing or unexpected):
                raise RuntimeError(f'Sapiens2 state mismatch missing={missing} unexpected={unexpected}')
            model = model.to(device=resolved_device, dtype=resolved_dtype)
    else:
        state = load_file(str(checkpoint), device='cpu')
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f'Sapiens2 state mismatch missing={missing} unexpected={unexpected}')
        model = model.to(device=resolved_device, dtype=resolved_dtype)
    model.eval()
    processor = Sapiens2PoseImageProcessor(preprocessor)
    return Sapiens2PoseArtifacts(model=model, processor=processor, config=config, status=status, device=resolved_device, dtype=resolved_dtype)


class Sapiens2PoseImageProcessor:
    def __init__(self, config: SimpleNamespace):
        self.config = config
        self.size = _size_hw(config.size)
        self.image_mean = torch.tensor(config.image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(config.image_std, dtype=torch.float32).view(3, 1, 1)
        self.rescale_factor = float(config.rescale_factor)

    def __call__(self, images: torch.Tensor | Any, *, return_tensors: str = 'pt') -> dict[str, torch.Tensor]:
        if return_tensors != 'pt':
            raise ValueError("Sapiens2PoseImageProcessor only supports return_tensors='pt'")
        tensor = self._to_bchw(images).float()
        if getattr(self.config, 'do_rescale', True) and tensor.max() > 2.0:
            tensor = tensor * self.rescale_factor
        if getattr(self.config, 'do_resize', True):
            tensor = F.interpolate(tensor, size=self.size, mode='bilinear', align_corners=False)
        if getattr(self.config, 'do_normalize', True):
            mean = self.image_mean.to(device=tensor.device, dtype=tensor.dtype)
            std = self.image_std.to(device=tensor.device, dtype=tensor.dtype)
            tensor = (tensor - mean) / std
        return {'pixel_values': tensor.contiguous()}

    @staticmethod
    def _to_bchw(images: torch.Tensor | Any) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            import numpy as np
            from PIL import Image

            if isinstance(images, Image.Image):
                images = np.asarray(images.convert('RGB'))
            images = torch.as_tensor(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f'expected image tensor with 3 or 4 dims, got shape {tuple(images.shape)}')
        if images.shape[1] == 3:
            return images
        if images.shape[-1] == 3:
            return images.permute(0, 3, 1, 2)
        raise ValueError(f'expected channel dimension of 3, got shape {tuple(images.shape)}')


class Sapiens2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight.to(dtype=dtype) * x_float.to(dtype=dtype)


class Sapiens2LayerScale(nn.Module):
    def __init__(self, hidden_size: int, value: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((hidden_size,), float(value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Sapiens2PatchEmbed(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class Sapiens2RopeEmbedding(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        periods = config.rope_theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32)
        self.register_buffer('periods', periods, persistent=True)
        self.patch_size = int(config.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        hp = height // self.patch_size
        wp = width // self.patch_size
        device = pixel_values.device
        y = (torch.arange(hp, device=device, dtype=torch.float32) + 0.5) / hp
        x = (torch.arange(wp, device=device, dtype=torch.float32) + 0.5) / wp
        coords = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1).reshape(-1, 2)
        coords = coords.mul(2.0).sub(1.0)
        angles = 2.0 * math.pi * coords[:, :, None] / self.periods.to(device=device, dtype=torch.float32)[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.cos(angles).to(dtype=pixel_values.dtype), torch.sin(angles).to(dtype=pixel_values.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_patches = cos.shape[-2]
    num_prefix = q.shape[-2] - num_patches
    q_prefix, q_patch = q.split((num_prefix, num_patches), dim=-2)
    k_prefix, k_patch = k.split((num_prefix, num_patches), dim=-2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_patch = q_patch * cos + _rotate_half(q_patch) * sin
    k_patch = k_patch * cos + _rotate_half(k_patch) * sin
    return torch.cat((q_prefix, q_patch), dim=-2), torch.cat((k_prefix, k_patch), dim=-2)


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


class Sapiens2Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.hidden_size) // self.num_heads
        self.num_key_value_heads = int(config.num_key_value_heads_per_layer[layer_idx])
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.wq = nn.Linear(config.hidden_size, config.hidden_size, bias=bool(config.query_bias))
        self.wk = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bool(config.key_bias))
        self.wv = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bool(config.value_bias))
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bool(config.proj_bias))
        self.q_norm = Sapiens2RMSNorm(self.head_dim, eps=float(config.rms_norm_eps)) if config.use_qk_norm else nn.Identity()
        self.k_norm = Sapiens2RMSNorm(self.head_dim, eps=float(config.rms_norm_eps)) if config.use_qk_norm else nn.Identity()
        self.gamma = Sapiens2LayerScale(config.hidden_size, value=float(config.layerscale_value))

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.wq(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rope(q, k, *position_embeddings)
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.gamma(self.proj(out))


class Sapiens2FeedForward(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.w12 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=bool(config.mlp_bias))
        self.w3 = nn.Linear(config.intermediate_size, config.hidden_size, bias=bool(config.mlp_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * up)


class Sapiens2Block(nn.Module):
    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.ln1 = Sapiens2RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        self.attn = Sapiens2Attention(config, layer_idx)
        self.ln2 = Sapiens2RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        self.ffn = Sapiens2FeedForward(config)

    def forward(self, x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), position_embeddings)
        x = x + self.ffn(self.ln2(x))
        return x


class Sapiens2Backbone(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.storage_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.patch_embed = Sapiens2PatchEmbed(config)
        self.rope_embed = Sapiens2RopeEmbedding(config)
        self.blocks = nn.ModuleList(Sapiens2Block(config, i) for i in range(config.num_hidden_layers))
        self.ln1 = Sapiens2RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = pixel_values.shape
        patches = self.patch_embed(pixel_values)
        cls = self.cls_token.expand(bsz, -1, -1)
        storage = self.storage_tokens.expand(bsz, -1, -1)
        x = torch.cat((cls, storage, patches), dim=1)
        position_embeddings = self.rope_embed(pixel_values)
        for block in self.blocks:
            x = block(x, position_embeddings)
        patch_tokens = self.ln1(x[:, 1 + self.config.num_register_tokens :, :])
        hp = height // self.config.patch_size
        wp = width // self.config.patch_size
        return patch_tokens.transpose(1, 2).reshape(bsz, self.config.hidden_size, hp, wp)


class Sapiens2DecodeHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        head = config.head_config
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, head.upsample_out_channels[0], kernel_size=head.upsample_kernel_sizes[0], stride=2, padding=1, bias=False),
            nn.Identity(),
            nn.SiLU(),
            nn.ConvTranspose2d(head.upsample_out_channels[0], head.upsample_out_channels[1], kernel_size=head.upsample_kernel_sizes[1], stride=2, padding=1, bias=False),
            nn.Identity(),
            nn.SiLU(),
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(head.upsample_out_channels[-1], head.conv_out_channels[0], kernel_size=head.conv_kernel_sizes[0], bias=True),
            nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(head.conv_out_channels[0], head.conv_out_channels[1], kernel_size=head.conv_kernel_sizes[1], bias=True),
            nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(head.conv_out_channels[1], head.conv_out_channels[2], kernel_size=head.conv_kernel_sizes[2], bias=True),
            nn.Identity(),
            nn.SiLU(),
        )
        self.conv_pose = nn.Conv2d(head.conv_out_channels[-1], config.num_labels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        return self.conv_pose(x)


class ModelStackSapiens2ForPoseEstimation(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.backbone = Sapiens2Backbone(config)
        self.decode_head = Sapiens2DecodeHead(config)

    @torch.no_grad()
    def predict_heatmaps(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(pixel_values)['heatmaps']

    def forward(self, pixel_values: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        features = self.backbone(pixel_values)
        heatmaps = self.decode_head(features)
        return {'heatmaps': heatmaps}


def _size_hw(size: Any) -> tuple[int, int]:
    if isinstance(size, dict):
        return int(size['height']), int(size['width'])
    return int(size.height), int(size.width)


def _load_config(path: Path) -> SimpleNamespace:
    data = json.loads(path.read_text(encoding='utf-8'))
    data.setdefault('num_labels', 308)
    data['head_config'] = _namespace(data.get('head_config', {}))
    return _namespace(data)


def _load_preprocessor_config(path: Path) -> SimpleNamespace:
    return _namespace(json.loads(path.read_text(encoding='utf-8')))


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'fp32': torch.float32,
        'float32': torch.float32,
    }.get(str(dtype).lower(), torch.bfloat16)


def status_to_json(status: Sapiens2PoseStatus) -> str:
    return json.dumps(asdict(status), indent=2, sort_keys=True)
