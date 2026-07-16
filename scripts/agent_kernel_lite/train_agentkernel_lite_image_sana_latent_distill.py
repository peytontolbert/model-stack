#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterator

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import utils


DEFAULT_TEACHER = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers"

IMAGENETTE_LABELS = {
    0: "tench fish",
    1: "english springer dog",
    2: "cassette player",
    3: "chain saw",
    4: "church building",
    5: "french horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute",
}


@dataclass
class SanaLatentStudentConfig:
    resolution: int = 512
    vae_scale_factor: int = 32
    latent_channels: int = 32
    patch_size: int = 1
    dim: int = 512
    depth: int = 10
    heads: int = 8
    mlp_ratio: int = 4
    prompt_dim: int = 2304
    max_sequence_length: int = 300

    @property
    def latent_size(self) -> int:
        return self.resolution // self.vae_scale_factor


def install_local_diffusers() -> None:
    candidates = [
        os.environ.get("DIFFUSERS_SRC", ""),
        "/data/webgl-game/repos/diffusers/src",
        "/data/repositories/diffusers/src",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def clean_prompt(value: Any, min_words: int = 5, max_chars: int = 420) -> str:
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item), "")
    text = " ".join(str(value or "").replace("\x00", "").split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    if len(text.split()) < min_words:
        return ""
    lowered = text.lower()
    if any(term in lowered for term in ("nsfw", "nude", "naked", "porn", "gore")):
        return ""
    return text


def prompt_matches_filter(prompt: str, require_any: str = "", exclude_any: str = "") -> bool:
    lowered = prompt.lower()
    required = [item.strip().lower() for item in require_any.split(",") if item.strip()]
    excluded = [item.strip().lower() for item in exclude_any.split(",") if item.strip()]

    def matches(term: str) -> bool:
        if " " in term:
            return term in lowered
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered) is not None

    if required and not any(matches(term) for term in required):
        return False
    if excluded and any(matches(term) for term in excluded):
        return False
    return True


def prompt_from_label(row: dict[str, Any], dataset_name: str) -> str:
    if "label" not in row:
        return ""
    try:
        label = int(row["label"])
    except (TypeError, ValueError):
        return ""
    if "imagenette" in dataset_name.lower():
        name = IMAGENETTE_LABELS.get(label, "")
        if name:
            return f"realistic photo of one {name}, single main object visible, physically plausible shape"
    return ""


def prompt_stream(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    if args.prompt_file:
        path = Path(args.prompt_file)
        index = 0
        while True:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    if path.suffix.lower() == ".jsonl":
                        row = json.loads(line)
                        prompt = clean_prompt(
                            row.get("prompt", row.get("text", row.get("caption", ""))),
                            min_words=args.min_prompt_words,
                            max_chars=args.max_prompt_chars,
                        )
                        row_seed = row.get("seed")
                    else:
                        prompt = clean_prompt(line, min_words=args.min_prompt_words, max_chars=args.max_prompt_chars)
                        row_seed = None
                    if prompt and prompt_matches_filter(prompt, args.prompt_require_any, args.prompt_exclude_any):
                        item = {"prompt": prompt, "source_index": index, "source_dataset": str(path)}
                        if row_seed is not None:
                            item["seed"] = int(row_seed)
                        if isinstance(row, dict):
                            for meta_key in ("label", "object_label", "class_label", "view", "view_label", "curation"):
                                if row.get(meta_key) is not None:
                                    item[meta_key] = row[meta_key]
                            for key in ("teacher_ref", "image_ref", "image_path", "path"):
                                if row.get(key):
                                    item["teacher_ref"] = str(row[key])
                                    break
                        yield item
                        index += 1

    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {"split": args.prompt_split, "streaming": True}
    if args.prompt_trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if args.prompt_config:
        dataset = load_dataset(args.prompt_dataset, args.prompt_config, **load_kwargs)
    else:
        dataset = load_dataset(args.prompt_dataset, **load_kwargs)
    if args.stream_shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.stream_shuffle_buffer)
    columns = [column.strip() for column in args.prompt_columns.split(",") if column.strip()]
    index = 0
    for row in dataset:
        prompt = ""
        for column in columns:
            if column in row:
                prompt = clean_prompt(row[column], min_words=args.min_prompt_words, max_chars=args.max_prompt_chars)
            image_nsfw = float(row.get("image_nsfw") or 0.0)
            prompt_nsfw = float(row.get("prompt_nsfw") or 0.0)
            if prompt and image_nsfw <= args.max_nsfw_score and prompt_nsfw <= args.max_nsfw_score:
                break
            prompt = ""
        if not prompt:
            prompt = clean_prompt(
                prompt_from_label(row, args.prompt_dataset),
                min_words=args.min_prompt_words,
                max_chars=args.max_prompt_chars,
            )
        if prompt and prompt_matches_filter(prompt, args.prompt_require_any, args.prompt_exclude_any):
            item = {"prompt": prompt, "source_index": index, "source_dataset": args.prompt_dataset}
            if args.stream_image_column and row.get(args.stream_image_column) is not None:
                item["teacher_image"] = row.get(args.stream_image_column)
            yield item
        index += 1


def linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def ternary_weight_ste(
    weight: torch.Tensor,
    threshold_ratio: float = 0.7,
    learned_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    flat = weight.flatten(1)
    base_scale = flat.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    threshold = float(threshold_ratio) * base_scale
    ternary = torch.where(flat > threshold, torch.ones_like(flat), torch.where(flat < -threshold, -torch.ones_like(flat), torch.zeros_like(flat)))
    if learned_scale is None:
        scale = base_scale
    else:
        scale = learned_scale.flatten().to(dtype=weight.dtype, device=weight.device).view(-1, 1).clamp_min(1e-6)
    quantized = ternary * scale
    quantized = quantized.view_as(weight)
    return weight + (quantized - weight).detach()


class TrainableBitNetLinear(nn.Linear):
    bitnet_qat = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold_ratio: float = 0.7,
        learned_scale: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.threshold_ratio = float(threshold_ratio)
        if learned_scale:
            self.weight_scale = nn.Parameter(torch.ones(out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, ternary_weight_ste(self.weight, self.threshold_ratio, getattr(self, "weight_scale", None)), self.bias)


class TrainableBitNetConv2d(nn.Conv2d):
    bitnet_qat = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        threshold_ratio: float = 0.7,
        learned_scale: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.threshold_ratio = float(threshold_ratio)
        if learned_scale:
            self.weight_scale = nn.Parameter(torch.ones(out_channels))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = ternary_weight_ste(self.weight, self.threshold_ratio, getattr(self, "weight_scale", None))
        return self._conv_forward(input, weight, self.bias)


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern and pattern in name for pattern in patterns)


def apply_bitnet_qat_modules(
    module: nn.Module,
    *,
    threshold_ratio: float,
    learned_scale: bool = False,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    prefix: str = "",
) -> int:
    converted = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        include_match = not include or _matches_any(full_name, include)
        exclude_match = _matches_any(full_name, exclude)
        if include_match and not exclude_match and isinstance(child, nn.Linear):
            replacement = TrainableBitNetLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                threshold_ratio=threshold_ratio,
                learned_scale=learned_scale,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                if hasattr(replacement, "weight_scale"):
                    replacement.weight_scale.copy_(child.weight.flatten(1).abs().mean(dim=1).clamp_min(1e-6))
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, child_name, replacement)
            converted += 1
            continue
        if include_match and not exclude_match and isinstance(child, nn.Conv2d):
            replacement = TrainableBitNetConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                threshold_ratio=threshold_ratio,
                learned_scale=learned_scale,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                if hasattr(replacement, "weight_scale"):
                    replacement.weight_scale.copy_(child.weight.flatten(1).abs().mean(dim=1).clamp_min(1e-6))
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, child_name, replacement)
            converted += 1
            continue
        converted += apply_bitnet_qat_modules(
            child,
            threshold_ratio=threshold_ratio,
            learned_scale=learned_scale,
            include=include,
            exclude=exclude,
            prefix=full_name,
        )
    return converted


@torch.no_grad()
def materialize_bitnet_qat_weights(module: nn.Module) -> int:
    converted = 0
    for child in module.children():
        if isinstance(child, (TrainableBitNetLinear, TrainableBitNetConv2d)):
            child.weight.copy_(ternary_weight_ste(child.weight, child.threshold_ratio, getattr(child, "weight_scale", None)))
            converted += 1
        converted += materialize_bitnet_qat_weights(child)
    return converted


def freeze_except_bitnet_qat_modules(module: nn.Module) -> int:
    trainable = 0
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for child in module.modules():
        if bool(getattr(child, "bitnet_qat", False)):
            for parameter in child.parameters(recurse=False):
                parameter.requires_grad_(True)
                trainable += int(parameter.numel())
    return trainable


def freeze_except_named_parameters(module: nn.Module, patterns: tuple[str, ...]) -> int:
    trainable = 0
    for name, parameter in module.named_parameters():
        enabled = _matches_any(name, patterns)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable += int(parameter.numel())
    return trainable


def enable_named_parameters(module: nn.Module, patterns: tuple[str, ...]) -> int:
    trainable = 0
    for name, parameter in module.named_parameters():
        if _matches_any(name, patterns):
            parameter.requires_grad_(True)
            trainable += int(parameter.numel())
    return trainable


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = linear(dim, dim)
        self.k = linear(dim, dim)
        self.v = linear(dim, dim)
        self.out = linear(dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, dim = x.shape
        context_tokens = context.shape[1]
        q = self.q(x).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(batch, context_tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(batch, context_tokens, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].to(torch.bool)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(y.transpose(1, 2).reshape(batch, tokens, dim))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = linear(dim, dim * 3)
        self.out = linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(batch, tokens, dim))


class SanaLatentBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(linear(dim, hidden), nn.GELU(), linear(hidden, dim))
        self.cond = nn.Sequential(nn.SiLU(), linear(dim, dim * 6))

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, prompt_mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2, shift3, scale3 = self.cond(cond).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        x = x + self.self_attn(y)
        y = self.norm2(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        x = x + self.cross_attn(y, prompt, prompt_mask)
        y = self.norm3(x) * (1 + scale3[:, None, :]) + shift3[:, None, :]
        return x + self.mlp(y)


class SanaLatentStudent(nn.Module):
    def __init__(self, config: SanaLatentStudentConfig) -> None:
        super().__init__()
        self.config = config
        latent_size = config.latent_size
        if latent_size % config.patch_size:
            raise ValueError("latent size must be divisible by patch size")
        patch_dim = config.latent_channels * config.patch_size * config.patch_size
        side = latent_size // config.patch_size
        self.side = side
        self.patch_in = linear(patch_dim, config.dim)
        self.patch_out = linear(config.dim, patch_dim)
        self.pos = nn.Parameter(torch.zeros(1, side * side, config.dim))
        self.time_mlp = nn.Sequential(linear(config.dim, config.dim), nn.SiLU(), linear(config.dim, config.dim))
        self.prompt_proj = nn.Sequential(linear(config.prompt_dim, config.dim), nn.GELU(), linear(config.dim, config.dim))
        self.blocks = nn.ModuleList([SanaLatentBlock(config.dim, config.heads, config.mlp_ratio) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.dim)
        nn.init.normal_(self.pos, std=0.02)

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        p = self.config.patch_size
        return F.unfold(z, kernel_size=p, stride=p).transpose(1, 2)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.config.patch_size
        return F.fold(
            patches.transpose(1, 2),
            output_size=(self.config.latent_size, self.config.latent_size),
            kernel_size=p,
            stride=p,
        )

    def forward(
        self,
        z: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.patch_in(self.patchify(z)) + self.pos
        prompt_tokens = self.prompt_proj(prompt_embeds.float())
        cond = self.time_mlp(timestep_embedding(timestep.float(), self.config.dim))
        for block in self.blocks:
            tokens = block(tokens, prompt_tokens, prompt_attention_mask, cond)
        return self.unpatchify(self.patch_out(self.norm(tokens)))


def load_teacher(args: argparse.Namespace) -> Any:
    install_local_diffusers()
    from diffusers import SanaPipeline

    dtype = torch.float16 if args.teacher_dtype == "float16" else torch.bfloat16 if args.teacher_dtype == "bfloat16" else torch.float32
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if args.local_files_only:
        kwargs["local_files_only"] = True
    teacher = SanaPipeline.from_pretrained(args.teacher_model, **kwargs).to(args.teacher_device)
    teacher.set_progress_bar_config(disable=True)
    teacher.transformer.eval()
    teacher.text_encoder.eval()
    teacher.vae.eval()
    for module in (teacher.transformer, teacher.text_encoder, teacher.vae):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return teacher


def make_student(config: SanaLatentStudentConfig, args: argparse.Namespace) -> nn.Module:
    if args.student_architecture == "sana_transformer":
        install_local_diffusers()
        from diffusers.models import SanaTransformer2DModel

        inner_dim = args.sana_num_attention_heads * args.sana_attention_head_dim
        return SanaTransformer2DModel(
            in_channels=config.latent_channels,
            out_channels=config.latent_channels,
            num_attention_heads=args.sana_num_attention_heads,
            attention_head_dim=args.sana_attention_head_dim,
            num_layers=args.sana_num_layers,
            num_cross_attention_heads=args.sana_num_cross_attention_heads,
            cross_attention_head_dim=args.sana_cross_attention_head_dim,
            cross_attention_dim=inner_dim,
            caption_channels=config.prompt_dim,
            mlp_ratio=args.sana_mlp_ratio,
            sample_size=config.latent_size,
            patch_size=config.patch_size,
            qk_norm=args.sana_qk_norm or None,
            timestep_scale=0.001,
        )
    return SanaLatentStudent(config)


def select_teacher_layers(teacher_layers: int, student_layers: int) -> list[int]:
    if student_layers < 1:
        raise ValueError("student_layers must be positive")
    if student_layers == 1:
        return [teacher_layers - 1]
    return [round(index * (teacher_layers - 1) / (student_layers - 1)) for index in range(student_layers)]


@torch.no_grad()
def initialize_sana_student_from_teacher(student: nn.Module, teacher_transformer: nn.Module) -> dict[str, Any]:
    teacher_blocks = len(teacher_transformer.transformer_blocks)
    student_blocks = len(student.transformer_blocks)
    layer_map = select_teacher_layers(teacher_blocks, student_blocks)
    teacher_state = teacher_transformer.state_dict()
    student_state = student.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    remapped_state: dict[str, torch.Tensor] = {}
    for key, value in student_state.items():
        source_key = key
        if key.startswith("transformer_blocks."):
            parts = key.split(".")
            student_layer = int(parts[1])
            parts[1] = str(layer_map[student_layer])
            source_key = ".".join(parts)
        source_value = teacher_state.get(source_key)
        if source_value is not None and tuple(source_value.shape) == tuple(value.shape):
            remapped_state[key] = source_value.to(device=value.device, dtype=value.dtype)
            copied.append(f"{source_key}->{key}" if source_key != key else key)
        else:
            remapped_state[key] = value
            skipped.append(key)
    student.load_state_dict(remapped_state, strict=True)
    return {
        "teacher_layers": teacher_blocks,
        "student_layers": student_blocks,
        "layer_map": layer_map,
        "copied_tensors": len(copied),
        "skipped_tensors": skipped,
    }


def student_predict(
    student: nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.student_architecture == "sana_transformer":
        dtype = next(student.parameters()).dtype
        return student(
            latents.to(dtype=dtype),
            encoder_hidden_states=prompt_embeds.to(dtype=dtype),
            encoder_attention_mask=prompt_mask,
            timestep=timesteps.float() * student.config.timestep_scale,
            return_dict=False,
        )[0].float()
    return student(latents, timesteps.float(), prompt_embeds, prompt_mask)


def student_predict_cfg(
    student: nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_mask: torch.Tensor,
    guidance_scale: float,
    args: argparse.Namespace,
) -> torch.Tensor:
    cond = student_predict(student, latents, timesteps, prompt_embeds, prompt_mask, args)
    if guidance_scale <= 1.0:
        return cond
    uncond = student_predict(student, latents, timesteps, negative_prompt_embeds, negative_prompt_mask, args)
    return uncond + float(guidance_scale) * (cond - uncond)


@torch.no_grad()
def encode_prompts(teacher: Any, prompts: list[str], args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = teacher.encode_prompt(
        prompts,
        do_classifier_free_guidance=True,
        negative_prompt="",
        num_images_per_prompt=1,
        device=torch.device(args.teacher_device),
        clean_caption=False,
        max_sequence_length=args.max_sequence_length,
    )
    return (
        prompt_embeds.float().clone(),
        prompt_attention_mask.clone(),
        negative_prompt_embeds.float().clone(),
        negative_prompt_attention_mask.clone(),
    )


@torch.no_grad()
def teacher_transformer_predict(
    teacher: Any,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> torch.Tensor:
    transformer_dtype = teacher.transformer.dtype
    scaled_t = timesteps.to(latents.device) * teacher.transformer.config.timestep_scale
    pred = teacher.transformer(
        latents.to(dtype=transformer_dtype),
        encoder_hidden_states=prompt_embeds.to(device=latents.device, dtype=transformer_dtype),
        encoder_attention_mask=prompt_attention_mask.to(latents.device),
        timestep=scaled_t,
        return_dict=False,
    )[0]
    pred = pred.float()
    if teacher.transformer.config.out_channels // 2 == teacher.transformer.config.in_channels:
        pred = pred.chunk(2, dim=1)[0]
    return pred.clone()


@torch.no_grad()
def teacher_predict(
    teacher: Any,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_prompt_attention_mask: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    cond = teacher_transformer_predict(teacher, latents, timesteps, prompt_embeds, prompt_attention_mask)
    if guidance_scale <= 1.0 or negative_prompt_embeds is None or negative_prompt_attention_mask is None:
        return cond
    uncond = teacher_transformer_predict(
        teacher,
        latents,
        timesteps,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    )
    return uncond + float(guidance_scale) * (cond - uncond)


@torch.no_grad()
def teacher_trajectory_targets(
    teacher: Any,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_attention_mask: torch.Tensor,
    args: argparse.Namespace,
    config: SanaLatentStudentConfig,
    step: int,
    seed_override: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    device = torch.device(args.teacher_device)
    seed = seed_override if seed_override is not None else args.fixed_teacher_seed if args.fixed_teacher_seed >= 0 else args.seed + step
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        prompt_embeds.shape[0],
        config.latent_channels,
        config.latent_size,
        config.latent_size,
        generator=generator,
        device=device,
    )
    teacher.scheduler.set_timesteps(args.trajectory_steps, device=device)
    examples = []
    for timestep_value in teacher.scheduler.timesteps:
        timesteps = timestep_value.expand(latents.shape[0]).to(device)
        pred = teacher_predict(
            teacher,
            latents,
            timesteps,
            prompt_embeds.to(device),
            prompt_attention_mask.to(device),
            negative_prompt_embeds.to(device),
            negative_prompt_attention_mask.to(device),
            args.teacher_guidance if args.distill_guided_targets else 1.0,
        )
        next_latents = teacher.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
        examples.append((latents.float().clone(), timesteps.float().clone(), pred.float().clone(), next_latents.float().clone()))
        latents = next_latents
    return examples


@torch.no_grad()
def generate_teacher_final_latents(
    teacher: Any,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    args: argparse.Namespace,
    step: int,
) -> torch.Tensor:
    seed = args.fixed_teacher_seed if args.fixed_teacher_seed >= 0 else args.seed + step
    generator = torch.Generator(device=args.teacher_device).manual_seed(seed)
    result = teacher(
        prompt=None,
        prompt_embeds=prompt_embeds.to(teacher.transformer.device, dtype=teacher.transformer.dtype),
        prompt_attention_mask=prompt_attention_mask.to(teacher.transformer.device),
        guidance_scale=args.teacher_guidance,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.teacher_steps,
        generator=generator,
        output_type="latent",
        return_dict=True,
        use_resolution_binning=not args.disable_resolution_binning,
        max_sequence_length=args.max_sequence_length,
    ).images
    return result.float().clone()


@torch.no_grad()
def save_latent_image(teacher: Any, latents: torch.Tensor, path: Path) -> None:
    image_latents = latents.to(teacher.vae.device, dtype=teacher.vae.dtype)
    decoded = teacher.vae.decode(image_latents / teacher.vae.config.scaling_factor, return_dict=False)[0]
    image = teacher.image_processor.postprocess(decoded, output_type="pt")[0]
    utils.save_image(image, path)


def scheduler_step_with_teacher_history(
    scheduler: Any,
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]],
    example_index: int,
    pred: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    replay_scheduler = copy.deepcopy(scheduler)
    replay_scheduler.set_timesteps(args.trajectory_steps, device=device)
    for replay_index in range(example_index):
        replay_latents, _, replay_target, _ = examples[replay_index]
        replay_scheduler.step(
            replay_target.to(device),
            replay_scheduler.timesteps[replay_index],
            replay_latents.to(device),
            return_dict=False,
        )
    latents, _, _, _ = examples[example_index]
    return replay_scheduler.step(
        pred,
        replay_scheduler.timesteps[example_index],
        latents.to(device),
        return_dict=False,
    )[0]


def student_rollout_latents(
    scheduler: Any,
    student: nn.Module,
    initial_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_prompt_mask: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    replay_scheduler = copy.deepcopy(scheduler)
    replay_scheduler.set_timesteps(args.trajectory_steps, device=device)
    latents = initial_latents.to(device)
    for timestep_value in replay_scheduler.timesteps:
        timesteps = timestep_value.expand(latents.shape[0]).to(device)
        if (
            args.train_student_cfg_guidance > 1.0
            and negative_prompt_embeds is not None
            and negative_prompt_mask is not None
        ):
            pred = student_predict_cfg(
                student,
                latents,
                timesteps.float(),
                prompt_embeds,
                prompt_mask,
                negative_prompt_embeds,
                negative_prompt_mask,
                args.train_student_cfg_guidance,
                args,
            )
        else:
            pred = student_predict(student, latents, timesteps.float(), prompt_embeds, prompt_mask, args)
        latents = replay_scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
    return latents


@torch.no_grad()
def save_student_sample(
    teacher: Any,
    student: SanaLatentStudent,
    prompt: str,
    step: int,
    args: argparse.Namespace,
    output_dir: Path,
    seed_override: int | None = None,
) -> None:
    device = torch.device(args.student_device)
    prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = encode_prompts(teacher, [prompt], args)
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)
    negative_prompt_embeds = negative_prompt_embeds.to(device)
    negative_prompt_mask = negative_prompt_mask.to(device)
    if seed_override is not None:
        sample_seed = int(seed_override)
    elif args.fixed_training_noise:
        sample_seed = args.seed + 101
    else:
        sample_seed = args.fixed_teacher_seed if args.fixed_teacher_seed >= 0 else args.seed + step
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    latent_channels = student.config.in_channels if args.student_architecture == "sana_transformer" else student.config.latent_channels
    latent_size = student.config.sample_size if args.student_architecture == "sana_transformer" else student.config.latent_size
    latents = torch.randn(
        1,
        latent_channels,
        latent_size,
        latent_size,
        generator=generator,
        device=device,
    )
    if args.direct_latent_prediction:
        t = torch.zeros((latents.shape[0],), device=device)
        latents = student_predict(student, latents, t, prompt_embeds, prompt_mask, args)
    elif args.teacher_noise_distill:
        scheduler = teacher.scheduler
        scheduler.set_timesteps(args.sample_steps, device=device)
        for t in scheduler.timesteps:
            timestep = t.expand(latents.shape[0]).to(device)
            pred = student_predict_cfg(student, latents, timestep, prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask, args.sample_guidance, args)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    elif args.final_latent_flow:
        for index in range(args.sample_steps):
            t = torch.full((latents.shape[0],), index / max(args.sample_steps - 1, 1), device=device)
            pred = student_predict(student, latents, t * 1000.0, prompt_embeds, prompt_mask, args)
            latents = latents + pred / float(args.sample_steps)
    else:
        scheduler = teacher.scheduler
        scheduler.set_timesteps(args.sample_steps, device=device)
        for t in scheduler.timesteps:
            timestep = t.expand(latents.shape[0]).to(device)
            pred = student_predict_cfg(student, latents, timestep, prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask, args.sample_guidance, args)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    save_latent_image(teacher, latents, output_dir / f"sana_latent_student_step_{step:06d}.png")


def save_checkpoint(output_dir: Path, config: SanaLatentStudentConfig, student: nn.Module, step: int, loss: float, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = student.state_dict()
    materialized_modules = 0
    if args.save_materialized_bitnet:
        materialized = make_student(config, args).to("cpu")
        materialized_modules = apply_bitnet_qat_modules(
            materialized,
            threshold_ratio=args.bitnet_qat_threshold_ratio,
            learned_scale=bool(args.bitnet_qat_learned_scale),
            include=tuple(item.strip() for item in args.bitnet_qat_include.split(",") if item.strip()),
            exclude=tuple(item.strip() for item in args.bitnet_qat_exclude.split(",") if item.strip()),
        )
        materialized.load_state_dict({key: value.detach().cpu() for key, value in state.items()}, strict=True)
        materialize_bitnet_qat_weights(materialized)
        state = {
            key: value
            for key, value in materialized.state_dict().items()
            if not key.endswith(".weight_scale")
        }
    payload = {
        "architecture": "agentkernel-lite-sana-latent-distill-v0",
        "student_architecture": args.student_architecture,
        "mode": "sana_teacher_latent_flow",
        "step": int(step),
        "loss": float(loss),
        "config": asdict(config),
        "teacher_model": args.teacher_model,
        "bitnet_qat": bool(args.bitnet_qat),
        "materialized_bitnet_qat_modules": int(materialized_modules),
        "student": student.state_dict(),
        "student_materialized": state if args.save_materialized_bitnet else None,
    }
    torch.save(payload, output_dir / "sana_latent_student.pt")
    if args.keep_step_checkpoints:
        torch.save(payload, output_dir / f"sana_latent_student_step_{int(step):06d}.pt")


def load_latent_cache_rows(cache_dir: Path) -> list[dict[str, Any]]:
    manifest = cache_dir / "manifest.jsonl"
    if not manifest.exists():
        return []
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("prompt") and row.get("latent_path"):
                    rows.append(row)
    return rows


def save_latent_cache_row(cache_dir: Path, prompt: str, final_latents: torch.Tensor, step: int, source: dict[str, Any]) -> None:
    latent_dir = cache_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    latent_name = f"latent_{step:09d}_{len(list(latent_dir.glob(f'latent_{step:09d}_*.pt'))):03d}.pt"
    latent_path = latent_dir / latent_name
    torch.save(final_latents.detach().cpu().half(), latent_path)
    row = {
        "prompt": prompt,
        "latent_path": str(latent_path.relative_to(cache_dir)),
        "step": int(step),
        "source_dataset": source.get("source_dataset", ""),
        "source_index": source.get("source_index", -1),
    }
    with (cache_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cached_latent(cache_dir: Path, row: dict[str, Any], device: torch.device) -> torch.Tensor:
    latent = torch.load(cache_dir / row["latent_path"], map_location=device)
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    return latent.float()


def tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().pow(2).mean(dim=tuple(range(1, value.ndim)), keepdim=True).sqrt().clamp_min(1e-6)


def direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.float().flatten(1)
    target_flat = target.float().flatten(1)
    return (1.0 - F.cosine_similarity(pred_flat, target_flat, dim=1)).mean()


def norm_ratio_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ratio = tensor_rms(pred) / tensor_rms(target)
    return (ratio - 1.0).pow(2).mean()


def normalized_target_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.float() / tensor_rms(pred), target.float() / tensor_rms(target))


def lowfreq_target_loss(pred: torch.Tensor, target: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = max(int(kernel_size), 1)
    if kernel <= 1:
        return F.mse_loss(pred.float(), target.float())
    return F.mse_loss(
        F.avg_pool2d(pred.float(), kernel_size=kernel, stride=1, padding=kernel // 2),
        F.avg_pool2d(target.float(), kernel_size=kernel, stride=1, padding=kernel // 2),
    )


def decode_latents_for_loss(teacher: Any, latents: torch.Tensor, size: int) -> torch.Tensor:
    image_latents = latents.to(teacher.vae.device, dtype=teacher.vae.dtype)
    decoded = teacher.vae.decode(image_latents / teacher.vae.config.scaling_factor, return_dict=False)[0]
    decoded = decoded.float()
    if size > 0 and decoded.shape[-1] != size:
        decoded = F.interpolate(decoded, size=(size, size), mode="bilinear", align_corners=False)
    return decoded


def load_reference_images_for_loss(rows: list[dict[str, Any]], teacher: Any, args: argparse.Namespace, size: int) -> torch.Tensor | None:
    raw_images = []
    for row in rows:
        if row.get("teacher_image") is not None:
            raw_images.append(row["teacher_image"])
            continue
        raw_path = row.get("teacher_ref")
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return None
        raw_images.append(Image.open(path))
    if not raw_images:
        return None
    images = []
    for raw_image in raw_images:
        image = raw_image.convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
        images.append(image)
    pixel_values = teacher.image_processor.preprocess(images, height=args.resolution, width=args.resolution).float()
    if size > 0 and pixel_values.shape[-1] != size:
        pixel_values = F.interpolate(pixel_values, size=(size, size), mode="bilinear", align_corners=False)
    return pixel_values


def load_reference_image_bank_for_loss(args: argparse.Namespace, teacher: Any, size: int) -> dict[str, dict[str, Any]]:
    if not args.prompt_file:
        return {}
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists() or prompt_path.suffix.lower() != ".jsonl":
        return {}
    bank: dict[str, dict[str, Any]] = {}
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            raw_path = next((row.get(key) for key in ("teacher_ref", "image_ref", "image_path", "path") if row.get(key)), None)
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = Path.cwd() / path
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
            pixel_values = teacher.image_processor.preprocess(
                [image],
                height=args.resolution,
                width=args.resolution,
            ).float()
            if size > 0 and pixel_values.shape[-1] != size:
                pixel_values = F.interpolate(pixel_values, size=(size, size), mode="bilinear", align_corners=False)
            bank[str(path)] = {
                "image": pixel_values[0].detach(),
                "label": str(row.get("label") or row.get("object_label") or row.get("class_label") or ""),
                "view": str(row.get("view") or row.get("view_label") or ""),
                "prompt": str(row.get("prompt") or row.get("text") or row.get("caption") or ""),
            }
    return bank


def select_negative_reference_images(
    reference_bank: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not reference_bank:
        return None
    positive_paths = set()
    for row in rows:
        raw_path = row.get("teacher_ref")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        positive_paths.add(str(path))

    positive_labels = {
        str(row.get("label") or row.get("object_label") or row.get("class_label") or "")
        for row in rows
        if row.get("label") or row.get("object_label") or row.get("class_label")
    }
    positive_views = {
        str(row.get("view") or row.get("view_label") or "")
        for row in rows
        if row.get("view") or row.get("view_label")
    }
    negatives = []
    same_label_different_view = []
    for path, entry in reference_bank.items():
        if path in positive_paths:
            continue
        image = entry["image"]
        label = str(entry.get("label") or "")
        view = str(entry.get("view") or "")
        if label and label in positive_labels:
            if view and positive_views and view not in positive_views:
                same_label_different_view.append(image)
            continue
        negatives.append(image)
    if not negatives and same_label_different_view:
        # Same-object/different-view rows are useful only as a fallback negative:
        # they separate view geometry without pushing object identity away from itself.
        negatives = same_label_different_view
    if not negatives:
        return None
    return torch.stack(negatives, dim=0).to(device=device, dtype=dtype)


@torch.no_grad()
def encode_reference_images_to_latents(rows: list[dict[str, Any]], teacher: Any, args: argparse.Namespace) -> torch.Tensor | None:
    pixel_values = load_reference_images_for_loss(rows, teacher, args, args.resolution)
    if pixel_values is None:
        return None
    pixel_values = pixel_values.to(device=teacher.vae.device, dtype=teacher.vae.dtype)
    latents = teacher.vae.encode(pixel_values).latent
    return (latents * teacher.vae.config.scaling_factor).float().clone()


@torch.no_grad()
def real_image_trajectory_targets(
    teacher: Any,
    clean_latents: torch.Tensor,
    args: argparse.Namespace,
    config: SanaLatentStudentConfig,
    step: int,
    seed_override: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    device = torch.device(args.teacher_device)
    clean_latents = clean_latents.to(device)
    seed = seed_override if seed_override is not None else args.fixed_teacher_seed if args.fixed_teacher_seed >= 0 else args.seed + step
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        clean_latents.shape,
        generator=generator,
        device=device,
    )
    teacher.scheduler.set_timesteps(args.trajectory_steps, device=device)
    replay_scheduler = copy.deepcopy(teacher.scheduler)
    replay_scheduler.set_timesteps(args.trajectory_steps, device=device)
    examples = []
    for index, timestep_value in enumerate(replay_scheduler.timesteps):
        sigma = replay_scheduler.sigmas[index].to(device=device, dtype=clean_latents.dtype)
        sigma = sigma.view(1, 1, 1, 1)
        if bool(getattr(replay_scheduler.config, "use_flow_sigmas", False)):
            alpha = 1.0 - sigma
        else:
            alpha = 1.0 / torch.sqrt(sigma * sigma + 1.0)
            sigma = sigma * alpha
        latents = alpha * clean_latents + sigma * noise
        target = noise - clean_latents
        timesteps = timestep_value.expand(clean_latents.shape[0]).to(device)
        next_latents = replay_scheduler.step(target, timestep_value, latents, return_dict=False)[0]
        examples.append((latents.float().clone(), timesteps.float().clone(), target.float().clone(), next_latents.float().clone()))
    return examples


def estimate_clean_latents_from_velocity(
    scheduler: Any,
    latents: torch.Tensor,
    velocity: torch.Tensor,
    example_index: int,
) -> torch.Tensor:
    sigma = scheduler.sigmas[example_index].to(device=latents.device, dtype=latents.dtype)
    sigma = sigma.view(1, 1, 1, 1)
    if bool(getattr(scheduler.config, "use_flow_sigmas", False)):
        alpha = 1.0 - sigma
        sigma_eff = sigma
    else:
        alpha = 1.0 / torch.sqrt(sigma * sigma + 1.0)
        sigma_eff = sigma * alpha
    # For SANA/flow targets v = noise - clean and z_t = alpha * clean + sigma * noise.
    # Rearranging gives clean = (z_t - sigma * v) / (alpha + sigma).
    return (latents - sigma_eff * velocity) / (alpha + sigma_eff).clamp_min(1e-6)


def decoded_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = max(int(kernel_size), 1)
    if kernel <= 1:
        return F.l1_loss(pred.float(), target.float())
    return F.l1_loss(
        F.avg_pool2d(pred.float(), kernel_size=kernel, stride=1, padding=kernel // 2),
        F.avg_pool2d(target.float(), kernel_size=kernel, stride=1, padding=kernel // 2),
    )


def decoded_moment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    reduce_dims = tuple(range(2, pred.ndim))
    pred_mean = pred.mean(dim=reduce_dims)
    target_mean = target.mean(dim=reduce_dims)
    pred_std = pred.std(dim=reduce_dims, unbiased=False)
    target_std = target.std(dim=reduce_dims, unbiased=False)
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def decoded_highfreq_loss(pred: torch.Tensor, target: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = max(int(kernel_size), 1)
    if kernel <= 1:
        return F.l1_loss(pred.float(), target.float())
    pred = pred.float()
    target = target.float()
    pred_low = F.avg_pool2d(pred, kernel_size=kernel, stride=1, padding=kernel // 2)
    target_low = F.avg_pool2d(target, kernel_size=kernel, stride=1, padding=kernel // 2)
    return F.l1_loss(pred - pred_low, target - target_low)


def decoded_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def decoded_teacher_edge_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_weight: float = 4.0,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    target_dx = F.pad((target[..., :, 1:] - target[..., :, :-1]).abs().mean(dim=1, keepdim=True), (0, 1, 0, 0))
    target_dy = F.pad((target[..., 1:, :] - target[..., :-1, :]).abs().mean(dim=1, keepdim=True), (0, 0, 0, 1))
    edge = target_dx + target_dy
    edge = edge / edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    weights = 1.0 + float(edge_weight) * edge.detach()
    return ((pred - target).abs() * weights).mean()


def decoded_laplacian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(pred.shape[1], 1, 3, 3)
    pred_edges = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    target_edges = F.conv2d(target, kernel, padding=1, groups=target.shape[1])
    return F.l1_loss(pred_edges, target_edges)


def decoded_foreground_mask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.08,
    sharpness: float = 18.0,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    pred_dist = (1.0 - pred).abs().mean(dim=1, keepdim=True)
    target_dist = (1.0 - target).abs().mean(dim=1, keepdim=True)
    pred_mask = torch.sigmoid((pred_dist - float(threshold)) * float(sharpness))
    target_mask = torch.sigmoid((target_dist - float(threshold)) * float(sharpness))
    mask_loss = F.l1_loss(pred_mask, target_mask)

    eps = 1e-6
    height, width = pred_mask.shape[-2:]
    y = torch.linspace(0.0, 1.0, height, device=pred.device, dtype=pred.dtype).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width)

    pred_area = pred_mask.mean(dim=(-2, -1))
    target_area = target_mask.mean(dim=(-2, -1))
    pred_mass = pred_mask.sum(dim=(-2, -1)).clamp_min(eps)
    target_mass = target_mask.sum(dim=(-2, -1)).clamp_min(eps)
    pred_cx = (pred_mask * x).sum(dim=(-2, -1)) / pred_mass
    target_cx = (target_mask * x).sum(dim=(-2, -1)) / target_mass
    pred_cy = (pred_mask * y).sum(dim=(-2, -1)) / pred_mass
    target_cy = (target_mask * y).sum(dim=(-2, -1)) / target_mass

    return (
        mask_loss
        + 0.25 * F.l1_loss(pred_area, target_area)
        + 0.25 * F.l1_loss(pred_cx, target_cx)
        + 0.25 * F.l1_loss(pred_cy, target_cy)
    )


def soft_foreground_mask(image: torch.Tensor, threshold: float, sharpness: float) -> torch.Tensor:
    distance_from_white = (1.0 - image.float()).abs().mean(dim=1, keepdim=True)
    return torch.sigmoid((distance_from_white - float(threshold)) * float(sharpness))


def decoded_structure_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.08,
    sharpness: float = 18.0,
) -> torch.Tensor:
    pred_mask = soft_foreground_mask(pred, threshold, sharpness)
    target_mask = soft_foreground_mask(target, threshold, sharpness).detach()
    eps = 1e-6

    intersection = (pred_mask * target_mask).sum(dim=(-2, -1))
    union = pred_mask.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1))
    dice_loss = 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()

    height, width = pred_mask.shape[-2:]
    y = torch.linspace(0.0, 1.0, height, device=pred.device, dtype=pred.dtype).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width)

    def moments(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = mask.sum(dim=(-2, -1)).clamp_min(eps)
        cx = (mask * x).sum(dim=(-2, -1)) / mass
        cy = (mask * y).sum(dim=(-2, -1)) / mass
        sx = torch.sqrt(((mask * (x - cx[..., None, None]).square()).sum(dim=(-2, -1)) / mass).clamp_min(eps))
        sy = torch.sqrt(((mask * (y - cy[..., None, None]).square()).sum(dim=(-2, -1)) / mass).clamp_min(eps))
        return cx, cy, sx, sy

    pred_cx, pred_cy, pred_sx, pred_sy = moments(pred_mask)
    target_cx, target_cy, target_sx, target_sy = moments(target_mask)
    bbox_loss = (
        F.l1_loss(pred_cx, target_cx)
        + F.l1_loss(pred_cy, target_cy)
        + F.l1_loss(pred_sx, target_sx)
        + F.l1_loss(pred_sy, target_sy)
        + F.l1_loss(pred_sx / pred_sy.clamp_min(eps), target_sx / target_sy.clamp_min(eps))
    )

    pred_x_profile = pred_mask.sum(dim=-2)
    target_x_profile = target_mask.sum(dim=-2)
    pred_y_profile = pred_mask.sum(dim=-1)
    target_y_profile = target_mask.sum(dim=-1)
    pred_x_profile = pred_x_profile / pred_x_profile.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_x_profile = target_x_profile / target_x_profile.sum(dim=-1, keepdim=True).clamp_min(eps)
    pred_y_profile = pred_y_profile / pred_y_profile.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_y_profile = target_y_profile / target_y_profile.sum(dim=-1, keepdim=True).clamp_min(eps)
    projection_loss = F.l1_loss(pred_x_profile, target_x_profile) + F.l1_loss(pred_y_profile, target_y_profile)

    pred_border = torch.cat(
        [
            pred_mask[..., :4, :].flatten(1),
            pred_mask[..., -4:, :].flatten(1),
            pred_mask[..., :, :4].flatten(1),
            pred_mask[..., :, -4:].flatten(1),
        ],
        dim=1,
    )
    target_border = torch.cat(
        [
            target_mask[..., :4, :].flatten(1),
            target_mask[..., -4:, :].flatten(1),
            target_mask[..., :, :4].flatten(1),
            target_mask[..., :, -4:].flatten(1),
        ],
        dim=1,
    )
    border_loss = F.l1_loss(pred_border, target_border)

    return dice_loss + 0.6 * bbox_loss + 2.0 * projection_loss + 0.5 * border_loss


def decoded_structure_embedding(
    image: torch.Tensor,
    threshold: float = 0.08,
    sharpness: float = 18.0,
    lower_dark_threshold: float = 0.45,
    lower_dark_sharpness: float = 16.0,
    grid_size: int = 32,
) -> torch.Tensor:
    image = image.float()
    eps = 1e-6
    mask = soft_foreground_mask(image, threshold, sharpness)
    height, width = mask.shape[-2:]
    y = torch.linspace(0.0, 1.0, height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)

    mass = mask.sum(dim=(-2, -1)).clamp_min(eps)
    area = mask.mean(dim=(-2, -1))
    cx = (mask * x).sum(dim=(-2, -1)) / mass
    cy = (mask * y).sum(dim=(-2, -1)) / mass
    sx = torch.sqrt(((mask * (x - cx[..., None, None]).square()).sum(dim=(-2, -1)) / mass).clamp_min(eps))
    sy = torch.sqrt(((mask * (y - cy[..., None, None]).square()).sum(dim=(-2, -1)) / mass).clamp_min(eps))
    aspect = sx / sy.clamp_min(eps)
    moments = torch.cat([area, cx, cy, sx, sy, aspect], dim=1)

    mask_grid = F.interpolate(mask, size=(grid_size, grid_size), mode="bilinear", align_corners=False).flatten(1)
    x_profile = mask.sum(dim=-2)
    y_profile = mask.sum(dim=-1)
    x_profile = x_profile / x_profile.sum(dim=-1, keepdim=True).clamp_min(eps)
    y_profile = y_profile / y_profile.sum(dim=-1, keepdim=True).clamp_min(eps)
    x_profile = F.interpolate(x_profile, size=grid_size, mode="linear", align_corners=False).flatten(1)
    y_profile = F.interpolate(y_profile, size=grid_size, mode="linear", align_corners=False).flatten(1)

    lower = image[..., height // 2 :, :]
    lower_dark = 1.0 - lower.mean(dim=1, keepdim=True)
    lower_dark_mask = torch.sigmoid((lower_dark - float(lower_dark_threshold)) * float(lower_dark_sharpness))
    lower_profile = lower_dark_mask.sum(dim=-2)
    lower_profile = lower_profile / lower_profile.sum(dim=-1, keepdim=True).clamp_min(eps)
    lower_profile = F.interpolate(lower_profile, size=grid_size, mode="linear", align_corners=False).flatten(1)

    embedding = torch.cat([moments, mask_grid, x_profile, y_profile, lower_profile], dim=1)
    return F.normalize(embedding, dim=1, eps=eps)


def decoded_structure_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    negatives: torch.Tensor | None,
    threshold: float = 0.08,
    sharpness: float = 18.0,
    lower_dark_threshold: float = 0.45,
    lower_dark_sharpness: float = 16.0,
    margin: float = 0.08,
) -> torch.Tensor:
    pred_embed = decoded_structure_embedding(
        pred,
        threshold,
        sharpness,
        lower_dark_threshold,
        lower_dark_sharpness,
    )
    target_embed = decoded_structure_embedding(
        target,
        threshold,
        sharpness,
        lower_dark_threshold,
        lower_dark_sharpness,
    ).detach()
    positive = (pred_embed - target_embed).square().mean(dim=1)
    if negatives is None or negatives.numel() == 0:
        return positive.mean()
    negative_embed = decoded_structure_embedding(
        negatives.to(device=pred.device, dtype=pred.dtype),
        threshold,
        sharpness,
        lower_dark_threshold,
        lower_dark_sharpness,
    ).detach()
    negative = (pred_embed[:, None, :] - negative_embed[None, :, :]).square().mean(dim=-1)
    return (positive.mean() + F.relu(float(margin) + positive[:, None] - negative).mean())


def spatial_structure_loss_from_maps(pred_map: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
    pred_mask = pred_map.float()
    target_mask = target_map.float().detach()
    eps = 1e-6
    pred_mask = pred_mask / pred_mask.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    target_mask = target_mask / target_mask.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)

    intersection = (pred_mask * target_mask).sum(dim=(-2, -1))
    union = pred_mask.sum(dim=(-2, -1)) + target_mask.sum(dim=(-2, -1))
    dice_loss = 1.0 - ((2.0 * intersection + eps) / (union + eps)).mean()

    height, width = pred_mask.shape[-2:]
    y = torch.linspace(0.0, 1.0, height, device=pred_mask.device, dtype=pred_mask.dtype).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=pred_mask.device, dtype=pred_mask.dtype).view(1, 1, 1, width)

    def moments(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = mask.sum(dim=(-2, -1)).clamp_min(eps)
        cx = (mask * x).sum(dim=(-2, -1)) / mass
        cy = (mask * y).sum(dim=(-2, -1)) / mass
        sx = torch.sqrt(((mask * (x - cx[..., None, None]).square()).sum(dim=(-2, -1)) / mass).clamp_min(eps))
        sy = torch.sqrt(((mask * (y - cy[..., None, None]).square()).sum(dim=(-2, -1)) / mass).clamp_min(eps))
        return cx, cy, sx, sy

    pred_cx, pred_cy, pred_sx, pred_sy = moments(pred_mask)
    target_cx, target_cy, target_sx, target_sy = moments(target_mask)
    moment_loss = (
        F.l1_loss(pred_cx, target_cx)
        + F.l1_loss(pred_cy, target_cy)
        + F.l1_loss(pred_sx, target_sx)
        + F.l1_loss(pred_sy, target_sy)
    )
    pred_x = pred_mask.sum(dim=-2)
    target_x = target_mask.sum(dim=-2)
    pred_y = pred_mask.sum(dim=-1)
    target_y = target_mask.sum(dim=-1)
    pred_x = pred_x / pred_x.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_x = target_x / target_x.sum(dim=-1, keepdim=True).clamp_min(eps)
    pred_y = pred_y / pred_y.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_y = target_y / target_y.sum(dim=-1, keepdim=True).clamp_min(eps)
    profile_loss = F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)
    return dice_loss + 0.5 * moment_loss + 2.0 * profile_loss


def latent_spatial_structure_loss(pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    pred_map = pred_latents.float().square().mean(dim=1, keepdim=True).sqrt()
    target_map = target_latents.float().square().mean(dim=1, keepdim=True).sqrt()
    return spatial_structure_loss_from_maps(pred_map, target_map)


def decoded_lower_dark_mask_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.45,
    sharpness: float = 16.0,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    height = pred.shape[-2]
    start = height // 2
    pred_lower = pred[..., start:, :]
    target_lower = target[..., start:, :]
    pred_dark = 1.0 - pred_lower.mean(dim=1, keepdim=True)
    target_dark = 1.0 - target_lower.mean(dim=1, keepdim=True)
    pred_mask = torch.sigmoid((pred_dark - float(threshold)) * float(sharpness))
    target_mask = torch.sigmoid((target_dark - float(threshold)) * float(sharpness))
    mask_loss = F.l1_loss(pred_mask, target_mask)

    eps = 1e-6
    lower_h, width = pred_mask.shape[-2:]
    y = torch.linspace(0.0, 1.0, lower_h, device=pred.device, dtype=pred.dtype).view(1, 1, lower_h, 1)
    x = torch.linspace(0.0, 1.0, width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width)
    pred_mass = pred_mask.sum(dim=(-2, -1)).clamp_min(eps)
    target_mass = target_mask.sum(dim=(-2, -1)).clamp_min(eps)
    pred_cx = (pred_mask * x).sum(dim=(-2, -1)) / pred_mass
    target_cx = (target_mask * x).sum(dim=(-2, -1)) / target_mass
    pred_cy = (pred_mask * y).sum(dim=(-2, -1)) / pred_mass
    target_cy = (target_mask * y).sum(dim=(-2, -1)) / target_mass
    return mask_loss + 0.2 * F.l1_loss(pred_cx, target_cx) + 0.2 * F.l1_loss(pred_cy, target_cy)


def decoded_lower_dark_support_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.45,
    sharpness: float = 16.0,
    dilation: int = 9,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    height = pred.shape[-2]
    start = height // 2
    pred_lower = pred[..., start:, :]
    target_lower = target[..., start:, :]
    pred_dark = 1.0 - pred_lower.mean(dim=1, keepdim=True)
    target_dark = 1.0 - target_lower.mean(dim=1, keepdim=True)
    pred_mask = torch.sigmoid((pred_dark - float(threshold)) * float(sharpness))
    target_mask = torch.sigmoid((target_dark - float(threshold)) * float(sharpness))

    kernel = max(int(dilation), 1)
    if kernel % 2 == 0:
        kernel += 1
    target_support = F.max_pool2d(target_mask, kernel_size=kernel, stride=1, padding=kernel // 2).detach()
    extra_dark = (pred_mask * (1.0 - target_support)).mean()
    missing_dark = (target_mask.detach() * (1.0 - pred_mask)).mean()
    return extra_dark + 0.5 * missing_dark


def decoded_lower_dark_profile_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.45,
    sharpness: float = 16.0,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    eps = 1e-6
    height = pred.shape[-2]
    start = height // 2
    pred_lower = pred[..., start:, :]
    target_lower = target[..., start:, :]
    pred_dark = 1.0 - pred_lower.mean(dim=1, keepdim=True)
    target_dark = 1.0 - target_lower.mean(dim=1, keepdim=True)
    pred_mask = torch.sigmoid((pred_dark - float(threshold)) * float(sharpness))
    target_mask = torch.sigmoid((target_dark - float(threshold)) * float(sharpness)).detach()

    pred_x = pred_mask.sum(dim=-2)
    target_x = target_mask.sum(dim=-2)
    pred_y = pred_mask.sum(dim=-1)
    target_y = target_mask.sum(dim=-1)
    pred_x = pred_x / pred_x.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_x = target_x / target_x.sum(dim=-1, keepdim=True).clamp_min(eps)
    pred_y = pred_y / pred_y.sum(dim=-1, keepdim=True).clamp_min(eps)
    target_y = target_y / target_y.sum(dim=-1, keepdim=True).clamp_min(eps)

    lower_h, width = pred_mask.shape[-2:]
    x = torch.linspace(0.0, 1.0, width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width)
    y = torch.linspace(0.0, 1.0, lower_h, device=pred.device, dtype=pred.dtype).view(1, 1, lower_h, 1)
    pred_mass = pred_mask.sum(dim=(-2, -1)).clamp_min(eps)
    target_mass = target_mask.sum(dim=(-2, -1)).clamp_min(eps)
    pred_cx = (pred_mask * x).sum(dim=(-2, -1)) / pred_mass
    target_cx = (target_mask * x).sum(dim=(-2, -1)) / target_mass
    pred_cy = (pred_mask * y).sum(dim=(-2, -1)) / pred_mass
    target_cy = (target_mask * y).sum(dim=(-2, -1)) / target_mass

    return (
        2.0 * F.l1_loss(pred_x, target_x)
        + 0.5 * F.l1_loss(pred_y, target_y)
        + 0.5 * F.l1_loss(pred_mask.mean(dim=(-2, -1)), target_mask.mean(dim=(-2, -1)))
        + 0.25 * F.l1_loss(pred_cx, target_cx)
        + 0.25 * F.l1_loss(pred_cy, target_cy)
    )


def decoded_lower_dark_strict_excess_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.45,
    sharpness: float = 16.0,
    support_threshold: float = 0.08,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    eps = 1e-6
    height = pred.shape[-2]
    start = height // 2
    pred_lower = pred[..., start:, :]
    target_lower = target[..., start:, :]
    pred_dark = 1.0 - pred_lower.mean(dim=1, keepdim=True)
    target_dark = 1.0 - target_lower.mean(dim=1, keepdim=True)
    pred_mask = torch.sigmoid((pred_dark - float(threshold)) * float(sharpness))
    target_mask = torch.sigmoid((target_dark - float(threshold)) * float(sharpness)).detach()

    target_x = target_mask.sum(dim=-2)
    target_x = target_x / target_x.amax(dim=-1, keepdim=True).clamp_min(eps)
    x_support = torch.sigmoid((target_x - float(support_threshold)) * 40.0).unsqueeze(-2).detach()
    extra_x = (pred_mask * (1.0 - x_support)).mean()

    target_2d_support = torch.sigmoid((target_mask - float(support_threshold)) * 40.0).detach()
    extra_2d = (pred_mask * (1.0 - target_2d_support)).mean()
    target_mass = target_mask.mean(dim=(-2, -1))
    pred_mass = pred_mask.mean(dim=(-2, -1))
    mass_overrun = F.relu(pred_mass - 1.08 * target_mass).mean()
    return extra_x + 0.5 * extra_2d + 0.5 * mass_overrun


def decoded_random_crop_loss(pred: torch.Tensor, target: torch.Tensor, crop_size: int) -> torch.Tensor:
    crop = int(crop_size)
    if crop <= 0 or crop >= pred.shape[-1] or crop >= pred.shape[-2]:
        return F.l1_loss(pred.float(), target.float())
    max_y = pred.shape[-2] - crop
    max_x = pred.shape[-1] - crop
    top = int(torch.randint(0, max_y + 1, (), device=pred.device).item())
    left = int(torch.randint(0, max_x + 1, (), device=pred.device).item())
    pred_crop = pred[..., top : top + crop, left : left + crop]
    target_crop = target[..., top : top + crop, left : left + crop]
    return F.l1_loss(pred_crop.float(), target_crop.float())


def decoded_multicrop_loss(pred: torch.Tensor, target: torch.Tensor, crop_size: int) -> torch.Tensor:
    crop = int(crop_size)
    if crop <= 0 or crop >= pred.shape[-1] or crop >= pred.shape[-2]:
        return F.l1_loss(pred.float(), target.float())
    height = pred.shape[-2]
    width = pred.shape[-1]
    positions = [
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    ]
    losses = []
    for top, left in positions:
        pred_crop = pred[..., top : top + crop, left : left + crop]
        target_crop = target[..., top : top + crop, left : left + crop]
        losses.append(F.l1_loss(pred_crop.float(), target_crop.float()))
    return torch.stack(losses).mean()


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher(args)
    config = SanaLatentStudentConfig(
        resolution=args.resolution,
        vae_scale_factor=int(teacher.vae_scale_factor),
        latent_channels=int(teacher.transformer.config.in_channels),
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        prompt_dim=int(teacher.transformer.config.caption_channels),
        max_sequence_length=args.max_sequence_length,
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    student_device = torch.device(args.student_device)
    student = make_student(config, args).to(student_device)
    start_step = 0
    bitnet_qat_modules = 0
    bitnet_qat_applied = False
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if args.bitnet_qat:
            bitnet_qat_modules = apply_bitnet_qat_modules(
                student,
                threshold_ratio=args.bitnet_qat_threshold_ratio,
                learned_scale=bool(args.bitnet_qat_learned_scale),
                include=tuple(item.strip() for item in args.bitnet_qat_include.split(",") if item.strip()),
                exclude=tuple(item.strip() for item in args.bitnet_qat_exclude.split(",") if item.strip()),
            )
            bitnet_qat_applied = True
        resume_state = checkpoint.get("student")
        if not args.bitnet_qat and checkpoint.get("student_materialized"):
            resume_state = checkpoint["student_materialized"]
        missing, unexpected = student.load_state_dict(resume_state, strict=False)
        if missing or unexpected:
            print(json.dumps({"resume_student_state": {"missing": missing, "unexpected": unexpected}}), flush=True)
        start_step = int(checkpoint.get("step") or 0)
    elif args.init_from_teacher_layers:
        if args.student_architecture != "sana_transformer":
            raise ValueError("--init-from-teacher-layers requires --student-architecture sana_transformer")
        init_report = initialize_sana_student_from_teacher(student, teacher.transformer)
        print(json.dumps({"teacher_layer_init": init_report}), flush=True)
    if args.bitnet_qat:
        if not bitnet_qat_applied:
            bitnet_qat_modules = apply_bitnet_qat_modules(
                student,
                threshold_ratio=args.bitnet_qat_threshold_ratio,
                learned_scale=bool(args.bitnet_qat_learned_scale),
                include=tuple(item.strip() for item in args.bitnet_qat_include.split(",") if item.strip()),
                exclude=tuple(item.strip() for item in args.bitnet_qat_exclude.split(",") if item.strip()),
            )
        trainable_bitnet_qat_parameters = 0
        if args.train_only_bitnet_qat:
            trainable_bitnet_qat_parameters = freeze_except_bitnet_qat_modules(student)
            extra_patterns = tuple(item.strip() for item in args.trainable_name_include.split(",") if item.strip())
            extra_trainable_parameters = enable_named_parameters(student, extra_patterns) if extra_patterns else 0
        elif args.trainable_name_include:
            trainable_bitnet_qat_parameters = freeze_except_named_parameters(
                student,
                tuple(item.strip() for item in args.trainable_name_include.split(",") if item.strip()),
            )
            extra_trainable_parameters = trainable_bitnet_qat_parameters
        else:
            extra_trainable_parameters = 0
        print(
            json.dumps(
                {
                    "bitnet_qat_enabled": {
                        "modules": bitnet_qat_modules,
                        "threshold_ratio": float(args.bitnet_qat_threshold_ratio),
                        "learned_scale": bool(args.bitnet_qat_learned_scale),
                        "include": args.bitnet_qat_include,
                        "exclude": args.bitnet_qat_exclude,
                        "train_only_bitnet_qat": bool(args.train_only_bitnet_qat),
                        "trainable_bitnet_qat_parameters": int(trainable_bitnet_qat_parameters),
                        "extra_trainable_parameters": int(extra_trainable_parameters),
                        "trainable_name_include": args.trainable_name_include,
                    }
                }
            ),
            flush=True,
        )
    elif args.trainable_name_include:
        trainable_parameters_count = freeze_except_named_parameters(
            student,
            tuple(item.strip() for item in args.trainable_name_include.split(",") if item.strip()),
        )
        print(
            json.dumps(
                {
                    "selective_trainable_parameters": {
                        "parameters": int(trainable_parameters_count),
                        "trainable_name_include": args.trainable_name_include,
                    }
                }
            ),
            flush=True,
        )
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters remain after freeze settings")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    prompts = prompt_stream(args)
    for _ in range(args.prompt_skip):
        next(prompts)
    structure_reference_bank = (
        load_reference_image_bank_for_loss(args, teacher, args.x0_decoded_size)
        if args.x0_decoded_structure_contrast_loss_weight > 0.0
        else {}
    )
    if structure_reference_bank:
        print(json.dumps({"structure_reference_bank": {"items": len(structure_reference_bank)}}), flush=True)
    replay_prompts = None
    if args.replay_prompt_file:
        replay_args = copy.copy(args)
        replay_args.prompt_file = args.replay_prompt_file
        replay_args.prompt_skip = 0
        replay_args.prompt_dataset = args.prompt_dataset
        replay_args.stream_image_column = ""
        replay_args.prompt_require_any = ""
        replay_args.prompt_exclude_any = ""
        replay_prompts = prompt_stream(replay_args)
    ledger_path = output_dir / "sana_latent_distill_ledger.jsonl"
    last_loss = 0.0
    final_latent_cache: dict[str, torch.Tensor] = {}
    latent_cache_dir = Path(args.latent_cache_dir) if args.latent_cache_dir else None
    cache_rows = load_latent_cache_rows(latent_cache_dir) if latent_cache_dir else []
    if args.train_from_latent_cache and not cache_rows:
        raise ValueError(f"--train-from-latent-cache requires cached rows in {latent_cache_dir}")
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        if args.train_from_latent_cache:
            rows = [random.choice(cache_rows) for _ in range(args.batch_size)]
        else:
            use_replay = replay_prompts is not None and random.random() < float(args.replay_probability)
            source_prompts = replay_prompts if use_replay else prompts
            rows = [next(source_prompts) for _ in range(args.batch_size)]
        batch_prompts = [row["prompt"] for row in rows]
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = encode_prompts(teacher, batch_prompts, args)
        prompt_embeds = prompt_embeds.to(student_device)
        prompt_mask = prompt_mask.to(student_device)
        negative_prompt_embeds = negative_prompt_embeds.to(student_device)
        negative_prompt_mask = negative_prompt_mask.to(student_device)
        if args.final_latent_flow:
            cache_key = "\n".join(batch_prompts)
            if args.train_from_latent_cache:
                loaded = [load_cached_latent(latent_cache_dir, row, teacher.transformer.device) for row in rows]
                final_latents = torch.cat(loaded, dim=0)
            elif args.fixed_teacher_seed >= 0 and cache_key in final_latent_cache:
                final_latents = final_latent_cache[cache_key].to(teacher.transformer.device)
            else:
                final_latents = generate_teacher_final_latents(
                    teacher,
                    prompt_embeds.to(teacher.transformer.device),
                    prompt_mask.to(teacher.transformer.device),
                    args,
                    step,
                )
                if args.fixed_teacher_seed >= 0:
                    final_latent_cache[cache_key] = final_latents.detach().cpu()
                teacher_reference_path = output_dir / "sana_teacher_reference.png"
                if args.save_teacher_reference and not teacher_reference_path.exists():
                    save_latent_image(teacher, final_latents, teacher_reference_path)
                if latent_cache_dir is not None:
                    for prompt_index, prompt in enumerate(batch_prompts):
                        save_latent_cache_row(
                            latent_cache_dir,
                            prompt,
                            final_latents[prompt_index : prompt_index + 1],
                            step,
                            rows[prompt_index],
                        )
                    cache_rows = load_latent_cache_rows(latent_cache_dir)
                if args.cache_only:
                    examples = []
                    last_loss = 0.0
                    ledger = {
                        "step": step,
                        "loss": last_loss,
                        "trajectory_examples": 0,
                        "batch_size": args.batch_size,
                        "resolution": args.resolution,
                        "teacher_model": args.teacher_model,
                        "source_dataset": rows[0]["source_dataset"],
                        "source_index": rows[0]["source_index"],
                        "prompt": rows[0]["prompt"],
                        "cache_rows": len(cache_rows),
                    }
                    with ledger_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
                    if step % args.log_every == 0 or step == 1:
                        print(json.dumps(ledger), flush=True)
                    continue
            examples = []
            if args.teacher_noise_distill:
                teacher.scheduler.set_timesteps(args.noise_train_steps, device=final_latents.device)
                train_timesteps = teacher.scheduler.timesteps
            for update_index in range(args.flow_updates_per_latent):
                if args.fixed_training_noise or args.direct_latent_prediction:
                    generator = torch.Generator(device=final_latents.device).manual_seed(args.seed + 101 + update_index)
                    z0 = torch.randn(final_latents.shape, generator=generator, device=final_latents.device)
                    t = torch.full((final_latents.shape[0],), update_index / max(args.flow_updates_per_latent - 1, 1), device=final_latents.device)
                else:
                    z0 = torch.randn_like(final_latents)
                    t = torch.rand(final_latents.shape[0], device=final_latents.device)
                if args.teacher_noise_distill:
                    timestep_index = torch.randint(0, len(train_timesteps), (final_latents.shape[0],), device=final_latents.device)
                    timesteps = train_timesteps[timestep_index].to(final_latents.device)
                    zt = teacher.scheduler.add_noise(final_latents, z0, timesteps)
                    target = teacher_predict(
                        teacher,
                        zt.to(teacher.transformer.device),
                        timesteps.to(teacher.transformer.device),
                        prompt_embeds.to(teacher.transformer.device),
                        prompt_mask.to(teacher.transformer.device),
                        negative_prompt_embeds.to(teacher.transformer.device),
                        negative_prompt_mask.to(teacher.transformer.device),
                        args.teacher_guidance if args.distill_guided_targets else 1.0,
                    )
                    examples.append((zt, timesteps.float(), target, None))
                elif args.direct_latent_prediction:
                    examples.append((z0, torch.zeros_like(t), final_latents, None))
                else:
                    zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * final_latents
                    target = final_latents - z0
                    examples.append((zt, t * 1000.0, target, None))
        elif args.trajectory_steps > 0:
            teacher_reference_path = output_dir / "sana_teacher_reference.png"
            if args.save_teacher_reference and not teacher_reference_path.exists():
                final_latents = generate_teacher_final_latents(
                    teacher,
                    prompt_embeds.to(teacher.transformer.device),
                    prompt_mask.to(teacher.transformer.device),
                    args,
                    step,
                )
                save_latent_image(teacher, final_latents, teacher_reference_path)
            reference_latents = encode_reference_images_to_latents(rows, teacher, args) if args.real_image_trajectory_targets else None
            if reference_latents is not None:
                examples = real_image_trajectory_targets(
                    teacher,
                    reference_latents,
                    args,
                    config,
                    step,
                    int(rows[0]["seed"]) if "seed" in rows[0] else None,
                )
            else:
                examples = teacher_trajectory_targets(
                    teacher,
                    prompt_embeds.to(teacher.transformer.device),
                    prompt_mask.to(teacher.transformer.device),
                    negative_prompt_embeds.to(teacher.transformer.device),
                    negative_prompt_mask.to(teacher.transformer.device),
                    args,
                    config,
                    step,
                    int(rows[0]["seed"]) if "seed" in rows[0] else None,
                )
        else:
            latents = torch.randn(
                args.batch_size,
                config.latent_channels,
                config.latent_size,
                config.latent_size,
                device=student_device,
            )
            timesteps = torch.randint(0, 1000, (args.batch_size,), device=student_device, dtype=torch.long)
            teacher_target = teacher_predict(
                teacher,
                latents.to(teacher.transformer.device),
                timesteps.to(teacher.transformer.device),
                prompt_embeds.to(teacher.transformer.device),
                prompt_mask.to(teacher.transformer.device),
                negative_prompt_embeds.to(teacher.transformer.device),
                negative_prompt_mask.to(teacher.transformer.device),
                args.teacher_guidance if args.distill_guided_targets else 1.0,
            )
            examples = [(latents, timesteps.float(), teacher_target, None)]
        losses = []
        base_losses = []
        direction_losses = []
        norm_losses = []
        normalized_losses = []
        lowfreq_losses = []
        pred_target_ratios = []
        next_latent_losses = []
        next_latent_decoded_losses = []
        next_latent_decoded_lowfreq_losses = []
        x0_decoded_losses = []
        x0_decoded_lowfreq_losses = []
        x0_decoded_foreground_losses = []
        x0_decoded_structure_losses = []
        x0_decoded_structure_contrast_losses = []
        x0_decoded_lower_dark_losses = []
        x0_decoded_lower_dark_support_losses = []
        x0_decoded_lower_dark_profile_losses = []
        x0_decoded_lower_dark_strict_excess_losses = []
        x0_decoded_highfreq_losses = []
        x0_decoded_gradient_losses = []
        x0_decoded_edge_weighted_losses = []
        x0_decoded_multicrop_losses = []
        rollout_losses = []
        rollout_structure_losses = []
        decoded_rollout_losses = []
        decoded_rollout_lowfreq_losses = []
        decoded_rollout_moment_losses = []
        decoded_rollout_structure_losses = []
        decoded_rollout_highfreq_losses = []
        decoded_rollout_gradient_losses = []
        decoded_rollout_laplacian_losses = []
        decoded_rollout_crop_losses = []
        decoded_rollout_multicrop_losses = []
        for example_index, (latents, timesteps, teacher_target, teacher_next_latents) in enumerate(examples):
            latents = latents.to(student_device)
            timesteps = timesteps.to(student_device)
            teacher_target = teacher_target.to(student_device)
            if args.train_student_cfg_guidance > 1.0:
                pred = student_predict_cfg(
                    student,
                    latents,
                    timesteps.float(),
                    prompt_embeds,
                    prompt_mask,
                    negative_prompt_embeds,
                    negative_prompt_mask,
                    args.train_student_cfg_guidance,
                    args,
                )
            else:
                pred = student_predict(student, latents, timesteps.float(), prompt_embeds, prompt_mask, args)
            base_loss = F.mse_loss(pred, teacher_target)
            loss = base_loss
            dir_loss = direction_loss(pred, teacher_target)
            nrm_loss = norm_ratio_loss(pred, teacher_target)
            normed_loss = normalized_target_loss(pred, teacher_target)
            low_loss = lowfreq_target_loss(pred, teacher_target, args.lowfreq_target_kernel_size)
            if args.direction_loss_weight > 0.0:
                loss = loss + float(args.direction_loss_weight) * dir_loss
            if args.norm_loss_weight > 0.0:
                loss = loss + float(args.norm_loss_weight) * nrm_loss
            if args.normalized_target_loss_weight > 0.0:
                loss = loss + float(args.normalized_target_loss_weight) * normed_loss
            if args.lowfreq_target_loss_weight > 0.0:
                loss = loss + float(args.lowfreq_target_loss_weight) * low_loss
            if args.next_latent_loss_weight > 0.0 and teacher_next_latents is not None:
                if timesteps.numel() != 1 or latents.shape[0] != 1:
                    raise ValueError("--next-latent-loss-weight currently requires batch-size=1 trajectory examples")
                teacher_next_latents = teacher_next_latents.to(student_device)
                student_next_latents = scheduler_step_with_teacher_history(
                    teacher.scheduler,
                    examples,
                    example_index,
                    pred,
                    args,
                    student_device,
                )
                next_latent_loss = F.mse_loss(student_next_latents, teacher_next_latents)
                loss = loss + float(args.next_latent_loss_weight) * next_latent_loss
                next_latent_losses.append(float(next_latent_loss.item()))
                if (
                    args.next_latent_decoded_loss_weight > 0.0
                    or args.next_latent_decoded_lowfreq_loss_weight > 0.0
                ):
                    student_next_decoded = decode_latents_for_loss(
                        teacher,
                        student_next_latents,
                        args.next_latent_decoded_size,
                    ).to(student_device)
                    with torch.no_grad():
                        reference_next_decoded = (
                            load_reference_images_for_loss(rows, teacher, args, args.next_latent_decoded_size)
                            if args.use_teacher_ref_decoded_targets
                            else None
                        )
                        if reference_next_decoded is not None:
                            teacher_next_decoded = reference_next_decoded.to(student_device)
                        else:
                            teacher_next_decoded = decode_latents_for_loss(
                                teacher,
                                teacher_next_latents,
                                args.next_latent_decoded_size,
                            ).to(student_device)
                    next_decoded_loss = F.l1_loss(student_next_decoded, teacher_next_decoded)
                    next_decoded_low_loss = decoded_lowfreq_loss(
                        student_next_decoded,
                        teacher_next_decoded,
                        args.next_latent_decoded_lowfreq_kernel_size,
                    )
                    loss = (
                        loss
                        + float(args.next_latent_decoded_loss_weight) * next_decoded_loss
                        + float(args.next_latent_decoded_lowfreq_loss_weight) * next_decoded_low_loss
                    )
                    next_latent_decoded_losses.append(float(next_decoded_loss.item()))
                    next_latent_decoded_lowfreq_losses.append(float(next_decoded_low_loss.item()))
            if (
                args.x0_decoded_loss_weight > 0.0
                or args.x0_decoded_lowfreq_loss_weight > 0.0
                or args.x0_decoded_foreground_loss_weight > 0.0
                or args.x0_decoded_structure_loss_weight > 0.0
                or args.x0_decoded_structure_contrast_loss_weight > 0.0
                or args.x0_decoded_lower_dark_loss_weight > 0.0
                or args.x0_decoded_lower_dark_support_loss_weight > 0.0
                or args.x0_decoded_lower_dark_profile_loss_weight > 0.0
                or args.x0_decoded_lower_dark_strict_excess_loss_weight > 0.0
                or args.x0_decoded_highfreq_loss_weight > 0.0
                or args.x0_decoded_gradient_loss_weight > 0.0
                or args.x0_decoded_edge_weighted_loss_weight > 0.0
                or args.x0_decoded_multicrop_loss_weight > 0.0
            ):
                student_x0_latents = estimate_clean_latents_from_velocity(
                    teacher.scheduler,
                    latents,
                    pred,
                    example_index,
                )
                student_x0_decoded = decode_latents_for_loss(
                    teacher,
                    student_x0_latents,
                    args.x0_decoded_size,
                ).to(student_device)
                with torch.no_grad():
                    reference_x0_decoded = (
                        load_reference_images_for_loss(rows, teacher, args, args.x0_decoded_size)
                        if args.use_teacher_ref_decoded_targets
                        else None
                    )
                    if reference_x0_decoded is not None:
                        teacher_x0_decoded = reference_x0_decoded.to(student_device)
                    else:
                        teacher_x0_latents = estimate_clean_latents_from_velocity(
                            teacher.scheduler,
                            latents,
                            teacher_target,
                            example_index,
                        )
                        teacher_x0_decoded = decode_latents_for_loss(
                            teacher,
                            teacher_x0_latents,
                            args.x0_decoded_size,
                        ).to(student_device)
                x0_decoded_loss = F.l1_loss(student_x0_decoded, teacher_x0_decoded)
                x0_decoded_low_loss = decoded_lowfreq_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_lowfreq_kernel_size,
                )
                x0_decoded_fg_loss = decoded_foreground_mask_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_foreground_threshold,
                    args.x0_decoded_foreground_sharpness,
                )
                x0_decoded_structure_loss = decoded_structure_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_structure_threshold,
                    args.x0_decoded_structure_sharpness,
                )
                negative_refs = select_negative_reference_images(
                    structure_reference_bank,
                    rows,
                    student_x0_decoded.device,
                    student_x0_decoded.dtype,
                )
                x0_decoded_structure_contrast_loss = decoded_structure_contrastive_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    negative_refs,
                    args.x0_decoded_structure_threshold,
                    args.x0_decoded_structure_sharpness,
                    args.x0_decoded_lower_dark_threshold,
                    args.x0_decoded_lower_dark_sharpness,
                    args.x0_decoded_structure_contrast_margin,
                )
                x0_decoded_lower_dark_loss = decoded_lower_dark_mask_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_lower_dark_threshold,
                    args.x0_decoded_lower_dark_sharpness,
                )
                x0_decoded_lower_dark_support_loss = decoded_lower_dark_support_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_lower_dark_threshold,
                    args.x0_decoded_lower_dark_sharpness,
                    args.x0_decoded_lower_dark_support_dilation,
                )
                x0_decoded_lower_dark_profile_loss = decoded_lower_dark_profile_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_lower_dark_threshold,
                    args.x0_decoded_lower_dark_sharpness,
                )
                x0_decoded_lower_dark_strict_excess_loss = decoded_lower_dark_strict_excess_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_lower_dark_threshold,
                    args.x0_decoded_lower_dark_sharpness,
                    args.x0_decoded_lower_dark_strict_support_threshold,
                )
                x0_decoded_high_loss = decoded_highfreq_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_highfreq_kernel_size,
                )
                x0_decoded_grad_loss = decoded_gradient_loss(student_x0_decoded, teacher_x0_decoded)
                x0_decoded_edge_loss = decoded_teacher_edge_weighted_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_edge_weight,
                )
                x0_decoded_crop_loss = decoded_multicrop_loss(
                    student_x0_decoded,
                    teacher_x0_decoded,
                    args.x0_decoded_multicrop_size,
                )
                loss = (
                    loss
                    + float(args.x0_decoded_loss_weight) * x0_decoded_loss
                    + float(args.x0_decoded_lowfreq_loss_weight) * x0_decoded_low_loss
                    + float(args.x0_decoded_foreground_loss_weight) * x0_decoded_fg_loss
                    + float(args.x0_decoded_structure_loss_weight) * x0_decoded_structure_loss
                    + float(args.x0_decoded_structure_contrast_loss_weight) * x0_decoded_structure_contrast_loss
                    + float(args.x0_decoded_lower_dark_loss_weight) * x0_decoded_lower_dark_loss
                    + float(args.x0_decoded_lower_dark_support_loss_weight) * x0_decoded_lower_dark_support_loss
                    + float(args.x0_decoded_lower_dark_profile_loss_weight) * x0_decoded_lower_dark_profile_loss
                    + float(args.x0_decoded_lower_dark_strict_excess_loss_weight) * x0_decoded_lower_dark_strict_excess_loss
                    + float(args.x0_decoded_highfreq_loss_weight) * x0_decoded_high_loss
                    + float(args.x0_decoded_gradient_loss_weight) * x0_decoded_grad_loss
                    + float(args.x0_decoded_edge_weighted_loss_weight) * x0_decoded_edge_loss
                    + float(args.x0_decoded_multicrop_loss_weight) * x0_decoded_crop_loss
                )
                x0_decoded_losses.append(float(x0_decoded_loss.item()))
                x0_decoded_lowfreq_losses.append(float(x0_decoded_low_loss.item()))
                x0_decoded_foreground_losses.append(float(x0_decoded_fg_loss.item()))
                x0_decoded_structure_losses.append(float(x0_decoded_structure_loss.item()))
                x0_decoded_structure_contrast_losses.append(float(x0_decoded_structure_contrast_loss.item()))
                x0_decoded_lower_dark_losses.append(float(x0_decoded_lower_dark_loss.item()))
                x0_decoded_lower_dark_support_losses.append(float(x0_decoded_lower_dark_support_loss.item()))
                x0_decoded_lower_dark_profile_losses.append(float(x0_decoded_lower_dark_profile_loss.item()))
                x0_decoded_lower_dark_strict_excess_losses.append(float(x0_decoded_lower_dark_strict_excess_loss.item()))
                x0_decoded_highfreq_losses.append(float(x0_decoded_high_loss.item()))
                x0_decoded_gradient_losses.append(float(x0_decoded_grad_loss.item()))
                x0_decoded_edge_weighted_losses.append(float(x0_decoded_edge_loss.item()))
                x0_decoded_multicrop_losses.append(float(x0_decoded_crop_loss.item()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            base_losses.append(float(base_loss.item()))
            direction_losses.append(float(dir_loss.item()))
            norm_losses.append(float(nrm_loss.item()))
            normalized_losses.append(float(normed_loss.item()))
            lowfreq_losses.append(float(low_loss.item()))
            pred_target_ratios.append(float((tensor_rms(pred).mean() / tensor_rms(teacher_target).mean()).item()))
        if (
            (args.trajectory_rollout_loss_weight > 0.0 or args.trajectory_rollout_structure_loss_weight > 0.0)
            and examples
            and examples[-1][3] is not None
        ):
            initial_latents = examples[0][0].to(student_device)
            teacher_final_latents = examples[-1][3].to(student_device)
            optimizer.zero_grad(set_to_none=True)
            student_final_latents = student_rollout_latents(
                teacher.scheduler,
                student,
                initial_latents,
                prompt_embeds,
                prompt_mask,
                negative_prompt_embeds,
                negative_prompt_mask,
                args,
                student_device,
            )
            rollout_loss = F.mse_loss(student_final_latents, teacher_final_latents)
            rollout_structure_loss = latent_spatial_structure_loss(student_final_latents, teacher_final_latents)
            rollout_total = (
                float(args.trajectory_rollout_loss_weight) * rollout_loss
                + float(args.trajectory_rollout_structure_loss_weight) * rollout_structure_loss
            )
            rollout_total.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            rollout_losses.append(float(rollout_loss.item()))
            rollout_structure_losses.append(float(rollout_structure_loss.item()))
        if (
            (
                args.decoded_rollout_loss_weight > 0.0
                or args.decoded_rollout_lowfreq_loss_weight > 0.0
                or args.decoded_rollout_moment_loss_weight > 0.0
                or args.decoded_rollout_structure_loss_weight > 0.0
                or args.decoded_rollout_highfreq_loss_weight > 0.0
                or args.decoded_rollout_gradient_loss_weight > 0.0
                or args.decoded_rollout_laplacian_loss_weight > 0.0
                or args.decoded_rollout_crop_loss_weight > 0.0
                or args.decoded_rollout_multicrop_loss_weight > 0.0
            )
            and examples
            and examples[-1][3] is not None
        ):
            initial_latents = examples[0][0].to(student_device)
            teacher_final_latents = examples[-1][3].to(student_device)
            optimizer.zero_grad(set_to_none=True)
            student_final_latents = student_rollout_latents(
                teacher.scheduler,
                student,
                initial_latents,
                prompt_embeds,
                prompt_mask,
                negative_prompt_embeds,
                negative_prompt_mask,
                args,
                student_device,
            )
            student_decoded = decode_latents_for_loss(teacher, student_final_latents, args.decoded_rollout_size).to(student_device)
            with torch.no_grad():
                reference_decoded = (
                    load_reference_images_for_loss(rows, teacher, args, args.decoded_rollout_size)
                    if args.use_teacher_ref_decoded_targets
                    else None
                )
                if reference_decoded is not None:
                    teacher_decoded = reference_decoded.to(student_device)
                else:
                    teacher_decoded = decode_latents_for_loss(teacher, teacher_final_latents, args.decoded_rollout_size).to(student_device)
            decoded_loss = F.l1_loss(student_decoded, teacher_decoded)
            decoded_low_loss = decoded_lowfreq_loss(
                student_decoded,
                teacher_decoded,
                args.decoded_rollout_lowfreq_kernel_size,
            )
            decoded_mom_loss = decoded_moment_loss(student_decoded, teacher_decoded)
            decoded_structure = decoded_structure_loss(
                student_decoded,
                teacher_decoded,
                args.decoded_rollout_structure_threshold,
                args.decoded_rollout_structure_sharpness,
            )
            decoded_high_loss = decoded_highfreq_loss(
                student_decoded,
                teacher_decoded,
                args.decoded_rollout_highfreq_kernel_size,
            )
            decoded_grad_loss = decoded_gradient_loss(student_decoded, teacher_decoded)
            decoded_lap_loss = decoded_laplacian_loss(student_decoded, teacher_decoded)
            decoded_crop_loss = decoded_random_crop_loss(
                student_decoded,
                teacher_decoded,
                args.decoded_rollout_crop_size,
            )
            decoded_multi_loss = decoded_multicrop_loss(
                student_decoded,
                teacher_decoded,
                args.decoded_rollout_multicrop_size,
            )
            decoded_total = (
                float(args.decoded_rollout_loss_weight) * decoded_loss
                + float(args.decoded_rollout_lowfreq_loss_weight) * decoded_low_loss
                + float(args.decoded_rollout_moment_loss_weight) * decoded_mom_loss
                + float(args.decoded_rollout_structure_loss_weight) * decoded_structure
                + float(args.decoded_rollout_highfreq_loss_weight) * decoded_high_loss
                + float(args.decoded_rollout_gradient_loss_weight) * decoded_grad_loss
                + float(args.decoded_rollout_laplacian_loss_weight) * decoded_lap_loss
                + float(args.decoded_rollout_crop_loss_weight) * decoded_crop_loss
                + float(args.decoded_rollout_multicrop_loss_weight) * decoded_multi_loss
            )
            decoded_total.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            decoded_rollout_losses.append(float(decoded_loss.item()))
            decoded_rollout_lowfreq_losses.append(float(decoded_low_loss.item()))
            decoded_rollout_moment_losses.append(float(decoded_mom_loss.item()))
            decoded_rollout_structure_losses.append(float(decoded_structure.item()))
            decoded_rollout_highfreq_losses.append(float(decoded_high_loss.item()))
            decoded_rollout_gradient_losses.append(float(decoded_grad_loss.item()))
            decoded_rollout_laplacian_losses.append(float(decoded_lap_loss.item()))
            decoded_rollout_crop_losses.append(float(decoded_crop_loss.item()))
            decoded_rollout_multicrop_losses.append(float(decoded_multi_loss.item()))
        last_loss = sum(losses) / max(len(losses), 1)
        ledger = {
            "step": step,
            "loss": last_loss,
            "base_mse_loss": sum(base_losses) / max(len(base_losses), 1) if base_losses else 0.0,
            "direction_loss": sum(direction_losses) / max(len(direction_losses), 1) if direction_losses else 0.0,
            "norm_loss": sum(norm_losses) / max(len(norm_losses), 1) if norm_losses else 0.0,
            "normalized_target_loss": sum(normalized_losses) / max(len(normalized_losses), 1) if normalized_losses else 0.0,
            "lowfreq_target_loss": sum(lowfreq_losses) / max(len(lowfreq_losses), 1) if lowfreq_losses else 0.0,
            "pred_target_rms_ratio": sum(pred_target_ratios) / max(len(pred_target_ratios), 1) if pred_target_ratios else 0.0,
            "next_latent_loss": sum(next_latent_losses) / max(len(next_latent_losses), 1) if next_latent_losses else 0.0,
            "next_latent_decoded_loss": sum(next_latent_decoded_losses) / max(len(next_latent_decoded_losses), 1) if next_latent_decoded_losses else 0.0,
            "next_latent_decoded_lowfreq_loss": sum(next_latent_decoded_lowfreq_losses) / max(len(next_latent_decoded_lowfreq_losses), 1) if next_latent_decoded_lowfreq_losses else 0.0,
            "x0_decoded_loss": sum(x0_decoded_losses) / max(len(x0_decoded_losses), 1) if x0_decoded_losses else 0.0,
            "x0_decoded_lowfreq_loss": sum(x0_decoded_lowfreq_losses) / max(len(x0_decoded_lowfreq_losses), 1) if x0_decoded_lowfreq_losses else 0.0,
            "x0_decoded_foreground_loss": sum(x0_decoded_foreground_losses) / max(len(x0_decoded_foreground_losses), 1) if x0_decoded_foreground_losses else 0.0,
            "x0_decoded_structure_loss": sum(x0_decoded_structure_losses) / max(len(x0_decoded_structure_losses), 1) if x0_decoded_structure_losses else 0.0,
            "x0_decoded_structure_contrast_loss": sum(x0_decoded_structure_contrast_losses) / max(len(x0_decoded_structure_contrast_losses), 1) if x0_decoded_structure_contrast_losses else 0.0,
            "x0_decoded_lower_dark_loss": sum(x0_decoded_lower_dark_losses) / max(len(x0_decoded_lower_dark_losses), 1) if x0_decoded_lower_dark_losses else 0.0,
            "x0_decoded_lower_dark_support_loss": sum(x0_decoded_lower_dark_support_losses) / max(len(x0_decoded_lower_dark_support_losses), 1) if x0_decoded_lower_dark_support_losses else 0.0,
            "x0_decoded_lower_dark_profile_loss": sum(x0_decoded_lower_dark_profile_losses) / max(len(x0_decoded_lower_dark_profile_losses), 1) if x0_decoded_lower_dark_profile_losses else 0.0,
            "x0_decoded_lower_dark_strict_excess_loss": sum(x0_decoded_lower_dark_strict_excess_losses) / max(len(x0_decoded_lower_dark_strict_excess_losses), 1) if x0_decoded_lower_dark_strict_excess_losses else 0.0,
            "x0_decoded_highfreq_loss": sum(x0_decoded_highfreq_losses) / max(len(x0_decoded_highfreq_losses), 1) if x0_decoded_highfreq_losses else 0.0,
            "x0_decoded_gradient_loss": sum(x0_decoded_gradient_losses) / max(len(x0_decoded_gradient_losses), 1) if x0_decoded_gradient_losses else 0.0,
            "x0_decoded_edge_weighted_loss": sum(x0_decoded_edge_weighted_losses) / max(len(x0_decoded_edge_weighted_losses), 1) if x0_decoded_edge_weighted_losses else 0.0,
            "x0_decoded_multicrop_loss": sum(x0_decoded_multicrop_losses) / max(len(x0_decoded_multicrop_losses), 1) if x0_decoded_multicrop_losses else 0.0,
            "trajectory_rollout_loss": sum(rollout_losses) / max(len(rollout_losses), 1) if rollout_losses else 0.0,
            "trajectory_rollout_structure_loss": sum(rollout_structure_losses) / max(len(rollout_structure_losses), 1) if rollout_structure_losses else 0.0,
            "decoded_rollout_loss": sum(decoded_rollout_losses) / max(len(decoded_rollout_losses), 1) if decoded_rollout_losses else 0.0,
            "decoded_rollout_lowfreq_loss": sum(decoded_rollout_lowfreq_losses) / max(len(decoded_rollout_lowfreq_losses), 1) if decoded_rollout_lowfreq_losses else 0.0,
            "decoded_rollout_moment_loss": sum(decoded_rollout_moment_losses) / max(len(decoded_rollout_moment_losses), 1) if decoded_rollout_moment_losses else 0.0,
            "decoded_rollout_structure_loss": sum(decoded_rollout_structure_losses) / max(len(decoded_rollout_structure_losses), 1) if decoded_rollout_structure_losses else 0.0,
            "decoded_rollout_highfreq_loss": sum(decoded_rollout_highfreq_losses) / max(len(decoded_rollout_highfreq_losses), 1) if decoded_rollout_highfreq_losses else 0.0,
            "decoded_rollout_gradient_loss": sum(decoded_rollout_gradient_losses) / max(len(decoded_rollout_gradient_losses), 1) if decoded_rollout_gradient_losses else 0.0,
            "decoded_rollout_laplacian_loss": sum(decoded_rollout_laplacian_losses) / max(len(decoded_rollout_laplacian_losses), 1) if decoded_rollout_laplacian_losses else 0.0,
            "decoded_rollout_crop_loss": sum(decoded_rollout_crop_losses) / max(len(decoded_rollout_crop_losses), 1) if decoded_rollout_crop_losses else 0.0,
            "decoded_rollout_multicrop_loss": sum(decoded_rollout_multicrop_losses) / max(len(decoded_rollout_multicrop_losses), 1) if decoded_rollout_multicrop_losses else 0.0,
            "trajectory_examples": len(examples),
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "teacher_model": args.teacher_model,
            "source_dataset": rows[0]["source_dataset"],
            "source_index": rows[0]["source_index"],
            "prompt": rows[0]["prompt"],
        }
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(ledger, ensure_ascii=False) + "\n")
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(ledger), flush=True)
        if step % args.sample_every == 0 or local_step == args.steps:
            row_seed = int(rows[0]["seed"]) if "seed" in rows[0] else None
            save_student_sample(teacher, student.eval(), batch_prompts[0], step, args, output_dir, row_seed)
            student.train()
        if step % args.checkpoint_every == 0 or local_step == args.steps:
            save_checkpoint(output_dir, config, student, step, last_loss, args)
    if not args.cache_only:
        save_checkpoint(output_dir, config, student, start_step + args.steps, last_loss, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill Sana-Sprint transformer targets into a small Sana-latent student.")
    parser.add_argument("--output-dir", default="checkpoints/agentkernel_lite_image_sana_latent_distill_v0")
    parser.add_argument("--resume", default="")
    parser.add_argument("--prompt-dataset", default="poloclub/diffusiondb")
    parser.add_argument("--prompt-config", default="2m_text_only")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--replay-prompt-file", default="")
    parser.add_argument("--replay-probability", type=float, default=0.0)
    parser.add_argument("--prompt-split", default="train")
    parser.add_argument("--prompt-columns", default="prompt,caption,text")
    parser.add_argument("--stream-image-column", default="")
    parser.add_argument("--stream-shuffle-buffer", type=int, default=0)
    parser.add_argument("--prompt-require-any", default="")
    parser.add_argument("--prompt-exclude-any", default="")
    parser.add_argument("--prompt-trust-remote-code", action="store_true")
    parser.add_argument("--prompt-skip", type=int, default=0)
    parser.add_argument("--min-prompt-words", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=420)
    parser.add_argument("--max-nsfw-score", type=float, default=0.2)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--trajectory-steps", type=int, default=0)
    parser.add_argument("--next-latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--next-latent-decoded-loss-weight", type=float, default=0.0)
    parser.add_argument("--next-latent-decoded-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--next-latent-decoded-size", type=int, default=96)
    parser.add_argument("--next-latent-decoded-lowfreq-kernel-size", type=int, default=7)
    parser.add_argument("--x0-decoded-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-foreground-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-foreground-threshold", type=float, default=0.08)
    parser.add_argument("--x0-decoded-foreground-sharpness", type=float, default=18.0)
    parser.add_argument("--x0-decoded-structure-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-structure-contrast-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-structure-contrast-margin", type=float, default=0.08)
    parser.add_argument("--x0-decoded-structure-threshold", type=float, default=0.08)
    parser.add_argument("--x0-decoded-structure-sharpness", type=float, default=18.0)
    parser.add_argument("--x0-decoded-lower-dark-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-lower-dark-threshold", type=float, default=0.45)
    parser.add_argument("--x0-decoded-lower-dark-sharpness", type=float, default=16.0)
    parser.add_argument("--x0-decoded-lower-dark-support-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-lower-dark-support-dilation", type=int, default=9)
    parser.add_argument("--x0-decoded-lower-dark-profile-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-lower-dark-strict-excess-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-lower-dark-strict-support-threshold", type=float, default=0.08)
    parser.add_argument("--x0-decoded-highfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-highfreq-kernel-size", type=int, default=9)
    parser.add_argument("--x0-decoded-gradient-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-edge-weighted-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-edge-weight", type=float, default=4.0)
    parser.add_argument("--x0-decoded-multicrop-loss-weight", type=float, default=0.0)
    parser.add_argument("--x0-decoded-multicrop-size", type=int, default=96)
    parser.add_argument("--x0-decoded-size", type=int, default=128)
    parser.add_argument("--x0-decoded-lowfreq-kernel-size", type=int, default=9)
    parser.add_argument("--trajectory-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-rollout-structure-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-lowfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-moment-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-structure-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-structure-threshold", type=float, default=0.08)
    parser.add_argument("--decoded-rollout-structure-sharpness", type=float, default=18.0)
    parser.add_argument("--decoded-rollout-highfreq-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-gradient-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-laplacian-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-crop-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-multicrop-loss-weight", type=float, default=0.0)
    parser.add_argument("--decoded-rollout-size", type=int, default=160)
    parser.add_argument("--decoded-rollout-lowfreq-kernel-size", type=int, default=9)
    parser.add_argument("--decoded-rollout-highfreq-kernel-size", type=int, default=9)
    parser.add_argument("--decoded-rollout-crop-size", type=int, default=96)
    parser.add_argument("--decoded-rollout-multicrop-size", type=int, default=96)
    parser.add_argument(
        "--use-teacher-ref-decoded-targets",
        action="store_true",
        help="Use per-row teacher_ref/image_ref pixels as decoded rollout targets when present.",
    )
    parser.add_argument(
        "--real-image-trajectory-targets",
        action="store_true",
        help="When rows include image pixels, encode them with the teacher VAE and train flow trajectories toward the real-image latents.",
    )
    parser.add_argument("--final-latent-flow", action="store_true")
    parser.add_argument("--direct-latent-prediction", action="store_true")
    parser.add_argument("--teacher-noise-distill", action="store_true")
    parser.add_argument("--noise-train-steps", type=int, default=20)
    parser.add_argument("--direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--norm-loss-weight", type=float, default=0.0)
    parser.add_argument("--normalized-target-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-target-loss-weight", type=float, default=0.0)
    parser.add_argument("--lowfreq-target-kernel-size", type=int, default=5)
    parser.add_argument("--teacher-steps", type=int, default=4)
    parser.add_argument("--teacher-guidance", type=float, default=1.0)
    parser.add_argument("--sample-guidance", type=float, default=4.5)
    parser.add_argument(
        "--train-student-cfg-guidance",
        type=float,
        default=0.0,
        help="When >1, train rollout and per-step losses through the same CFG student path used at inference.",
    )
    parser.add_argument("--distill-guided-targets", action="store_true")
    parser.add_argument("--disable-resolution-binning", action="store_true")
    parser.add_argument("--fixed-teacher-seed", type=int, default=-1)
    parser.add_argument("--save-teacher-reference", action="store_true")
    parser.add_argument("--fixed-training-noise", action="store_true")
    parser.add_argument("--flow-updates-per-latent", type=int, default=1)
    parser.add_argument("--latent-cache-dir", default="")
    parser.add_argument("--train-from-latent-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--student-architecture", choices=("custom", "sana_transformer"), default="custom")
    parser.add_argument("--sana-num-layers", type=int, default=12)
    parser.add_argument("--sana-num-attention-heads", type=int, default=16)
    parser.add_argument("--sana-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-num-cross-attention-heads", type=int, default=16)
    parser.add_argument("--sana-cross-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-mlp-ratio", type=float, default=2.5)
    parser.add_argument("--sana-qk-norm", default="")
    parser.add_argument("--init-from-teacher-layers", action="store_true")
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--bitnet-qat-include", default="")
    parser.add_argument("--bitnet-qat-exclude", default="")
    parser.add_argument("--save-materialized-bitnet", action="store_true")
    parser.add_argument("--train-only-bitnet-qat", action="store_true")
    parser.add_argument("--trainable-name-include", default="")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--keep-step-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
