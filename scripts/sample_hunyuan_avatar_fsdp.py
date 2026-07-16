"""Generate Avatar clips from FSDP-local transformer shards on two GPUs."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoFeatureExtractor, WhisperModel


class FSDPTransformerProxy(nn.Module):
    """Expose Avatar's cache and configuration attributes through an FSDP model."""

    def __init__(self, fsdp_model: nn.Module) -> None:
        super().__init__()
        self.fsdp_model = fsdp_model

    @property
    def cache_out(self):
        return self.fsdp_model.module.cache_out

    @cache_out.setter
    def cache_out(self, value) -> None:
        self.fsdp_model.module.cache_out = value

    def forward(self, *args, **kwargs):
        return self.fsdp_model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fsdp_model.module, name)


class BroadcastTextEncoder:
    """Run heavyweight text encoders on rank zero and broadcast their outputs.

    Avatar's denoising loop is collective across FSDP ranks, but its prompt
    encoders are not. Loading LLaVA independently per rank doubles a very slow
    CPU-side model-store read without improving inference. Both ranks call
    ``encode`` in the same order; rank zero broadcasts the resulting tensors.
    """

    def __init__(self, inner, *, rank: int, device: torch.device, dtype: torch.dtype) -> None:
        self.inner = inner
        self.rank = rank
        self.device = device
        self.dtype = dtype if inner is None else inner.dtype
        image_token_index = [getattr(inner.model.config, "image_token_index", None) if inner is not None else None]
        dist.broadcast_object_list(image_token_index, src=0, device=device)
        self.model = SimpleNamespace(config=SimpleNamespace(image_token_index=image_token_index[0]))

    def text2tokens(self, *args, **kwargs):
        payload = [self.inner.text2tokens(*args, **kwargs) if self.inner is not None else None]
        dist.broadcast_object_list(payload, src=0, device=self.device)
        return payload[0]

    def encode(self, *args, **kwargs):
        payload = [None]
        if self.inner is not None:
            output = self.inner.encode(*args, **kwargs)
            payload[0] = (output.hidden_state.cpu(), output.attention_mask.cpu() if output.attention_mask is not None else None)
        dist.broadcast_object_list(payload, src=0, device=self.device)
        if self.inner is not None:
            return output
        hidden_state, attention_mask = payload[0]
        return SimpleNamespace(hidden_state=hidden_state, attention_mask=attention_mask)


def parse_launcher_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shard-dir", type=Path, required=True)
    return parser.parse_known_args()


def main() -> None:
    launcher_args, avatar_argv = parse_launcher_args()
    sys.argv = [sys.argv[0], *avatar_argv]
    avatar_root = os.environ["HUNYUAN_AVATAR_ROOT"]
    model_base = Path(os.environ["MODEL_BASE"])
    if avatar_root not in sys.path:
        sys.path.insert(0, avatar_root)

    from hymm_sp.config import parse_args
    from hymm_sp.constants import PROMPT_TEMPLATE
    from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
    from hymm_sp.data_kits.face_align import AlignImage
    from hymm_sp.sample_inference_audio import HunyuanVideoSampler
    from hymm_sp.text_encoder import TextEncoder
    from hymm_sp.vae import load_vae
    from runtime.hunyuan_avatar_fsdp import (
        AvatarFSDPConfig,
        build_avatar_transformer_cpu,
        load_rank_local_shard,
        normalize_floating_point_dtype,
        wrap_avatar_transformer_fsdp,
    )

    args = parse_args()
    if args.use_fp8:
        raise ValueError("This path consumes validated BF16 FSDP shards; FP8 shard loading is not implemented.")
    if not args.cpu_offload:
        raise ValueError("Pass --cpu-offload so VAE/text components remain explicitly off GPU between stages.")

    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"rank={rank} constructing and loading local FSDP transformer shard", flush=True)
    transformer = build_avatar_transformer_cpu(args, avatar_root)
    normalize_floating_point_dtype(transformer, torch.bfloat16)
    transformer = wrap_avatar_transformer_fsdp(
        transformer,
        avatar_root=avatar_root,
        config=AvatarFSDPConfig(device_id=rank),
    )
    load_rank_local_shard(transformer, launcher_args.shard_dir, rank=rank)
    transformer.eval()
    transformer_proxy = FSDPTransformerProxy(transformer)

    # Auxiliary modules are intentionally CPU resident until their short use.
    vae, _, s_ratio, t_ratio = load_vae(args.vae, args.vae_precision, logger=logger, device="cpu")
    vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
    crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
    prompt_template_video = PROMPT_TEMPLATE[args.prompt_template_video]
    if rank == 0:
        text_encoder_inner = TextEncoder(
            text_encoder_type=args.text_encoder,
            max_length=args.text_len + crop_start,
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            use_attention_mask=args.use_attention_mask,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=logger,
            device="cpu",
        )
        text_encoder_2_inner = TextEncoder(
            text_encoder_type=args.text_encoder_2,
            max_length=args.text_len_2,
            text_encoder_precision=args.text_encoder_precision_2,
            tokenizer_type=args.tokenizer_2,
            use_attention_mask=args.use_attention_mask,
            reproduce=args.reproduce,
            logger=logger,
            device="cpu",
        )
    else:
        text_encoder_inner = None
        text_encoder_2_inner = None
    text_encoder = BroadcastTextEncoder(text_encoder_inner, rank=rank, device=device, dtype=torch.float16)
    text_encoder_2 = BroadcastTextEncoder(text_encoder_2_inner, rank=rank, device=device, dtype=torch.float16)
    sampler = HunyuanVideoSampler(
        args, vae, vae_kwargs, text_encoder, transformer_proxy,
        text_encoder_2=text_encoder_2, device=device, logger=logger,
    )
    wav2vec = WhisperModel.from_pretrained(model_base / "ckpts" / "whisper-tiny").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_base / "ckpts" / "whisper-tiny")
    align_instance = AlignImage("cuda", det_path=str(model_base / "ckpts" / "det_align" / "detface.pt"))
    dataset = VideoAudioTextLoaderVal(
        image_size=args.image_size,
        meta_file=args.input,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        feature_extractor=feature_extractor,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False),
        drop_last=False,
    )
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    for batch in loader:
        video_id = batch["videoid"][0]
        samples = sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
        if rank == 0 and samples is not None:
            sample = samples["samples"][0].unsqueeze(0)[:, :, :batch["audio_len"][0]]
            frames = (rearrange(sample[0], "c f h w -> f h w c") * 255.0).cpu().numpy().astype(np.uint8)
            silent_output = Path(args.save_path) / f"{video_id}.mp4"
            imageio.mimsave(silent_output, frames, fps=batch["fps"].item())
            audio_path = str(batch["audio_path"][0])
            output = Path(args.save_path) / f"{video_id}_audio.mp4"
            os.system(f"ffmpeg -y -loglevel error -i '{silent_output}' -i '{audio_path}' -shortest '{output}'")
            silent_output.unlink(missing_ok=True)
            print(f"wrote {output}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
