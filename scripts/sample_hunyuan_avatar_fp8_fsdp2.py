"""Generate Avatar clips from native-FP8 FSDP2 shards on two GPUs."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from datetime import timedelta
from types import SimpleNamespace
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from loguru import logger
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
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
        self.encode_calls = 0
        image_token_index = [getattr(inner.model.config, "image_token_index", None) if inner is not None else None]
        dist.broadcast_object_list(image_token_index, src=0, device=device)
        # The Avatar pipeline expands the LLaVA image placeholder before calling
        # encode(). Rank one needs this small piece of model configuration, not a
        # second 8B LLaVA instance.
        self.model = SimpleNamespace(config=SimpleNamespace(image_token_index=image_token_index[0]))

    def text2tokens(self, *args, **kwargs):
        print(f"rank={self.rank} text2tokens start", flush=True)
        payload = [self.inner.text2tokens(*args, **kwargs) if self.inner is not None else None]
        print(f"rank={self.rank} text2tokens broadcast", flush=True)
        dist.broadcast_object_list(payload, src=0, device=self.device)
        print(f"rank={self.rank} text2tokens complete", flush=True)
        return payload[0]

    def encode(self, *args, **kwargs):
        payload = [None]
        if self.inner is not None:
            print(f"rank={self.rank} text encode start", flush=True)
            output = self.inner.encode(*args, **kwargs)
            print(f"rank={self.rank} text encode complete", flush=True)
            payload[0] = (output.hidden_state.cpu(), output.attention_mask.cpu() if output.attention_mask is not None else None)
        print(f"rank={self.rank} text encode broadcast", flush=True)
        dist.broadcast_object_list(payload, src=0, device=self.device)
        print(f"rank={self.rank} text encode broadcast complete", flush=True)
        hidden_state, attention_mask = payload[0]
        self.encode_calls += 1
        if self.inner is not None and getattr(self.inner.model, "_avatar_stage_local_int8", False) and self.encode_calls >= 2:
            # LLaVA is used for positive and negative prompt conditioning only.
            # Its outputs are now CPU tensors, so reclaim rank 0 before Avatar's
            # FSDP transformer begins denoising.
            print(f"rank={self.rank} releasing stage-local INT8 LLaVA", flush=True)
            del self.inner.model
            self.inner = None
            gc.collect()
            torch.cuda.empty_cache()
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
    from hymm_sp.modules.parallel_states import initialize_sequence_parallel_state
    from hymm_sp.sample_inference_audio import HunyuanVideoSampler
    from hymm_sp.text_encoder import TextEncoder
    from hymm_sp.vae import load_vae
    from runtime.hunyuan_avatar_fsdp import (
        build_avatar_transformer_cpu,
        convert_avatar_linears_to_fp8_cpu,
        load_rank_local_fsdp2_shard,
        normalize_residual_fp32_parameters_for_fsdp2,
        split_avatar_fp8_linear_parameters,
        wrap_avatar_transformer_fsdp2,
    )

    args = parse_args()
    if not args.cpu_offload:
        raise ValueError("Pass --cpu-offload so VAE/text components remain explicitly off GPU between stages.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    initialize_sequence_parallel_state(world_size)

    print(f"rank={rank} constructing and loading local FSDP2 FP8 transformer shard", flush=True)
    transformer = build_avatar_transformer_cpu(args, avatar_root)
    fp8_shards = "fp8" in launcher_args.shard_dir.name.lower()
    if fp8_shards:
        convert_avatar_linears_to_fp8_cpu(transformer, args.ckpt, args, avatar_root)
        normalize_residual_fp32_parameters_for_fsdp2(transformer)
        split_avatar_fp8_linear_parameters(transformer, torch.bfloat16)
    else:
        print(f"rank={rank} using homogeneous BF16 FSDP2 shards", flush=True)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
    transformer = wrap_avatar_transformer_fsdp2(transformer, avatar_root=avatar_root, mesh=mesh)
    load_rank_local_fsdp2_shard(transformer, launcher_args.shard_dir, rank=rank)
    transformer.eval()

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
        args, vae, vae_kwargs, text_encoder, transformer,
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
