"""Materialize Avatar reference VAE conditioning before distributed denoising.

This deliberately runs without Avatar's transformer. The resulting CPU artifact
lets the FSDP process reserve both GPUs for the FP8 transformer and temporary
rank-zero LLaVA conditioning.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from PIL import Image
from loguru import logger


def resize_reference(image: Image.Image, image_size: int) -> torch.Tensor:
    width, height = image.size
    scale = image_size / min(width, height)
    new_width = round(width * scale / 64) * 64
    new_height = round(height * scale / 64) * 64
    long_edge = {704: 1216, 512: 768, 384: 576, 256: 384}.get(image_size, int(image_size * 1.5))
    if new_width * new_height > image_size * long_edge:
        scale = (image_size * long_edge / width / height) ** 0.5
        new_width = round(width * scale / 64) * 64
        new_height = round(height * scale / 64) * 64
    pixels = np.array(image.resize((new_width, new_height), Image.LANCZOS).convert("RGB"), copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(torch.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--frames", type=int, default=129, help="Reference frames; Avatar requires 129 to form its 33-latent conditioning chunk.")
    parser.add_argument("--vae", default="884-16c-hy0801")
    parser.add_argument("--vae-precision", default="fp16")
    args = parser.parse_args()

    avatar_root = Path(os.environ["HUNYUAN_AVATAR_ROOT"])
    import sys
    sys.path.insert(0, str(avatar_root))
    from hymm_sp.data_kits.audio_preprocessor import get_facemask
    from hymm_sp.data_kits.face_align import AlignImage
    from hymm_sp.vae import load_vae

    rows = pd.read_csv(args.input)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    vae, _, _, _ = load_vae(args.vae, args.vae_precision, logger=logger, device="cpu")
    vae.to("cuda:0")
    vae.eval()
    align = AlignImage("cuda:0", det_path=str(Path(os.environ["MODEL_BASE"]) / "ckpts" / "det_align" / "detface.pt"))

    for row in rows.itertuples(index=False):
        video_id = str(row.videoid)
        output = args.cache_dir / f"{video_id}.pt"
        if output.exists():
            print(f"using existing VAE cache {output}", flush=True)
            continue
        image_path = Path(row.image)
        if not image_path.is_absolute():
            working_path = Path.cwd() / image_path
            image_path = working_path if working_path.exists() else args.input.parent / image_path
        source = resize_reference(Image.open(image_path), args.image_size).to("cuda:0")
        face_masks = get_facemask(source.clone(), align, area=3.0)
        reference = source.repeat(1, args.frames, 1, 1, 1) / 127.5 - 1.0
        unconditional = torch.zeros_like(reference) * 2.0 - 1.0
        reference = rearrange(reference, "b f c h w -> b c f h w")
        unconditional = rearrange(unconditional, "b f c h w -> b c f h w")
        with torch.no_grad(), torch.autocast("cuda", dtype=vae.dtype, enabled=vae.dtype != torch.float32):
            vae.enable_tiling()
            ref_latents = vae.encode(reference).latent_dist.sample()
            uncond_latents = vae.encode(unconditional).latent_dist.sample()
            vae.disable_tiling()
        if getattr(vae.config, "shift_factor", None):
            ref_latents.sub_(vae.config.shift_factor).mul_(vae.config.scaling_factor)
            uncond_latents.sub_(vae.config.shift_factor).mul_(vae.config.scaling_factor)
        else:
            ref_latents.mul_(vae.config.scaling_factor)
            uncond_latents.mul_(vae.config.scaling_factor)
        face_masks = torch.nn.functional.interpolate(
            face_masks.float().squeeze(2), ref_latents.shape[-2:], mode="bilinear"
        ).unsqueeze(2).to(dtype=ref_latents.dtype)
        temporary = output.with_suffix(".tmp")
        torch.save({
            "ref_latents": ref_latents.cpu(),
            "uncond_ref_latents": uncond_latents.cpu(),
            "face_masks": face_masks.cpu(),
            "image_path": str(image_path),
            "image_size": args.image_size,
            "frames": args.frames,
        }, temporary)
        temporary.replace(output)
        print(f"wrote VAE cache {output}", flush=True)
        del source, face_masks, reference, unconditional, ref_latents, uncond_latents
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
