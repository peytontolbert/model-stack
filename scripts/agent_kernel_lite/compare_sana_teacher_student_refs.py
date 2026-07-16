#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import utils


def import_training_module():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_agentkernel_lite_image_sana_latent_distill import (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        encode_prompts,
        load_teacher,
        make_student,
        student_predict_cfg,
    )

    return SanaLatentStudentConfig, apply_bitnet_qat_modules, encode_prompts, load_teacher, make_student, student_predict_cfg


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                row = json.loads(text)
            else:
                row = {"prompt": text}
            prompt = row.get("prompt") or row.get("caption") or row.get("text")
            if prompt:
                row["prompt"] = str(prompt)
                rows.append(row)
    return rows


def resolve_path(path_text: str | None, root: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    return path if path.exists() else None


def load_reference(row: dict[str, Any], root: Path, size: int) -> Image.Image | None:
    for key in ("teacher_ref", "image_ref", "image_path", "path"):
        path = resolve_path(row.get(key), root)
        if path is not None:
            return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return None


@torch.no_grad()
def sample_student(
    *,
    teacher: Any,
    student: torch.nn.Module,
    prompt: str,
    seed: int,
    args: argparse.Namespace,
    encode_prompts,
    student_predict_cfg,
) -> torch.Tensor:
    device = torch.device(args.student_device)
    prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = encode_prompts(teacher, [prompt], args)
    prompt_embeds = prompt_embeds.to(device)
    prompt_mask = prompt_mask.to(device)
    negative_prompt_embeds = negative_prompt_embeds.to(device)
    negative_prompt_mask = negative_prompt_mask.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_channels = student.config.in_channels if args.student_architecture == "sana_transformer" else student.config.latent_channels
    latent_size = student.config.sample_size if args.student_architecture == "sana_transformer" else student.config.latent_size
    latents = torch.randn(1, latent_channels, latent_size, latent_size, generator=generator, device=device)
    teacher.scheduler.set_timesteps(args.sample_steps, device=device)
    for timestep_value in teacher.scheduler.timesteps:
        timestep = timestep_value.expand(latents.shape[0]).to(device)
        pred = student_predict_cfg(
            student,
            latents,
            timestep,
            prompt_embeds,
            prompt_mask,
            negative_prompt_embeds,
            negative_prompt_mask,
            args.sample_guidance,
            args,
        )
        latents = teacher.scheduler.step(pred, timestep_value, latents, return_dict=False)[0]
    image_latents = latents.to(teacher.vae.device, dtype=teacher.vae.dtype)
    decoded = teacher.vae.decode(image_latents / teacher.vae.config.scaling_factor, return_dict=False)[0]
    return teacher.image_processor.postprocess(decoded, output_type="pt")[0].detach().cpu()


@torch.no_grad()
def sample_teacher(teacher: Any, prompt: str, seed: int, args: argparse.Namespace) -> Image.Image:
    generator = torch.Generator(device=args.teacher_device).manual_seed(seed)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": args.resolution,
        "width": args.resolution,
        "num_inference_steps": args.teacher_steps,
        "guidance_scale": args.teacher_guidance,
        "generator": generator,
    }
    if args.max_sequence_length:
        kwargs["max_sequence_length"] = args.max_sequence_length
    try:
        result = teacher(**kwargs)
    except TypeError:
        kwargs.pop("guidance_scale", None)
        result = teacher(**kwargs)
    return result.images[0].convert("RGB")


def image_l1(a: Image.Image, b: Image.Image) -> float:
    at = torch.ByteTensor(torch.ByteStorage.from_buffer(a.tobytes())).float().view(a.height, a.width, 3) / 255.0
    bt = torch.ByteTensor(torch.ByteStorage.from_buffer(b.tobytes())).float().view(b.height, b.width, 3) / 255.0
    return float((at - bt).abs().mean().item())


def make_sheet(
    *,
    rows: list[dict[str, Any]],
    refs: list[Image.Image | None],
    teachers: list[Image.Image],
    students: list[Image.Image],
    metrics: list[dict[str, Any]],
    output_path: Path,
    cell: int,
) -> None:
    label_h = 92
    cols = 3
    sheet = Image.new("RGB", (cols * cell, len(rows) * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    headers = ("dataset_ref", "sana_teacher", "student")
    for row_i, row in enumerate(rows):
        y = row_i * (cell + label_h)
        images = [
            refs[row_i].resize((cell, cell), Image.Resampling.LANCZOS) if refs[row_i] is not None else Image.new("RGB", (cell, cell), (235, 235, 235)),
            teachers[row_i].resize((cell, cell), Image.Resampling.LANCZOS),
            students[row_i].resize((cell, cell), Image.Resampling.LANCZOS),
        ]
        for col_i, image in enumerate(images):
            x = col_i * cell
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + cell + 3), headers[col_i], fill=(0, 0, 0), font=font)
        prompt = row["prompt"]
        meta = metrics[row_i]
        label = f"{row_i:02d} seed={meta['seed']} t/ref={meta.get('teacher_ref_l1')} s/ref={meta.get('student_ref_l1')}"
        wrapped = textwrap.wrap(label + " " + prompt, width=92)[:4]
        draw.text((4, y + cell + 18), "\n".join(wrapped), fill=(0, 0, 0), font=font)
    sheet.save(output_path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dataset refs, SANA teacher generations, and SANA student outputs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_teacher_diffusers")
    parser.add_argument("--teacher-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--student-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--teacher-steps", type=int, default=12)
    parser.add_argument("--teacher-guidance", type=float, default=4.5)
    parser.add_argument("--sample-steps", type=int, default=12)
    parser.add_argument("--sample-guidance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--use-materialized-bitnet", action="store_true")
    parser.add_argument("--bitnet-qat", action="store_true")
    parser.add_argument("--bitnet-qat-threshold-ratio", type=float, default=0.7)
    parser.add_argument("--bitnet-qat-learned-scale", action="store_true")
    parser.add_argument("--bitnet-qat-include", default="")
    parser.add_argument("--bitnet-qat-exclude", default="")
    parser.add_argument("--student-architecture", choices=("custom", "sana_transformer"), default="sana_transformer")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--sana-num-layers", type=int, default=8)
    parser.add_argument("--sana-num-attention-heads", type=int, default=12)
    parser.add_argument("--sana-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-num-cross-attention-heads", type=int, default=12)
    parser.add_argument("--sana-cross-attention-head-dim", type=int, default=32)
    parser.add_argument("--sana-mlp-ratio", type=float, default=2.5)
    parser.add_argument("--sana-qk-norm", default="")
    args = parser.parse_args()

    (
        SanaLatentStudentConfig,
        apply_bitnet_qat_modules,
        encode_prompts,
        load_teacher,
        make_student,
        student_predict_cfg,
    ) = import_training_module()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.prompt_file))
    rows = rows[args.prompt_offset :]
    if args.max_prompts:
        rows = rows[: args.max_prompts]
    if not rows:
        raise ValueError("no prompts found")

    teacher = load_teacher(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = checkpoint.get("config", {})
    config = SanaLatentStudentConfig(**{**config_dict, "resolution": args.resolution, "patch_size": args.patch_size})
    student = make_student(config, args).to(args.student_device)
    if args.bitnet_qat:
        apply_bitnet_qat_modules(
            student,
            threshold_ratio=args.bitnet_qat_threshold_ratio,
            learned_scale=bool(args.bitnet_qat_learned_scale),
            include=tuple(item.strip() for item in args.bitnet_qat_include.split(",") if item.strip()),
            exclude=tuple(item.strip() for item in args.bitnet_qat_exclude.split(",") if item.strip()),
        )
    student_state = checkpoint.get("student_materialized") if args.use_materialized_bitnet else None
    if student_state is None:
        student_state = checkpoint["student"]
    student.load_state_dict(student_state, strict=True)
    student.eval()

    refs: list[Image.Image | None] = []
    teachers: list[Image.Image] = []
    students: list[Image.Image] = []
    metrics: list[dict[str, Any]] = []
    root = Path.cwd()
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        seed = int(row.get("seed") if row.get("seed") is not None else args.seed + args.prompt_offset + index)
        ref = load_reference(row, root, args.resolution)
        teacher_image = sample_teacher(teacher, prompt, seed, args)
        student_tensor = sample_student(
            teacher=teacher,
            student=student,
            prompt=prompt,
            seed=seed,
            args=args,
            encode_prompts=encode_prompts,
            student_predict_cfg=student_predict_cfg,
        )
        student_path = output_dir / f"student_{index:03d}.png"
        teacher_path = output_dir / f"teacher_{index:03d}.png"
        ref_path = output_dir / f"ref_{index:03d}.png"
        utils.save_image(student_tensor, student_path)
        teacher_image.save(teacher_path)
        student_image = Image.open(student_path).convert("RGB")
        if ref is not None:
            ref.save(ref_path)
        metric: dict[str, Any] = {"index": index, "prompt": prompt, "seed": seed, "source_ref": row.get("teacher_ref")}
        if ref is not None:
            metric["teacher_ref_l1"] = round(image_l1(teacher_image.resize((args.resolution, args.resolution)), ref), 5)
            metric["student_ref_l1"] = round(image_l1(student_image.resize((args.resolution, args.resolution)), ref), 5)
        else:
            metric["teacher_ref_l1"] = None
            metric["student_ref_l1"] = None
        refs.append(ref)
        teachers.append(teacher_image)
        students.append(student_image)
        metrics.append(metric)
        print(json.dumps(metric), flush=True)

    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric) + "\n")
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    make_sheet(rows=rows, refs=refs, teachers=teachers, students=students, metrics=metrics, output_path=output_dir / "contact_sheet.png", cell=180)
    print(output_dir / "contact_sheet.png")


if __name__ == "__main__":
    main()
