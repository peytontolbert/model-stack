#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

LIGHTX2V_ROOT = Path("/data/clone/third_party/LightX2V")
HY3DGEN_ROOT = Path("/data/clone/third_party/Hunyuan3D-2")
TRELLIS_ROOT = Path("/data/clone/third_party/TRELLIS")
TRELLIS2_ROOT = Path("/data/clone/third_party/TRELLIS.2")
HUNYUAN3D_CONFIG = LIGHTX2V_ROOT / "configs/hunyuan3d/hunyuan3d_shape.json"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_paths(req: dict[str, Any]) -> dict[str, str]:
    out_dir = Path(req["output_dir"])
    extra = req.get("extra_args") or {}
    stem = extra.get("output_stem", "mesh")
    fmt = req.get("output_format", "glb")
    artifacts = {fmt: str(out_dir / f"{stem}.{fmt}")}
    if fmt == "glb":
        artifacts = {"glb": str(out_dir / f"{stem}.glb")}
    if req.get("texture"):
        artifacts["textured_glb"] = str(out_dir / f"{stem}_textured.glb")
    return artifacts


def _hunyuan_variant_dir(req: dict[str, Any]) -> Path:
    model_path = Path(req.get("model_path") or "/arxiv/models/Hunyuan3D-2mv")
    variant = req.get("variant")
    if variant:
        candidate = model_path / variant
        if candidate.is_dir():
            return candidate
    if (model_path / "config.yaml").is_file():
        return model_path
    for child in sorted(model_path.iterdir() if model_path.is_dir() else []):
        if child.is_dir() and child.name.startswith("hunyuan3d-dit-") and (child / "config.yaml").is_file():
            return child
    return model_path


def _hunyuan_schema_info(variant_dir: Path) -> dict[str, Any]:
    cfg = variant_dir / "config.yaml"
    ckpt = None
    for name in ("model.fp16.safetensors", "model.fp16.ckpt", "model.safetensors", "model.ckpt"):
        if (variant_dir / name).is_file():
            ckpt = variant_dir / name
            break
    target = ""
    if cfg.is_file():
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if "target:" in line and not target:
                target = line.split("target:", 1)[1].strip()
                break
    contains_x_embedder = None
    contains_double_blocks = None
    if ckpt and ckpt.suffix == ".ckpt":
        try:
            with zipfile.ZipFile(ckpt) as zf:
                data_name = next((n for n in zf.namelist() if n.endswith("/data.pkl")), None)
                if data_name:
                    data = zf.read(data_name)
                    contains_x_embedder = b"x_embedder.weight" in data
                    contains_double_blocks = b"double_blocks.0" in data
        except zipfile.BadZipFile:
            pass
    elif ckpt and ckpt.suffix == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(ckpt), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
            contains_x_embedder = "x_embedder.weight" in keys
            contains_double_blocks = any(key.startswith("double_blocks.0") for key in keys)
        except Exception:
            contains_x_embedder = None
            contains_double_blocks = None
    return {
        "variant_dir": str(variant_dir),
        "config": str(cfg),
        "checkpoint": str(ckpt) if ckpt else None,
        "target": target,
        "contains_x_embedder_weight": contains_x_embedder,
        "contains_double_blocks": contains_double_blocks,
    }


def _lightx2v_hunyuan_command(req: dict[str, Any], artifacts: dict[str, str], variant_dir: Path) -> list[str]:
    output = artifacts.get("glb") or next(iter(artifacts.values()))
    return [
        sys.executable,
        "-m",
        "lightx2v.infer",
        "--model_cls",
        "hunyuan3d",
        "--task",
        "i23d",
        "--model_path",
        str(variant_dir),
        "--config_json",
        str(HUNYUAN3D_CONFIG),
        "--image_path",
        req["image_path"],
        "--save_result_path",
        output,
        "--seed",
        str(req.get("seed", 42)),
    ]


def _run_hunyuan3d(req: dict[str, Any]) -> dict[str, Any]:
    artifacts = _artifact_paths(req)
    variant_dir = _hunyuan_variant_dir(req)
    schema = _hunyuan_schema_info(variant_dir)
    cmd = _lightx2v_hunyuan_command(req, artifacts, variant_dir)
    if not Path(req["image_path"]).is_file():
        return {
            "backend": "hunyuan3d",
            "status": "input_missing",
            "env": "ai",
            "artifacts": artifacts,
            "command": cmd,
            "schema": schema,
            "error": f"image_path does not exist: {req['image_path']}",
        }
    extra = req.get("extra_args") or {}
    model_path = req.get("model_path") or "/arxiv/models/Hunyuan3D-2mv"
    subfolder = req.get("variant") or variant_dir.name
    output = artifacts.get("glb") or next(iter(artifacts.values()))
    steps = int(extra.get("num_inference_steps", 1))
    octree_resolution = int(extra.get("octree_resolution", 64))
    num_chunks = int(extra.get("num_chunks", 1000))
    guidance_scale = float(extra.get("guidance_scale", 5.0))
    load_only = bool(extra.get("load_only", False))
    official_cmd = [
        sys.executable,
        "-c",
        "<hy3dgen worker inline>",
        "--model_path",
        str(model_path),
        "--subfolder",
        subfolder,
        "--image_path",
        req["image_path"],
        "--output",
        output,
    ]
    start = time.monotonic()
    try:
        sys.path.insert(0, str(HY3DGEN_ROOT))
        import torch
        from PIL import Image
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        source_image = Image.open(req["image_path"])
        has_alpha_cutout = (
            source_image.mode == "RGBA"
            and source_image.getchannel("A").getextrema()[0] < 255
        )
        image = source_image.convert("RGBA") if has_alpha_cutout else BackgroundRemover()(source_image.convert("RGB"))
        cutout_path = Path(output).with_name(f"{Path(output).stem}_cutout.png")
        cutout_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(cutout_path)
        artifacts["cutout_png"] = str(cutout_path)
        # Variants are named `...-mv`, `...-mv-fast`, and `...-mv-turbo`.
        # Every multiview variant requires the view-keyed image mapping.
        pipeline_image = {"front": image} if "-mv" in subfolder else image
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            variant="fp16",
            use_safetensors=True,
        )
        if load_only:
            elapsed = time.monotonic() - start
            return {
                "backend": "hunyuan3d",
                "status": "loaded",
                "env": "ai",
                "artifacts": artifacts,
                "command": official_cmd,
                "schema": schema,
                "elapsed_sec": elapsed,
                "model_path": str(model_path),
                "subfolder": subfolder,
            }
        generator = torch.manual_seed(int(req.get("seed", 42)))
        meshes = pipe(
            image=pipeline_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            generator=generator,
            output_type="trimesh",
            enable_pbar=bool(extra.get("enable_pbar", False)),
        )
        mesh = meshes[0]
        if isinstance(mesh, list):
            mesh = mesh[0] if mesh else None
        if mesh is None or not hasattr(mesh, "export"):
            return {
                "backend": "hunyuan3d",
                "status": "empty_mesh",
                "env": "ai",
                "artifacts": artifacts,
                "command": official_cmd,
                "schema": schema,
                "elapsed_sec": time.monotonic() - start,
                "error": "hy3dgen returned no exportable mesh; increase inference steps/octree resolution or provide stronger multiview inputs",
                "model_path": str(model_path),
                "subfolder": subfolder,
                "num_inference_steps": steps,
                "octree_resolution": octree_resolution,
            }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output)
        elapsed = time.monotonic() - start
        return {
            "backend": "hunyuan3d",
            "status": "ok",
            "env": "ai",
            "artifacts": artifacts,
            "command": official_cmd,
            "schema": schema,
            "elapsed_sec": elapsed,
            "model_path": str(model_path),
            "subfolder": subfolder,
            "num_inference_steps": steps,
            "octree_resolution": octree_resolution,
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        return {
            "backend": "hunyuan3d",
            "status": "failed",
            "env": "ai",
            "artifacts": artifacts,
            "command": official_cmd,
            "schema": schema,
            "elapsed_sec": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _trellis2_run_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    if extra.get("num_inference_steps") is not None and extra.get("steps") is None:
        extra = {**extra, "steps": extra["num_inference_steps"]}
    run_kwargs: dict[str, Any] = {
        "seed": int(extra.get("seed", 42)),
        "preprocess_image": bool(extra.get("preprocess_image", True)),
    }
    if extra.get("pipeline_type"):
        run_kwargs["pipeline_type"] = str(extra["pipeline_type"])

    def sampler_params(prefix: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        steps = extra.get(f"{prefix}_steps", extra.get("steps"))
        if steps is not None:
            params["steps"] = int(steps)
        guidance = extra.get(f"{prefix}_guidance_strength", extra.get("guidance_strength"))
        if guidance is not None:
            params["guidance_strength"] = float(guidance)
        guidance_rescale = extra.get(f"{prefix}_guidance_rescale")
        if guidance_rescale is not None:
            params["guidance_rescale"] = float(guidance_rescale)
        rescale_t = extra.get(f"{prefix}_rescale_t")
        if rescale_t is not None:
            params["rescale_t"] = float(rescale_t)
        return params

    ss = sampler_params("ss")
    shape = sampler_params("shape")
    tex = sampler_params("tex")
    if ss:
        run_kwargs["sparse_structure_sampler_params"] = ss
    if shape:
        run_kwargs["shape_slat_sampler_params"] = shape
    if tex:
        run_kwargs["tex_slat_sampler_params"] = tex
    return run_kwargs


def _export_trellis2_glb(mesh: Any, output: str, extra: dict[str, Any]) -> None:
    import o_voxel

    aabb = extra.get("aabb", [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    if isinstance(aabb, str):
        values = [float(x) for x in aabb.split(",")]
        if len(values) != 6:
            raise ValueError("aabb string must contain 6 comma-separated floats")
        aabb = [[values[0], values[1], values[2]], [values[3], values[4], values[5]]]
    if int(extra.get("simplify", 0)) > 0 and hasattr(mesh, "simplify"):
        mesh.simplify(int(extra["simplify"]))
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=aabb,
        decimation_target=int(extra.get("decimation_target", 1000000)),
        texture_size=int(extra.get("texture_size", 1024)),
        remesh=bool(extra.get("remesh", True)),
        remesh_band=int(extra.get("remesh_band", 1)),
        remesh_project=int(extra.get("remesh_project", 0)),
        verbose=bool(extra.get("verbose_export", False)),
    )
    glb.export(output, extension_webp=bool(extra.get("extension_webp", True)))


def _run_trellis(req: dict[str, Any]) -> dict[str, Any]:
    artifacts = _artifact_paths(req)
    model_path = req.get("model_path") or "/data/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots/af44b45f2e35a493886929c6d786e563ec68364d"
    extra = req.get("extra_args") or {}
    load_only = bool(extra.get("load_only", False))
    output = artifacts.get("glb") or next(iter(artifacts.values()))
    cmd = [
        sys.executable,
        "-c",
        "<trellis2 worker inline>",
        "--model_path",
        str(model_path),
        "--image_path",
        req["image_path"],
        "--output",
        output,
    ]
    if not Path(req["image_path"]).is_file():
        return {
            "backend": "trellis",
            "status": "input_missing",
            "env": "trellis",
            "artifacts": artifacts,
            "command": cmd,
            "error": f"image_path does not exist: {req['image_path']}",
        }
    start = time.monotonic()
    try:
        sys.path.insert(0, str(TRELLIS2_ROOT))
        from PIL import Image
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(model_path))
        if hasattr(pipe, "cuda"):
            pipe.cuda()
        if load_only:
            return {
                "backend": "trellis",
                "status": "loaded",
                "env": "trellis",
                "artifacts": artifacts,
                "command": cmd,
                "elapsed_sec": time.monotonic() - start,
                "model_path": str(model_path),
                "runtime_source": str(TRELLIS2_ROOT),
            }
        image = Image.open(req["image_path"])
        trellis_extra = {**extra, "seed": req.get("seed", 42)}
        if req.get("variant") and not trellis_extra.get("pipeline_type"):
            trellis_extra["pipeline_type"] = req["variant"]
        run_kwargs = _trellis2_run_kwargs(trellis_extra)
        result = pipe.run(image, **run_kwargs)
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        if mesh is None:
            return {
                "backend": "trellis",
                "status": "empty_mesh",
                "env": "trellis",
                "artifacts": artifacts,
                "command": cmd,
                "elapsed_sec": time.monotonic() - start,
                "error": "TRELLIS.2 returned no mesh",
                "model_path": str(model_path),
                "runtime_source": str(TRELLIS2_ROOT),
            }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        if hasattr(mesh, "export"):
            mesh.export(output)
        else:
            _export_trellis2_glb(mesh, output, extra)
        return {
            "backend": "trellis",
            "status": "ok",
            "env": "trellis",
            "artifacts": artifacts,
            "command": cmd,
            "elapsed_sec": time.monotonic() - start,
            "model_path": str(model_path),
            "runtime_source": str(TRELLIS2_ROOT),
        }
    except Exception as exc:
        return {
            "backend": "trellis",
            "status": "failed",
            "env": "trellis",
            "artifacts": artifacts,
            "command": cmd,
            "elapsed_sec": time.monotonic() - start,
            "error": f"{type(exc).__name__}: {exc}",
            "model_path": str(model_path),
            "runtime_source": str(TRELLIS2_ROOT),
        }


def run_request(req: dict[str, Any]) -> dict[str, Any]:
    backend = req.get("backend")
    if backend == "hunyuan3d":
        return _run_hunyuan3d(req)
    if backend == "trellis":
        return _run_trellis(req)
    return {"backend": backend, "status": "unsupported_backend", "error": f"unsupported backend: {backend}"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Env-local worker for model-stack 3D generation backends.")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--result-json")
    args = parser.parse_args()
    req = _load_json(args.request_json)
    result = run_request(req)
    _write_json(args.result_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("status") in {"failed", "unsupported_backend"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
