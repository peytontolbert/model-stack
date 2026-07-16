from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_LIGHTX2V_ROOT = Path("/data/clone/third_party/LightX2V")


@dataclass(frozen=True)
class Hunyuan3DLightX2VPaths:
    model_path: Path
    lightx2v_root: Path = DEFAULT_LIGHTX2V_ROOT


@dataclass(frozen=True)
class Hunyuan3DLightX2VStatus:
    model_id: str
    model_path: str
    lightx2v_root: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str
    supports_image: bool
    supports_3d: bool
    detail: str
    blockers: tuple[str, ...] = ()
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    shape_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hunyuan3DLightX2VProbe:
    status: Hunyuan3DLightX2VStatus
    imports: dict[str, bool]
    errors: dict[str, str]


def register_lightx2v_runtime(paths: Hunyuan3DLightX2VPaths) -> Hunyuan3DLightX2VPaths:
    root = str(paths.lightx2v_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return paths


def hunyuan3d_lightx2v_status(
    paths: Hunyuan3DLightX2VPaths,
    *,
    model_id: str | None = None,
) -> Hunyuan3DLightX2VStatus:
    resolved_id = model_id or paths.model_path.name
    lower_id = resolved_id.lower()
    if "hunyuan3d-omni" in lower_id:
        return _omni_status(paths, resolved_id)

    expected = _expected_common_runtime(paths)
    variants = _shape_variants(paths.model_path)
    if variants:
        for variant in variants:
            expected[f"{variant}/config.yaml"] = paths.model_path / variant / "config.yaml"
            expected[f"{variant}/model"] = _first_existing(
                paths.model_path / variant / "model.fp16.safetensors",
                paths.model_path / variant / "model.fp16.ckpt",
                paths.model_path / variant / "model_fp16.ckpt",
                paths.model_path / variant / "model.safetensors",
                paths.model_path / variant / "model.ckpt",
            )
    else:
        expected["shape variant"] = paths.model_path / "hunyuan3d-dit-v2-1" / "config.yaml"

    present = tuple(name for name, path in expected.items() if path.exists())
    missing = tuple(name for name, path in expected.items() if not path.exists())
    if missing:
        status = "incomplete_hunyuan3d_lightx2v_assets"
        runnable = False
        detail = "Hunyuan3D shape checkpoint or LightX2V Hunyuan3D runtime files are missing."
        blockers = tuple(f"missing {name}: {expected[name]}" for name in missing)
    else:
        status = "candidate_hunyuan3d_lightx2v_bridge"
        runnable = True
        detail = (
            "Hunyuan3D shape assets are present and LightX2V has a Hunyuan3D image-to-3D runner. "
            "Use the LightX2V bridge in ai for shape mesh generation; paint/PBR texturing remains a separate postprocess bridge."
        )
        blockers = (
            "full image-to-3D mesh generation has not been run from model-stack yet",
            "optional flash_attn/sageattention kernels are absent in ai; LightX2V imports fall back or log missing kernels",
        )
    return Hunyuan3DLightX2VStatus(
        model_id=resolved_id,
        model_path=str(paths.model_path),
        lightx2v_root=str(paths.lightx2v_root),
        status=status,
        runnable=runnable,
        preferred_env="ai",
        loader="runtime.hunyuan3d_lightx2v_bridge",
        recommended_dtype="float16",
        supports_image=True,
        supports_3d=True,
        detail=detail,
        blockers=blockers,
        present_artifacts=present,
        missing_artifacts=missing,
        shape_variants=tuple(variants),
    )


def probe_hunyuan3d_lightx2v_runtime(
    paths: Hunyuan3DLightX2VPaths,
    *,
    model_id: str | None = None,
) -> Hunyuan3DLightX2VProbe:
    register_lightx2v_runtime(paths)
    status = hunyuan3d_lightx2v_status(paths, model_id=model_id)
    modules = (
        "lightx2v",
        "lightx2v.models.runners.hunyuan3d.hunyuan3d_shape_runner",
        "lightx2v.models.networks.hunyuan3d.model",
        "lightx2v.models.schedulers.hunyuan3d.scheduler",
        "lightx2v.models.input_encoders.hf.hunyuan3d.encoder",
        "lightx2v.models.video_encoders.hf.hunyuan3d.decoder",
    )
    imports: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            imports[module_name] = True
        except Exception as exc:
            imports[module_name] = False
            errors[module_name] = f"{type(exc).__name__}:{exc}"
    return Hunyuan3DLightX2VProbe(status=status, imports=imports, errors=errors)


def status_to_json(status: Hunyuan3DLightX2VStatus) -> str:
    return json.dumps(asdict(status), indent=2, sort_keys=True)


def probe_to_json(probe: Hunyuan3DLightX2VProbe) -> str:
    return json.dumps(asdict(probe), indent=2, sort_keys=True)


def _expected_common_runtime(paths: Hunyuan3DLightX2VPaths) -> dict[str, Path]:
    return {
        "LightX2V source root": paths.lightx2v_root / "lightx2v" / "__init__.py",
        "LightX2V Hunyuan3D runner": paths.lightx2v_root / "lightx2v" / "models" / "runners" / "hunyuan3d" / "hunyuan3d_shape_runner.py",
        "LightX2V Hunyuan3D transformer": paths.lightx2v_root / "lightx2v" / "models" / "networks" / "hunyuan3d" / "model.py",
        "LightX2V Hunyuan3D scheduler": paths.lightx2v_root / "lightx2v" / "models" / "schedulers" / "hunyuan3d" / "scheduler.py",
        "LightX2V Hunyuan3D condition encoder": paths.lightx2v_root / "lightx2v" / "models" / "input_encoders" / "hf" / "hunyuan3d" / "encoder.py",
        "LightX2V Hunyuan3D VAE decoder": paths.lightx2v_root / "lightx2v" / "models" / "video_encoders" / "hf" / "hunyuan3d" / "decoder.py",
        "LightX2V Hunyuan3D config": paths.lightx2v_root / "configs" / "hunyuan3d" / "hunyuan3d_shape.json",
    }


def _shape_variants(model_path: Path) -> list[str]:
    names = []
    for child in model_path.iterdir() if model_path.exists() else ():
        if child.is_dir() and child.name.startswith("hunyuan3d-dit-") and (child / "config.yaml").is_file():
            names.append(child.name)
    return sorted(names)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _omni_status(paths: Hunyuan3DLightX2VPaths, model_id: str) -> Hunyuan3DLightX2VStatus:
    expected = {
        "config.json": paths.model_path / "config.json",
        "model config": paths.model_path / "model" / "config.json",
        "model weights": paths.model_path / "model" / "pytorch_model.bin",
        "cond encoder": paths.model_path / "cond_encoder" / "pytorch_model.bin",
        "vae": paths.model_path / "vae" / "pytorch_model.bin",
    }
    present = tuple(name for name, path in expected.items() if path.exists())
    missing = tuple(name for name, path in expected.items() if not path.exists())
    detail = (
        "Hunyuan3D-Omni assets are present, but this checkpoint needs the Hunyuan3D-Omni runtime/control bridge; "
        "the current LightX2V Hunyuan3D bridge targets 2.x shape DiT checkpoints."
    )
    return Hunyuan3DLightX2VStatus(
        model_id=model_id,
        model_path=str(paths.model_path),
        lightx2v_root=str(paths.lightx2v_root),
        status="needs_hunyuan3d_omni_bridge" if not missing else "incomplete_hunyuan3d_omni_assets",
        runnable=False,
        preferred_env="hunyuan3d_omni_or_custom_bridge",
        loader="Hunyuan3D-Omni custom bridge required",
        recommended_dtype="float16",
        supports_image=True,
        supports_3d=True,
        detail=detail,
        blockers=tuple(f"missing {name}: {expected[name]}" for name in missing) or (detail,),
        present_artifacts=present,
        missing_artifacts=missing,
    )
