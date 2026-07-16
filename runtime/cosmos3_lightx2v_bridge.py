from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_LIGHTX2V_ROOT = Path("/data/clone/third_party/LightX2V")
DEFAULT_MODEL_ROOT = Path("/arxiv/models")


@dataclass(frozen=True)
class Cosmos3LightX2VPaths:
    model_path: Path
    lightx2v_root: Path = DEFAULT_LIGHTX2V_ROOT

    @property
    def transformer_dir(self) -> Path:
        return self.model_path / "transformer"

    @property
    def vae_dir(self) -> Path:
        return self.model_path / "vae"

    @property
    def vision_encoder_dir(self) -> Path:
        return self.model_path / "vision_encoder"

    @property
    def sound_tokenizer_dir(self) -> Path:
        return self.model_path / "sound_tokenizer"


@dataclass(frozen=True)
class Cosmos3LightX2VStatus:
    model_id: str
    model_path: str
    lightx2v_root: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str
    supports_text: bool
    supports_image: bool
    supports_video: bool
    supports_audio: bool
    detail: str
    blockers: tuple[str, ...] = ()
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    transformer_shards: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cosmos3LightX2VProbe:
    status: Cosmos3LightX2VStatus
    imports: dict[str, bool]
    errors: dict[str, str]
    config: dict[str, Any]


@dataclass(frozen=True)
class Cosmos3LightX2VLaunchPlan:
    env: dict[str, str]
    command: tuple[str, ...]
    cwd: str
    mode: str


def register_lightx2v_runtime(paths: Cosmos3LightX2VPaths) -> Cosmos3LightX2VPaths:
    root = str(paths.lightx2v_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return paths


def cosmos3_lightx2v_status(paths: Cosmos3LightX2VPaths, *, model_id: str | None = None) -> Cosmos3LightX2VStatus:
    resolved_id = model_id or paths.model_path.name
    expected = {
        "LightX2V source root": paths.lightx2v_root / "lightx2v" / "__init__.py",
        "LightX2V Cosmos3 runner": paths.lightx2v_root / "lightx2v" / "models" / "runners" / "cosmos3" / "cosmos3_runner.py",
        "LightX2V Cosmos3 transformer": paths.lightx2v_root / "lightx2v" / "models" / "networks" / "cosmos3" / "model.py",
        "LightX2V Cosmos3 scheduler": paths.lightx2v_root / "lightx2v" / "models" / "schedulers" / "cosmos3" / "scheduler.py",
        "LightX2V Cosmos3 config": paths.lightx2v_root / "configs" / "cosmos3" / "cosmos3_super_omni_t2v.json",
        "model_index.json": paths.model_path / "model_index.json",
        "config.json": paths.model_path / "config.json",
        "transformer config": paths.transformer_dir / "config.json",
        "transformer index": paths.transformer_dir / "diffusion_pytorch_model.safetensors.index.json",
        "vae config": paths.vae_dir / "config.json",
        "vae weights": paths.vae_dir / "diffusion_pytorch_model.safetensors",
        "vision encoder config": paths.vision_encoder_dir / "config.json",
        "vision encoder weights": paths.vision_encoder_dir / "model.safetensors",
    }
    has_sound = paths.sound_tokenizer_dir.exists()
    if has_sound:
        expected["sound tokenizer config"] = paths.sound_tokenizer_dir / "config.json"
        expected["sound tokenizer weights"] = paths.sound_tokenizer_dir / "diffusion_pytorch_model.safetensors"
    present = tuple(name for name, path in expected.items() if path.exists())
    missing = tuple(name for name, path in expected.items() if not path.exists())
    shards = _expected_transformer_shards(paths.transformer_dir)
    missing_shards = tuple(str(path) for path in shards if not path.is_file())
    if missing or missing_shards:
        status = "incomplete_cosmos3_lightx2v_assets"
        runnable = False
        detail = "Cosmos3 local checkpoint or LightX2V runtime files are missing."
        blockers = tuple(f"missing {name}: {expected[name]}" for name in missing) + tuple(f"missing transformer shard: {path}" for path in missing_shards)
    else:
        status = "candidate_cosmos3_lightx2v_bridge"
        runnable = True
        detail = (
            "Cosmos3 Diffusers snapshot is complete and LightX2V Cosmos3 runtime source is present. "
            "Use LightX2V in ai as the verified bounded-generation fallback; upstream Diffusers metadata is handled by runtime.cosmos3_omni_diffusers_pipeline."
        )
        blockers = (
            "full generation has not been run from model-stack in this pass",
            "optional fast attention kernels flash_attn3/sageattention are absent; LightX2V imports fall back or log missing kernels",
        )
    return Cosmos3LightX2VStatus(
        model_id=resolved_id,
        model_path=str(paths.model_path),
        lightx2v_root=str(paths.lightx2v_root),
        status=status,
        runnable=runnable,
        preferred_env="ai",
        loader="runtime.cosmos3_lightx2v_bridge",
        recommended_dtype="bfloat16",
        supports_text=True,
        supports_image=True,
        supports_video=True,
        supports_audio=has_sound,
        detail=detail,
        blockers=blockers,
        present_artifacts=present,
        missing_artifacts=missing,
        transformer_shards=tuple(str(path) for path in shards),
    )


def probe_cosmos3_lightx2v_runtime(paths: Cosmos3LightX2VPaths, *, model_id: str | None = None) -> Cosmos3LightX2VProbe:
    register_lightx2v_runtime(paths)
    status = cosmos3_lightx2v_status(paths, model_id=model_id)
    modules = (
        "lightx2v",
        "lightx2v.infer",
        "lightx2v.pipeline",
        "lightx2v.models.runners.cosmos3.cosmos3_runner",
        "lightx2v.models.networks.cosmos3.model",
        "lightx2v.models.schedulers.cosmos3.scheduler",
        "lightx2v.models.video_encoders.hf.cosmos3.vae",
    )
    if status.supports_audio:
        modules += ("lightx2v.models.audio_encoders.hf.cosmos3.sound_tokenizer",)
    imports: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            imports[module_name] = True
        except Exception as exc:
            imports[module_name] = False
            errors[module_name] = f"{type(exc).__name__}:{exc}"
    config = _read_json(paths.model_path / "model_index.json")
    return Cosmos3LightX2VProbe(status=status, imports=imports, errors=errors, config=config)


def build_cosmos3_lightx2v_launch_plan(
    paths: Cosmos3LightX2VPaths,
    *,
    task: str = "t2v",
    config_name: str = "cosmos3_super_omni_t2v.json",
    config_json: str | None = None,
    prompt: str = "A robot arm carefully picks up a small object on a table.",
    negative_prompt: str = "",
    save_result_path: str = "results/cosmos3_lightx2v_bridge.mp4",
    seed: int = 123,
    use_lazy_wrapper: bool = False,
) -> Cosmos3LightX2VLaunchPlan:
    resolved_config_json = Path(config_json) if config_json else paths.lightx2v_root / "configs" / "cosmos3" / config_name
    env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(paths.lightx2v_root),
        "HF_HOME": "/data/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data/huggingface/hub",
        "TRANSFORMERS_CACHE": "/data/huggingface/hub",
    }
    entrypoint = ("scripts/lightx2v_cosmos3_lazy_infer.py",) if use_lazy_wrapper else ("-m", "lightx2v.infer")
    command = (
        "conda", "run", "--no-capture-output", "-n", "ai",
        "python", *entrypoint,
        "--model_cls", "cosmos3",
        "--task", task,
        "--model_path", str(paths.model_path),
        "--config_json", str(resolved_config_json),
        "--prompt", prompt,
        "--negative_prompt", negative_prompt,
        "--save_result_path", save_result_path,
        "--seed", str(seed),
    )
    return Cosmos3LightX2VLaunchPlan(env=env, command=command, cwd="/data/transformer_10", mode="lightx2v_cosmos3")


def status_to_json(status: Cosmos3LightX2VStatus) -> str:
    return json.dumps(asdict(status), indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _expected_transformer_shards(transformer_dir: Path) -> tuple[Path, ...]:
    index_path = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    data = _read_json(index_path)
    filenames = sorted({str(value) for value in data.get("weight_map", {}).values()})
    if not filenames:
        filenames = sorted(path.name for path in transformer_dir.glob("*.safetensors"))
    return tuple(transformer_dir / filename for filename in filenames)
