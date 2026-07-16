from __future__ import annotations

import importlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_LIGHTX2V_ROOT = Path("/data/clone/third_party/LightX2V")
DEFAULT_T5_CHECKPOINT = Path("/arxiv/models/models_t5_umt5-xxl-enc-bf16.pth")
DEFAULT_WORK_DIR = Path("/data/tmp/model-stack-smokes/lingbot-world")


@dataclass(frozen=True)
class LingBotWorldPaths:
    model_path: Path
    lightx2v_root: Path = DEFAULT_LIGHTX2V_ROOT
    t5_checkpoint: Path = DEFAULT_T5_CHECKPOINT

    @property
    def transformer_dir(self) -> Path:
        if (self.model_path / "transformers").is_dir():
            return self.model_path / "transformers"
        if (self.model_path / "high_noise_model").is_dir():
            return self.model_path / "high_noise_model"
        return self.model_path

    @property
    def vae_checkpoint(self) -> Path:
        return self.model_path / "Wan2.1_VAE.pth"

    @property
    def tokenizer_dir(self) -> Path:
        return self.model_path / "google" / "umt5-xxl"


@dataclass(frozen=True)
class LingBotWorldStatus:
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
    detail: str
    blockers: tuple[str, ...] = ()
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    transformer_shards: tuple[str, ...] = ()
    config_class: str | None = None


@dataclass(frozen=True)
class LingBotWorldProbe:
    status: LingBotWorldStatus
    imports: dict[str, bool]
    errors: dict[str, str]


@dataclass(frozen=True)
class LingBotWorldLaunchPlan:
    env: dict[str, str]
    command: tuple[str, ...]
    cwd: str
    config_json: str
    mode: str


def lingbot_world_status(paths: LingBotWorldPaths, *, model_id: str | None = None) -> LingBotWorldStatus:
    resolved_id = model_id or _model_id_from_path(paths.model_path)
    root_config = paths.model_path / "config.json"
    if not root_config.is_file() and (paths.model_path / "configuration.json").is_file():
        root_config = paths.model_path / "configuration.json"
    config = _read_json(root_config)
    model_config = _read_json(paths.model_path / "config.json") or _read_json(paths.model_path / "low_noise_model" / "config.json") or _read_json(paths.model_path / "high_noise_model" / "config.json")
    class_name = str((model_config or config or {}).get("_class_name") or "")
    expected = {
        "LightX2V source root": paths.lightx2v_root / "lightx2v" / "__init__.py",
        "LightX2V infer CLI": paths.lightx2v_root / "lightx2v" / "infer.py",
        "LightX2V LingBot fast runner": paths.lightx2v_root / "lightx2v" / "models" / "runners" / "wan" / "wan_lingbot_fast_runner.py",
        "LightX2V LingBot fast model": paths.lightx2v_root / "lightx2v" / "models" / "networks" / "wan" / "lingbot_fast_model.py",
        "README.md": paths.model_path / "README.md",
        "root config": root_config,
        "Wan2.1_VAE.pth": paths.vae_checkpoint,
        "google/umt5-xxl tokenizer": paths.tokenizer_dir / "tokenizer.json",
        "shared UMT5 checkpoint": paths.t5_checkpoint,
        "transformer index": paths.transformer_dir / "diffusion_pytorch_model.safetensors.index.json",
    }
    two_stage = (paths.model_path / "high_noise_model").is_dir() and (paths.model_path / "low_noise_model").is_dir()
    if two_stage:
        expected.pop("transformer index", None)
        expected["high_noise_model index"] = paths.model_path / "high_noise_model" / "diffusion_pytorch_model.safetensors.index.json"
        expected["low_noise_model index"] = paths.model_path / "low_noise_model" / "diffusion_pytorch_model.safetensors.index.json"
    present = tuple(name for name, path in expected.items() if path.exists())
    missing = tuple(name for name, path in expected.items() if not path.exists())
    shards = (_safetensor_shards(paths.model_path / "high_noise_model") + _safetensor_shards(paths.model_path / "low_noise_model")) if two_stage else _safetensor_shards(paths.transformer_dir)
    if missing or not shards:
        status = "incomplete_lingbot_world_lightx2v_assets"
        runnable = False
        blockers = tuple(f"missing {name}: {expected[name]}" for name in missing)
        if not shards:
            blockers += (f"missing transformer safetensors shards in {paths.transformer_dir}",)
        detail = "LingBot World local checkpoint or LightX2V runtime files are missing."
    else:
        status = "candidate_lingbot_world_lightx2v_bridge"
        runnable = True
        blockers = (
            "bounded generation has not been run from model-stack yet; start with a tiny frame/step config and cached/shared T5 path",
            "LightX2V logs missing optional flash_attn/sageattention kernels in ai, so attention falls back unless those kernels are installed",
            "full 480p/720p README examples expect multi-GPU torchrun; single-GPU bridge should use offload and reduced frames for first validation",
        )
        if two_stage:
            detail = (
                "LingBot World base preview is a two-stage Wan/LingBot checkpoint. Local LightX2V exposes "
                "model_cls=lingbot_world for high_noise_model/low_noise_model layouts, and required local assets are present."
            )
        else:
            detail = (
                "LingBot World v2 causal-fast is a Wan-style custom checkpoint. Local LightX2V exposes "
                "model_cls=lingbot_world_fast and imports successfully in ai; model-stack can build a launch config "
                "using the local transformer shards plus the shared UMT5 checkpoint."
            )
    return LingBotWorldStatus(
        model_id=resolved_id,
        model_path=str(paths.model_path),
        lightx2v_root=str(paths.lightx2v_root),
        status=status,
        runnable=runnable,
        preferred_env="ai",
        loader="runtime.lingbot_world_bridge",
        recommended_dtype="bfloat16",
        supports_text=True,
        supports_image=True,
        supports_video=True,
        detail=detail,
        blockers=blockers,
        present_artifacts=present,
        missing_artifacts=missing,
        transformer_shards=tuple(str(path) for path in shards),
        config_class=class_name,
    )


def probe_lingbot_world_runtime(paths: LingBotWorldPaths, *, model_id: str | None = None) -> LingBotWorldProbe:
    status = lingbot_world_status(paths, model_id=model_id)
    _register_lightx2v(paths.lightx2v_root)
    modules = (
        "lightx2v",
        "lightx2v.infer",
        "lightx2v.models.runners.wan.wan_lingbot_fast_runner",
        "lightx2v.models.networks.wan.lingbot_fast_model",
        "lightx2v.models.networks.wan.infer.lingbot_fast.transformer_infer",
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
    return LingBotWorldProbe(status=status, imports=imports, errors=errors)


def write_lingbot_world_config(
    paths: LingBotWorldPaths,
    output_path: str | Path,
    *,
    target_height: int = 256,
    target_width: int = 448,
    target_video_length: int = 13,
    infer_steps: int = 1,
    cpu_offload: bool = True,
    vae_cpu_offload: bool = True,
    t5_cpu_offload: bool = True,
) -> Path:
    base_config = paths.lightx2v_root / "configs" / "lingbot_fast" / "lingbot_fast_i2v.json"
    data = json.loads(base_config.read_text(encoding="utf-8"))
    data.update(
        {
            "model_cls": "lingbot_world_fast",
            "task": "i2v",
            "model_path": str(paths.model_path),
            "dit_original_ckpt": str(paths.transformer_dir),
            "t5_original_ckpt": str(paths.t5_checkpoint),
            "vae_name": paths.vae_checkpoint.name,
            "target_height": int(target_height),
            "target_width": int(target_width),
            "target_video_length": int(target_video_length),
            "infer_steps": int(infer_steps),
            "cpu_offload": bool(cpu_offload),
            "vae_cpu_offload": bool(vae_cpu_offload),
            "t5_cpu_offload": bool(t5_cpu_offload),
            "use_image_encoder": False,
            "feature_caching": "NoCaching",
            "self_attn_1_type": "torch_sdpa",
            "cross_attn_1_type": "torch_sdpa",
            "cross_attn_2_type": "torch_sdpa",
        }
    )
    ar_config = dict(data.get("ar_config", {}))
    ar_config.setdefault("num_frame_per_chunk", 3)
    ar_config.setdefault("sink_size", 3)
    ar_config.setdefault("local_attn_size", 21)
    ar_config["kv_offload"] = bool(cpu_offload)
    data["ar_config"] = ar_config
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_lingbot_world_launch_plan(
    paths: LingBotWorldPaths,
    *,
    config_json: str | Path,
    prompt: str,
    image_path: str | Path,
    action_path: str | Path | None = None,
    save_result_path: str | Path | None = None,
    seed: int = 42,
) -> LingBotWorldLaunchPlan:
    command = [
        "python",
        "-m",
        "lightx2v.infer",
        "--seed",
        str(seed),
        "--model_cls",
        "lingbot_world_fast",
        "--task",
        "i2v",
        "--model_path",
        str(paths.model_path),
        "--config_json",
        str(config_json),
        "--prompt",
        prompt,
        "--image_path",
        str(image_path),
    ]
    if action_path is not None:
        command.extend(["--action_path", str(action_path)])
    if save_result_path is not None:
        command.extend(["--save_result_path", str(save_result_path)])
    return LingBotWorldLaunchPlan(
        env={"PYTHONPATH": str(paths.lightx2v_root)},
        command=tuple(command),
        cwd=str(paths.lightx2v_root),
        config_json=str(config_json),
        mode="lightx2v_lingbot_world_fast_i2v",
    )


def status_to_json(status: LingBotWorldStatus) -> dict[str, Any]:
    return asdict(status)


def probe_to_json(probe: LingBotWorldProbe) -> dict[str, Any]:
    return asdict(probe)


def _register_lightx2v(root: Path) -> None:
    import sys

    value = str(root.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safetensor_shards(path: Path) -> list[Path]:
    return sorted(path.glob("*.safetensors")) if path.is_dir() else []


def _model_id_from_path(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 2 and parts[-2] == "robbyant":
        return f"robbyant/{path.name}"
    return path.name
