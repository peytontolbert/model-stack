from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path("/arxiv/models/HunyuanVideo-Avatar")
DEFAULT_AVATAR_ROOT = Path("/data/clone/hunyuanvideo-avatar")
DEFAULT_BF16_SHARD_DIR = Path("/data/transformer_10/checkpoints/hunyuan_avatar_bf16_fsdp2")
DEFAULT_FP8_SHARD_DIR = Path("/data/transformer_10/checkpoints/hunyuan_avatar_fp8_fsdp2")


@dataclass(frozen=True)
class HunyuanAvatarPaths:
    model_path: Path = DEFAULT_MODEL_PATH
    avatar_root: Path = DEFAULT_AVATAR_ROOT
    bf16_shard_dir: Path = DEFAULT_BF16_SHARD_DIR
    fp8_shard_dir: Path = DEFAULT_FP8_SHARD_DIR

    @property
    def ckpt_dir(self) -> Path:
        return self.model_path / "ckpts"

    @property
    def transformer_dir(self) -> Path:
        return self.ckpt_dir / "hunyuan-video-t2v-720p" / "transformers"

    @property
    def bf16_transformer_ckpt(self) -> Path:
        return self.transformer_dir / "mp_rank_00_model_states.pt"

    @property
    def fp8_transformer_ckpt(self) -> Path:
        return self.transformer_dir / "mp_rank_00_model_states_fp8.pt"

    @property
    def fp8_map_ckpt(self) -> Path:
        return self.transformer_dir / "mp_rank_00_model_states_fp8_map.pt"

    @property
    def vae_dir(self) -> Path:
        return self.ckpt_dir / "hunyuan-video-t2v-720p" / "vae"

    @property
    def llava_dir(self) -> Path:
        return self.ckpt_dir / "llava_llama_image"

    @property
    def clip_dir(self) -> Path:
        return self.ckpt_dir / "text_encoder_2"

    @property
    def whisper_dir(self) -> Path:
        return self.ckpt_dir / "whisper-tiny"

    @property
    def face_detector(self) -> Path:
        return self.ckpt_dir / "det_align" / "detface.pt"


@dataclass(frozen=True)
class HunyuanAvatarStatus:
    model_id: str
    model_path: str
    avatar_root: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str
    supports_audio: bool
    supports_image: bool
    supports_video: bool
    detail: str
    blockers: tuple[str, ...] = ()
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    available_shard_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class HunyuanAvatarRuntimeProbe:
    status: HunyuanAvatarStatus
    imports: dict[str, bool]
    constants: dict[str, Any]
    errors: dict[str, str]


@dataclass(frozen=True)
class HunyuanAvatarLaunchPlan:
    env: dict[str, str]
    command: tuple[str, ...]
    cwd: str
    mode: str
    shard_dir: str | None = None


def register_hunyuan_avatar_runtime(paths: HunyuanAvatarPaths | None = None) -> HunyuanAvatarPaths:
    selected = paths or HunyuanAvatarPaths()
    root = str(selected.avatar_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ.setdefault("MODEL_BASE", str(selected.model_path.resolve()))
    apply_hunyuan_avatar_compatibility_patches()
    return selected


def apply_hunyuan_avatar_compatibility_patches() -> tuple[str, ...]:
    patches: list[str] = []
    try:
        import transformers.utils as transformers_utils
    except Exception:
        return tuple(patches)
    if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
        transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
        patches.append("compat:transformers_utils_flax_weights_name")
    return tuple(patches)


def hunyuan_avatar_status(paths: HunyuanAvatarPaths | None = None) -> HunyuanAvatarStatus:
    selected = paths or HunyuanAvatarPaths()
    expected = {
        "upstream runtime package": selected.avatar_root / "hymm_sp" / "__init__.py",
        "upstream config parser": selected.avatar_root / "hymm_sp" / "config.py",
        "upstream sampler": selected.avatar_root / "hymm_sp" / "sample_inference_audio.py",
        "ckpts/config.json": selected.ckpt_dir / "config.json",
        "bf16 transformer checkpoint": selected.bf16_transformer_ckpt,
        "fp8 transformer checkpoint": selected.fp8_transformer_ckpt,
        "fp8 transformer map": selected.fp8_map_ckpt,
        "vae weights": selected.vae_dir / "pytorch_model.pt",
        "vae config": selected.vae_dir / "config.json",
        "llava config": selected.llava_dir / "config.json",
        "llava weight index": selected.llava_dir / "model.safetensors.index.json",
        "clip text encoder": selected.clip_dir / "model.safetensors",
        "whisper encoder": selected.whisper_dir / "model.safetensors",
        "face detector": selected.face_detector,
    }
    present = tuple(name for name, path in expected.items() if path.exists())
    missing = tuple(name for name, path in expected.items() if not path.exists())
    shard_dirs = tuple(str(path) for path in (selected.bf16_shard_dir, selected.fp8_shard_dir) if _valid_shard_dir(path))
    if missing:
        status = "incomplete_hunyuan_avatar_bridge_assets"
        runnable = False
        detail = "HunyuanVideo-Avatar local checkpoint/runtime layout is missing required bridge artifacts."
        blockers = tuple(f"missing {name}: {expected[name]}" for name in missing)
    elif not shard_dirs:
        status = "verified_hunyuan_avatar_runtime_assets_needs_shards"
        runnable = False
        detail = (
            "HunyuanVideo-Avatar checkpoint and upstream runtime are present. FSDP/FSDP2 rank-local shards "
            "are not available, so generation should materialize shards before using the memory-safe bridge."
        )
        blockers = ("rank-local FSDP/FSDP2 shard directory missing or incomplete",)
    else:
        status = "verified_hunyuan_avatar_custom_bridge_assets"
        runnable = True
        detail = (
            "HunyuanVideo-Avatar checkpoint, upstream hymm_sp runtime, and rank-local FSDP/FSDP2 shards are present. "
            "Use py311build with MODEL_BASE pointing at the model root and HUNYUAN_AVATAR_ROOT pointing at the upstream checkout."
        )
        blockers = (
            "full generation still has the known LLaVA image-token alignment blocker unless using a fixed prompt/image path",
        )
    return HunyuanAvatarStatus(
        model_id="HunyuanVideo-Avatar",
        model_path=str(selected.model_path),
        avatar_root=str(selected.avatar_root),
        status=status,
        runnable=runnable,
        preferred_env="py311build",
        loader="runtime.hunyuan_avatar_bridge",
        recommended_dtype="fp8_fsdp2_or_bfloat16_fsdp",
        supports_audio=True,
        supports_image=True,
        supports_video=True,
        detail=detail,
        blockers=blockers,
        present_artifacts=present,
        missing_artifacts=missing,
        available_shard_dirs=shard_dirs,
    )


def probe_hunyuan_avatar_runtime(paths: HunyuanAvatarPaths | None = None) -> HunyuanAvatarRuntimeProbe:
    selected = register_hunyuan_avatar_runtime(paths)
    status = hunyuan_avatar_status(selected)
    modules = (
        "hymm_sp.config",
        "hymm_sp.constants",
        "hymm_sp.modules.models_audio",
        "hymm_sp.modules.token_refiner",
        "hymm_sp.sample_inference_audio",
        "hymm_sp.text_encoder",
        "hymm_sp.vae",
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
    constants: dict[str, Any] = {}
    try:
        constants_mod = importlib.import_module("hymm_sp.constants")
        constants = {
            "MODEL_BASE": getattr(constants_mod, "MODEL_BASE", None),
            "VAE_PATH": dict(getattr(constants_mod, "VAE_PATH", {})),
            "TEXT_ENCODER_PATH": dict(getattr(constants_mod, "TEXT_ENCODER_PATH", {})),
            "TOKENIZER_PATH": dict(getattr(constants_mod, "TOKENIZER_PATH", {})),
        }
    except Exception as exc:
        constants = {"error": f"{type(exc).__name__}:{exc}"}
    return HunyuanAvatarRuntimeProbe(status=status, imports=imports, constants=constants, errors=errors)


def build_hunyuan_avatar_launch_plan(
    *,
    paths: HunyuanAvatarPaths | None = None,
    mode: str = "fp8_fsdp2",
    input_csv: str = "input/peyton_avatar_test.csv",
    save_path: str = "results/hunyuan_avatar_model_stack",
    infer_steps: int = 20,
    sample_frames: int = 129,
    seed: int = 1024,
    image_size: int = 704,
    flow_shift: float = 5.0,
    cfg_scale: float = 7.5,
    nproc_per_node: int = 2,
) -> HunyuanAvatarLaunchPlan:
    selected = paths or HunyuanAvatarPaths()
    env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": ".",
        "HUNYUAN_AVATAR_ROOT": str(selected.avatar_root),
        "MODEL_BASE": str(selected.model_path),
    }
    if mode == "fp8_fsdp2":
        shard_dir = selected.fp8_shard_dir
        command = (
            "conda", "run", "--no-capture-output", "-n", "py311build",
            "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
            "scripts/sample_hunyuan_avatar_fp8_fsdp2.py",
            "--shard-dir", str(shard_dir),
            "--use-fp8",
            "--ckpt", str(selected.fp8_transformer_ckpt),
            "--cpu-offload",
        )
    elif mode == "bf16_fsdp":
        shard_dir = selected.bf16_shard_dir
        command = (
            "conda", "run", "--no-capture-output", "-n", "py311build",
            "torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}",
            "scripts/sample_hunyuan_avatar_fsdp.py",
            "--shard-dir", str(shard_dir),
            "--ckpt", str(selected.bf16_transformer_ckpt),
            "--cpu-offload",
        )
    else:
        raise ValueError("mode must be 'fp8_fsdp2' or 'bf16_fsdp'")
    command = command + (
        "--input", input_csv,
        "--save-path", save_path,
        "--infer-steps", str(infer_steps),
        "--sample-n-frames", str(sample_frames),
        "--seed", str(seed),
        "--image-size", str(image_size),
        "--cfg-scale", str(cfg_scale),
        "--flow-shift-eval-video", str(flow_shift),
    )
    return HunyuanAvatarLaunchPlan(env=env, command=command, cwd="/data/transformer_10", mode=mode, shard_dir=str(shard_dir))


def status_to_json(status: HunyuanAvatarStatus) -> str:
    return json.dumps(asdict(status), indent=2, sort_keys=True)


def _valid_shard_dir(path: Path) -> bool:
    manifest = path / "avatar_transformer.manifest.json"
    if not manifest.is_file():
        manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    world_size = int(data.get("world_size", 0) or 0)
    shards = data.get("shards") or [f"avatar_transformer.rank{rank:02d}.pt" for rank in range(world_size)]
    return world_size > 0 and all((path / str(shard)).is_file() for shard in shards)
