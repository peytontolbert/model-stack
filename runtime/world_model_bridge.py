from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorldModelBridgeOptions:
    device: str | None = None
    dtype: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = True
    eval_mode: bool = True


@dataclass(frozen=True)
class WorldModelStatus:
    model_id: str
    model_path: str
    family: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str | None
    supports_text: bool = False
    supports_video: bool = False
    projection_dim: int | None = None
    detail: str = ""
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CosmosEmbed1Artifacts:
    model: Any
    processor: Any
    config: Any
    device: Any
    dtype: Any
    status: WorldModelStatus

    def embed_text(self, text: str | list[str]) -> Any:
        texts = [text] if isinstance(text, str) else text
        inputs = self.processor(text=texts).to(self.device, dtype=self.dtype)
        return self.model.get_text_embeddings(**inputs)

    def embed_video(self, videos: Any) -> Any:
        if not self.status.supports_video:
            raise WorldModelBridgeUnsupported(self.status)
        inputs = self.processor(videos=videos).to(self.device, dtype=self.dtype)
        return self.model.get_video_embeddings(**inputs)


@dataclass(frozen=True)
class TransformersWorldModelArtifacts:
    model: Any
    processor: Any | None
    config: Any
    device: Any
    dtype: Any
    status: WorldModelStatus


class WorldModelBridgeUnsupported(RuntimeError):
    def __init__(self, status: WorldModelStatus) -> None:
        blockers = "; ".join(status.blockers or (status.detail,))
        super().__init__(
            f"{status.model_id} is not runnable by world_model_bridge "
            f"(family={status.family}, status={status.status}, preferred_env={status.preferred_env}). {blockers}"
        )
        self.status = status


_COSMOS_EMBED1_PROFILES: dict[str, dict[str, Any]] = {
    "Cosmos-Embed1-224p": {
        "env": "ai",
        "dtype": "bfloat16",
        "supports_video": True,
        "projection_dim": 256,
        "detail": "Verified with AutoProcessor + AutoModel on CUDA using BF16 text and video embeddings.",
        "runnable": True,
    },
    "Cosmos-Embed1-448p": {
        "env": "ai",
        "dtype": "float32",
        "supports_video": True,
        "projection_dim": 768,
        "detail": "Verified with AutoProcessor + AutoModel on CUDA using FP32 text and video embeddings.",
        "runnable": True,
        "blockers": ("BF16 video embedding fails in ViT layer norm: expected scalar type Float but found BFloat16.",),
    },
    "Cosmos-Embed1-448p-anomaly-detection": {
        "env": "py311build",
        "dtype": "float32",
        "supports_video": False,
        "projection_dim": 768,
        "detail": "Verified with AutoProcessor + AutoModel text embeddings under py311build.",
        "runnable": True,
        "blockers": (
            "ai/Transformers 4.57 lacks transformers.modeling_utils.apply_chunking_to_forward required by the local remote-code model.",
            "AutoTokenizer raises KeyError:'CosmosEmbed1Config'; AutoProcessor is the supported path.",
        ),
    },
}

_DIFFUSERS_VIDEO_PROFILES: dict[str, dict[str, Any]] = {
    "ChronoEdit-14B-Diffusers": {
        "env": "ai",
        "dtype": "bfloat16",
        "pipeline_class": "WanImageToVideoPipeline",
        "status": "candidate_diffusers_video_snapshot",
        "detail": (
            "14B Wan image-to-video Diffusers snapshot; base folder is complete after reusing the matching "
            "Wan VAE weights from lingbot-video-dense-1.3b, and VAE/image_encoder component smokes pass "
            "on CUDA with BF16 in ai."
        ),
    },
    "black-forest-labs--FLUX.2-dev": {
        "env": "ai",
        "dtype": "bfloat16",
        "pipeline_class": "Flux2Pipeline",
        "status": "needs_flux2_component_level_transformer_placement",
        "supports_text": True,
        "supports_video": False,
        "detail": (
            "FLUX.2-dev component schemas validate in ai, but generic Diffusers pipeline-level device_map places "
            "the entire ~60GiB BF16 Flux2Transformer2DModel on CPU. model-stack needs a custom Flux2 bridge: "
            "load the transformer at component level with a submodule device_map, skip text_encoder/tokenizer on the hot path, "
            "and pass cached prompt_embeds."
        ),
    },
    "black-forest-labs--FLUX.2-klein-9B": {
        "env": "ai",
        "dtype": "bfloat16",
        "pipeline_class": "Flux2KleinPipeline",
        "status": "verified_flux2_klein_latent_generation",
        "supports_text": True,
        "supports_video": False,
        "detail": (
            "FLUX.2-klein-9B full pipeline placement and 256x256 1-step latent generation pass in ai with "
            "device_map=balanced and explicit max_memory. Cold load is slow without cache, but the bounded warmed run loaded "
            "in 5.46s and generated in 5.03s."
        ),
    },
    "AnyFlow-Wan2.1-T2V-1.3B-Diffusers": {
        "env": "ai",
        "dtype": "bfloat16",
        "pipeline_class": "AnyFlowPipeline",
        "status": "verified_cached_prompt_embeds_cuda_bridge",
        "skip_components": ("text_encoder", "tokenizer"),
        "device_map": "cuda",
        "detail": (
            "1.3B AnyFlow Wan2.1 text-to-video Diffusers snapshot; no-text CUDA BF16 pipeline load "
            "and tiny latent prompt-embeds inference pass in ai with device_map='cuda'. Use cached "
            "prompt_embeds to avoid UMT5 hot-path load."
        ),
    },
    "AnyFlow-FAR-Wan2.1-1.3B-Diffusers": {
        "env": "ai",
        "dtype": "bfloat16",
        "pipeline_class": "AnyFlowFARPipeline",
        "status": "verified_cached_prompt_embeds_cuda_bridge",
        "skip_components": ("text_encoder", "tokenizer"),
        "detail": (
            "1.3B AnyFlow FAR Wan2.1 Diffusers snapshot; no-text CUDA BF16 pipeline load and tiny "
            "latent prompt-embeds inference pass in ai. Use cached prompt_embeds to avoid UMT5 hot-path load."
        ),
    },
    "zai-org--CogVideoX-2b": {
        "env": "ai",
        "dtype": "bfloat16",
        "pipeline_class": "CogVideoXPipeline",
        "status": "candidate_diffusers_video_snapshot",
        "detail": "2B CogVideoX Diffusers snapshot; pipeline class is available in ai, py311build, and trellis, with ai preferred for the current bridge stack.",
    },
}


def world_model_status(model_path: str | Path, *, model_id: str | None = None) -> WorldModelStatus:
    path = Path(model_path)
    resolved_id = model_id or path.name
    if not path.exists():
        return WorldModelStatus(
            model_id=resolved_id,
            model_path=str(path),
            family="missing",
            status="missing_local_path",
            runnable=False,
            preferred_env="manual",
            loader="none",
            recommended_dtype=None,
            detail="Local model path does not exist.",
            blockers=("download or resolver alias required before runtime verification",),
        )

    cosmos_profile = _cosmos_embed1_profile(resolved_id, path)
    if cosmos_profile is not None:
        return WorldModelStatus(
            model_id=resolved_id,
            model_path=str(path),
            family="cosmos_embed1",
            status=str(cosmos_profile.get("status", "verified_transformers_remote_code")),
            runnable=bool(cosmos_profile.get("runnable", False)),
            preferred_env=str(cosmos_profile["env"]),
            loader="runtime.world_model_bridge.load_cosmos_embed1",
            recommended_dtype=str(cosmos_profile["dtype"]),
            supports_text=True,
            supports_video=bool(cosmos_profile["supports_video"]),
            projection_dim=int(cosmos_profile["projection_dim"]),
            detail=str(cosmos_profile["detail"]),
            blockers=tuple(cosmos_profile.get("blockers") or ()),
        )

    model_index = path / "model_index.json"
    if model_index.is_file():
        return _diffusers_world_status(path, resolved_id, model_index)

    if (path / "adapter_config.json").is_file():
        return WorldModelStatus(
            model_id=resolved_id,
            model_path=str(path),
            family="world_adapter",
            status="adapter_needs_base_model",
            runnable=False,
            preferred_env="match_base_model_env",
            loader="peft.PeftModel.from_pretrained(base_model, adapter_path)",
            recommended_dtype=None,
            detail="PEFT adapter files are present, but a runnable base model must be selected first.",
            blockers=("explicit base model missing",),
        )

    lower_id = resolved_id.lower()
    if lower_id.startswith("pe-av-") or lower_id in {"pe-av-small", "pe-av-base", "pe-av-large", "pe-av-base-16-frame"}:
        return _unsupported_custom_status(
            path,
            resolved_id,
            family="pe_av",
            status="needs_pe_av_transformers_bridge",
            env="transformers_main_or_perception_models_env",
            detail="PE-AV requires PeAudioVideoModel and PeAudioVideoProcessor from Transformers main or facebookresearch/perception_models; current ai, py311build, and trellis envs do not expose those classes.",
        )
    if "lightx2v" in lower_id or "int8" in lower_id and "wan2.2" in lower_id:
        return _unsupported_custom_status(
            path,
            resolved_id,
            family="wan_lightx2v",
            status="needs_wan_lightx2v_loader",
            env="ai_or_custom_wan_env",
            detail="Wan2.2 LightX2V int8 split-block checkpoint needs a custom Wan/LightX2V loader and block assembly path.",
        )
    if lower_id in {"egm-4b", "egm-4b-sft"}:
        return WorldModelStatus(
            model_id=resolved_id,
            model_path=str(path),
            family="egm_video",
            status="candidate_transformers_image_text_to_text",
            runnable=True,
            preferred_env="ai",
            loader="runtime.world_model_bridge.load_transformers_image_text_model",
            recommended_dtype="bfloat16",
            supports_text=True,
            supports_video=True,
            detail=(
                f"{resolved_id} constructs as Qwen3VLForConditionalGeneration through "
                "AutoModelForImageTextToText in ai; generic AutoModel is not valid for this checkpoint."
            ),
        )
    if lower_id in {"abot-world-0-5b-lf", "acvlab--abot-world-0-5b-lf"} or "abot-world" in lower_id:
        expected = (
            "Wan2.2_VAE.pth",
            "taew2_2.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "diffusion_pytorch_model.safetensors",
            "google/umt5-xxl/tokenizer.json",
        )
        missing = tuple(name for name in expected if not (path / name).exists())
        return WorldModelStatus(
            model_id=resolved_id,
            model_path=str(path),
            family="abot_world",
            status="verified_generator_cuda_bridge" if not missing else "incomplete_abot_world_checkpoint",
            runnable=not missing,
            preferred_env="abot_world",
            loader="runtime.abot_world_bridge.load_abot_generator_cuda",
            recommended_dtype="bfloat16",
            supports_text=True,
            supports_video=True,
            detail=(
                "ABot-World checkpoint layout is Wan2.2-TI2V based and is not a standalone Diffusers pipeline. "
                "The model-stack ABot bridge verifies direct CUDA BF16 generator load plus one action-conditioned forward; "
                "full rollout still needs cached prompt embeddings/lazy T5 handling."
            ),
            blockers=tuple(f"missing checkpoint file: {name}" for name in missing)
            or (
                "full interactive rollout still needs cached prompt embeddings or lazy T5 construction",
                "optional flash_attn/sageattention/sageattn3 kernels are absent; SDPA fallback is verified but slower",
            ),
        )
    if "cosmos-predict2.5" in lower_id or "cosmos-transfer2.5" in lower_id:
        return _cosmos25_status(path, resolved_id)
    if "hunyuan3d" in lower_id:
        return _hunyuan3d_status(path, resolved_id)
    if "hunyuanworld" in lower_id:
        return _unsupported_custom_status(
            path,
            resolved_id,
            family="hunyuanworld",
            status="needs_hunyuanworld_custom_bridge",
            env="py311build_or_custom",
            detail="Custom HunyuanWorld layout needs a HunyuanWorld-specific loader before generation.",
        )
    if lower_id in {"hunyuanvideo-i2v", "hunyuanvideo-avatar"} or lower_id.startswith("hunyuanvideo-"):
        return _hunyuan_video_status(path, resolved_id, lower_id)
    if lower_id in {"wan-ai--wan2.2-animate-14b", "wan2.2-animate-14b"} or "wan2.2-animate" in lower_id:
        return _wan_animate_status(path, resolved_id)
    if lower_id in {"dam-3b", "dam-3b-video", "dam-3b-self-contained"}:
        return _dam_status(path, resolved_id, lower_id)
    if lower_id.startswith("pixeldit-"):
        return _unsupported_custom_status(
            path,
            resolved_id,
            family="pixeldit",
            status="needs_pixeldit_custom_bridge",
            env="pixeldit_or_custom_bridge",
            detail=(
                "PixelDiT has model_type='pixeldit' and no auto_map; Transformers 4.57 does not recognize "
                "the architecture, so model-stack needs a PixelDiT runtime bridge rather than a generic AutoModel load."
            ),
        )
    if "lingbot-world" in lower_id:
        return _lingbot_world_status(path, resolved_id)
    if "gen3c-cosmos" in lower_id:
        return _gen3c_cosmos_status(path, resolved_id)

    return _unsupported_custom_status(
        path,
        resolved_id,
        family="manual_world_model",
        status="manual_world_triage",
        env="manual",
        detail="No supported world-model bridge family matched this local layout.",
    )


def _lingbot_world_status(path: Path, model_id: str) -> WorldModelStatus:
    try:
        from runtime.lingbot_world_bridge import LingBotWorldPaths, lingbot_world_status
    except Exception as exc:  # pragma: no cover - import failure should still be graceful
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="lingbot_world",
            status="needs_lingbot_world_lightx2v_bridge",
            runnable=False,
            preferred_env="ai",
            loader="runtime.lingbot_world_bridge",
            recommended_dtype="bfloat16",
            supports_text=True,
            supports_video=True,
            detail="LingBot World should route through the LightX2V lingbot_world_fast bridge.",
            blockers=(f"failed to import LingBot World bridge: {type(exc).__name__}:{exc}",),
        )

    status = lingbot_world_status(LingBotWorldPaths(model_path=path), model_id=model_id)
    return WorldModelStatus(
        model_id=status.model_id,
        model_path=status.model_path,
        family="lingbot_world",
        status=status.status,
        runnable=status.runnable,
        preferred_env=status.preferred_env,
        loader=status.loader,
        recommended_dtype=status.recommended_dtype,
        supports_text=status.supports_text,
        supports_video=status.supports_video,
        detail=status.detail,
        blockers=status.blockers,
    )


def _gen3c_cosmos_status(path: Path, model_id: str) -> WorldModelStatus:
    try:
        from runtime.gen3c_cosmos_bridge import gen3c_cosmos_status
    except Exception as exc:  # pragma: no cover - import failure should still be graceful
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="gen3c_cosmos",
            status="needs_gen3c_cosmos_predict1_runtime",
            runnable=False,
            preferred_env="gen3c_cosmos_predict1_or_custom_bridge",
            loader="runtime.gen3c_cosmos_bridge",
            recommended_dtype="bfloat16",
            supports_video=True,
            detail="GEN3C-Cosmos requires the GEN3C/Cosmos-Predict1 runtime bridge.",
            blockers=(f"failed to import GEN3C bridge: {type(exc).__name__}:{exc}",),
        )

    status = gen3c_cosmos_status(path, model_id=model_id)
    return WorldModelStatus(
        model_id=status.model_id,
        model_path=status.model_path,
        family="gen3c_cosmos",
        status=status.status,
        runnable=status.runnable,
        preferred_env=status.preferred_env,
        loader=status.loader,
        recommended_dtype=status.recommended_dtype,
        supports_video=True,
        detail=status.detail,
        blockers=status.blockers,
    )


def _hunyuan3d_status(path: Path, model_id: str) -> WorldModelStatus:
    try:
        from runtime.hunyuan3d_lightx2v_bridge import Hunyuan3DLightX2VPaths, hunyuan3d_lightx2v_status

        h3d_status = hunyuan3d_lightx2v_status(Hunyuan3DLightX2VPaths(model_path=path), model_id=model_id)
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="hunyuan3d_lightx2v" if h3d_status.status != "needs_hunyuan3d_omni_bridge" else "hunyuan3d_omni",
            status=h3d_status.status,
            runnable=h3d_status.runnable,
            preferred_env=h3d_status.preferred_env,
            loader=h3d_status.loader,
            recommended_dtype=h3d_status.recommended_dtype,
            supports_text=False,
            supports_video=False,
            detail=h3d_status.detail,
            blockers=h3d_status.blockers,
        )
    except Exception as exc:
        return _unsupported_custom_status(
            path,
            model_id,
            family="hunyuan3d",
            status="hunyuan3d_lightx2v_bridge_status_failed",
            env="ai",
            detail=f"Hunyuan3D LightX2V bridge status failed: {type(exc).__name__}:{exc}",
        )


def _hunyuan_video_status(path: Path, model_id: str, lower_id: str) -> WorldModelStatus:
    if lower_id == "hunyuanvideo-avatar":
        try:
            from runtime.hunyuan_avatar_bridge import HunyuanAvatarPaths, hunyuan_avatar_status

            avatar_status = hunyuan_avatar_status(HunyuanAvatarPaths(model_path=path))
            return WorldModelStatus(
                model_id=model_id,
                model_path=str(path),
                family="hunyuan_avatar",
                status=avatar_status.status,
                runnable=avatar_status.runnable,
                preferred_env=avatar_status.preferred_env,
                loader=avatar_status.loader,
                recommended_dtype=avatar_status.recommended_dtype,
                supports_text=True,
                supports_video=True,
                detail=avatar_status.detail,
                blockers=avatar_status.blockers,
            )
        except Exception as exc:
            return _unsupported_custom_status(
                path,
                model_id,
                family="hunyuan_avatar",
                status="hunyuan_avatar_bridge_status_failed",
                env="py311build",
                detail=f"Hunyuan Avatar bridge status failed: {type(exc).__name__}:{exc}",
            )
    if lower_id == "hunyuanvideo-1.5":
        has_transformer_weights = any((path / "transformer").glob("**/*.safetensors")) or any((path / "transformer").glob("**/*.pt"))
        status = "incomplete_hunyuanvideo_1_5_snapshot" if not has_transformer_weights else "needs_hunyuanvideo_1_5_bridge"
        detail = (
            "HunyuanVideo-1.5 has a Diffusers-style config with _class_name='HunyuanVideo_1_5_Pipeline', "
            "but the local folder currently contains only scheduler/config metadata and no transformer weights."
            if not has_transformer_weights
            else "HunyuanVideo-1.5 has a model-specific Diffusers pipeline config and needs a HunyuanVideo 1.5 bridge."
        )
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="hunyuan_video",
            status=status,
            runnable=False,
            preferred_env="hunyuanvideo_or_custom_bridge",
            loader="model-specific bridge required",
            recommended_dtype="bfloat16",
            supports_text=True,
            supports_video=True,
            detail=detail,
            blockers=(detail,),
        )
    config_detail = (
        "config.json is not strict JSON; upstream Hunyuan config files use Python-style trailing commas/comments, "
        "so model-stack must load them through a Hunyuan runtime/config adapter."
    )
    return WorldModelStatus(
        model_id=model_id,
        model_path=str(path),
        family="hunyuan_video",
        status="needs_hunyuanvideo_custom_bridge",
        runnable=False,
        preferred_env="py311build_or_hunyuanvideo_env",
        loader="model-specific bridge required",
        recommended_dtype="bfloat16",
        supports_text=True,
        supports_video=True,
        detail=config_detail,
        blockers=(config_detail,),
    )


def _wan_animate_status(path: Path, model_id: str) -> WorldModelStatus:
    expected = (
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00004.safetensors",
        "diffusion_pytorch_model-00002-of-00004.safetensors",
        "diffusion_pytorch_model-00003-of-00004.safetensors",
        "diffusion_pytorch_model-00004-of-00004.safetensors",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    )
    missing = tuple(name for name in expected if not (path / name).is_file())
    detail = (
        "Wan Animate 14B custom checkpoint is complete on disk; model-stack should route it through a Wan Animate "
        "bridge with cached T5 prompt embeddings, cached VAE control latents, and optional int8/offload controls."
        if not missing
        else "Wan Animate 14B custom checkpoint is missing required diffusion or encoder artifacts."
    )
    return WorldModelStatus(
        model_id=model_id,
        model_path=str(path),
        family="wan_animate",
        status="candidate_wan_animate_custom_bridge" if not missing else "incomplete_wan_animate_checkpoint",
        runnable=False,
        preferred_env="py311build_or_custom_wan_env",
        loader="WanAnimate custom bridge required",
        recommended_dtype="bfloat16",
        supports_text=True,
        supports_video=True,
        detail=detail,
        blockers=tuple(f"missing checkpoint file: {name}" for name in missing)
        or ("custom Wan Animate bridge not wired into runtime.world_model_bridge yet",),
    )


def _dam_status(path: Path, model_id: str, lower_id: str) -> WorldModelStatus:
    try:
        from runtime.dam_bridge import dam_status
    except Exception as exc:  # pragma: no cover - import failure should remain graceful
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="dam",
            status="needs_dam_lazy_submodule_bridge",
            runnable=False,
            preferred_env="ai",
            loader="runtime.dam_bridge",
            recommended_dtype="bfloat16",
            supports_text=True,
            supports_video=True,
            detail="DAM should route through the lazy submodule bridge.",
            blockers=(f"failed to import DAM bridge: {type(exc).__name__}:{exc}",),
        )

    status = dam_status(path, model_id=model_id)
    return WorldModelStatus(
        model_id=status.model_id,
        model_path=status.model_path,
        family="dam",
        status=status.status,
        runnable=status.runnable,
        preferred_env=status.preferred_env,
        loader=status.loader,
        recommended_dtype=status.recommended_dtype,
        supports_text=True,
        supports_video=True,
        detail=status.detail,
        blockers=status.blockers,
    )


def _cosmos25_status(path: Path, model_id: str) -> WorldModelStatus:
    try:
        from runtime.cosmos25_bridge import cosmos25_status

        cosmos_status = cosmos25_status(path, model_id=model_id)
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family=cosmos_status.family,
            status=cosmos_status.status,
            runnable=cosmos_status.runnable,
            preferred_env=cosmos_status.preferred_env,
            loader=cosmos_status.loader,
            recommended_dtype=cosmos_status.recommended_dtype,
            supports_text=cosmos_status.supports_text,
            supports_video=cosmos_status.supports_video,
            detail=cosmos_status.detail,
            blockers=cosmos_status.blockers,
        )
    except Exception as exc:
        return _unsupported_custom_status(
            path,
            model_id,
            family="cosmos25",
            status="cosmos25_bridge_status_failed",
            env="cosmos25_or_custom_bridge",
            detail=f"Cosmos 2.5 bridge status failed: {type(exc).__name__}:{exc}",
        )


def load_world_model(model_path: str | Path, *, model_id: str | None = None, options: WorldModelBridgeOptions | None = None) -> Any:
    status = world_model_status(model_path, model_id=model_id)
    if status.family == "cosmos_embed1" and status.runnable:
        return load_cosmos_embed1(model_path, model_id=model_id, options=options)
    if status.family == "diffusers_world" and status.runnable:
        return load_diffusers_world_model(model_path, model_id=model_id, options=options)
    if status.family == "egm_video" and status.runnable:
        return load_transformers_image_text_model(model_path, model_id=model_id, options=options)
    raise WorldModelBridgeUnsupported(status)


def load_transformers_image_text_model(
    model_path: str | Path,
    *,
    model_id: str | None = None,
    options: WorldModelBridgeOptions | None = None,
) -> TransformersWorldModelArtifacts:
    status = world_model_status(model_path, model_id=model_id)
    if status.family != "egm_video" or not status.runnable:
        raise WorldModelBridgeUnsupported(status)
    selected = options or WorldModelBridgeOptions(dtype=status.recommended_dtype)
    try:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # pragma: no cover - optional env packages
        blocked = WorldModelStatus(**{**status.__dict__, "status": "missing_dependency", "runnable": False, "blockers": (f"{type(exc).__name__}:{exc}",)})
        raise WorldModelBridgeUnsupported(blocked) from exc

    dtype = _resolve_torch_dtype(torch, selected.dtype or status.recommended_dtype or "bfloat16")
    device = torch.device(selected.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = str(model_path)
    config = AutoConfig.from_pretrained(path, trust_remote_code=selected.trust_remote_code, local_files_only=selected.local_files_only)
    try:
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=selected.trust_remote_code, local_files_only=selected.local_files_only)
    except Exception:
        processor = None
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        trust_remote_code=selected.trust_remote_code,
        local_files_only=selected.local_files_only,
        dtype=dtype,
    )
    model = model.to(device)
    if selected.eval_mode and hasattr(model, "eval"):
        model.eval()
    return TransformersWorldModelArtifacts(model=model, processor=processor, config=config, device=device, dtype=dtype, status=status)


def load_diffusers_world_model(
    model_path: str | Path,
    *,
    model_id: str | None = None,
    options: WorldModelBridgeOptions | None = None,
) -> Any:
    status = world_model_status(model_path, model_id=model_id)
    if status.family != "diffusers_world" or not status.runnable:
        raise WorldModelBridgeUnsupported(status)
    selected = options or WorldModelBridgeOptions(dtype=status.recommended_dtype)
    profile = _diffusers_video_profile(model_id or Path(model_path).name, Path(model_path)) or {}
    try:
        from runtime.diffusers_bridge import DiffusersBridgeOptions, load_diffusers_pipeline

        bridge_options = DiffusersBridgeOptions(
            device=selected.device,
            dtype=selected.dtype or status.recommended_dtype,
            local_files_only=selected.local_files_only,
            trust_remote_code=selected.trust_remote_code,
            enable_vae_slicing=True,
            channels_last=True,
            device_map=profile.get("device_map"),
            skip_components=tuple(profile.get("skip_components", ())),
        )
        return load_diffusers_pipeline(str(model_path), options=bridge_options)
    except Exception as exc:  # pragma: no cover - depends on optional env packages and GPU memory
        blocked = WorldModelStatus(
            **{
                **status.__dict__,
                "status": "load_failed",
                "runnable": False,
                "blockers": (f"{type(exc).__name__}:{exc}",),
            }
        )
        raise WorldModelBridgeUnsupported(blocked) from exc


def load_cosmos_embed1(
    model_path: str | Path,
    *,
    model_id: str | None = None,
    options: WorldModelBridgeOptions | None = None,
) -> CosmosEmbed1Artifacts:
    status = world_model_status(model_path, model_id=model_id)
    if status.family != "cosmos_embed1" or not status.runnable:
        raise WorldModelBridgeUnsupported(status)

    selected = options or WorldModelBridgeOptions(dtype=status.recommended_dtype)
    dtype_name = selected.dtype or status.recommended_dtype or "bfloat16"

    try:
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor
    except Exception as exc:  # pragma: no cover - depends on optional env packages
        blocked = WorldModelStatus(
            **{**status.__dict__, "status": "missing_dependency", "runnable": False, "blockers": (f"{type(exc).__name__}:{exc}",)}
        )
        raise WorldModelBridgeUnsupported(blocked) from exc

    dtype = _resolve_torch_dtype(torch, dtype_name)
    device = torch.device(selected.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = str(model_path)
    config = AutoConfig.from_pretrained(path, trust_remote_code=selected.trust_remote_code, local_files_only=selected.local_files_only)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=selected.trust_remote_code, local_files_only=selected.local_files_only)
    model = AutoModel.from_pretrained(
        path,
        trust_remote_code=selected.trust_remote_code,
        local_files_only=selected.local_files_only,
        torch_dtype=dtype,
    )
    model = model.to(device)
    if selected.eval_mode and hasattr(model, "eval"):
        model.eval()
    return CosmosEmbed1Artifacts(model=model, processor=processor, config=config, device=device, dtype=dtype, status=status)


def _cosmos_embed1_profile(model_id: str, path: Path) -> dict[str, Any] | None:
    names = (model_id, path.name)
    for name in names:
        if name in _COSMOS_EMBED1_PROFILES:
            return _COSMOS_EMBED1_PROFILES[name]
    if any("Cosmos-Embed1" in name for name in names):
        return {
            "env": "ai_or_py311build",
            "dtype": "float32",
            "supports_video": False,
            "projection_dim": 768 if "448" in " ".join(names) else 256,
            "detail": "Cosmos Embed1 variant detected but not in the verified profile table; run a smoke before enabling it.",
            "status": "candidate_transformers_remote_code",
            "runnable": False,
            "blockers": ("unverified Cosmos Embed1 variant",),
        }
    return None


def _diffusers_video_profile(model_id: str, path: Path) -> dict[str, Any] | None:
    names = (model_id, path.name, model_id.split("/")[-1])
    for name in names:
        if name in _DIFFUSERS_VIDEO_PROFILES:
            return _DIFFUSERS_VIDEO_PROFILES[name]
    return None


def _diffusers_world_status(path: Path, model_id: str, model_index: Path) -> WorldModelStatus:
    try:
        data = json.loads(model_index.read_text(encoding="utf-8"))
    except Exception as exc:
        return _unsupported_custom_status(
            path,
            model_id,
            family="diffusers_world",
            status="invalid_model_index_json",
            env="ai_or_model_specific_env",
            detail=f"model_index.json is not readable: {type(exc).__name__}:{exc}",
        )
    class_name = data.get("_class_name")
    missing_components = tuple(_missing_diffusers_components(path, data))
    class_available = _diffusers_pipeline_available(class_name)
    profile = _diffusers_video_profile(model_id, path)
    if profile is not None and not missing_components and str(class_name) == str(profile.get("pipeline_class")):
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="diffusers_world",
            status=str(profile.get("status", "candidate_diffusers_video_snapshot")),
            runnable=True,
            preferred_env=str(profile["env"]),
            loader="runtime.world_model_bridge.load_diffusers_world_model",
            recommended_dtype=str(profile["dtype"]),
            supports_text=bool(profile.get("supports_text", False)),
            supports_video=bool(profile.get("supports_video", True)),
            detail=str(profile["detail"]),
            blockers=() if class_available else (f"pipeline class {class_name} not importable in the current Python env; use preferred env {profile['env']}",),
        )
    if str(class_name) == "Cosmos3OmniDiffusersPipeline" and not missing_components:
        try:
            from runtime.cosmos3_lightx2v_bridge import Cosmos3LightX2VPaths, cosmos3_lightx2v_status

            cosmos_status = cosmos3_lightx2v_status(Cosmos3LightX2VPaths(model_path=path), model_id=model_id)
            return WorldModelStatus(
                model_id=model_id,
                model_path=str(path),
                family="cosmos3_lightx2v",
                status=cosmos_status.status,
                runnable=cosmos_status.runnable,
                preferred_env=cosmos_status.preferred_env,
                loader=cosmos_status.loader,
                recommended_dtype=cosmos_status.recommended_dtype,
                supports_text=True,
                supports_video=True,
                detail=cosmos_status.detail,
                blockers=cosmos_status.blockers,
            )
        except Exception as exc:
            return _unsupported_custom_status(
                path,
                model_id,
                family="cosmos3_lightx2v",
                status="cosmos3_lightx2v_bridge_status_failed",
                env="ai",
                detail=f"Cosmos3 LightX2V bridge status failed: {type(exc).__name__}:{exc}",
            )
    if str(class_name) == "LingBotVideoPipeline" and not missing_components:
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="lingbot_video",
            status="needs_lingbot_video_pipeline",
            runnable=False,
            preferred_env="ai_or_custom_wan_env",
            loader="LingBotVideoPipeline custom pipeline",
            recommended_dtype="bfloat16",
            supports_video=True,
            detail="Diffusers-style LingBot video snapshot is complete, but current envs do not expose LingBotVideoPipeline.",
            blockers=("missing diffusers pipeline class: LingBotVideoPipeline",),
        )
    if missing_components:
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="diffusers_world",
            status="incomplete_diffusers_world_snapshot",
            runnable=False,
            preferred_env="ai_or_model_specific_env",
            loader="diffusers.DiffusionPipeline.from_pretrained",
            recommended_dtype="bfloat16",
            detail="Diffusers world snapshot is missing required component files or directories.",
            blockers=tuple(f"missing component: {name}" for name in missing_components),
        )
    if class_available:
        return WorldModelStatus(
            model_id=model_id,
            model_path=str(path),
            family="diffusers_world",
            status="candidate_diffusers_world_snapshot",
            runnable=True,
            preferred_env="ai",
            loader="runtime.diffusers_bridge.load_diffusers_pipeline",
            recommended_dtype="bfloat16",
            supports_video=True,
            detail=f"Diffusers snapshot is complete and pipeline class is importable: {class_name}",
        )
    return WorldModelStatus(
        model_id=model_id,
        model_path=str(path),
        family="diffusers_world",
        status="needs_diffusers_pipeline_implementation",
        runnable=False,
        preferred_env="ai_or_cosmos_env",
        loader="diffusers custom pipeline",
        recommended_dtype="bfloat16",
        supports_video=True,
        detail=f"Diffusers layout is complete, but the installed diffusers package lacks pipeline class {class_name}.",
        blockers=(f"missing diffusers pipeline class: {class_name}",),
    )


def _missing_diffusers_components(path: Path, data: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for name, value in data.items():
        if str(name).startswith("_") or not isinstance(value, list):
            continue
        component_path = path / str(name)
        if not component_path.exists():
            missing.append(str(name))
            continue
        for filename in _missing_component_weight_files(component_path, value):
            missing.append(f"{name}/{filename}")
    return missing


def _missing_component_weight_files(component_path: Path, component_spec: Any | None = None) -> tuple[str, ...]:
    if not component_path.is_dir():
        return ()
    missing: list[str] = []
    index_names = ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
    for index_name in index_names:
        index_path = component_path / index_name
        if not index_path.is_file():
            continue
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        filenames = sorted({str(filename) for filename in data.get("weight_map", {}).values()})
        missing.extend(filename for filename in filenames if not (component_path / filename).is_file())
    if missing or any((component_path / name).is_file() for name in index_names):
        return tuple(dict.fromkeys(missing))
    expected = _expected_component_weight_names(component_spec)
    has_component_config = (component_path / "config.json").is_file()
    if has_component_config and expected and not any((component_path / name).is_file() for name in expected):
        missing.append(expected[0])
    return tuple(dict.fromkeys(missing))


def _expected_component_weight_names(component_spec: Any | None) -> tuple[str, ...]:
    if not isinstance(component_spec, list) or len(component_spec) < 2:
        return ()
    library, class_name = str(component_spec[0]), str(component_spec[1]).lower()
    if library == "diffusers" and any(token in class_name for token in ("autoencoder", "transformer", "unet", "model")):
        return ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    if library == "transformers" and any(token in class_name for token in ("model", "encoder", "decoder", "for")):
        return ("model.safetensors", "pytorch_model.bin", "model.bin")
    return ()


def _diffusers_pipeline_available(class_name: Any) -> bool:
    if not class_name or importlib.util.find_spec("diffusers") is None:
        return False
    try:
        import diffusers
        return hasattr(diffusers, str(class_name))
    except Exception:
        return False


def _unsupported_custom_status(path: Path, model_id: str, *, family: str, status: str, env: str, detail: str) -> WorldModelStatus:
    return WorldModelStatus(
        model_id=model_id,
        model_path=str(path),
        family=family,
        status=status,
        runnable=False,
        preferred_env=env,
        loader="model-specific bridge required",
        recommended_dtype=None,
        detail=detail,
        blockers=(detail,),
    )


def _resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    return {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }.get(str(dtype).lower(), torch_module.bfloat16)
