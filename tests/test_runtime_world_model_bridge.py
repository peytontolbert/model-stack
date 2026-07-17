from __future__ import annotations

import json
import sys
import types

import pytest

from runtime.world_model_bridge import (
    WorldModelBridgeOptions,
    WorldModelBridgeUnsupported,
    load_diffusers_world_model,
    load_transformers_image_text_model,
    load_world_model,
    world_model_status,
)


def test_cosmos_embed1_224p_status_is_runnable(tmp_path):
    model_dir = tmp_path / "Cosmos-Embed1-224p"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "cosmos_embed1"}), encoding="utf-8")

    status = world_model_status(model_dir)

    assert status.family == "cosmos_embed1"
    assert status.status == "verified_transformers_remote_code"
    assert status.runnable is True
    assert status.preferred_env == "ai"
    assert status.recommended_dtype == "bfloat16"
    assert status.supports_text is True
    assert status.supports_video is True
    assert status.projection_dim == 256
    assert status.loader == "runtime.world_model_bridge.load_cosmos_embed1"


def test_cosmos_embed1_anomaly_routes_to_py311build_with_blockers(tmp_path):
    model_dir = tmp_path / "Cosmos-Embed1-448p-anomaly-detection"
    model_dir.mkdir()

    status = world_model_status(model_dir)

    assert status.runnable is True
    assert status.preferred_env == "py311build"
    assert status.recommended_dtype == "float32"
    assert status.supports_text is True
    assert status.supports_video is False
    assert any("apply_chunking_to_forward" in blocker for blocker in status.blockers)
    assert any("AutoTokenizer" in blocker for blocker in status.blockers)


def test_unknown_cosmos_embed1_variant_is_not_auto_runnable(tmp_path):
    model_dir = tmp_path / "Cosmos-Embed1-999p"
    model_dir.mkdir()

    status = world_model_status(model_dir)

    assert status.family == "cosmos_embed1"
    assert status.runnable is False
    assert status.status == "candidate_transformers_remote_code"
    assert status.blockers == ("unverified Cosmos Embed1 variant",)
    with pytest.raises(WorldModelBridgeUnsupported):
        load_world_model(model_dir)


def test_cosmos3_diffusers_snapshot_routes_to_lightx2v_bridge(tmp_path):
    model_dir = tmp_path / "Cosmos3-Nano"
    for component in ("transformer", "vae", "vision_encoder", "scheduler"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Cosmos3OmniDiffusersPipeline",
                "transformer": ["diffusers", "Cosmos3OmniTransformer"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "vision_encoder": ["transformers", "Qwen3VLVisionModel"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }
        ),
        encoding="utf-8",
    )

    status = world_model_status(model_dir)

    assert status.family == "cosmos3_lightx2v"
    assert status.status in {"candidate_cosmos3_lightx2v_bridge", "incomplete_cosmos3_lightx2v_assets"}
    assert status.preferred_env == "ai"
    assert status.loader == "runtime.cosmos3_lightx2v_bridge"

    with pytest.raises(WorldModelBridgeUnsupported) as exc:
        load_world_model(model_dir)
    assert exc.value.status is status or exc.value.status.family == status.family


def test_incomplete_diffusers_world_snapshot_lists_missing_components(tmp_path):
    model_dir = tmp_path / "world-partial"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "WanImageToVideoPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]}),
        encoding="utf-8",
    )

    status = world_model_status(model_dir)

    assert status.status == "incomplete_diffusers_world_snapshot"
    assert status.runnable is False
    assert status.blockers == ("missing component: transformer",)


def test_world_adapter_requires_base_model(tmp_path):
    model_dir = tmp_path / "world-planner-adapter"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "unknown"}), encoding="utf-8")

    status = world_model_status(model_dir, model_id="repository_library/world-planner-adapter")

    assert status.family == "world_adapter"
    assert status.status == "adapter_needs_base_model"
    assert status.runnable is False
    assert status.preferred_env == "match_base_model_env"
    assert status.blockers == ("explicit base model missing",)


def test_hunyuanworld_custom_layout_reports_model_specific_bridge(tmp_path):
    model_dir = tmp_path / "HunyuanWorld-Voyager"
    model_dir.mkdir()

    status = world_model_status(model_dir)

    assert status.family == "hunyuanworld"
    assert status.status == "needs_hunyuanworld_custom_bridge"
    assert status.runnable is False
    assert status.preferred_env == "py311build_or_custom"


def test_gen3c_cosmos_routes_to_predict1_runtime_bridge(tmp_path):
    model_dir = tmp_path / "GEN3C-Cosmos-7B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"input_types": ["Cosmos_GEN3C"], "model_size": "7b"}), encoding="utf-8")
    (model_dir / "README.md").write_text("GEN3C", encoding="utf-8")
    (model_dir / "model.pt").write_bytes(b"not-a-zip-but-present")

    status = world_model_status(model_dir)

    assert status.family == "gen3c_cosmos"
    assert status.status == "needs_gen3c_cosmos_predict1_runtime"
    assert status.runnable is False
    assert status.preferred_env == "gen3c_cosmos_predict1_or_custom_bridge"
    assert status.loader == "runtime.gen3c_cosmos_bridge"
    assert any("Cosmos-Predict1" in blocker for blocker in status.blockers)

def test_anyflow_13b_diffusers_video_profile_uses_cached_prompt_bridge(tmp_path):
    model_dir = tmp_path / "AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
    for component in ("transformer", "vae", "tokenizer", "scheduler", "text_encoder"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "AnyFlowPipeline",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "tokenizer": ["transformers", "AutoTokenizer"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }
        ),
        encoding="utf-8",
    )

    status = world_model_status(model_dir)

    assert status.family == "diffusers_world"
    assert status.status == "verified_cached_prompt_embeds_cuda_bridge"
    assert status.runnable is True
    assert status.preferred_env == "ai"
    assert status.recommended_dtype == "bfloat16"
    assert status.supports_video is True
    assert status.loader == "runtime.world_model_bridge.load_diffusers_world_model"
    assert "cached prompt_embeds" in status.detail


def test_anyflow_far_diffusers_video_profile_uses_cached_prompt_bridge(tmp_path):
    model_dir = tmp_path / "AnyFlow-FAR-Wan2.1-1.3B-Diffusers"
    for component in ("transformer", "vae", "tokenizer", "scheduler", "text_encoder"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "AnyFlowFARPipeline",
                "transformer": ["diffusers", "AnyFlowFARTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "tokenizer": ["transformers", "T5TokenizerFast"],
                "text_encoder": ["transformers", "UMT5EncoderModel"],
                "scheduler": ["diffusers", "FlowMapEulerDiscreteScheduler"],
            }
        ),
        encoding="utf-8",
    )

    status = world_model_status(model_dir)

    assert status.family == "diffusers_world"
    assert status.status == "verified_cached_prompt_embeds_cuda_bridge"
    assert status.runnable is True
    assert status.preferred_env == "ai"
    assert "cached prompt_embeds" in status.detail


def test_cogvideox_2b_diffusers_video_profile_is_runnable_candidate(tmp_path):
    model_dir = tmp_path / "zai-org--CogVideoX-2b"
    for component in ("transformer", "vae", "tokenizer", "scheduler", "text_encoder"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "CogVideoXPipeline",
                "transformer": ["diffusers", "CogVideoXTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLCogVideoX"],
                "tokenizer": ["transformers", "T5Tokenizer"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "scheduler": ["diffusers", "CogVideoXDDIMScheduler"],
            }
        ),
        encoding="utf-8",
    )

    status = world_model_status(model_dir)

    assert status.status == "candidate_diffusers_video_snapshot"
    assert status.runnable is True
    assert status.preferred_env == "ai"
    assert "CogVideoX" in status.detail


def test_sapiens2_pose_routes_to_custom_bridge(tmp_path):
    model_dir = tmp_path / "sapiens2-pose-1b"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "sapiens2", "architectures": ["Sapiens2ForPoseEstimation"]}),
        encoding="utf-8",
    )
    (model_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("", encoding="utf-8")

    status = world_model_status(model_dir, model_id="facebook/sapiens2-pose-1b")

    assert status.family == "sapiens2_pose"
    assert status.status == "verified_sapiens2_pose_load_bridge"
    assert status.runnable is True
    assert status.preferred_env == "ai"
    assert status.loader == "runtime.sapiens2_pose_bridge.load_sapiens2_pose_model"


def test_pe_av_and_lightx2v_have_specific_blockers(tmp_path):
    pe = tmp_path / "pe-av-small"
    pe.mkdir()
    (pe / "config.json").write_text("{}", encoding="utf-8")
    (pe / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")
    pe_status = world_model_status(pe)
    assert pe_status.family == "pe_av"
    assert pe_status.status == "needs_pe_av_transformers_bridge"

    light = tmp_path / "wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split"
    light.mkdir()
    (light / "block_0.safetensors").write_text("", encoding="utf-8")
    light_status = world_model_status(light)
    assert light_status.family == "wan_lightx2v"
    assert light_status.status == "needs_wan_lightx2v_loader"


def test_abot_world_checkpoint_routes_to_generator_bridge(tmp_path):
    model_dir = tmp_path / "acvlab--ABot-World-0-5B-LF"
    (model_dir / "google" / "umt5-xxl").mkdir(parents=True)
    for filename in (
        "Wan2.2_VAE.pth",
        "taew2_2.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors",
        "google/umt5-xxl/tokenizer.json",
    ):
        (model_dir / filename).write_text("", encoding="utf-8")

    status = world_model_status(model_dir)

    assert status.family == "abot_world"
    assert status.status == "verified_generator_cuda_bridge"
    assert status.runnable is True
    assert status.preferred_env == "abot_world"
    assert status.recommended_dtype == "bfloat16"
    assert status.supports_text is True
    assert status.supports_video is True
    assert "ABot-World" in status.detail




def test_wan_animate_complete_snapshot_routes_to_custom_bridge(tmp_path):
    model_dir = tmp_path / "Wan-AI--Wan2.2-Animate-14B"
    model_dir.mkdir()
    for filename in (
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00004.safetensors",
        "diffusion_pytorch_model-00002-of-00004.safetensors",
        "diffusion_pytorch_model-00003-of-00004.safetensors",
        "diffusion_pytorch_model-00004-of-00004.safetensors",
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    ):
        (model_dir / filename).write_text("", encoding="utf-8")

    status = world_model_status(model_dir)

    assert status.family == "wan_animate"
    assert status.status == "candidate_wan_animate_custom_bridge"
    assert status.preferred_env == "py311build_or_custom_wan_env"
    assert status.supports_video is True
    assert any("custom Wan Animate bridge" in blocker for blocker in status.blockers)


def test_hunyuanvideo_i2v_reports_config_adapter_needed(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-I2V"
    model_dir.mkdir()

    status = world_model_status(model_dir)

    assert status.family == "hunyuan_video"
    assert status.status == "needs_hunyuanvideo_custom_bridge"
    assert status.preferred_env == "py311build_or_hunyuanvideo_env"
    assert any("not strict JSON" in blocker for blocker in status.blockers)




def test_hunyuanvideo_avatar_routes_to_custom_bridge_status(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-Avatar"
    (model_dir / "ckpts" / "hunyuan-video-t2v-720p" / "transformers").mkdir(parents=True)

    status = world_model_status(model_dir)

    assert status.family == "hunyuan_avatar"
    assert status.preferred_env == "py311build"
    assert status.loader == "runtime.hunyuan_avatar_bridge" or status.status == "incomplete_hunyuan_avatar_bridge_assets"

def test_hunyuanvideo_15_incomplete_snapshot_is_explicit(tmp_path):
    model_dir = tmp_path / "HunyuanVideo-1.5"
    (model_dir / "transformer" / "1080p_sr_distilled").mkdir(parents=True)
    (model_dir / "scheduler").mkdir()

    status = world_model_status(model_dir)

    assert status.family == "hunyuan_video"
    assert status.status == "incomplete_hunyuanvideo_1_5_snapshot"
    assert status.preferred_env == "hunyuanvideo_or_custom_bridge"
    assert any("no transformer weights" in blocker for blocker in status.blockers)


def test_dam_routes_to_lazy_submodule_bridge(tmp_path):
    model_dir = tmp_path / "DAM-3B-Self-Contained"
    model_dir.mkdir()
    (model_dir / "llava_llama.py").write_text("", encoding="utf-8")

    status = world_model_status(model_dir)

    assert status.family == "dam"
    assert status.status == "incomplete_dam_checkpoint"
    assert status.preferred_env == "ai"
    assert status.loader == "runtime.dam_bridge"
    assert any("missing required DAM artifacts" in blocker for blocker in status.blockers)


def test_pixeldit_reports_custom_bridge_needed(tmp_path):
    model_dir = tmp_path / "PixelDiT-1300M-1024px"
    model_dir.mkdir()

    status = world_model_status(model_dir)

    assert status.family == "pixeldit"
    assert status.status == "needs_pixeldit_custom_bridge"
    assert status.preferred_env == "pixeldit_or_custom_bridge"


def test_abot_world_missing_checkpoint_file_is_explicit(tmp_path):
    model_dir = tmp_path / "ABot-World-0-5B-LF"
    model_dir.mkdir()
    (model_dir / "Wan2.2_VAE.pth").write_text("", encoding="utf-8")

    status = world_model_status(model_dir)

    assert status.family == "abot_world"
    assert status.status == "incomplete_abot_world_checkpoint"
    assert status.runnable is False
    assert "missing checkpoint file: diffusion_pytorch_model.safetensors" in status.blockers


def test_egm_4b_routes_to_image_text_transformers_loader(tmp_path):
    for model_name in ("EGM-4B", "EGM-4B-SFT"):
        model_dir = tmp_path / model_name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")

        status = world_model_status(model_dir)

        assert status.family == "egm_video"
        assert status.status == "candidate_transformers_image_text_to_text"
        assert status.runnable is True
        assert status.preferred_env == "ai"
        assert status.loader == "runtime.world_model_bridge.load_transformers_image_text_model"


def test_egm_loader_forwards_transformers_placement_options(monkeypatch, tmp_path):
    model_dir = tmp_path / "EGM-4B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    calls = {}

    class FakeTorch:
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

        class cuda:
            @staticmethod
            def is_available():
                return True

        @staticmethod
        def device(value):
            return value

    class FakeModel:
        def __init__(self):
            self.to_called = False
            self.eval_called = False

        def to(self, device):
            self.to_called = True
            return self

        def eval(self):
            self.eval_called = True
            return self

    fake_model = FakeModel()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["config"] = (path, kwargs)
            return {"ok": True}

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["processor"] = (path, kwargs)
            return "processor"

    class FakeAutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model"] = (path, kwargs)
            return fake_model

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoConfig=FakeAutoConfig,
            AutoModelForImageTextToText=FakeAutoModelForImageTextToText,
            AutoProcessor=FakeAutoProcessor,
        ),
    )

    artifacts = load_transformers_image_text_model(
        model_dir,
        options=WorldModelBridgeOptions(
            device="cuda:1",
            dtype="bfloat16",
            device_map="balanced",
            max_memory={1: "20GiB", "cpu": "64GiB"},
            offload_folder="/data/tmp/egm-offload",
        ),
    )

    assert artifacts.model is fake_model
    assert fake_model.to_called is False
    assert fake_model.eval_called is True
    assert calls["model"][1]["device_map"] == "balanced"
    assert calls["model"][1]["max_memory"] == {1: "20GiB", "cpu": "64GiB"}
    assert calls["model"][1]["low_cpu_mem_usage"] is True
    assert calls["model"][1]["offload_folder"] == "/data/tmp/egm-offload"


def test_diffusers_world_loader_forwards_profile_and_override_optimizations(monkeypatch, tmp_path):
    model_dir = tmp_path / "AnyFlow-Wan2.1-T2V-1.3B-Diffusers"
    for component in ("transformer", "vae", "tokenizer", "scheduler", "text_encoder"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "AnyFlowPipeline",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "tokenizer": ["transformers", "AutoTokenizer"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    def fake_load(path, *, options):
        calls["path"] = path
        calls["options"] = options
        return "pipeline"

    import runtime.diffusers_bridge as diffusers_bridge

    monkeypatch.setattr(diffusers_bridge, "load_diffusers_pipeline", fake_load)

    loaded = load_diffusers_world_model(
        model_dir,
        options=WorldModelBridgeOptions(
            device="cuda:1",
            dtype="bfloat16",
            max_memory={1: "20GiB", "cpu": "96GiB"},
            use_safetensors=True,
            enable_model_cpu_offload=True,
        ),
    )

    assert loaded == "pipeline"
    assert calls["path"] == str(model_dir)
    assert calls["options"].device_map == "cuda"
    assert calls["options"].max_memory == {1: "20GiB", "cpu": "96GiB"}
    assert calls["options"].use_safetensors is True
    assert calls["options"].enable_model_cpu_offload is True
    assert calls["options"].skip_components == ("text_encoder", "tokenizer")


def test_lingbot_video_pipeline_has_specific_blocker(tmp_path):
    model_dir = tmp_path / "lingbot-video-dense-1.3b"
    for component in ("transformer", "vae", "processor", "scheduler", "text_encoder"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "LingBotVideoPipeline",
                "transformer": ["diffusers", "LingBotVideoTransformer"],
                "vae": ["diffusers", "AutoencoderKLWan"],
                "processor": ["transformers", "AutoProcessor"],
                "text_encoder": ["transformers", "AutoModel"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }
        ),
        encoding="utf-8",
    )

    status = world_model_status(model_dir)

    assert status.family == "lingbot_video"
    assert status.status == "needs_lingbot_video_pipeline"
    assert status.runnable is False
    assert status.blockers == ("missing diffusers pipeline class: LingBotVideoPipeline",)

def test_diffusers_world_snapshot_detects_missing_indexed_shard(tmp_path):
    model_dir = tmp_path / "zai-org--CogVideoX-2b"
    for component in ("transformer", "vae", "tokenizer", "scheduler", "text_encoder"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "text_encoder" / "model-00001-of-00002.safetensors").write_text("", encoding="utf-8")
    (model_dir / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "CogVideoXPipeline",
                "transformer": ["diffusers", "CogVideoXTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLCogVideoX"],
                "tokenizer": ["transformers", "T5Tokenizer"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "scheduler": ["diffusers", "CogVideoXDDIMScheduler"],
            }
        ),
        encoding="utf-8",
    )

    status = world_model_status(model_dir)

    assert status.status == "incomplete_diffusers_world_snapshot"
    assert status.runnable is False
    assert status.blockers == ("missing component: text_encoder/model-00002-of-00002.safetensors",)

def test_cosmos25_predict_routes_to_repo_checkpoint_status(tmp_path):
    model_dir = tmp_path / "Cosmos-Predict2.5-14B"
    (model_dir / "base" / "pre-trained").mkdir(parents=True)
    (model_dir / "base" / "post-trained").mkdir(parents=True)
    (model_dir / "README.md").write_text("", encoding="utf-8")
    (model_dir / "base" / "pre-trained" / "54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt").write_text("", encoding="utf-8")
    (model_dir / "base" / "post-trained" / "e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt").write_text("", encoding="utf-8")

    status = world_model_status(model_dir, model_id="nvidia/Cosmos-Predict2.5-14B")

    assert status.family == "cosmos25_predict"
    assert status.status == "candidate_cosmos25_repo_checkpoint"
    assert status.preferred_env == "cosmos25_py310"
    assert status.recommended_dtype == "bfloat16"
    assert "Cosmos 2.5 runtime bridge not wired into load_world_model yet." in status.blockers


def test_cosmos25_transfer_routes_to_repo_checkpoint_status(tmp_path):
    model_dir = tmp_path / "Cosmos-Transfer2.5-2B"
    (model_dir / "auto" / "multiview").mkdir(parents=True)
    (model_dir / "distilled" / "general" / "edge").mkdir(parents=True)
    (model_dir / "README.md").write_text("", encoding="utf-8")
    (model_dir / "auto" / "multiview" / "4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt").write_text("", encoding="utf-8")
    (model_dir / "auto" / "multiview" / "b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt").write_text("", encoding="utf-8")
    (model_dir / "distilled" / "general" / "edge" / "41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt").write_text("", encoding="utf-8")

    status = world_model_status(model_dir, model_id="nvidia/Cosmos-Transfer2.5-2B")

    assert status.family == "cosmos25_transfer"
    assert status.status == "candidate_cosmos25_repo_checkpoint"
    assert status.supports_text is True
    assert status.supports_video is True

def test_hunyuan3d_routes_to_lightx2v_shape_bridge(tmp_path):
    model_dir = tmp_path / "Hunyuan3D-2mini"
    (model_dir / "hunyuan3d-dit-v2-mini").mkdir(parents=True)
    (model_dir / "hunyuan3d-dit-v2-mini" / "config.yaml").write_text("", encoding="utf-8")
    (model_dir / "hunyuan3d-dit-v2-mini" / "model.fp16.safetensors").write_text("", encoding="utf-8")

    status = world_model_status(model_dir, model_id="Hunyuan3D-2mini")

    assert status.family == "hunyuan3d_lightx2v"
    assert status.status in {"candidate_hunyuan3d_lightx2v_bridge", "incomplete_hunyuan3d_lightx2v_assets"}
    assert status.preferred_env == "ai"
    assert status.recommended_dtype == "float16"


def test_hunyuan3d_omni_routes_to_custom_bridge_status(tmp_path):
    model_dir = tmp_path / "Hunyuan3D-Omni"
    for rel in ("config.json", "model/config.json", "model/pytorch_model.bin", "cond_encoder/pytorch_model.bin", "vae/pytorch_model.bin"):
        path = model_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    status = world_model_status(model_dir, model_id="Hunyuan3D-Omni")

    assert status.family == "hunyuan3d_omni"
    assert status.status == "needs_hunyuan3d_omni_bridge"
    assert status.runnable is False
