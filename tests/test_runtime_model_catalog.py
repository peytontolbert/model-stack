from __future__ import annotations

import json

import pytest
from runtime.diffusers_bridge import DiffusersBridgeOptions
from runtime.model_catalog import (
    ModelCatalogRecord,
    ModelStackRuntimeUnsupported,
    diffusers_catalog_adapter_status,
    diffusers_catalog_status,
    find_catalog_record,
    infer_integration_lane,
    load_catalog_diffusers_component,
    load_catalog_diffusers_lora_adapter,
    load_catalog_diffusers_pipeline,
    load_catalog_model,
    load_model_catalog,
    plan_model_integration,
    primary_lane_records,
)


def _write_index(path):
    data = {
        "models": [
            {
                "id": "llama-ok",
                "relative_path": "llama-ok",
                "path": "llama-ok",
                "library_name": "transformers",
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "tasks": ["text-generation"],
            },
            {
                "id": "qwen-custom",
                "relative_path": "qwen-custom",
                "path": "qwen-custom",
                "library_name": "transformers",
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "tasks": ["text-generation"],
            },
            {
                "id": "flux",
                "relative_path": "flux",
                "path": "flux",
                "library_name": "diffusers",
                "pipeline_tag": "image-to-image",
                "class_name": "FluxPipeline",
                "tasks": ["image-to-image"],
            },
            {
                "id": "asr",
                "relative_path": "asr",
                "path": "asr",
                "library_name": "nemo",
                "pipeline_tag": "automatic-speech-recognition",
                "tasks": ["automatic-speech-recognition"],
            },
            {
                "id": "adapter",
                "relative_path": "adapter",
                "path": "adapter",
                "library_name": "peft",
                "pipeline_tag": "text2text-generation",
                "tasks": ["text2text-generation"],
            },
            {
                "id": "classifier",
                "relative_path": "classifier",
                "path": "classifier",
                "library_name": "transformers",
                "pipeline_tag": "text-classification",
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "tasks": ["text-classification"],
            },
        ],
        "model_stack_runtime_audit": {
            "not_runnable_today": [
                {"id": "qwen-custom", "integration_lane": "transformers_causal_lm_bridge", "status": "not_runnable_in_model_stack_today", "reasons": ["unsupported_model_type"]},
                {"id": "flux", "integration_lane": "diffusers_cuda_bridge", "status": "not_runnable_in_model_stack_today", "reasons": ["custom_pipeline_or_model_class"]},
                {"id": "asr", "integration_lane": "nemo_asr_bridge", "status": "not_runnable_in_model_stack_today", "reasons": ["concrete_asr_backend_missing"]},
                {"id": "adapter", "integration_lane": "peft_adapter_bridge", "status": "not_runnable_in_model_stack_today", "reasons": ["adapter_or_lora_not_standalone"]},
                {"id": "classifier", "integration_lane": "encoder_classifier_bridge", "status": "not_runnable_in_model_stack_today", "reasons": ["encoder_classifier_loader_missing"]},
            ]
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_catalog_preserves_audit_lanes(tmp_path):
    index = tmp_path / "model_index.json"
    _write_index(index)

    records = load_model_catalog(index)
    lanes = {record.id: record.integration_lane for record in records}

    assert lanes["llama-ok"] == "candidate_runnable_today"
    assert lanes["qwen-custom"] == "transformers_causal_lm_bridge"
    assert lanes["flux"] == "diffusers_cuda_bridge"
    assert lanes["asr"] == "nemo_asr_bridge"
    assert lanes["adapter"] == "peft_adapter_bridge"
    assert lanes["classifier"] == "encoder_classifier_bridge"


def test_primary_lane_records_filters_requested_first_wave(tmp_path):
    index = tmp_path / "model_index.json"
    _write_index(index)

    selected = primary_lane_records(load_model_catalog(index))

    assert {record.id for record in selected} == {"qwen-custom", "flux", "asr", "adapter", "classifier"}


def test_plan_model_integration_uses_model_root(tmp_path, monkeypatch):
    index = tmp_path / "model_index.json"
    _write_index(index)
    record = find_catalog_record("flux", index_path=index)

    plan = plan_model_integration(record, model_root="/models")

    assert plan.backend == "diffusers"
    assert plan.loader == "diffusers.DiffusionPipeline.from_pretrained"
    assert plan.local_path == "/models/flux"
    assert "runtime.native" in plan.performance_path

    monkeypatch.setenv("MODEL_STACK_MODEL_ROOT", "/env-models")
    assert plan_model_integration(record).local_path == "/env-models/flux"


def test_peft_requires_base_model_before_importing_optional_dependency(tmp_path):
    index = tmp_path / "model_index.json"
    _write_index(index)

    with pytest.raises(ModelStackRuntimeUnsupported) as exc:
        load_catalog_model("adapter", index_path=index)

    assert exc.value.plan.backend == "peft"
    assert exc.value.plan.lane == "peft_adapter_bridge"


def test_infer_integration_lane_without_audit():
    assert infer_integration_lane({"library_name": "peft"}) == "peft_adapter_bridge"
    assert infer_integration_lane({"library_name": "nemo"}) == "nemo_asr_bridge"
    assert infer_integration_lane({"library_name": "diffusers"}) == "diffusers_cuda_bridge"
    assert infer_integration_lane({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}) == "transformers_causal_lm_bridge"
    assert infer_integration_lane({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}) == "candidate_runnable_today"
    assert infer_integration_lane({"pipeline_tag": "text-classification", "tasks": ["classification"]}) == "encoder_classifier_bridge"


def test_plan_for_manual_record_is_not_runnable():
    record = ModelCatalogRecord(
        id="unknown",
        path="unknown",
        relative_path="unknown",
        library_name=None,
        pipeline_tag=None,
        model_type=None,
        architectures=(),
        class_name=None,
        tasks=(),
        integration_lane="manual_runtime_triage",
        status="not_runnable_in_model_stack_today",
        reasons=("no_standard_hf_model_type_detected",),
        raw={},
    )

    plan = plan_model_integration(record)

    assert not plan.runnable
    assert plan.backend == "manual"
    assert plan.notes == ("no_standard_hf_model_type_detected",)


def test_load_catalog_model_dispatches_diffusers_before_generic_prep(tmp_path, monkeypatch):
    index = tmp_path / "model_index.json"
    _write_index(index)
    calls = {}

    def fake_load_diffusers_pipeline(path, *, options, **kwargs):
        calls["path"] = path
        calls["options"] = options
        calls["kwargs"] = kwargs
        return {"artifact": "diffusers", "path": path}

    monkeypatch.setattr("runtime.diffusers_bridge.load_diffusers_pipeline", fake_load_diffusers_pipeline)

    out = load_catalog_model(
        "flux",
        index_path=index,
        model_root="/models",
        device="cuda:0",
        dtype="bfloat16",
        enable_xformers=True,
        compile_transformer=True,
        variant="bf16",
        device_map="balanced",
        max_memory={0: "20GiB", "cpu": "96GiB"},
        enable_model_cpu_offload=True,
        custom_pipeline="pipeline_flux2",
    )

    assert out == {"artifact": "diffusers", "path": "/models/flux"}
    assert calls["path"] == "/models/flux"
    assert isinstance(calls["options"], DiffusersBridgeOptions)
    assert calls["options"].device == "cuda:0"
    assert calls["options"].dtype == "bfloat16"
    assert calls["options"].enable_xformers is True
    assert calls["options"].compile_transformer is True
    assert calls["options"].variant == "bf16"
    assert calls["options"].device_map == "balanced"
    assert calls["options"].max_memory == {0: "20GiB", "cpu": "96GiB"}
    assert calls["options"].enable_model_cpu_offload is True
    assert calls["kwargs"] == {"custom_pipeline": "pipeline_flux2"}




def test_plan_model_integration_resolves_arxiv_model_root(tmp_path, monkeypatch):
    root = tmp_path / "arxiv_models"
    model_dir = root / "MOSS-SoundEffect-v2.0"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MODEL_STACK_ARXIV_MODEL_ROOT", str(root))

    record = ModelCatalogRecord(
        id="MOSS-SoundEffect-v2.0",
        path="MOSS-SoundEffect-v2.0",
        relative_path="MOSS-SoundEffect-v2.0",
        library_name="diffusers",
        pipeline_tag="text-to-audio",
        model_type=None,
        architectures=(),
        class_name="MossSoundEffectPipeline",
        tasks=("text-to-audio",),
        integration_lane="diffusers_cuda_bridge",
        status="not_runnable_in_model_stack_today",
        reasons=("custom_pipeline_or_model_class",),
        raw={},
    )

    assert plan_model_integration(record).local_path == str(model_dir)


def test_plan_model_integration_resolves_arxiv_provider_subdirectory(tmp_path, monkeypatch):
    root = tmp_path / "arxiv_models"
    model_dir = root / "nvidia" / "ChronoEdit-14B-Diffusers"
    model_dir.mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MODEL_STACK_ARXIV_MODEL_ROOT", str(root))

    record = ModelCatalogRecord(
        id="ChronoEdit-14B-Diffusers",
        path="ChronoEdit-14B-Diffusers",
        relative_path="ChronoEdit-14B-Diffusers",
        library_name="diffusers",
        pipeline_tag="image-to-video",
        model_type=None,
        architectures=(),
        class_name="WanImageToVideoPipeline",
        tasks=("image-to-video",),
        integration_lane="diffusers_cuda_bridge",
        status="not_runnable_in_model_stack_today",
        reasons=("custom_pipeline_or_model_class",),
        raw={"provider": "nvidia"},
    )

    assert plan_model_integration(record).local_path == str(model_dir)

def test_plan_model_integration_resolves_hf_cache_snapshot(tmp_path, monkeypatch):
    index = tmp_path / "model_index.json"
    _write_index(index)
    snapshot = tmp_path / "hf" / "models--black-forest-labs--FLUX.2-dev" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf"))

    record = ModelCatalogRecord(
        id="black-forest-labs--FLUX.2-dev",
        path="black-forest-labs--FLUX.2-dev",
        relative_path="black-forest-labs--FLUX.2-dev",
        library_name="diffusers",
        pipeline_tag="image-to-image",
        model_type=None,
        architectures=(),
        class_name="Flux2Pipeline",
        tasks=("image-to-image",),
        integration_lane="diffusers_cuda_bridge",
        status="not_runnable_in_model_stack_today",
        reasons=("custom_pipeline_or_model_class",),
        raw={},
    )

    assert plan_model_integration(record).local_path == str(snapshot)


def test_plan_model_integration_prefers_complete_hf_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "hf" / "models--black-forest-labs--FLUX.2-dev" / "snapshots"
    partial = cache / "newer-partial"
    complete = cache / "older-complete"
    partial.mkdir(parents=True)
    complete.mkdir(parents=True)
    (partial / "vae").mkdir()
    (complete / "scheduler").mkdir()
    (complete / "vae").mkdir()
    (complete / "model_index.json").write_text(
        json.dumps({"_class_name": "Flux2Pipeline", "scheduler": ["diffusers", "Scheduler"], "vae": ["diffusers", "VAE"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf"))

    record = ModelCatalogRecord(
        id="black-forest-labs--FLUX.2-dev",
        path="black-forest-labs--FLUX.2-dev",
        relative_path="black-forest-labs--FLUX.2-dev",
        library_name="diffusers",
        pipeline_tag="image-to-image",
        model_type=None,
        architectures=(),
        class_name="Flux2Pipeline",
        tasks=("image-to-image",),
        integration_lane="diffusers_cuda_bridge",
        status="not_runnable_in_model_stack_today",
        reasons=("custom_pipeline_or_model_class",),
        raw={},
    )

    assert plan_model_integration(record).local_path == str(complete)




def test_plan_model_integration_prefers_snapshot_with_all_indexed_shards(tmp_path, monkeypatch):
    cache = tmp_path / "hf" / "models--black-forest-labs--FLUX.2-dev" / "snapshots"
    partial = cache / "newer-partial"
    complete = cache / "older-complete"
    for snapshot in (partial, complete):
        transformer = snapshot / "transformer"
        transformer.mkdir(parents=True)
        (snapshot / "model_index.json").write_text(
            json.dumps({"_class_name": "Flux2Pipeline", "transformer": ["diffusers", "Flux2Transformer2DModel"]}),
            encoding="utf-8",
        )
        (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}),
            encoding="utf-8",
        )
        (transformer / "part-1.safetensors").write_bytes(b"present")
    (complete / "transformer" / "part-2.safetensors").write_bytes(b"present")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf"))

    record = ModelCatalogRecord(
        id="black-forest-labs--FLUX.2-dev",
        path="black-forest-labs--FLUX.2-dev",
        relative_path="black-forest-labs--FLUX.2-dev",
        library_name="diffusers",
        pipeline_tag="image-to-image",
        model_type=None,
        architectures=(),
        class_name="Flux2Pipeline",
        tasks=("image-to-image",),
        integration_lane="diffusers_cuda_bridge",
        status="not_runnable_in_model_stack_today",
        reasons=("custom_pipeline_or_model_class",),
        raw={},
    )

    assert plan_model_integration(record).local_path == str(complete)

def test_catalog_diffusers_status_uses_planned_path(tmp_path, monkeypatch):
    index = tmp_path / "model_index.json"
    _write_index(index)
    calls = {}

    def fake_status(path):
        calls["path"] = path
        return {"complete": True}

    monkeypatch.setattr("runtime.diffusers_bridge.diffusers_snapshot_status", fake_status)

    out = diffusers_catalog_status("flux", index_path=index, model_root="/models")

    assert out == {"complete": True}
    assert calls == {"path": "/models/flux"}


def test_catalog_diffusers_component_uses_planned_path_and_options(tmp_path, monkeypatch):
    index = tmp_path / "model_index.json"
    _write_index(index)
    calls = {}

    def fake_load_component(path, component, *, options, component_class=None, **kwargs):
        calls["path"] = path
        calls["component"] = component
        calls["options"] = options
        calls["component_class"] = component_class
        calls["kwargs"] = kwargs
        return {"component": component}

    monkeypatch.setattr("runtime.diffusers_bridge.load_diffusers_component", fake_load_component)

    out = load_catalog_diffusers_component(
        "flux",
        "vae",
        index_path=index,
        model_root="/models",
        device="cuda:1",
        dtype="bfloat16",
        variant="bf16",
        custom=True,
    )

    assert out == {"component": "vae"}
    assert calls["path"] == "/models/flux"
    assert calls["component"] == "vae"
    assert calls["options"].device == "cuda:1"
    assert calls["options"].dtype == "bfloat16"
    assert calls["options"].variant == "bf16"
    assert calls["kwargs"] == {"custom": True}


def test_catalog_diffusers_pipeline_helper_uses_planned_path(tmp_path, monkeypatch):
    index = tmp_path / "model_index.json"
    _write_index(index)
    calls = {}

    def fake_load_pipeline(path, *, options, **kwargs):
        calls["path"] = path
        calls["options"] = options
        calls["kwargs"] = kwargs
        return {"pipeline": path}

    monkeypatch.setattr("runtime.diffusers_bridge.load_diffusers_pipeline", fake_load_pipeline)

    out = load_catalog_diffusers_pipeline(
        "flux",
        index_path=index,
        model_root="/models",
        device="cuda:0",
        dtype="bfloat16",
        device_map="balanced",
        max_memory={0: "20GiB"},
    )

    assert out == {"pipeline": "/models/flux"}
    assert calls["path"] == "/models/flux"
    assert calls["options"].device_map == "balanced"
    assert calls["options"].max_memory == {0: "20GiB"}


def test_catalog_diffusers_helpers_reject_non_diffusers(tmp_path):
    index = tmp_path / "model_index.json"
    _write_index(index)

    with pytest.raises(ModelStackRuntimeUnsupported):
        diffusers_catalog_status("qwen-custom", index_path=index)


class _FakeLoraPipeline:
    def __init__(self):
        self.calls = []

    def load_lora_weights(self, adapter_path, **kwargs):
        self.calls.append((adapter_path, kwargs))




def test_plan_model_integration_resolves_catalog_provider_hf_cache(tmp_path, monkeypatch):
    snapshot = tmp_path / "hf" / "models--briaai--RMBG-2.0" / "snapshots" / "rmbg123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf"))

    record = ModelCatalogRecord(
        id="RMBG-2.0",
        path="RMBG-2.0",
        relative_path="RMBG-2.0",
        library_name="transformers",
        pipeline_tag="image-segmentation",
        model_type=None,
        architectures=(),
        class_name=None,
        tasks=("image-segmentation",),
        integration_lane="encoder_classifier_bridge",
        status="not_runnable_in_model_stack_today",
        reasons=("encoder_classifier_loader_missing",),
        raw={"provider": "briaai"},
    )

    assert plan_model_integration(record).local_path == str(snapshot)

def test_plan_model_integration_resolves_un_namespaced_nvidia_hf_cache(tmp_path, monkeypatch):
    snapshot = tmp_path / "hf" / "models--nvidia--ChronoEdit-14B-Diffusers-Paint-Brush-Lora" / "snapshots" / "adapter123"
    snapshot.mkdir(parents=True)
    (snapshot / "config_paintbrush.json").write_text("{}", encoding="utf-8")
    (snapshot / "paintbrush_lora_diffusers.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hf"))

    record = ModelCatalogRecord(
        id="ChronoEdit-14B-Diffusers-Paint-Brush-Lora",
        path="ChronoEdit-14B-Diffusers-Paint-Brush-Lora",
        relative_path="ChronoEdit-14B-Diffusers-Paint-Brush-Lora",
        library_name="diffusers",
        pipeline_tag="image-editing",
        model_type=None,
        architectures=(),
        class_name=None,
        tasks=("image-editing",),
        integration_lane="diffusers_cuda_bridge",
        status="not_runnable_in_model_stack_today",
        reasons=("adapter_or_lora_not_standalone",),
        raw={},
    )

    assert plan_model_integration(record).local_path == str(snapshot)


def test_catalog_diffusers_adapter_status_uses_planned_path(tmp_path, monkeypatch):
    index = tmp_path / "model_index.json"
    _write_index(index)
    calls = {}

    def fake_status(path):
        calls["path"] = path
        return {"complete": True}

    monkeypatch.setattr("runtime.diffusers_bridge.diffusers_adapter_status", fake_status)

    out = diffusers_catalog_adapter_status("flux", index_path=index, model_root="/models")

    assert out == {"complete": True}
    assert calls == {"path": "/models/flux"}


def test_catalog_diffusers_lora_adapter_uses_planned_path(tmp_path):
    index = tmp_path / "model_index.json"
    _write_index(index)
    adapter = tmp_path / "models" / "flux"
    adapter.mkdir(parents=True)
    (adapter / "paintbrush_lora_diffusers.safetensors").write_bytes(b"weights")
    pipe = _FakeLoraPipeline()

    artifacts = load_catalog_diffusers_lora_adapter(
        pipe,
        "flux",
        index_path=index,
        model_root=str(tmp_path / "models"),
        adapter_name="paintbrush",
    )

    assert artifacts.adapter_path == str(adapter)
    assert artifacts.weight_name == "paintbrush_lora_diffusers.safetensors"
    assert pipe.calls == [
        (str(adapter), {"weight_name": "paintbrush_lora_diffusers.safetensors", "adapter_name": "paintbrush"})
    ]
