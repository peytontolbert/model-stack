from __future__ import annotations

import json

from runtime import diffusers_bridge as bridge


class _FakeDType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"torch.{self.name}"


class _FakeDevice:
    def __init__(self, value):
        self.value = str(value)
        self.type = self.value.split(":", 1)[0]

    def __str__(self):
        return self.value


class _FakeCuda:
    @staticmethod
    def is_available():
        return True


class _FakeTorch:
    device = _FakeDevice
    dtype = _FakeDType
    bfloat16 = _FakeDType("bfloat16")
    float16 = _FakeDType("float16")
    float32 = _FakeDType("float32")
    channels_last = "channels_last"
    cuda = _FakeCuda()

    @staticmethod
    def compile(module, mode=None, fullgraph=None):
        module.compiled = (mode, fullgraph)
        return module


class _FakeModule:
    def __init__(self):
        self.to_calls = []
        self.eval_called = False

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        return self

    def eval(self):
        self.eval_called = True
        return self


class _FakePipeline:
    config = {"transformer": ["Fake", {}], "vae": ["Fake", {}]}

    def __init__(self):
        self.transformer = _FakeModule()
        self.vae = _FakeModule()
        self.to_calls = []
        self.vae_slicing = False
        self.attention_slicing = False
        self.xformers = False
        self.model_cpu_offload = None
        self.sequential_cpu_offload = None

    @property
    def components(self):
        return {"transformer": self.transformer, "vae": self.vae}

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        return self

    def enable_vae_slicing(self):
        self.vae_slicing = True

    def enable_attention_slicing(self):
        self.attention_slicing = True

    def enable_xformers_memory_efficient_attention(self):
        self.xformers = True

    def enable_model_cpu_offload(self, device=None):
        self.model_cpu_offload = device

    def enable_sequential_cpu_offload(self, device=None):
        self.sequential_cpu_offload = device


def test_prepare_diffusers_pipeline_applies_cuda_optimizations(monkeypatch):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)
    pipe = _FakePipeline()

    artifacts = bridge.prepare_diffusers_pipeline(
        pipe,
        options=bridge.DiffusersBridgeOptions(
            device="cuda:0",
            dtype="bfloat16",
            enable_attention_slicing=True,
            enable_xformers=True,
            compile_transformer=True,
        ),
    )

    assert artifacts.pipeline is pipe
    assert str(artifacts.device) == "cuda:0"
    assert artifacts.dtype is _FakeTorch.bfloat16
    assert pipe.to_calls == [{"device": artifacts.device, "dtype": _FakeTorch.bfloat16}]
    assert pipe.vae_slicing is True
    assert pipe.attention_slicing is True
    assert pipe.xformers is True
    assert pipe.transformer.eval_called is True
    assert pipe.vae.eval_called is True
    assert "torch_compile:transformer" in artifacts.enabled_optimizations
    assert "channels_last:transformer" in artifacts.enabled_optimizations


def test_build_diffusers_load_kwargs_keeps_local_cuda_defaults(monkeypatch):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)

    kwargs = bridge.build_diffusers_load_kwargs(
        bridge.DiffusersBridgeOptions(dtype="fp16", variant="fp16", use_safetensors=True),
        custom_pipeline="pipeline_custom",
    )

    assert kwargs == {
        "torch_dtype": _FakeTorch.float16,
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "variant": "fp16",
        "use_safetensors": True,
        "custom_pipeline": "pipeline_custom",
    }


def test_build_diffusers_load_kwargs_can_skip_components(monkeypatch):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)

    kwargs = bridge.build_diffusers_load_kwargs(
        bridge.DiffusersBridgeOptions(skip_components=("text_encoder", "tokenizer")),
    )

    assert kwargs["text_encoder"] is None
    assert kwargs["tokenizer"] is None



def test_diffusers_component_report_lists_known_components():
    pipe = _FakePipeline()

    assert bridge.diffusers_component_report(pipe) == {"transformer": "_FakeModule", "vae": "_FakeModule"}


class AnyFlowFARPipeline:
    def __init__(self):
        self.transformer = _FakeModule()


def test_prepare_diffusers_pipeline_patches_anyflow_far_single_tuple_return(monkeypatch):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)
    pipe = AnyFlowFARPipeline()

    def forward(*args, **kwargs):
        return ("sample",)

    pipe.transformer.forward = forward

    artifacts = bridge.prepare_diffusers_pipeline(
        pipe,
        options=bridge.DiffusersBridgeOptions(device="cuda:0", dtype="bfloat16", channels_last=False),
    )

    assert pipe.transformer.forward(return_dict=False) == ("sample", None)
    assert pipe.transformer.forward(return_dict=True) == ("sample",)
    assert "compat:anyflow_far_return_tuple_padding" in artifacts.enabled_optimizations


def test_diffusers_snapshot_status_detects_missing_components(tmp_path):
    model_dir = tmp_path / "flux"
    (model_dir / "scheduler").mkdir(parents=True)
    (model_dir / "text_encoder").mkdir()
    (model_dir / "tokenizer").mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2Pipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Mistral3ForConditionalGeneration"],
                "tokenizer": ["transformers", "PixtralProcessor"],
                "transformer": ["diffusers", "Flux2Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            }
        ),
        encoding="utf-8",
    )

    status = bridge.diffusers_snapshot_status(str(model_dir))

    assert status.class_name == "Flux2Pipeline"
    assert status.required_components == ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
    assert status.present_components == ("scheduler", "text_encoder", "tokenizer")
    assert status.missing_components == ("transformer", "vae")
    assert not status.complete




def test_diffusers_snapshot_status_ignores_scalar_model_index_metadata(tmp_path):
    model_dir = tmp_path / "moss"
    for component in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
        (model_dir / component).mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MossSoundEffectPipeline",
                "transformer": ["WanAudioModel", "transformer"],
                "vae": ["DAC", "vae"],
                "text_encoder": ["Qwen3TextEncoder", "text_encoder"],
                "tokenizer": ["AutoTokenizer", "tokenizer"],
                "scheduler": ["FlowMatchScheduler", "scheduler"],
                "sample_rate": 48000,
                "dit_variant": "1.3B",
            }
        ),
        encoding="utf-8",
    )

    status = bridge.diffusers_snapshot_status(str(model_dir))

    assert status.complete
    assert status.required_components == ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")

def test_load_diffusers_pipeline_rejects_incomplete_snapshot(tmp_path):
    model_dir = tmp_path / "partial"
    model_dir.mkdir()
    (model_dir / "model_index.json").write_text(json.dumps({"_class_name": "X", "unet": ["diffusers", "UNet"]}), encoding="utf-8")

    try:
        bridge.load_diffusers_pipeline(str(model_dir))
    except FileNotFoundError as exc:
        assert "missing unet" in str(exc)
    else:
        raise AssertionError("expected incomplete snapshot to fail before importing weights")


class _FakeParam:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class _FakeComponent:
    loaded = None

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        obj = cls()
        obj.model_path = model_path
        obj.kwargs = kwargs
        obj.to_calls = []
        obj.eval_called = False
        cls.loaded = obj
        return obj

    def to(self, **kwargs):
        self.to_calls.append(kwargs)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return [_FakeParam(2), _FakeParam(3)]


def test_diffusers_component_specs_reads_model_index(tmp_path):
    model_dir = tmp_path / "pipe"
    (model_dir / "vae").mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "Pipe", "vae": ["diffusers", "AutoencoderKL"], "tokenizer": ["transformers", "Tok"]}),
        encoding="utf-8",
    )

    specs = bridge.diffusers_component_specs(str(model_dir))

    assert specs == (
        bridge.DiffusersComponentSpec("vae", "diffusers", "AutoencoderKL", True),
        bridge.DiffusersComponentSpec("tokenizer", "transformers", "Tok", False),
    )


def test_load_diffusers_component_uses_component_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)
    model_dir = tmp_path / "pipe"
    (model_dir / "vae").mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "Pipe", "vae": ["diffusers", "AutoencoderKL"]}),
        encoding="utf-8",
    )

    artifacts = bridge.load_diffusers_component(
        str(model_dir),
        "vae",
        options=bridge.DiffusersBridgeOptions(device="cuda:1", dtype="bfloat16", variant="bf16"),
        component_class=_FakeComponent,
    )

    assert artifacts.name == "vae"
    assert artifacts.class_name == "_FakeComponent"
    assert str(artifacts.device) == "cuda:1"
    assert artifacts.dtype is _FakeTorch.bfloat16
    assert artifacts.parameter_count == 5
    assert artifacts.component.kwargs["subfolder"] == "vae"
    assert artifacts.component.kwargs["variant"] == "bf16"
    assert artifacts.component.to_calls == [{"device": artifacts.device, "dtype": _FakeTorch.bfloat16}]
    assert artifacts.component.eval_called is True


def test_build_diffusers_load_kwargs_includes_device_map_controls(monkeypatch):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)

    kwargs = bridge.build_diffusers_load_kwargs(
        bridge.DiffusersBridgeOptions(
            device_map="balanced",
            max_memory={0: "20GiB", "cpu": "96GiB"},
            low_cpu_mem_usage=True,
        )
    )

    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"] == {0: "20GiB", "cpu": "96GiB"}
    assert kwargs["low_cpu_mem_usage"] is True


def test_load_diffusers_component_respects_managed_placement(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)
    model_dir = tmp_path / "pipe"
    (model_dir / "transformer").mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "Pipe", "transformer": ["diffusers", "WanTransformer3DModel"]}),
        encoding="utf-8",
    )

    artifacts = bridge.load_diffusers_component(
        str(model_dir),
        "transformer",
        options=bridge.DiffusersBridgeOptions(
            device="cuda:0",
            dtype="bfloat16",
            device_map="balanced",
            max_memory={0: "20GiB", "cpu": "96GiB"},
        ),
        component_class=_FakeComponent,
    )

    assert artifacts.component.kwargs["device_map"] == "balanced"
    assert artifacts.component.kwargs["max_memory"] == {0: "20GiB", "cpu": "96GiB"}
    assert artifacts.component.to_calls == []
    assert artifacts.component.eval_called is True


def test_prepare_diffusers_pipeline_uses_offload_without_whole_pipeline_to(monkeypatch):
    monkeypatch.setattr(bridge, "torch", _FakeTorch)
    pipe = _FakePipeline()

    artifacts = bridge.prepare_diffusers_pipeline(
        pipe,
        options=bridge.DiffusersBridgeOptions(
            device="cuda:1",
            dtype="bfloat16",
            device_map="balanced",
            enable_model_cpu_offload=True,
        ),
    )

    assert pipe.to_calls == []
    assert str(pipe.model_cpu_offload) == "cuda:1"
    assert "device_map:balanced" in artifacts.enabled_optimizations
    assert "model_cpu_offload:cuda:1" in artifacts.enabled_optimizations
    assert not any(item.startswith("channels_last") for item in artifacts.enabled_optimizations)


def test_diffusers_snapshot_status_detects_missing_indexed_shards(tmp_path):
    model_dir = tmp_path / "pipe"
    text_encoder = model_dir / "text_encoder"
    text_encoder.mkdir(parents=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "Pipe", "text_encoder": ["transformers", "Encoder"]}),
        encoding="utf-8",
    )
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (text_encoder / "model-00001-of-00002.safetensors").write_bytes(b"present")

    status = bridge.diffusers_snapshot_status(str(model_dir))

    assert status.present_components == ("text_encoder",)
    assert status.missing_components == ("text_encoder/model-00002-of-00002.safetensors",)
    assert not status.complete


def test_diffusers_snapshot_status_detects_missing_unsharded_diffusers_weight(tmp_path):
    model_dir = tmp_path / "wan"
    vae = model_dir / "vae"
    vae.mkdir(parents=True)
    (vae / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "WanImageToVideoPipeline", "vae": ["diffusers", "AutoencoderKLWan"]}),
        encoding="utf-8",
    )

    status = bridge.diffusers_snapshot_status(str(model_dir))

    assert status.present_components == ("vae",)
    assert status.missing_components == ("vae/diffusion_pytorch_model.safetensors",)
    assert not status.complete


class _FakeLoraPipeline:
    def __init__(self):
        self.calls = []

    def load_lora_weights(self, adapter_path, **kwargs):
        self.calls.append((adapter_path, kwargs))




def test_diffusers_adapter_status_ignores_plain_diffusion_root_weights(tmp_path):
    model_dir = tmp_path / "wan"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"weights")

    status = bridge.diffusers_adapter_status(str(model_dir))

    assert not status.complete
    assert status.weight_files == ()

def test_diffusers_adapter_status_detects_lora_files(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "config_paintbrush.json").write_text("{}", encoding="utf-8")
    (adapter / "paintbrush_lora_diffusers.safetensors").write_bytes(b"weights")

    status = bridge.diffusers_adapter_status(str(adapter))

    assert status.complete
    assert status.config_files == ("config_paintbrush.json",)
    assert status.weight_files == ("paintbrush_lora_diffusers.safetensors",)


def test_load_diffusers_lora_adapter_dispatches_to_pipeline(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "config_paintbrush.json").write_text("{}", encoding="utf-8")
    (adapter / "paintbrush_lora_diffusers.safetensors").write_bytes(b"weights")
    pipe = _FakeLoraPipeline()

    artifacts = bridge.load_diffusers_lora_adapter(pipe, str(adapter), adapter_name="paintbrush")

    assert artifacts.pipeline is pipe
    assert artifacts.weight_name == "paintbrush_lora_diffusers.safetensors"
    assert pipe.calls == [
        (str(adapter), {"weight_name": "paintbrush_lora_diffusers.safetensors", "adapter_name": "paintbrush"})
    ]
