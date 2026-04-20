import json
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime as runtime_pkg
import runtime.cache as cache_mod
import runtime.blocks as runtime_blocks_mod
import runtime.factory as runtime_factory_mod
import runtime.loader as runtime_loader_mod
import runtime.generation as runtime_generation_mod
import runtime.kv_cache as runtime_kv_cache_mod
import runtime.positional as runtime_positional_mod
import runtime.sampling as runtime_sampling_mod
import blocks.block_sparse_attn_block as block_sparse_attn_block_mod
import blocks.local_attn_block as local_attn_block_mod
import blocks.native_fusion as blocks_native_fusion_mod
import blocks.cross_attn_block as cross_attn_block_mod
import blocks.moe_block as moe_block_mod
import blocks.parallel_block as parallel_block_mod
import blocks.segment_bidir_attn_block as segment_bidir_attn_block_mod
import blocks.transformer_block as transformer_block_mod
import blocks.encoder_block as encoder_block_mod
import eval.bench as eval_bench_mod
import eval.cli as eval_cli_mod
import eval.latency as eval_latency_mod
import eval.llama_hf_parity as eval_llama_hf_parity_mod
import eval.suite as eval_suite_mod
import model.causal as causal_model_mod
import model.checkpoint as model_checkpoint_mod
import model.encoder as encoder_model_mod
import model.factory as model_factory_mod
import model.generate as model_generate_mod
import model.hf_llama_loader as model_hf_llama_loader_mod
import model.hf_snapshot as model_hf_snapshot_mod
import model.llama_bootstrap as model_llama_bootstrap_mod
import model.registry as model_registry_mod
import model.seq2seq as seq2seq_model_mod
import model as model_pkg
import runtime.checkpoint as runtime_checkpoint_mod
import runtime.prep as runtime_prep_mod
import export.exporter as export_exporter_mod
import runtime.modeling as runtime_modeling_mod
import serve.api as serve_api_mod
import serve.engine as serve_engine_mod
import serve.generate as serve_generate_mod
import serve.runtime as serve_runtime_mod
import tensor.positional as tensor_positional_mod
import tensor.sampling as tensor_sampling_mod
from attn.kv_cache import ContiguousKVCache, PagedKVCache
from serve.runtime import ModelRuntime
from specs.config import ModelConfig


def _cfg() -> ModelConfig:
    return ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        vocab_size=64,
        dtype="float32",
    )


def test_resolve_kv_cache_backend_prefers_native_paged(monkeypatch):
    monkeypatch.delenv("MODEL_STACK_KV_BACKEND", raising=False)
    monkeypatch.setattr(cache_mod, "native_paged_kv_cache_available", lambda: True)
    monkeypatch.setattr(cache_mod, "has_native_op", lambda name: False)
    assert cache_mod.resolve_kv_cache_backend() == "native-paged"


def test_create_kv_cache_wraps_native_cache_state(monkeypatch):
    fake_native_state = object()
    monkeypatch.setattr(cache_mod, "resolve_kv_cache_backend", lambda requested=None: "native-paged")
    monkeypatch.setattr(
        cache_mod,
        "create_native_paged_kv_cache_state",
        lambda **kwargs: (fake_native_state, None),
    )
    spec = cache_mod.KVCacheSpec(
        batch=2,
        n_layers=3,
        n_kv_heads=4,
        head_dim=8,
        pagesize=128,
        dtype=torch.float32,
        device=torch.device("cpu"),
        backend="native-paged",
    )
    cache = cache_mod.create_kv_cache(spec)
    assert isinstance(cache, PagedKVCache)
    assert cache.backend == "native-paged"
    assert cache._native_cache is fake_native_state
    assert cache._native_layers is None


def test_allocate_model_kv_cache_derives_spec_from_model(monkeypatch):
    model = torch.nn.Linear(16, 16, bias=False)
    model.cfg = _cfg()
    seen = {}

    def fake_create(spec):
        seen["spec"] = spec
        return "cache"

    monkeypatch.setattr(cache_mod, "create_kv_cache", fake_create)
    cache = cache_mod.allocate_model_kv_cache(model, batch_size=5, pagesize=96, backend="paged")
    assert cache == "cache"
    spec = seen["spec"]
    assert spec.batch == 5
    assert spec.n_layers == 2
    assert spec.n_kv_heads == 4
    assert spec.head_dim == 4
    assert spec.pagesize == 96
    assert spec.backend == "paged"


def test_model_runtime_allocate_cache_uses_runtime_factory(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=192)
    seen = {}

    def fake_spec(cfg_in, *, batch_size, dtype, device, pagesize, backend):
        seen["spec_args"] = {
            "cfg": cfg_in,
            "batch_size": batch_size,
            "dtype": dtype,
            "device": device,
            "pagesize": pagesize,
            "backend": backend,
        }
        return "spec"

    def fake_create(spec):
        seen["spec"] = spec
        return {"cache": "ok"}

    monkeypatch.setattr(serve_runtime_mod, "kv_cache_spec_from_config", fake_spec)
    monkeypatch.setattr(serve_runtime_mod, "create_kv_cache", fake_create)

    cache = rt.allocate_cache(batch_size=7, backend="contiguous")

    assert cache == {"cache": "ok"}
    assert seen["spec"] == "spec"
    assert seen["spec_args"]["cfg"] is cfg
    assert seen["spec_args"]["batch_size"] == 7
    assert seen["spec_args"]["dtype"] == torch.float32
    assert seen["spec_args"]["device"] == torch.device("cpu")
    assert seen["spec_args"]["pagesize"] == 192
    assert seen["spec_args"]["backend"] == "contiguous"


def test_clone_kv_cache_rows_uses_native_cache_clone_rows_fast_path():
    class FakeNativeClone:
        def __init__(self):
            self.ids = None

        def clone_rows(self, row_ids):
            self.ids = row_ids.clone()
            return "native-clone"

    native = FakeNativeClone()
    cache = runtime_kv_cache_mod.PagedKVCache(
        batch=2,
        n_layers=1,
        n_kv_heads=1,
        head_dim=2,
        pagesize=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        native_cache_state=native,
        native_layer_states=None,
        backend_name="native-paged",
    )

    cloned = runtime_kv_cache_mod.clone_kv_cache_rows(cache, torch.tensor([1, 0, 1], dtype=torch.long))

    assert native.ids.tolist() == [1, 0, 1]
    assert isinstance(cloned, runtime_kv_cache_mod.PagedKVCache)
    assert cloned._native_cache == "native-clone"
    assert cloned.batch == 3


def test_clone_kv_cache_rows_preserves_native_paged_lengths():
    if not runtime_pkg.native_paged_kv_cache_available():
        pytest.skip("native paged KV cache unavailable")

    spec = cache_mod.KVCacheSpec(
        batch=1,
        n_layers=1,
        n_kv_heads=1,
        head_dim=2,
        pagesize=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        backend="native-paged",
    )
    cache = cache_mod.create_kv_cache(spec)
    assert isinstance(cache, runtime_kv_cache_mod.PagedKVCache)

    cache.append_batch(
        0,
        torch.tensor([[[[1.0, 10.0], [2.0, 20.0]]]], dtype=torch.float32),
        torch.tensor([[[[11.0, 110.0], [12.0, 120.0]]]], dtype=torch.float32),
        block_ids=torch.tensor([0], dtype=torch.long),
    )

    cloned = runtime_kv_cache_mod.clone_kv_cache_rows(cache, torch.tensor([0], dtype=torch.long))

    assert cloned.layer_lengths(0).tolist() == [2]
    assert cloned.layer_max_length(0) == 2
    k, v = cloned.read_batch(0, 0, 2)
    assert torch.equal(k, torch.tensor([[[[1.0, 10.0], [2.0, 20.0]]]], dtype=torch.float32))
    assert torch.equal(v, torch.tensor([[[[11.0, 110.0], [12.0, 120.0]]]], dtype=torch.float32))


def test_clone_kv_cache_rows_clones_selected_contiguous_rows():
    cache = runtime_kv_cache_mod.ContiguousKVCache(
        batch=2,
        n_layers=1,
        n_kv_heads=1,
        head_dim=2,
        pagesize=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        backend_name="contiguous",
    )
    cache.append_batch(
        0,
        torch.tensor([[[[1.0, 10.0], [2.0, 20.0]]]], dtype=torch.float32),
        torch.tensor([[[[11.0, 110.0], [12.0, 120.0]]]], dtype=torch.float32),
        block_ids=torch.tensor([0], dtype=torch.long),
    )
    cache.append_batch(
        0,
        torch.tensor([[[[3.0, 30.0]]]], dtype=torch.float32),
        torch.tensor([[[[13.0, 130.0]]]], dtype=torch.float32),
        block_ids=torch.tensor([1], dtype=torch.long),
    )

    cloned = runtime_kv_cache_mod.clone_kv_cache_rows(cache, torch.tensor([1, 0, 1], dtype=torch.long))

    assert cloned.batch == 3
    assert cloned.backend == "contiguous"
    assert cloned.layer_lengths(0).tolist() == [1, 2, 1]
    k, v = cloned.read_batch(0, 0, 2)
    assert torch.equal(k[0, :, 0, :], torch.tensor([[3.0, 30.0]], dtype=torch.float32))
    assert torch.equal(k[1], torch.tensor([[[1.0, 10.0], [2.0, 20.0]]], dtype=torch.float32))
    assert torch.equal(v[2, :, 0, :], torch.tensor([[13.0, 130.0]], dtype=torch.float32))
    assert torch.equal(k[0, :, 1, :], torch.zeros((1, 2), dtype=torch.float32))


def test_reorder_kv_cache_rows_uses_native_cache_reorder_rows_fast_path():
    class FakeNativeCache:
        def __init__(self):
            self.ids = None

        def reorder_rows_(self, row_ids):
            self.ids = row_ids.clone()

    native = FakeNativeCache()
    cache = runtime_kv_cache_mod.PagedKVCache(
        batch=2,
        n_layers=1,
        n_kv_heads=1,
        head_dim=2,
        pagesize=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        native_cache_state=native,
        native_layer_states=None,
        backend_name="native-paged",
    )

    out = runtime_kv_cache_mod.reorder_kv_cache_rows_(cache, torch.tensor([1, 0, 1], dtype=torch.long))

    assert out is cache
    assert native.ids.tolist() == [1, 0, 1]
    assert cache.batch == 3
    assert cache._native_cache is native


def test_model_runtime_health_info_reports_runtime_owned_payload():
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=192)
    rt.default_kv_cache_backend = "native-paged"
    rt.native_runtime_info = {"ops": ["attention_prefill"]}

    info = rt.health_info()

    assert info["status"] == "ok"
    assert info["device"] == "cpu"
    assert info["dtype"] == "torch.float32"
    assert info["kv_cache_backend"] == "native-paged"
    assert info["native_runtime"] == {"ops": ["attention_prefill"]}


def test_model_runtime_build_generation_config_delegates_to_runtime_builder(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=192)
    seen = {}

    def fake_builder(**kwargs):
        seen["kwargs"] = kwargs
        return "cfg"

    monkeypatch.setattr(serve_runtime_mod, "runtime_build_generation_config", fake_builder)

    built = rt.build_generation_config(max_new_tokens=12, top_k=5, presence_penalty=0.3, beam_size=4, length_penalty=0.8)

    assert built == "cfg"
    assert seen["kwargs"]["max_new_tokens"] == 12
    assert seen["kwargs"]["do_sample"] is True
    assert seen["kwargs"]["top_k"] == 5
    assert seen["kwargs"]["presence_penalty"] == 0.3
    assert seen["kwargs"]["beam_size"] == 4
    assert seen["kwargs"]["length_penalty"] == 0.8


def test_model_runtime_build_generation_config_honors_explicit_greedy_override(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=192)
    seen = {}

    def fake_builder(**kwargs):
        seen["kwargs"] = kwargs
        return "cfg"

    monkeypatch.setattr(serve_runtime_mod, "runtime_build_generation_config", fake_builder)

    built = rt.build_generation_config(do_sample=False, temperature=0.7, top_k=4, top_p=0.9)

    assert built == "cfg"
    assert seen["kwargs"]["do_sample"] is False


def test_model_runtime_generate_token_lists_uses_runtime_owned_coercion_and_generation(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=160)
    seen = {}

    monkeypatch.setattr(rt, "allocate_cache", lambda batch_size, backend=None: ("cache", batch_size, backend))

    def fake_generate(input_ids, **kwargs):
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.tensor([[7, 8, 9]], dtype=torch.long)

    monkeypatch.setattr(rt, "generate", fake_generate)

    out = rt.generate_token_lists([[1, 2, 3]], config="cfg", cache_backend="paged")

    assert out == [[7, 8, 9]]
    assert torch.equal(seen["input_ids"], torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert seen["kwargs"]["cache"] == ("cache", 1, "paged")
    assert seen["kwargs"]["config"] == "cfg"
    assert seen["kwargs"]["cache_backend"] == "paged"


def test_model_runtime_generate_token_ids_rejects_invalid_rank():
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=160)

    try:
        rt.generate_token_ids([1, 2, 3])
    except ValueError as exc:
        assert "rank-2" in str(exc)
    else:
        raise AssertionError("expected ValueError for rank-1 input_ids")


def test_runtime_model_loader_owns_config_build_and_weight_load(monkeypatch):
    cfg = _cfg()
    cfg.dtype = "float16"
    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_calls = []
            self.eval_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls.append({"args": args, "kwargs": kwargs})
            return self

        def eval(self):
            self.eval_calls += 1
            return self

    model = FakeModel()

    def fake_load_config(indir):
        seen["config_dir"] = indir
        return cfg

    def fake_build_model(cfg_in, **kwargs):
        seen["build"] = {"cfg": cfg_in, "kwargs": kwargs}
        return model

    monkeypatch.setattr(runtime_checkpoint_mod, "load_config", fake_load_config)
    monkeypatch.setattr(runtime_factory_mod, "build_model", fake_build_model)

    def fake_load_pretrained(model_in, indir, *, strict=True):
        seen["pretrained"] = {"model": model_in, "indir": indir, "strict": strict}
        return model_in

    monkeypatch.setattr(runtime_checkpoint_mod, "load_pretrained", fake_load_pretrained)

    loaded = runtime_modeling_mod.load_model_dir("/tmp/model-dir", device="cpu", strict=False)

    assert loaded.cfg is cfg
    assert loaded.model is model
    assert loaded.device == torch.device("cpu")
    assert loaded.dtype == torch.float16
    assert seen["config_dir"] == "/tmp/model-dir"
    assert seen["build"]["cfg"] is cfg
    assert seen["build"]["kwargs"]["task"] == "causal-lm"
    assert seen["build"]["kwargs"]["block"] == "llama"
    assert seen["pretrained"] == {"model": model, "indir": "/tmp/model-dir", "strict": False}
    assert model.to_calls[0]["kwargs"]["device"] == torch.device("cpu")
    assert model.to_calls[0]["kwargs"]["dtype"] == torch.float16
    assert model.eval_calls == 1


def test_runtime_model_factory_spec_loader_owns_factory_import_and_runtime_prep(monkeypatch):
    cfg = _cfg()
    cfg.dtype = "float16"
    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_calls = []
            self.eval_calls = 0
            self.cfg = cfg

        def to(self, *args, **kwargs):
            self.to_calls.append({"args": args, "kwargs": kwargs})
            return self

        def eval(self):
            self.eval_calls += 1
            return self

    model = FakeModel()
    fake_module = type("FakeModule", (), {"build_small": lambda self=None: (model, object())})()

    def fake_import_module(name):
        seen["module"] = name
        return fake_module

    monkeypatch.setattr(runtime_loader_mod.importlib, "import_module", fake_import_module)

    loaded = runtime_modeling_mod.load_model_factory_spec("demo.builders:build_small", device="cpu")

    assert seen["module"] == "demo.builders"
    assert loaded.cfg is cfg
    assert loaded.model is model
    assert loaded.device == torch.device("cpu")
    assert loaded.dtype == torch.float16
    assert model.to_calls[0]["kwargs"]["device"] == torch.device("cpu")
    assert model.to_calls[0]["kwargs"]["dtype"] == torch.float16
    assert model.eval_calls == 1


def test_model_runtime_from_dir_delegates_to_runtime_model_loader(monkeypatch):
    cfg = _cfg()
    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_calls = []
            self.eval_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls.append({"args": args, "kwargs": kwargs})
            return self

        def eval(self):
            self.eval_calls += 1
            return self

    model = FakeModel()

    def fake_load_model_dir(model_dir, *, device=None, dtype=None):
        seen["loader"] = {"model_dir": model_dir, "device": device, "dtype": dtype}
        return runtime_modeling_mod.RuntimeModelArtifacts(
            cfg=cfg,
            model=model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    monkeypatch.setattr(serve_runtime_mod, "runtime_load_model_dir", fake_load_model_dir)

    rt = ModelRuntime.from_dir("/models/demo", device="cpu", dtype="float32", kv_pagesize=320)

    assert seen["loader"] == {"model_dir": "/models/demo", "device": "cpu", "dtype": "float32"}
    assert rt.cfg is cfg
    assert rt.model is model
    assert rt.device == torch.device("cpu")
    assert rt.dtype == torch.float32
    assert rt.kv_pagesize == 320
    assert model.to_calls[-1]["args"] == (torch.device("cpu"),)
    assert model.eval_calls >= 1


def test_model_runtime_from_model_delegates_to_runtime_modeling_helpers(monkeypatch):
    cfg = _cfg()
    seen = {}
    input_model = torch.nn.Linear(4, 4, bias=False)
    prepared_model = torch.nn.Linear(4, 4, bias=False)

    def fake_resolve_model_config(model, *, fallback=None):
        seen["resolve"] = {"model": model, "fallback": fallback}
        return cfg

    monkeypatch.setattr(serve_runtime_mod, "runtime_resolve_model_config", fake_resolve_model_config)

    def fake_prepare(model, *, device=None, dtype=None, config_dtype=None):
        seen["prepare"] = {
            "model": model,
            "device": device,
            "dtype": dtype,
            "config_dtype": config_dtype,
        }
        return prepared_model, torch.device("cpu"), torch.float32

    monkeypatch.setattr(serve_runtime_mod, "runtime_prepare_model_for_runtime", fake_prepare)

    rt = ModelRuntime.from_model(input_model, device="cpu", dtype="float32", kv_pagesize=224)

    assert seen["resolve"] == {"model": input_model, "fallback": None}
    assert seen["prepare"]["model"] is input_model
    assert seen["prepare"]["device"] == "cpu"
    assert seen["prepare"]["dtype"] == "float32"
    assert seen["prepare"]["config_dtype"] == "float32"
    assert rt.cfg is cfg
    assert rt.model is prepared_model
    assert rt.device == torch.device("cpu")
    assert rt.dtype == torch.float32
    assert rt.kv_pagesize == 224


def test_eval_cli_load_or_build_uses_runtime_loader_for_model_dirs(monkeypatch):
    fake_model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    def fake_load_runtime_model(*, model_dir=None, factory_spec=None, device=None):
        seen["loader"] = {"model_dir": model_dir, "factory_spec": factory_spec, "device": device}
        return runtime_modeling_mod.RuntimeModelArtifacts(
            cfg=_cfg(),
            model=fake_model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    monkeypatch.setattr(eval_cli_mod, "runtime_load_runtime_model", fake_load_runtime_model)

    args = type("Args", (), {"model_dir": "/models/demo", "model": None})()
    out = eval_cli_mod._load_or_build(args, device="cpu")

    assert out is fake_model
    assert seen["loader"] == {"model_dir": "/models/demo", "factory_spec": None, "device": "cpu"}


def test_eval_cli_load_or_build_uses_runtime_loader_for_factory_specs(monkeypatch):
    fake_model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    def fake_load_runtime_model(*, model_dir=None, factory_spec=None, device=None):
        seen["loader"] = {"model_dir": model_dir, "factory_spec": factory_spec, "device": device}
        return runtime_modeling_mod.RuntimeModelArtifacts(
            cfg=None,
            model=fake_model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    monkeypatch.setattr(eval_cli_mod, "runtime_load_runtime_model", fake_load_runtime_model)

    args = type("Args", (), {"model_dir": None, "model": "demo.builders:build_small"})()
    out = eval_cli_mod._load_or_build(args, device="cpu")

    assert out is fake_model
    assert seen["loader"] == {"model_dir": None, "factory_spec": "demo.builders:build_small", "device": "cpu"}


def test_eval_cli_bench_generate_uses_runtime_from_model_for_cache_factory(monkeypatch):
    fake_model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    class FakeRuntime:
        @classmethod
        def from_model(cls, model, *, device=None):
            seen["from_model"] = {"model": model, "device": device}
            inst = cls()
            inst.allocate_cache = lambda batch_size, backend=None: ("cache", batch_size, backend)
            return inst

    monkeypatch.setattr(eval_cli_mod, "_load_or_build", lambda args, device=None: fake_model)
    monkeypatch.setattr(serve_runtime_mod, "ModelRuntime", FakeRuntime)

    def fake_benchmark_generate(model, **kwargs):
        seen["bench"] = {"model": model, "kwargs": kwargs}
        cache = (
            kwargs["kv_cache_factory"](batch_size=3, backend=kwargs.get("cache_backend"))
            if kwargs.get("kv_cache_factory") is not None
            else None
        )
        seen["cache"] = cache
        return eval_bench_mod.ThroughputResult(
            tokens_per_sec=1.0,
            latency_ms=2.0,
            total_tokens=3,
            total_time_s=4.0,
        )

    monkeypatch.setattr(eval_cli_mod, "benchmark_generate", fake_benchmark_generate)

    args = type(
        "Args",
        (),
        {
            "device": "cpu",
            "batch_size": 1,
            "seq_len": 2,
            "max_new_tokens": 3,
            "vocab_size": 8,
            "warmup": 0,
            "repeats": 1,
            "do_sample": None,
            "temperature": 0.7,
            "top_k": 5,
            "top_p": 0.9,
            "eos_id": 2,
            "no_repeat_ngram": 3,
            "repetition_penalty": 1.2,
            "presence_penalty": 0.4,
            "frequency_penalty": 0.2,
            "sliding_window": 32,
            "cache_backend": "native-paged",
            "outdir": None,
            "viz_log_dir": None,
            "model_dir": None,
            "model": "demo.builders:build_small",
        },
    )()

    eval_cli_mod.cmd_bench_gen(args)

    assert seen["from_model"]["model"] is fake_model
    assert seen["from_model"]["device"] == torch.device("cpu")
    assert seen["bench"]["model"] is fake_model
    assert seen["cache"] == ("cache", 3, "native-paged")
    assert seen["bench"]["kwargs"]["temperature"] == 0.7
    assert seen["bench"]["kwargs"]["top_k"] == 5
    assert seen["bench"]["kwargs"]["top_p"] == 0.9
    assert seen["bench"]["kwargs"]["repetition_penalty"] == 1.2
    assert seen["bench"]["kwargs"]["cache_backend"] == "native-paged"


def test_eval_cli_latency_generate_uses_runtime_from_model_for_cache_factory(monkeypatch):
    fake_model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    class FakeRuntime:
        @classmethod
        def from_model(cls, model, *, device=None):
            seen["from_model"] = {"model": model, "device": device}
            inst = cls()
            inst.allocate_cache = lambda batch_size, backend=None: ("cache", batch_size, backend)
            return inst

    monkeypatch.setattr(eval_cli_mod, "_load_or_build", lambda args, device=None: fake_model)
    monkeypatch.setattr(serve_runtime_mod, "ModelRuntime", FakeRuntime)

    def fake_latency_generate(model, **kwargs):
        seen["latency"] = {"model": model, "kwargs": kwargs}
        cache = (
            kwargs["kv_cache_factory"](batch_size=2, backend=kwargs.get("cache_backend"))
            if kwargs.get("kv_cache_factory") is not None
            else None
        )
        seen["cache"] = cache
        return eval_latency_mod.LatencyDist(p50_ms=1.0, p95_ms=2.0, p99_ms=3.0, samples=4)

    monkeypatch.setattr(eval_cli_mod, "latency_generate", fake_latency_generate)

    args = type(
        "Args",
        (),
        {
            "mode": "generate",
            "device": "cpu",
            "repeats": 3,
            "batch_size": 1,
            "seq_len": 2,
            "max_new_tokens": 4,
            "vocab_size": 8,
            "do_sample": True,
            "temperature": 0.5,
            "top_k": 6,
            "top_p": 0.8,
            "eos_id": 7,
            "no_repeat_ngram": 2,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.2,
            "sliding_window": 24,
            "cache_backend": "paged",
            "outdir": None,
            "model_dir": None,
            "model": "demo.builders:build_small",
        },
    )()

    eval_cli_mod.cmd_latency(args)

    assert seen["from_model"]["model"] is fake_model
    assert seen["from_model"]["device"] == torch.device("cpu")
    assert seen["latency"]["model"] is fake_model
    assert seen["cache"] == ("cache", 2, "paged")
    assert seen["latency"]["kwargs"]["do_sample"] is True
    assert seen["latency"]["kwargs"]["top_k"] == 6
    assert seen["latency"]["kwargs"]["top_p"] == 0.8
    assert seen["latency"]["kwargs"]["cache_backend"] == "paged"


def test_export_from_dir_uses_runtime_modeling_loaders(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    def fake_load_config(model_dir):
        seen["config"] = model_dir
        return cfg

    def fake_build_model(cfg_in):
        seen["build"] = cfg_in
        return model

    monkeypatch.setattr(export_exporter_mod, "runtime_load_config", fake_load_config)
    monkeypatch.setattr(export_exporter_mod, "runtime_build_model", fake_build_model)

    def fake_load_pretrained(model_in, model_dir):
        seen["pretrained"] = {"model": model_in, "model_dir": model_dir}
        return model_in

    monkeypatch.setattr(export_exporter_mod, "runtime_load_pretrained", fake_load_pretrained)

    def fake_export_model(model_in, export_cfg, *, model_cfg=None):
        seen["export"] = {"model": model_in, "export_cfg": export_cfg, "model_cfg": model_cfg}
        return "artifact"

    monkeypatch.setattr(export_exporter_mod, "export_model", fake_export_model)
    def fake_export_model_delta(model_in, path):
        seen["delta"] = {"model": model_in, "path": path}

    monkeypatch.setattr(export_exporter_mod, "export_model_delta", fake_export_model_delta)

    out = export_exporter_mod.export_from_dir(
        "/models/demo",
        type("ExportCfg", (), {"outdir": "/tmp/out"})(),
    )

    assert out == "artifact"
    assert seen["config"] == "/models/demo"
    assert seen["build"] is cfg
    assert seen["pretrained"] == {"model": model, "model_dir": "/models/demo"}
    assert seen["export"]["model"] is model
    assert seen["export"]["model_cfg"] is cfg
    assert seen["delta"]["model"] is model
    assert seen["delta"]["path"] == "/tmp/out/delta.pt"


def test_export_model_torchscript_uses_two_tensor_wrapper(monkeypatch, tmp_path):
    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, attn_mask):
            del attn_mask
            return input_ids

    model = FakeModel()
    seen = {}

    def fake_trace(module, args, **kwargs):
        seen["module"] = module
        seen["args"] = args
        seen["kwargs"] = kwargs

        class FakeScripted:
            def save(self, path):
                seen["save"] = path

        return FakeScripted()

    monkeypatch.setattr(export_exporter_mod.torch.jit, "trace", fake_trace)
    monkeypatch.setattr(export_exporter_mod, "_write_card", lambda *args, **kwargs: None)

    out = export_exporter_mod.export_model(
        model,
        type("ExportCfg", (), {"target": "torchscript", "outdir": str(tmp_path), "quantize": None})(),
        model_cfg=_cfg(),
    )

    assert out == tmp_path / "model.ts"
    assert len(seen["args"]) == 2
    assert all(isinstance(arg, torch.Tensor) for arg in seen["args"])
    assert isinstance(seen["module"], export_exporter_mod._ExportWrapper)
    assert seen["kwargs"]["check_trace"] is False
    assert seen["save"] == str(tmp_path / "model.ts")


def test_export_model_onnx_uses_two_tensor_wrapper(monkeypatch, tmp_path):
    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, attn_mask):
            del attn_mask
            return input_ids

    model = FakeModel()
    seen = {}

    def fake_export(module, args, path, **kwargs):
        seen["module"] = module
        seen["args"] = args
        seen["path"] = path
        seen["kwargs"] = kwargs

    monkeypatch.setattr(export_exporter_mod.torch.onnx, "export", fake_export)
    monkeypatch.setattr(export_exporter_mod, "_write_card", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_exporter_mod.importlib.util, "find_spec", lambda name: object() if name == "onnx" else None)

    out = export_exporter_mod.export_model(
        model,
        type(
            "ExportCfg",
            (),
            {"target": "onnx", "outdir": str(tmp_path), "quantize": None, "dynamic_axes": True, "opset": 19},
        )(),
        model_cfg=_cfg(),
    )

    assert out == tmp_path / "model.onnx"
    assert len(seen["args"]) == 2
    assert all(isinstance(arg, torch.Tensor) for arg in seen["args"])
    assert isinstance(seen["module"], export_exporter_mod._ExportWrapper)
    assert seen["path"] == str(tmp_path / "model.onnx")
    assert seen["kwargs"]["input_names"] == ["input_ids", "attn_mask"]
    assert seen["kwargs"]["output_names"] == ["logits"]
    assert seen["kwargs"]["dynamo"] is False


def test_export_model_onnx_requires_onnx_dependency(tmp_path):
    model = torch.nn.Linear(4, 4, bias=False)

    try:
        export_exporter_mod.export_model(
            model,
            type(
                "ExportCfg",
                (),
                {"target": "onnx", "outdir": str(tmp_path), "quantize": None, "dynamic_axes": True, "opset": 19},
            )(),
            model_cfg=_cfg(),
        )
    except RuntimeError as exc:
        assert "requires the 'onnx' package" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when onnx dependency is missing")


def test_runtime_layer_view_is_runtime_owned(monkeypatch):
    monkeypatch.setattr(
        cache_mod,
        "create_native_paged_kv_cache_state",
        lambda **kwargs: (None, None),
    )
    cache = PagedKVCache(
        2,
        2,
        4,
        8,
        64,
        torch.float32,
        torch.device("cpu"),
        backend_name="paged",
    )
    layer = cache.layer(1)
    assert isinstance(layer, cache_mod.RuntimeLayerCacheView)
    assert layer.layer_idx == 1
    assert layer.parent is cache


def test_runtime_layer_view_uses_native_layer_max_length_without_touching_lengths():
    class FakeNativeLayer:
        def max_length(self):
            return 7

        def lengths(self):
            raise AssertionError("lengths() should not be used when max_length() is available")

    cache = runtime_kv_cache_mod.PagedKVCache(
        batch=2,
        n_layers=1,
        n_kv_heads=1,
        head_dim=2,
        pagesize=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        native_cache_state=None,
        native_layer_states=[FakeNativeLayer()],
        backend_name="native-paged",
    )

    assert cache.layer(0).length() == 7


def test_evict_kv_cache_delegates_to_cache_object():
    class FakeCache:
        def __init__(self):
            self.calls = []

        def evict(self, max_tokens: int, policy: str = "fifo") -> None:
            self.calls.append((max_tokens, policy))

    cache = FakeCache()
    cache_mod.evict_kv_cache(cache, 33, policy="sliding-window")
    assert cache.calls == [(33, "sliding-window")]


def test_attn_kv_cache_is_runtime_shim():
    assert PagedKVCache is runtime_kv_cache_mod.PagedKVCache
    assert ContiguousKVCache is runtime_kv_cache_mod.ContiguousKVCache


def test_blocks_native_fusion_is_runtime_shim():
    assert blocks_native_fusion_mod.apply_native_norm is runtime_blocks_mod.apply_native_norm
    assert blocks_native_fusion_mod.fused_add_norm is runtime_blocks_mod.fused_add_norm
    assert blocks_native_fusion_mod.apply_residual_update is runtime_blocks_mod.apply_residual_update
    assert blocks_native_fusion_mod.stack_native_execution_info is runtime_blocks_mod.stack_native_execution_info


def test_tensor_sampling_is_runtime_shim():
    assert tensor_sampling_mod.apply_temperature is runtime_sampling_mod.apply_temperature
    assert tensor_sampling_mod.apply_topk_mask is runtime_sampling_mod.apply_topk_mask
    assert tensor_sampling_mod.apply_topp_mask is runtime_sampling_mod.apply_topp_mask
    assert tensor_sampling_mod.apply_no_repeat_ngram_mask is runtime_sampling_mod.apply_no_repeat_ngram_mask


def test_tensor_positional_is_runtime_shim():
    assert tensor_positional_mod.apply_rotary is runtime_positional_mod.apply_rotary
    assert tensor_positional_mod.build_rope_cache is runtime_positional_mod.build_rope_cache
    assert tensor_positional_mod.build_alibi_bias is runtime_positional_mod.build_alibi_bias
    assert tensor_positional_mod.rope_yarn_factors is runtime_positional_mod.rope_yarn_factors


def test_runtime_generation_imports_runtime_sampling_helpers():
    assert runtime_generation_mod.apply_temperature is runtime_sampling_mod.apply_temperature
    assert runtime_generation_mod.apply_topk_mask is runtime_sampling_mod.apply_topk_mask
    assert runtime_generation_mod.apply_topp_mask is runtime_sampling_mod.apply_topp_mask
    assert runtime_generation_mod.apply_no_repeat_ngram_mask is runtime_sampling_mod.apply_no_repeat_ngram_mask


def test_runtime_blocks_imports_runtime_positional_helpers():
    assert runtime_blocks_mod.build_alibi_bias is runtime_positional_mod.build_alibi_bias
    assert runtime_blocks_mod.build_relative_position_indices is runtime_positional_mod.build_relative_position_indices
    assert runtime_blocks_mod.relative_position_bias_from_table is runtime_positional_mod.relative_position_bias_from_table


def test_model_factory_is_runtime_shim():
    assert model_factory_mod.build_model is runtime_factory_mod.build_model
    assert model_factory_mod.build_causal_lm is runtime_factory_mod.build_causal_lm
    assert model_factory_mod.build_prefix_lm is runtime_factory_mod.build_prefix_lm
    assert model_factory_mod.build_encoder is runtime_factory_mod.build_encoder
    assert model_factory_mod.build_seq2seq is runtime_factory_mod.build_seq2seq


def test_runtime_build_encoder_accepts_compress_config(monkeypatch):
    seen = {}

    def fake_apply(model, *, lora=None, quant=None):
        seen["model"] = model
        seen["lora"] = lora
        seen["quant"] = quant
        return {"kind": "encoder"}

    monkeypatch.setattr(encoder_model_mod, "apply_compression", fake_apply)

    model = runtime_modeling_mod.build_encoder(
        _cfg(),
        compress={"lora": {"rank": 2}, "quant": {"include": ["embed"]}},
    )

    assert isinstance(model, encoder_model_mod.EncoderModel)
    assert model._compression_summary == {"kind": "encoder"}
    assert seen["model"] is model
    assert seen["lora"] == {"rank": 2}
    assert seen["quant"] == {"include": ["embed"]}


def test_runtime_build_seq2seq_accepts_compress_config(monkeypatch):
    seen = {}

    def fake_apply(model, *, lora=None, quant=None):
        seen["model"] = model
        seen["lora"] = lora
        seen["quant"] = quant
        return {"kind": "seq2seq"}

    monkeypatch.setattr(seq2seq_model_mod, "apply_compression", fake_apply)

    model = runtime_modeling_mod.build_seq2seq(
        _cfg(),
        compress={"lora": {"rank": 4}, "quant": {"exclude": ["lm_head"]}},
    )

    assert isinstance(model, seq2seq_model_mod.EncoderDecoderLM)
    assert model._compression_summary == {"kind": "seq2seq"}
    assert seen["model"] is model
    assert seen["lora"] == {"rank": 4}
    assert seen["quant"] == {"exclude": ["lm_head"]}


def test_runtime_build_prefix_lm_uses_runtime_safe_default_block():
    model = runtime_modeling_mod.build_model(_cfg(), task="prefix-lm")
    out = model(torch.tensor([[1, 2, 3]], dtype=torch.long), prefix_lengths=2)
    assert out.shape == (1, 3, _cfg().vocab_size)


def test_model_checkpoint_is_runtime_shim():
    assert model_checkpoint_mod.load_config is runtime_checkpoint_mod.load_config
    assert model_checkpoint_mod.load_pretrained is runtime_checkpoint_mod.load_pretrained
    assert model_checkpoint_mod.save_pretrained is runtime_checkpoint_mod.save_pretrained


def test_model_hf_loader_is_runtime_shim():
    assert model_hf_llama_loader_mod.load_hf_llama_weights_into_local is runtime_checkpoint_mod.load_hf_llama_weights_into_local


def test_model_llama_bootstrap_is_runtime_shim():
    assert model_llama_bootstrap_mod.build_local_llama_from_snapshot is runtime_checkpoint_mod.build_local_llama_from_snapshot


def test_model_hf_snapshot_is_runtime_shim():
    assert model_hf_snapshot_mod.ensure_snapshot is runtime_checkpoint_mod.ensure_snapshot


def test_runtime_modeling_reexports_checkpoint_surface():
    assert runtime_modeling_mod.save_pretrained is runtime_checkpoint_mod.save_pretrained
    assert runtime_modeling_mod.load_pretrained is runtime_checkpoint_mod.load_pretrained
    assert runtime_modeling_mod.load_config is runtime_checkpoint_mod.load_config
    assert runtime_modeling_mod.ensure_snapshot is runtime_checkpoint_mod.ensure_snapshot
    assert runtime_modeling_mod.load_hf_llama_weights_into_local is runtime_checkpoint_mod.load_hf_llama_weights_into_local
    assert runtime_modeling_mod.model_config_from_hf_llama_snapshot_config is runtime_checkpoint_mod.model_config_from_hf_llama_snapshot_config
    assert runtime_modeling_mod.model_config_from_hf_llama_transformers_config is runtime_checkpoint_mod.model_config_from_hf_llama_transformers_config
    assert runtime_modeling_mod.build_local_llama_from_hf_config is runtime_checkpoint_mod.build_local_llama_from_hf_config
    assert runtime_modeling_mod.build_local_llama_from_snapshot is runtime_checkpoint_mod.build_local_llama_from_snapshot


def test_runtime_modeling_reexports_loader_surface():
    assert runtime_modeling_mod.load_model_dir is runtime_loader_mod.load_model_dir
    assert runtime_modeling_mod.load_model_factory_spec is runtime_loader_mod.load_model_factory_spec
    assert runtime_modeling_mod.load_runtime_model is runtime_loader_mod.load_runtime_model


def test_runtime_modeling_reexports_factory_surface():
    assert runtime_modeling_mod.build_model is runtime_factory_mod.build_model
    assert runtime_modeling_mod.build_registered_model is runtime_factory_mod.build_registered_model
    assert runtime_modeling_mod.build_causal_lm is runtime_factory_mod.build_causal_lm
    assert runtime_modeling_mod.build_prefix_lm is runtime_factory_mod.build_prefix_lm
    assert runtime_modeling_mod.build_encoder is runtime_factory_mod.build_encoder
    assert runtime_modeling_mod.build_seq2seq is runtime_factory_mod.build_seq2seq
    assert runtime_modeling_mod.register_model is runtime_factory_mod.register_model
    assert runtime_modeling_mod.get_model_builder is runtime_factory_mod.get_model_builder


def test_runtime_package_exports_checkpoint_surface():
    assert runtime_pkg.save_pretrained is runtime_checkpoint_mod.save_pretrained
    assert runtime_pkg.load_pretrained is runtime_checkpoint_mod.load_pretrained
    assert runtime_pkg.load_config is runtime_checkpoint_mod.load_config
    assert runtime_pkg.ensure_snapshot is runtime_checkpoint_mod.ensure_snapshot
    assert runtime_pkg.load_hf_llama_weights_into_local is runtime_checkpoint_mod.load_hf_llama_weights_into_local
    assert runtime_pkg.build_local_llama_from_hf_config is runtime_checkpoint_mod.build_local_llama_from_hf_config
    assert runtime_pkg.build_local_llama_from_snapshot is runtime_checkpoint_mod.build_local_llama_from_snapshot


def test_runtime_package_exports_loader_surface():
    assert runtime_pkg.load_model_dir is runtime_loader_mod.load_model_dir
    assert runtime_pkg.load_model_factory_spec is runtime_loader_mod.load_model_factory_spec
    assert runtime_pkg.load_runtime_model is runtime_loader_mod.load_runtime_model


def test_runtime_package_exports_factory_surface():
    assert runtime_pkg.build_model is runtime_factory_mod.build_model
    assert runtime_pkg.build_registered_model is runtime_factory_mod.build_registered_model
    assert runtime_pkg.build_causal_lm is runtime_factory_mod.build_causal_lm
    assert runtime_pkg.build_prefix_lm is runtime_factory_mod.build_prefix_lm
    assert runtime_pkg.build_encoder is runtime_factory_mod.build_encoder
    assert runtime_pkg.build_seq2seq is runtime_factory_mod.build_seq2seq
    assert runtime_pkg.register_model is runtime_factory_mod.register_model
    assert runtime_pkg.get_model_builder is runtime_factory_mod.get_model_builder


def test_model_registry_is_runtime_shim():
    assert model_registry_mod.register_model is runtime_factory_mod.register_model
    assert model_registry_mod.get_model_builder is runtime_factory_mod.get_model_builder
    assert model_registry_mod.build is runtime_factory_mod.build_registered_model


def test_runtime_modeling_is_prep_compatibility_shim():
    assert runtime_modeling_mod.RuntimeModelArtifacts is runtime_prep_mod.RuntimeModelArtifacts
    assert runtime_modeling_mod.prepare_model_for_runtime is runtime_prep_mod.prepare_model_for_runtime
    assert runtime_modeling_mod.resolve_model_config is runtime_prep_mod.resolve_model_config
    assert runtime_modeling_mod.resolve_model_device is runtime_prep_mod.resolve_model_device
    assert runtime_modeling_mod.resolve_model_dtype is runtime_prep_mod.resolve_model_dtype


def test_model_package_reexports_runtime_modeling_surface():
    assert model_pkg.RuntimeModelArtifacts is runtime_prep_mod.RuntimeModelArtifacts
    assert model_pkg.build is runtime_factory_mod.build_registered_model
    assert model_pkg.build_model is runtime_factory_mod.build_model
    assert model_pkg.build_registered_model is runtime_factory_mod.build_registered_model
    assert model_pkg.build_causal_lm is runtime_factory_mod.build_causal_lm
    assert model_pkg.build_prefix_lm is runtime_factory_mod.build_prefix_lm
    assert model_pkg.build_encoder is runtime_factory_mod.build_encoder
    assert model_pkg.build_seq2seq is runtime_factory_mod.build_seq2seq
    assert model_pkg.build_local_llama_from_hf_config is runtime_checkpoint_mod.build_local_llama_from_hf_config
    assert model_pkg.build_local_llama_from_snapshot is runtime_checkpoint_mod.build_local_llama_from_snapshot
    assert model_pkg.ensure_snapshot is runtime_checkpoint_mod.ensure_snapshot
    assert model_pkg.load_hf_llama_weights_into_local is runtime_checkpoint_mod.load_hf_llama_weights_into_local
    assert model_pkg.load_model_dir is runtime_loader_mod.load_model_dir
    assert model_pkg.load_model_factory_spec is runtime_loader_mod.load_model_factory_spec
    assert model_pkg.load_runtime_model is runtime_loader_mod.load_runtime_model
    assert model_pkg.model_config_from_hf_llama_snapshot_config is runtime_checkpoint_mod.model_config_from_hf_llama_snapshot_config
    assert model_pkg.model_config_from_hf_llama_transformers_config is runtime_checkpoint_mod.model_config_from_hf_llama_transformers_config
    assert model_pkg.prepare_model_for_runtime is runtime_prep_mod.prepare_model_for_runtime
    assert model_pkg.resolve_model_config is runtime_prep_mod.resolve_model_config
    assert model_pkg.resolve_model_device is runtime_prep_mod.resolve_model_device
    assert model_pkg.resolve_model_dtype is runtime_prep_mod.resolve_model_dtype
    assert model_pkg.save_pretrained is runtime_checkpoint_mod.save_pretrained
    assert model_pkg.load_pretrained is runtime_checkpoint_mod.load_pretrained
    assert model_pkg.load_config is runtime_checkpoint_mod.load_config


def test_runtime_model_registry_supports_custom_builders():
    seen = {}

    def fake_builder(cfg, **kwargs):
        seen["cfg"] = cfg
        seen["kwargs"] = kwargs
        return {"cfg": cfg, "kwargs": kwargs}

    runtime_factory_mod.register_model("__unit_test_runtime_registry__", fake_builder)
    out = model_registry_mod.build("__unit_test_runtime_registry__", _cfg(), block="custom")

    assert out == {"cfg": _cfg(), "kwargs": {"block": "custom"}}
    assert seen["cfg"] == _cfg()
    assert seen["kwargs"] == {"block": "custom"}


def test_eval_llama_parity_uses_runtime_hf_helpers():
    assert eval_llama_hf_parity_mod.build_local_llama_from_hf_config is runtime_checkpoint_mod.build_local_llama_from_hf_config
    assert eval_llama_hf_parity_mod.load_hf_llama_weights_into_local is runtime_checkpoint_mod.load_hf_llama_weights_into_local


def test_runtime_ensure_snapshot_returns_direct_model_dir(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    out = runtime_modeling_mod.ensure_snapshot(str(model_dir), str(tmp_path / "cache"))

    assert out == str(model_dir)


def test_runtime_ensure_snapshot_prefers_latest_cached_snapshot(tmp_path):
    cache_dir = tmp_path / "cache"
    older = cache_dir / "models--meta-llama--Llama-3.1-8B" / "snapshots" / "old"
    newer = cache_dir / "models--meta-llama--Llama-3.1-8B" / "snapshots" / "new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "config.json").write_text("{}", encoding="utf-8")
    (newer / "config.json").write_text("{}", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    out = runtime_modeling_mod.ensure_snapshot("meta-llama/Llama-3.1-8B", str(cache_dir))

    assert out == str(newer)


def test_runtime_ensure_snapshot_falls_back_to_hf_download(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    seen = {}

    class FakeHub:
        @staticmethod
        def snapshot_download(*, repo_id, cache_dir):
            seen["download"] = {"repo_id": repo_id, "cache_dir": cache_dir}
            return "/downloaded/snapshot"

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)

    out = runtime_modeling_mod.ensure_snapshot("meta-llama/Llama-3.1-8B", str(cache_dir))

    assert out == "/downloaded/snapshot"
    assert seen["download"] == {
        "repo_id": "meta-llama/Llama-3.1-8B",
        "cache_dir": str(cache_dir),
    }


def test_snapshot_hf_llama_config_helper_derives_runtime_model_config():
    cfg_obj = {
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "intermediate_size": 96,
        "vocab_size": 128,
        "head_dim": 8,
        "num_key_value_heads": 2,
        "rope_theta": 10000.0,
        "rope_parameters": {"rope_theta": 250000.0},
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }

    cfg, n_kv_heads, tie_weights = runtime_modeling_mod.model_config_from_hf_llama_snapshot_config(
        cfg_obj,
        torch_dtype=torch.bfloat16,
    )

    assert cfg.d_model == 32
    assert cfg.n_layers == 3
    assert cfg.n_heads == 4
    assert cfg.d_ff == 96
    assert cfg.vocab_size == 128
    assert cfg.head_dim == 8
    assert cfg.rope_theta == 250000.0
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.dtype == "bfloat16"
    assert n_kv_heads == 2
    assert tie_weights is False


def test_transformers_hf_llama_config_helper_derives_runtime_model_config():
    cfg_hf = type(
        "Cfg",
        (),
        {
            "hidden_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 6,
            "head_dim": 8,
            "num_key_value_heads": 3,
            "intermediate_size": 128,
            "vocab_size": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "rope_parameters": {"rope_theta": 750000.0},
            "rope_scaling": {
                "type": "linear",
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
            },
            "tie_word_embeddings": True,
            "max_position_embeddings": 4096,
        },
    )()

    cfg, n_kv_heads, tie_weights = runtime_modeling_mod.model_config_from_hf_llama_transformers_config(
        cfg_hf,
        dtype=torch.float16,
        seq_len=2048,
    )

    assert cfg.d_model == 48
    assert cfg.n_layers == 2
    assert cfg.n_heads == 6
    assert cfg.d_ff == 128
    assert cfg.vocab_size == 256
    assert cfg.head_dim == 8
    assert cfg.rope_theta == 750000.0
    assert cfg.dtype == "float16"
    assert cfg.rope_scaling_type == "linear"
    assert cfg.rope_scaling_factor == 8.0
    assert cfg.rope_scaling_original_max_position_embeddings == 8192
    assert cfg.rope_scaling_low_freq_factor == 1.0
    assert cfg.rope_scaling_high_freq_factor == 4.0
    assert n_kv_heads == 3
    assert tie_weights is True


def test_build_local_llama_from_snapshot_uses_runtime_hf_loader(monkeypatch, tmp_path):
    cfg_obj = {
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 128,
        "num_key_value_heads": 2,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg_obj), encoding="utf-8")
    seen = {}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_calls = []
            self.eval_calls = 0

        def to(self, *args, **kwargs):
            self.to_calls.append({"args": args, "kwargs": kwargs})
            return self

        def eval(self):
            self.eval_calls += 1
            return self

    model = FakeModel()

    def fake_build_causal_lm(cfg, **kwargs):
        seen["build"] = {"cfg": cfg, "kwargs": kwargs}
        return model

    def fake_load_hf(model_in, ckpt_dir):
        seen["load"] = {"model": model_in, "ckpt_dir": ckpt_dir}

    monkeypatch.setattr(runtime_factory_mod, "build_causal_lm", fake_build_causal_lm)
    monkeypatch.setattr(runtime_checkpoint_mod, "load_hf_llama_weights_into_local", fake_load_hf)

    built_model, loaded_cfg = runtime_modeling_mod.build_local_llama_from_snapshot(
        str(tmp_path),
        "cpu",
        torch.float16,
    )

    assert built_model is model
    assert loaded_cfg == cfg_obj
    assert seen["build"]["cfg"].d_model == 32
    assert seen["build"]["kwargs"]["block"] == "llama"
    assert seen["build"]["kwargs"]["n_kv_heads"] == 2
    assert seen["build"]["kwargs"]["tie_weights"] is False
    assert seen["load"] == {"model": model, "ckpt_dir": str(tmp_path)}
    assert model.to_calls[0]["kwargs"]["dtype"] == torch.float16
    assert model.to_calls[-1]["args"] == ("cpu",)
    assert model.eval_calls == 1


def test_apply_attention_biases_combines_bool_mask_alibi_and_rpb(monkeypatch):
    x = torch.zeros(1, 3, 8)
    base_mask = torch.tensor(
        [[[[False, True, False], [False, False, True], [True, False, False]]]],
        dtype=torch.bool,
    )
    rpb_table = torch.zeros(4, 7)

    monkeypatch.setattr(
        runtime_blocks_mod,
        "build_alibi_bias",
        lambda num_heads, seq_len, device=None: torch.ones(1, num_heads, seq_len, seq_len, device=device),
    )
    monkeypatch.setattr(
        runtime_blocks_mod,
        "relative_position_bias_from_table",
        lambda indices, table: torch.full(
            (1, table.shape[0], indices.shape[0], indices.shape[1]),
            2.0,
            device=table.device,
        ),
    )

    out = runtime_blocks_mod.apply_attention_biases(
        x,
        base_mask,
        num_heads=4,
        use_alibi=True,
        rpb_table=rpb_table,
        rpb_max_distance=4,
    )

    assert out.shape == (1, 4, 3, 3)
    assert torch.isneginf(out[0, 0, 0, 1])
    assert torch.isneginf(out[0, 0, 1, 2])
    assert out[0, 0, 0, 0].item() == 3.0
    assert out[0, 0, 2, 1].item() == 3.0


def test_prepare_attention_mask_for_heads_broadcasts_padding_mask():
    mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    out = runtime_blocks_mod.prepare_attention_mask_for_heads(
        mask,
        batch_size=1,
        num_heads=4,
        tgt_len=2,
        src_len=3,
    )
    assert out.shape == (1, 4, 2, 3)
    assert out.dtype == torch.bool
    assert torch.equal(out[0, 0, 0], torch.tensor([False, True, False]))


def test_prepare_cross_attention_mask_broadcasts_batched_padding_mask():
    q = torch.zeros(2, 3, 8)
    memory = torch.zeros(2, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 1, 0, 1]], dtype=torch.long)
    out = runtime_blocks_mod.prepare_cross_attention_mask(q, memory, mask, num_heads=4)
    assert out.shape == (2, 4, 3, 5)
    assert out.dtype == torch.bool
    assert torch.equal(out[0, 0, 0], torch.tensor([False, False, True, True, True]))
    assert torch.equal(out[1, 0, 1], torch.tensor([False, True, False, True, False]))


def test_prepare_pattern_attention_mask_combines_pattern_and_external_mask():
    x = torch.zeros(1, 3, 8)
    external = torch.tensor([[1, 0, 1]], dtype=torch.long)
    pattern = torch.tensor(
        [[False, True, False], [False, False, True], [True, False, False]],
        dtype=torch.bool,
    )
    out = runtime_blocks_mod.prepare_pattern_attention_mask(
        x,
        external,
        num_heads=2,
        pattern_mask=pattern,
    )
    assert out.shape == (1, 2, 3, 3)
    assert torch.equal(out[0, 0, 0], torch.tensor([False, True, False]))
    assert torch.equal(out[0, 0, 1], torch.tensor([False, True, True]))


def test_prepare_segment_bidir_attention_mask_expands_segments():
    x = torch.zeros(1, 3, 8)
    segment_ids = torch.tensor([[0, 0, 1]], dtype=torch.long)
    out = runtime_blocks_mod.prepare_segment_bidir_attention_mask(
        x,
        segment_ids,
        None,
        num_heads=2,
    )
    assert out.shape == (1, 2, 3, 3)
    assert torch.equal(out[0, 0, 0], torch.tensor([False, False, True]))
    assert torch.equal(out[0, 1, 2], torch.tensor([True, True, False]))


def test_execute_attention_mlp_block_matches_prenorm_residual_flow():
    x = torch.zeros(1, 2, 3)
    out = runtime_blocks_mod.execute_attention_mlp_block(
        x,
        attn_fn=lambda y: y + 1.0,
        mlp_fn=lambda y: y + 2.0,
        n1=torch.nn.Identity(),
        n2=torch.nn.Identity(),
        resid_dropout=torch.nn.Identity(),
        drop_path=torch.nn.Identity(),
        residual_scale=0.5,
        norm_policy="prenorm",
    )
    assert torch.allclose(out, torch.full_like(x, 1.75))


def test_execute_parallel_attention_mlp_block_matches_parallel_residual_flow():
    x = torch.zeros(1, 2, 3)
    out = runtime_blocks_mod.execute_parallel_attention_mlp_block(
        x,
        attn_fn=lambda y: y + 1.0,
        mlp_fn=lambda y: y + 2.0,
        norm=torch.nn.Identity(),
        resid_dropout=torch.nn.Identity(),
        drop_path=torch.nn.Identity(),
        residual_scale=0.5,
    )
    assert torch.allclose(out, torch.full_like(x, 1.5))


def test_execute_block_stack_routes_layer_cache_and_positions():
    seen: list[dict[str, object]] = []

    class FakeBlock(torch.nn.Module):
        def __init__(self, idx: int):
            super().__init__()
            self.idx = idx

        def forward(self, x, mask, cache=None, position_embeddings=None, position_ids=None):
            seen.append(
                {
                    "idx": self.idx,
                    "mask": mask,
                    "cache": cache,
                    "position_embeddings": position_embeddings,
                    "position_ids": position_ids,
                }
            )
            return x + float(self.idx + 1)

    class FakeCache:
        def layer(self, idx: int):
            return f"layer-{idx}"

    x = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 1, 2, 2)
    cos = torch.zeros(2, 4)
    sin = torch.ones(2, 4)
    position_ids = torch.tensor([[0, 1]], dtype=torch.long)

    out = runtime_blocks_mod.execute_block_stack(
        [FakeBlock(0), FakeBlock(1)],
        x,
        mask,
        FakeCache(),
        position_embeddings=(cos, sin),
        position_ids=position_ids,
    )

    assert torch.equal(out, torch.full_like(x, 3.0))
    assert [call["cache"] for call in seen] == ["layer-0", "layer-1"]
    assert all(call["mask"] is mask for call in seen)
    assert all(call["position_embeddings"][0] is cos for call in seen)
    assert all(call["position_embeddings"][1] is sin for call in seen)
    assert all(call["position_ids"] is position_ids for call in seen)


def test_transformer_block_delegates_to_runtime_block_helpers(monkeypatch):
    helper_calls = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, k, v, mask, cache, *, position_embeddings=None, position_ids=None):
            del k, v
            self.calls.append(
                {
                    "x": x,
                    "mask": mask,
                    "cache": cache,
                    "position_embeddings": position_embeddings,
                    "position_ids": position_ids,
                }
            )
            return x

    fake_attn = FakeAttention()

    def fake_apply_attention_biases(x, mask, **kwargs):
        helper_calls["bias"] = {"x": x, "mask": mask, "kwargs": kwargs}
        return "biased-mask"

    monkeypatch.setattr(transformer_block_mod, "apply_attention_biases", fake_apply_attention_biases)

    def fake_execute(x, **kwargs):
        helper_calls["execute"] = kwargs
        kwargs["attn_fn"](x)
        return x + 1.0

    monkeypatch.setattr(transformer_block_mod, "execute_attention_mlp_block", fake_execute)

    block = transformer_block_mod.TransformerBlock(_cfg(), fake_attn)
    x = torch.zeros(1, 2, 16)
    out = block(
        x,
        None,
        cache="cache",
        position_embeddings=("cos", "sin"),
        position_ids="pos",
    )

    assert torch.equal(out, torch.ones_like(x))
    assert helper_calls["bias"]["kwargs"]["num_heads"] == 4
    assert helper_calls["execute"]["residual_scale"] == block.bc.residual_scale
    assert helper_calls["execute"]["norm_policy"] == block.bc.norm_policy
    assert fake_attn.calls[0]["cache"] == "cache"
    assert fake_attn.calls[0]["position_embeddings"] == ("cos", "sin")
    assert fake_attn.calls[0]["position_ids"] == "pos"


def test_encoder_block_delegates_mask_prep_to_runtime(monkeypatch):
    helper_calls = {}

    class FakeAttention(torch.nn.Module):
        def forward(self, x, k, v, mask, cache, **kwargs):
            del x, k, v, cache, kwargs
            helper_calls["attn_mask"] = mask
            return torch.zeros(1, 2, 16)

    monkeypatch.setattr(encoder_block_mod, "build_attention", lambda *args, **kwargs: FakeAttention())

    def fake_prepare_encoder_attention_mask(x, padding_mask, **kwargs):
        helper_calls["prepared"] = {"x_shape": tuple(x.shape), "padding_mask": padding_mask, "kwargs": kwargs}
        return "enc-mask"

    monkeypatch.setattr(encoder_block_mod, "prepare_encoder_attention_mask", fake_prepare_encoder_attention_mask)

    block = encoder_block_mod.EncoderBlock(_cfg())
    x = torch.zeros(1, 2, 16)
    padding_mask = torch.tensor([[1, 0]], dtype=torch.long)
    block(x, padding_mask)

    assert helper_calls["prepared"]["x_shape"] == (1, 2, 16)
    assert helper_calls["prepared"]["padding_mask"] is padding_mask
    assert helper_calls["prepared"]["kwargs"]["num_heads"] == 4
    assert helper_calls["attn_mask"] == "enc-mask"


def test_cross_attention_block_prepares_memory_padding_mask(monkeypatch):
    helper_calls = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, k, v, mask, cache=None, **kwargs):
            del cache, kwargs
            self.calls.append({"x": x, "k": k, "v": v, "mask": mask})
            return x

    fake_attn = FakeAttention()
    monkeypatch.setattr(cross_attn_block_mod, "build_attention", lambda *args, **kwargs: fake_attn)

    def fake_prepare(x, memory, enc_mask, **kwargs):
        helper_calls["prepared"] = {
            "x_shape": tuple(x.shape),
            "memory_shape": tuple(memory.shape),
            "enc_mask": enc_mask,
            "kwargs": kwargs,
        }
        return "prepared-mask"

    monkeypatch.setattr(cross_attn_block_mod, "prepare_cross_attention_mask", fake_prepare)

    block = cross_attn_block_mod.CrossAttentionBlock(_cfg())
    x = torch.zeros(1, 2, 16)
    memory = torch.zeros(1, 3, 16)
    enc_mask = torch.tensor([[1, 0, 1]], dtype=torch.long)
    block(x, memory, enc_mask=enc_mask)

    assert helper_calls["prepared"]["x_shape"] == (1, 2, 16)
    assert helper_calls["prepared"]["memory_shape"] == (1, 3, 16)
    assert helper_calls["prepared"]["enc_mask"] is enc_mask
    assert helper_calls["prepared"]["kwargs"]["num_heads"] == 4
    assert fake_attn.calls[0]["mask"] == "prepared-mask"


def test_local_attention_block_delegates_to_runtime_mask_helper(monkeypatch):
    helper_calls = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, k, v, mask, cache=None, **kwargs):
            del k, v, cache, kwargs
            self.calls.append({"x": x, "mask": mask})
            return x

    fake_attn = FakeAttention()
    monkeypatch.setattr(local_attn_block_mod, "build_attention", lambda *args, **kwargs: fake_attn)

    def fake_prepare(x, mask, **kwargs):
        helper_calls["prepared"] = {"x_shape": tuple(x.shape), "mask": mask, "kwargs": kwargs}
        return torch.zeros(x.shape[0], kwargs["num_heads"], x.shape[1], x.shape[1], dtype=torch.bool)

    monkeypatch.setattr(local_attn_block_mod, "prepare_local_attention_mask", fake_prepare)

    block = local_attn_block_mod.LocalAttentionBlock(_cfg(), window_size=5)
    x = torch.zeros(1, 3, 16)
    block(x, None)

    assert helper_calls["prepared"]["x_shape"] == (1, 3, 16)
    assert helper_calls["prepared"]["kwargs"]["window_size"] == 5
    assert helper_calls["prepared"]["kwargs"]["num_heads"] == 4
    assert fake_attn.calls[0]["mask"].shape == (1, 4, 3, 3)


def test_block_sparse_attention_block_delegates_to_runtime_mask_helper(monkeypatch):
    helper_calls = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, k, v, mask, cache=None, **kwargs):
            del k, v, cache, kwargs
            self.calls.append({"x": x, "mask": mask})
            return x

    fake_attn = FakeAttention()
    monkeypatch.setattr(block_sparse_attn_block_mod, "build_attention", lambda *args, **kwargs: fake_attn)

    def fake_prepare(x, mask, **kwargs):
        helper_calls["prepared"] = {"x_shape": tuple(x.shape), "mask": mask, "kwargs": kwargs}
        return torch.zeros(x.shape[0], kwargs["num_heads"], x.shape[1], x.shape[1], dtype=torch.bool)

    monkeypatch.setattr(block_sparse_attn_block_mod, "prepare_block_sparse_attention_mask", fake_prepare)

    block = block_sparse_attn_block_mod.BlockSparseAttentionBlock(_cfg(), block_size=7)
    x = torch.zeros(1, 3, 16)
    block(x, None)

    assert helper_calls["prepared"]["kwargs"]["block_size"] == 7
    assert helper_calls["prepared"]["kwargs"]["num_heads"] == 4
    assert helper_calls["prepared"]["kwargs"]["pattern"].shape == (1, 1)
    assert fake_attn.calls[0]["mask"].shape == (1, 4, 3, 3)


def test_segment_bidir_attention_block_delegates_to_runtime_mask_helper(monkeypatch):
    helper_calls = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, k, v, mask, cache=None, **kwargs):
            del k, v, cache, kwargs
            self.calls.append({"x": x, "mask": mask})
            return x

    fake_attn = FakeAttention()
    monkeypatch.setattr(segment_bidir_attn_block_mod, "build_attention", lambda *args, **kwargs: fake_attn)

    def fake_prepare(x, segment_ids, mask, **kwargs):
        helper_calls["prepared"] = {
            "x_shape": tuple(x.shape),
            "segment_ids": segment_ids,
            "mask": mask,
            "kwargs": kwargs,
        }
        return torch.zeros(x.shape[0], kwargs["num_heads"], x.shape[1], x.shape[1], dtype=torch.bool)

    monkeypatch.setattr(segment_bidir_attn_block_mod, "prepare_segment_bidir_attention_mask", fake_prepare)

    block = segment_bidir_attn_block_mod.SegmentBidirAttentionBlock(_cfg())
    x = torch.zeros(1, 3, 16)
    segment_ids = torch.tensor([[0, 1, 1]], dtype=torch.long)
    block(x, segment_ids, None)

    assert helper_calls["prepared"]["x_shape"] == (1, 3, 16)
    assert helper_calls["prepared"]["segment_ids"] is segment_ids
    assert helper_calls["prepared"]["kwargs"]["num_heads"] == 4
    assert fake_attn.calls[0]["mask"].shape == (1, 4, 3, 3)


def test_parallel_block_delegates_to_runtime_parallel_helper(monkeypatch):
    helper_calls = {}

    def fake_execute(x, **kwargs):
        helper_calls["kwargs"] = kwargs
        return x + 2.0

    monkeypatch.setattr(parallel_block_mod, "execute_parallel_attention_mlp_block", fake_execute)

    block = parallel_block_mod.ParallelTransformerBlock(_cfg())
    x = torch.zeros(1, 2, 16)
    out = block(x, None, cache="cache")

    assert torch.equal(out, torch.full_like(x, 2.0))
    assert helper_calls["kwargs"]["norm"] is block.n
    assert helper_calls["kwargs"]["mlp_fn"] is block.mlp


def test_contiguous_cache_supports_runtime_layer_view():
    cache = ContiguousKVCache(
        2,
        1,
        2,
        4,
        8,
        torch.float32,
        torch.device("cpu"),
    )
    k = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).view(2, 2, 3, 4)
    v = (k + 1000).clone()
    layer = cache.layer(0)
    layer.append(k, v)
    got_k, got_v = layer.read(0, 3)
    assert layer.length() == 3
    assert got_k.shape == (2, 2, 3, 4)
    assert got_v.shape == (2, 2, 3, 4)
    assert torch.equal(got_k, k)
    assert torch.equal(got_v, v)


def test_causal_model_uses_runtime_block_stack(monkeypatch):
    seen = {}

    def fake_execute(blocks, x, mask=None, cache=None, *, position_embeddings=None, position_ids=None):
        seen["blocks"] = blocks
        seen["mask"] = mask
        seen["cache"] = cache
        seen["position_embeddings"] = position_embeddings
        seen["position_ids"] = position_ids
        return x

    monkeypatch.setattr(causal_model_mod, "execute_block_stack", fake_execute)

    model = causal_model_mod.CausalLM(_cfg(), block_variant="llama")
    out = model(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert out.shape == (1, 3, _cfg().vocab_size)
    assert seen["blocks"] is model.blocks
    assert seen["cache"] is None
    assert seen["position_embeddings"][0].shape == (3, 4)
    assert seen["position_embeddings"][1].shape == (3, 4)
    assert torch.equal(seen["position_ids"], torch.tensor([[0, 1, 2]], dtype=torch.long))


def test_moe_block_forward_accepts_runtime_position_args(monkeypatch):
    helper_calls = {}

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, k, v, mask, cache, *, position_embeddings=None, position_ids=None):
            del k, v
            self.calls.append(
                {
                    "mask": mask,
                    "cache": cache,
                    "position_embeddings": position_embeddings,
                    "position_ids": position_ids,
                }
            )
            return x

    fake_attn = FakeAttention()
    monkeypatch.setattr(moe_block_mod, "build_attention", lambda *args, **kwargs: fake_attn)
    monkeypatch.setattr(moe_block_mod, "apply_attention_biases", lambda x, mask, **kwargs: "moe-mask")

    def fake_execute(x, **kwargs):
        helper_calls["execute"] = kwargs
        kwargs["attn_fn"](x)
        return x

    monkeypatch.setattr(moe_block_mod, "execute_attention_mlp_block", fake_execute)

    block = moe_block_mod.MoEBlock(_cfg(), num_experts=2, k=1)
    x = torch.zeros(1, 2, 16)
    out = block(
        x,
        None,
        cache="cache",
        position_embeddings=("cos", "sin"),
        position_ids="pos",
    )

    assert torch.equal(out, x)
    assert helper_calls["execute"]["norm_policy"] == block.bc.norm_policy
    assert fake_attn.calls[0]["mask"] == "moe-mask"
    assert fake_attn.calls[0]["cache"] == "cache"
    assert fake_attn.calls[0]["position_embeddings"] == ("cos", "sin")
    assert fake_attn.calls[0]["position_ids"] == "pos"


def test_serve_engine_generate_delegates_to_runtime_generate(monkeypatch):
    seen = {}

    def fake_runtime_generate(model, input_ids, **kwargs):
        seen["model"] = model
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.full((1, 2), 7, dtype=torch.long)

    monkeypatch.setattr(serve_engine_mod, "runtime_generate", fake_runtime_generate)
    model = object()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = serve_engine_mod.generate(model, ids, cache_backend="native-paged")
    assert torch.equal(out, torch.full((1, 2), 7, dtype=torch.long))
    assert seen["model"] is model
    assert torch.equal(seen["input_ids"], ids)
    assert isinstance(seen["kwargs"]["config"], serve_engine_mod.GenerationConfig)
    assert serve_engine_mod.GenerationConfig is runtime_generation_mod.GenerationConfig
    assert seen["kwargs"]["cache_backend"] == "native-paged"


def test_serve_engine_generate_builds_default_config_via_runtime_builder(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["builder_kwargs"] = kwargs
        return "cfg"

    def fake_runtime_generate(model, input_ids, **kwargs):
        seen["generate_kwargs"] = kwargs
        return torch.full((1, 2), 5, dtype=torch.long)

    monkeypatch.setattr(serve_engine_mod, "runtime_build_generation_config", fake_builder)
    monkeypatch.setattr(serve_engine_mod, "runtime_generate", fake_runtime_generate)

    out = serve_engine_mod.generate(object(), torch.tensor([[1, 2]], dtype=torch.long))

    assert torch.equal(out, torch.full((1, 2), 5, dtype=torch.long))
    assert seen["builder_kwargs"] == {}
    assert seen["generate_kwargs"]["config"] == "cfg"


def test_serve_generate_builds_config_via_runtime_builder(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["builder_kwargs"] = kwargs
        return "cfg"

    def fake_engine_generate(model, input_ids, **kwargs):
        seen["engine_kwargs"] = kwargs
        return torch.full((1, 3), 4, dtype=torch.long)

    monkeypatch.setattr(serve_generate_mod, "runtime_build_generation_config", fake_builder)
    monkeypatch.setattr(serve_generate_mod, "_engine_generate", fake_engine_generate)

    out = serve_generate_mod.generate(
        object(),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=9,
        top_k=7,
        beam_size=3,
        length_penalty=0.6,
    )

    assert torch.equal(out, torch.full((1, 3), 4, dtype=torch.long))
    assert seen["builder_kwargs"]["max_new_tokens"] == 9
    assert seen["builder_kwargs"]["top_k"] == 7
    assert seen["builder_kwargs"]["beam_size"] == 3
    assert seen["builder_kwargs"]["length_penalty"] == 0.6
    assert seen["engine_kwargs"]["config"] == "cfg"


def test_serve_generate_infers_sampling_mode_and_passes_runtime_args(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["builder_kwargs"] = kwargs
        return "cfg"

    def fake_engine_generate(model, input_ids, **kwargs):
        seen["engine_kwargs"] = kwargs
        return torch.full((1, 3), 2, dtype=torch.long)

    monkeypatch.setattr(serve_generate_mod, "runtime_build_generation_config", fake_builder)
    monkeypatch.setattr(serve_generate_mod, "_engine_generate", fake_engine_generate)

    out = serve_generate_mod.generate(
        object(),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=4,
        top_p=0.9,
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        cache_backend="native-paged",
    )

    assert torch.equal(out, torch.full((1, 3), 2, dtype=torch.long))
    assert seen["builder_kwargs"]["do_sample"] is True
    assert seen["builder_kwargs"]["top_p"] == 0.9
    assert torch.equal(seen["engine_kwargs"]["attention_mask"], torch.tensor([[1, 1]], dtype=torch.long))
    assert seen["engine_kwargs"]["cache_backend"] == "native-paged"


def test_model_runtime_create_generation_session_uses_runtime_defaults(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=144)
    rt.default_kv_cache_backend = "native-paged"
    seen = {}

    def fake_from_model(cls, model_in, input_ids, **kwargs):
        del cls
        seen["model"] = model_in
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return "session"

    monkeypatch.setattr(runtime_generation_mod.RuntimeGenerationSession, "from_model", classmethod(fake_from_model))
    session = rt.create_generation_session(torch.tensor([[1, 2]], dtype=torch.long))

    assert session == "session"
    assert seen["model"] is model
    assert torch.equal(seen["input_ids"], torch.tensor([[1, 2]], dtype=torch.long))
    assert seen["kwargs"]["cache_pagesize"] == 144
    assert seen["kwargs"]["cache_backend"] == "native-paged"


def test_model_runtime_generate_uses_runtime_generate(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=160)
    rt.default_kv_cache_backend = "paged"
    seen = {}

    def fake_generate(model_in, input_ids, **kwargs):
        seen["model"] = model_in
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.full((1, 4), 9, dtype=torch.long)

    monkeypatch.setattr(serve_runtime_mod, "runtime_generate", fake_generate)
    cfg_obj = object()
    out = rt.generate(torch.tensor([[3, 4]], dtype=torch.long), config=cfg_obj)

    assert torch.equal(out, torch.full((1, 4), 9, dtype=torch.long))
    assert seen["model"] is model
    assert torch.equal(seen["input_ids"], torch.tensor([[3, 4]], dtype=torch.long))
    assert seen["kwargs"]["config"] is cfg_obj
    assert seen["kwargs"]["cache_pagesize"] == 160
    assert seen["kwargs"]["cache_backend"] == "paged"


def test_model_causal_generate_builds_config_via_runtime_builder(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["builder_kwargs"] = kwargs
        return "cfg"

    def fake_engine_generate(model, input_ids, **kwargs):
        seen["engine_kwargs"] = kwargs
        return torch.full((1, 4), 3, dtype=torch.long)

    monkeypatch.setattr(causal_model_mod, "runtime_build_generation_config", fake_builder)
    monkeypatch.setattr(causal_model_mod, "engine_generate", fake_engine_generate)

    model = causal_model_mod.CausalLM(_cfg(), block_variant="llama")
    out = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=6,
        do_sample=None,
        temperature=0.7,
        top_p=0.9,
        top_k=5,
        eos_token_id=[9],
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        presence_penalty=0.3,
        frequency_penalty=0.4,
        attn_mask=torch.tensor([[1, 1]], dtype=torch.long),
        sliding_window=32,
        beam_size=4,
        length_penalty=0.7,
        cache_backend="native-paged",
        return_dict=False,
    )

    assert torch.equal(out, torch.full((1, 4), 3, dtype=torch.long))
    assert seen["builder_kwargs"]["max_new_tokens"] == 6
    assert seen["builder_kwargs"]["do_sample"] is True
    assert seen["builder_kwargs"]["temperature"] == 0.7
    assert seen["builder_kwargs"]["top_p"] == 0.9
    assert seen["builder_kwargs"]["top_k"] == 5
    assert seen["builder_kwargs"]["eos_id"] == 9
    assert seen["builder_kwargs"]["no_repeat_ngram"] == 2
    assert seen["builder_kwargs"]["repetition_penalty"] == 1.2
    assert seen["builder_kwargs"]["presence_penalty"] == 0.3
    assert seen["builder_kwargs"]["frequency_penalty"] == 0.4
    assert seen["builder_kwargs"]["sliding_window"] == 32
    assert seen["builder_kwargs"]["beam_size"] == 4
    assert seen["builder_kwargs"]["length_penalty"] == 0.7
    assert seen["engine_kwargs"]["config"] == "cfg"
    assert torch.equal(seen["engine_kwargs"]["attention_mask"], torch.tensor([[1, 1]], dtype=torch.long))
    assert seen["engine_kwargs"]["cache_backend"] == "native-paged"


def test_model_causal_generate_honors_explicit_greedy_override(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["builder_kwargs"] = kwargs
        return "cfg"

    monkeypatch.setattr(causal_model_mod, "runtime_build_generation_config", fake_builder)
    monkeypatch.setattr(
        causal_model_mod,
        "engine_generate",
        lambda model, input_ids, **kwargs: torch.full((1, 3), 4, dtype=torch.long),
    )

    model = causal_model_mod.CausalLM(_cfg(), block_variant="llama")
    out = model.generate(
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=5,
        do_sample=False,
        temperature=0.7,
        top_k=6,
        top_p=0.8,
        return_dict=False,
    )

    assert torch.equal(out, torch.full((1, 3), 4, dtype=torch.long))
    assert seen["builder_kwargs"]["do_sample"] is False


def test_model_generate_greedy_is_runtime_shim(monkeypatch):
    seen = {}

    def fake_runtime_generate(model, input_ids, **kwargs):
        seen["model"] = model
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.full((1, 3), 6, dtype=torch.long)

    monkeypatch.setattr(model_generate_mod, "runtime_greedy_generate", fake_runtime_generate)

    out = model_generate_mod.greedy_generate(
        object(),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=5,
        eos_id=9,
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        no_repeat_ngram=2,
        repetition_penalty=1.1,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        sliding_window=16,
        cache_backend="native-paged",
    )

    assert torch.equal(out, torch.full((1, 3), 6, dtype=torch.long))
    assert seen["kwargs"]["max_new_tokens"] == 5
    assert seen["kwargs"]["eos_id"] == 9
    assert torch.equal(seen["kwargs"]["attention_mask"], torch.tensor([[1, 1]], dtype=torch.long))
    assert seen["kwargs"]["no_repeat_ngram"] == 2
    assert seen["kwargs"]["repetition_penalty"] == 1.1
    assert seen["kwargs"]["presence_penalty"] == 0.2
    assert seen["kwargs"]["frequency_penalty"] == 0.3
    assert seen["kwargs"]["sliding_window"] == 16
    assert seen["kwargs"]["cache_backend"] == "native-paged"


def test_model_generate_sample_is_runtime_shim(monkeypatch):
    seen = {}

    def fake_runtime_generate(model, input_ids, **kwargs):
        seen["model"] = model
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.full((1, 4), 8, dtype=torch.long)

    monkeypatch.setattr(model_generate_mod, "runtime_sample_generate", fake_runtime_generate)

    out = model_generate_mod.sample_generate(
        object(),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=7,
        temperature=0.5,
        top_k=3,
        top_p=0.8,
        eos_id=4,
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        no_repeat_ngram=2,
        repetition_penalty=1.3,
        presence_penalty=0.4,
        frequency_penalty=0.1,
        sliding_window=24,
        cache_backend="paged",
    )

    assert torch.equal(out, torch.full((1, 4), 8, dtype=torch.long))
    assert seen["kwargs"]["max_new_tokens"] == 7
    assert seen["kwargs"]["temperature"] == 0.5
    assert seen["kwargs"]["top_k"] == 3
    assert seen["kwargs"]["top_p"] == 0.8
    assert seen["kwargs"]["eos_id"] == 4
    assert torch.equal(seen["kwargs"]["attention_mask"], torch.tensor([[1, 1]], dtype=torch.long))
    assert seen["kwargs"]["no_repeat_ngram"] == 2
    assert seen["kwargs"]["repetition_penalty"] == 1.3
    assert seen["kwargs"]["presence_penalty"] == 0.4
    assert seen["kwargs"]["frequency_penalty"] == 0.1
    assert seen["kwargs"]["sliding_window"] == 24
    assert seen["kwargs"]["cache_backend"] == "paged"


def test_runtime_greedy_generate_builds_runtime_config_and_cache_backend(monkeypatch):
    seen = {}

    def fake_generate(model, input_ids, **kwargs):
        seen["model"] = model
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.full((1, 4), 3, dtype=torch.long)

    monkeypatch.setattr(runtime_generation_mod, "generate", fake_generate)

    out = runtime_generation_mod.greedy_generate(
        object(),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=6,
        eos_id=9,
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        no_repeat_ngram=2,
        repetition_penalty=1.2,
        presence_penalty=0.3,
        frequency_penalty=0.4,
        sliding_window=32,
        cache_backend="native-paged",
    )

    cfg = seen["kwargs"]["config"]
    assert torch.equal(out, torch.full((1, 4), 3, dtype=torch.long))
    assert cfg.max_new_tokens == 6
    assert cfg.do_sample is False
    assert cfg.eos_id == 9
    assert cfg.no_repeat_ngram == 2
    assert cfg.repetition_penalty == 1.2
    assert cfg.presence_penalty == 0.3
    assert cfg.frequency_penalty == 0.4
    assert cfg.sliding_window == 32
    assert torch.equal(seen["kwargs"]["attention_mask"], torch.tensor([[1, 1]], dtype=torch.long))
    assert seen["kwargs"]["cache_backend"] == "native-paged"


def test_runtime_sample_generate_builds_runtime_config_and_cache_backend(monkeypatch):
    seen = {}

    def fake_generate(model, input_ids, **kwargs):
        seen["model"] = model
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.full((1, 5), 4, dtype=torch.long)

    monkeypatch.setattr(runtime_generation_mod, "generate", fake_generate)

    out = runtime_generation_mod.sample_generate(
        object(),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=8,
        temperature=0.6,
        top_k=5,
        top_p=0.85,
        eos_id=7,
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        no_repeat_ngram=3,
        repetition_penalty=1.1,
        presence_penalty=0.2,
        frequency_penalty=0.5,
        sliding_window=48,
        cache_backend="paged",
    )

    cfg = seen["kwargs"]["config"]
    assert torch.equal(out, torch.full((1, 5), 4, dtype=torch.long))
    assert cfg.max_new_tokens == 8
    assert cfg.do_sample is True
    assert cfg.temperature == 0.6
    assert cfg.top_k == 5
    assert cfg.top_p == 0.85
    assert cfg.eos_id == 7
    assert cfg.no_repeat_ngram == 3
    assert cfg.repetition_penalty == 1.1
    assert cfg.presence_penalty == 0.2
    assert cfg.frequency_penalty == 0.5
    assert cfg.sliding_window == 48
    assert torch.equal(seen["kwargs"]["attention_mask"], torch.tensor([[1, 1]], dtype=torch.long))
    assert seen["kwargs"]["cache_backend"] == "paged"


def test_serve_api_generate_delegates_to_runtime_transport_helpers(monkeypatch):
    cfg_obj = object()
    seen = {}

    class FakeRuntime:
        def build_generation_config(self, **kwargs):
            seen["config_kwargs"] = kwargs
            return cfg_obj

        def generate_token_lists(self, input_ids, *, config, attention_mask=None, cache_backend=None):
            seen["input_ids"] = input_ids
            seen["config"] = config
            seen["attention_mask"] = attention_mask
            seen["cache_backend"] = cache_backend
            return [[1, 2, 3, 4]]

    monkeypatch.setattr(serve_api_mod, "get_runtime", lambda: FakeRuntime())

    resp = serve_api_mod.generate(
        serve_api_mod.GenerateRequest(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
            max_new_tokens=5,
            temperature=0.7,
            top_k=10,
            top_p=0.9,
            eos_id=2,
            no_repeat_ngram=3,
            repetition_penalty=1.2,
            presence_penalty=0.4,
            frequency_penalty=0.2,
            sliding_window=32,
            beam_size=5,
            length_penalty=0.75,
            cache_backend="native-paged",
        )
    )

    assert resp.output_ids == [[1, 2, 3, 4]]
    assert seen["input_ids"] == [[1, 2, 3]]
    assert seen["config"] is cfg_obj
    assert seen["attention_mask"] == [[1, 1, 1]]
    assert seen["cache_backend"] == "native-paged"
    assert seen["config_kwargs"]["max_new_tokens"] == 5
    assert seen["config_kwargs"]["do_sample"] is None
    assert seen["config_kwargs"]["temperature"] == 0.7
    assert seen["config_kwargs"]["top_k"] == 10
    assert seen["config_kwargs"]["top_p"] == 0.9
    assert seen["config_kwargs"]["eos_id"] == 2
    assert seen["config_kwargs"]["no_repeat_ngram"] == 3
    assert seen["config_kwargs"]["repetition_penalty"] == 1.2
    assert seen["config_kwargs"]["presence_penalty"] == 0.4
    assert seen["config_kwargs"]["frequency_penalty"] == 0.2
    assert seen["config_kwargs"]["sliding_window"] == 32
    assert seen["config_kwargs"]["beam_size"] == 5
    assert seen["config_kwargs"]["length_penalty"] == 0.75


def test_serve_api_generate_honors_explicit_greedy_override(monkeypatch):
    seen = {}

    class FakeRuntime:
        def build_generation_config(self, **kwargs):
            seen["config_kwargs"] = kwargs
            return object()

        def generate_token_lists(self, input_ids, *, config, attention_mask=None, cache_backend=None):
            del input_ids, config, attention_mask, cache_backend
            return [[1, 2, 3]]

    monkeypatch.setattr(serve_api_mod, "get_runtime", lambda: FakeRuntime())

    resp = serve_api_mod.generate(
        serve_api_mod.GenerateRequest(
            input_ids=[[1, 2]],
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
        )
    )

    assert resp.output_ids == [[1, 2, 3]]
    assert seen["config_kwargs"]["do_sample"] is False


def test_eval_benchmark_generate_builds_config_via_runtime_builder(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["builder_kwargs"] = kwargs
        return "cfg"

    cache_ids = []

    monkeypatch.setattr(eval_bench_mod, "runtime_build_generation_config", fake_builder)
    monkeypatch.setattr(
        eval_bench_mod,
        "decode_tokens",
        lambda model, x, **kwargs: cache_ids.append(kwargs.get("cache")) or x,
    )

    model = torch.nn.Linear(16, 16, bias=False)
    counter = {"count": 0}

    def fake_cache_factory(*, batch_size, backend=None):
        counter["count"] += 1
        return {"id": counter["count"], "batch_size": batch_size, "backend": backend}

    eval_bench_mod.benchmark_generate(
        model,
        batch_size=1,
        seq_len=2,
        max_new_tokens=3,
        warmup_steps=1,
        repeats=2,
        vocab_size=8,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        sliding_window=16,
        beam_size=3,
        length_penalty=0.9,
        cache_backend="native-paged",
        kv_cache_factory=fake_cache_factory,
    )

    assert seen["builder_kwargs"]["max_new_tokens"] == 3
    assert seen["builder_kwargs"]["do_sample"] is True
    assert seen["builder_kwargs"]["temperature"] == 0.7
    assert seen["builder_kwargs"]["top_p"] == 0.9
    assert seen["builder_kwargs"]["repetition_penalty"] == 1.2
    assert seen["builder_kwargs"]["sliding_window"] == 16
    assert seen["builder_kwargs"]["beam_size"] == 3
    assert seen["builder_kwargs"]["length_penalty"] == 0.9
    assert [cache["id"] for cache in cache_ids] == [1, 2, 3]
    assert all(cache["backend"] == "native-paged" for cache in cache_ids)


def test_eval_benchmark_generate_skips_cache_for_beam_search(monkeypatch):
    cache_ids = []

    monkeypatch.setattr(
        eval_bench_mod,
        "runtime_build_generation_config",
        lambda **kwargs: serve_runtime_mod.GenerationConfig(max_new_tokens=3, beam_size=2),
    )
    monkeypatch.setattr(
        eval_bench_mod,
        "decode_tokens",
        lambda model, x, **kwargs: cache_ids.append(kwargs.get("cache")) or x,
    )

    model = torch.nn.Linear(16, 16, bias=False)

    def fake_cache_factory(*, batch_size, backend=None):
        raise AssertionError("beam search benchmark should not allocate cache")

    eval_bench_mod.benchmark_generate(
        model,
        batch_size=1,
        seq_len=2,
        max_new_tokens=3,
        warmup_steps=1,
        repeats=2,
        vocab_size=8,
        beam_size=2,
        kv_cache_factory=fake_cache_factory,
    )

    assert cache_ids == [None, None, None]


def test_eval_latency_generate_builds_config_via_runtime_builder(monkeypatch):
    seen = {}

    def fake_builder(**kwargs):
        seen["builder_kwargs"] = kwargs
        return "cfg"

    cache_ids = []

    monkeypatch.setattr(eval_latency_mod, "runtime_build_generation_config", fake_builder)
    monkeypatch.setattr(
        eval_latency_mod,
        "decode_tokens",
        lambda model, x, **kwargs: cache_ids.append(kwargs.get("cache")) or x,
    )

    model = torch.nn.Linear(16, 16, bias=False)
    counter = {"count": 0}

    def fake_cache_factory(*, batch_size, backend=None):
        counter["count"] += 1
        return {"id": counter["count"], "batch_size": batch_size, "backend": backend}

    eval_latency_mod.latency_generate(
        model,
        repeats=2,
        batch_size=1,
        seq_len=2,
        max_new_tokens=4,
        vocab_size=8,
        do_sample=True,
        top_k=4,
        presence_penalty=0.5,
        beam_size=2,
        length_penalty=0.65,
        cache_backend="paged",
        kv_cache_factory=fake_cache_factory,
    )

    assert seen["builder_kwargs"]["max_new_tokens"] == 4
    assert seen["builder_kwargs"]["do_sample"] is True
    assert seen["builder_kwargs"]["top_k"] == 4
    assert seen["builder_kwargs"]["presence_penalty"] == 0.5
    assert seen["builder_kwargs"]["beam_size"] == 2
    assert seen["builder_kwargs"]["length_penalty"] == 0.65
    assert [cache["id"] for cache in cache_ids] == [1, 2]
    assert all(cache["backend"] == "paged" for cache in cache_ids)


def test_eval_latency_generate_skips_cache_for_beam_search(monkeypatch):
    cache_ids = []

    monkeypatch.setattr(
        eval_latency_mod,
        "runtime_build_generation_config",
        lambda **kwargs: serve_runtime_mod.GenerationConfig(max_new_tokens=4, beam_size=2),
    )
    monkeypatch.setattr(
        eval_latency_mod,
        "decode_tokens",
        lambda model, x, **kwargs: cache_ids.append(kwargs.get("cache")) or x,
    )

    model = torch.nn.Linear(16, 16, bias=False)

    def fake_cache_factory(*, batch_size, backend=None):
        raise AssertionError("beam search latency path should not allocate cache")

    eval_latency_mod.latency_generate(
        model,
        repeats=2,
        batch_size=1,
        seq_len=2,
        max_new_tokens=4,
        vocab_size=8,
        beam_size=2,
        kv_cache_factory=fake_cache_factory,
    )

    assert cache_ids == [None, None]


def test_eval_suite_run_basic_suite_passes_runtime_decode_kwargs(monkeypatch):
    model = torch.nn.Linear(16, 16, bias=False)
    loader = object()
    seen = {}

    monkeypatch.setattr(
        eval_suite_mod,
        "evaluate_lm_next_token",
        lambda model_in, loader_in, **kwargs: type("Ppl", (), {"nll": 1.0, "ppl": 2.0, "acc": 0.5, "num_tokens": 10})(),
    )
    monkeypatch.setattr(
        eval_suite_mod,
        "benchmark_forward",
        lambda model_in, **kwargs: eval_bench_mod.ThroughputResult(tokens_per_sec=11.0, latency_ms=12.0, total_tokens=13, total_time_s=14.0),
    )

    def fake_benchmark_generate(model_in, **kwargs):
        seen["gen_kwargs"] = kwargs
        return eval_bench_mod.ThroughputResult(tokens_per_sec=21.0, latency_ms=22.0, total_tokens=23, total_time_s=24.0)

    monkeypatch.setattr(eval_suite_mod, "benchmark_generate", fake_benchmark_generate)
    monkeypatch.setattr(
        eval_suite_mod,
        "evaluate_ece",
        lambda model_in, loader_in, **kwargs: type("Ece", (), {"ece": 0.1, "num_tokens": 9})(),
    )

    result = eval_suite_mod.run_basic_suite(
        model,
        loader,
        device="cpu",
        kv_cache_factory="kvf",
        do_sample=None,
        temperature=0.7,
        top_k=4,
        top_p=0.9,
        repetition_penalty=1.2,
        sliding_window=32,
        beam_size=6,
        length_penalty=0.55,
        cache_backend="native-paged",
    )

    assert result.bench_generate["tokens_per_sec"] == 21.0
    assert seen["gen_kwargs"]["kv_cache_factory"] == "kvf"
    assert seen["gen_kwargs"]["temperature"] == 0.7
    assert seen["gen_kwargs"]["top_k"] == 4
    assert seen["gen_kwargs"]["top_p"] == 0.9
    assert seen["gen_kwargs"]["repetition_penalty"] == 1.2
    assert seen["gen_kwargs"]["sliding_window"] == 32
    assert seen["gen_kwargs"]["beam_size"] == 6
    assert seen["gen_kwargs"]["length_penalty"] == 0.55
    assert seen["gen_kwargs"]["cache_backend"] == "native-paged"


def test_eval_cli_suite_uses_runtime_from_model_and_generation_kwargs(monkeypatch):
    fake_model = torch.nn.Linear(4, 4, bias=False)
    seen = {}

    class FakeRuntime:
        @classmethod
        def from_model(cls, model, *, device=None):
            seen["from_model"] = {"model": model, "device": device}
            inst = cls()
            inst.allocate_cache = lambda batch_size, backend=None: ("cache", batch_size, backend)
            return inst

    monkeypatch.setattr(eval_cli_mod, "_load_or_build", lambda args, device=None: fake_model)
    monkeypatch.setattr(serve_runtime_mod, "ModelRuntime", FakeRuntime)
    monkeypatch.setattr(eval_cli_mod, "build_dataloader", lambda *args, **kwargs: "loader")

    def fake_run_basic_suite(model, loader, **kwargs):
        seen["suite"] = {"model": model, "loader": loader, "kwargs": kwargs}
        cache = (
            kwargs["kv_cache_factory"](batch_size=2, backend=kwargs.get("cache_backend"))
            if kwargs.get("kv_cache_factory") is not None
            else None
        )
        seen["cache"] = cache
        return eval_suite_mod.SuiteResult(
            ppl={"nll": 1.0, "ppl": 2.0, "acc": 0.5, "tokens": 10},
            bench_forward={"tokens_per_sec": 3.0, "latency_ms": 4.0},
            bench_generate={"tokens_per_sec": 5.0, "latency_ms": 6.0},
            ece={"ece": 0.1, "tokens": 9},
        )

    monkeypatch.setattr(eval_cli_mod, "run_basic_suite", fake_run_basic_suite)

    args = type(
        "Args",
        (),
        {
            "device": "cpu",
            "shards": "demo-shards",
            "batch_size": 1,
            "seq_len": 2,
            "num_workers": 0,
            "seed": 1337,
            "streaming": False,
            "do_sample": None,
            "temperature": 0.8,
            "top_k": 7,
            "top_p": 0.85,
            "eos_id": 2,
            "no_repeat_ngram": 3,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.4,
            "sliding_window": 40,
            "beam_size": 3,
            "length_penalty": 0.6,
            "cache_backend": "paged",
            "outdir": None,
            "model_dir": None,
            "model": "demo.builders:build_small",
        },
    )()

    eval_cli_mod.cmd_suite(args)

    assert seen["from_model"]["model"] is fake_model
    assert seen["from_model"]["device"] == torch.device("cpu")
    assert seen["suite"]["model"] is fake_model
    assert seen["suite"]["loader"] == "loader"
    assert seen["cache"] == ("cache", 2, "paged")
    assert seen["suite"]["kwargs"]["temperature"] == 0.8
    assert seen["suite"]["kwargs"]["top_k"] == 7
    assert seen["suite"]["kwargs"]["top_p"] == 0.85
    assert seen["suite"]["kwargs"]["repetition_penalty"] == 1.1
    assert seen["suite"]["kwargs"]["beam_size"] == 3
    assert seen["suite"]["kwargs"]["length_penalty"] == 0.6
    assert seen["suite"]["kwargs"]["cache_backend"] == "paged"


def test_serve_api_healthz_delegates_to_runtime_health_info(monkeypatch):
    monkeypatch.setattr(
        serve_api_mod,
        "get_runtime",
        lambda: type("FakeRuntime", (), {"health_info": lambda self: {"status": "ok", "device": "cpu"}})(),
    )
    assert serve_api_mod.healthz() == {"status": "ok", "device": "cpu"}


def test_clone_kv_cache_rows_uses_native_layer_clone_rows_fast_path():
    class FakeNativeLayer:
        def __init__(self):
            self.ids = None

        def clone_rows(self, row_ids):
            self.ids = row_ids.clone()
            return f"layer-clone:{id(self)}"

    layers = [FakeNativeLayer(), FakeNativeLayer()]
    cache = runtime_kv_cache_mod.PagedKVCache(
        batch=2,
        n_layers=2,
        n_kv_heads=1,
        head_dim=2,
        pagesize=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        native_cache_state=None,
        native_layer_states=layers,
        backend_name="native-paged",
    )

    cloned = runtime_kv_cache_mod.clone_kv_cache_rows(cache, torch.tensor([0, 1, 1], dtype=torch.long))

    assert [layer.ids.tolist() for layer in layers] == [[0, 1, 1], [0, 1, 1]]
    assert isinstance(cloned, runtime_kv_cache_mod.PagedKVCache)
    assert cloned._native_layers == [f"layer-clone:{id(layer)}" for layer in layers]
    assert cloned.batch == 3


def test_reorder_kv_cache_rows_reorders_contiguous_cache_in_place():
    cache = runtime_kv_cache_mod.ContiguousKVCache(
        batch=2,
        n_layers=1,
        n_kv_heads=1,
        head_dim=2,
        pagesize=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
        backend_name="contiguous",
    )
    cache.append_batch(
        0,
        torch.tensor([[[[1.0, 10.0], [2.0, 20.0]]]], dtype=torch.float32),
        torch.tensor([[[[11.0, 110.0], [12.0, 120.0]]]], dtype=torch.float32),
        block_ids=torch.tensor([0], dtype=torch.long),
    )
    cache.append_batch(
        0,
        torch.tensor([[[[3.0, 30.0]]]], dtype=torch.float32),
        torch.tensor([[[[13.0, 130.0]]]], dtype=torch.float32),
        block_ids=torch.tensor([1], dtype=torch.long),
    )

    out = runtime_kv_cache_mod.reorder_kv_cache_rows_(cache, torch.tensor([1, 0, 1], dtype=torch.long))

    assert out is cache
    assert cache.batch == 3
    assert cache.layer_lengths(0).tolist() == [1, 2, 1]
    k, v = cache.read_batch(0, 0, 2)
    assert torch.equal(k[0, :, 0, :], torch.tensor([[3.0, 30.0]], dtype=torch.float32))
    assert torch.equal(k[1], torch.tensor([[[1.0, 10.0], [2.0, 20.0]]], dtype=torch.float32))
    assert torch.equal(v[2, :, 0, :], torch.tensor([[13.0, 130.0]], dtype=torch.float32))


def test_runtime_generate_uses_cached_beam_decode_when_available(monkeypatch):
    seen = {"reorder_rows": []}
    fake_cache = object()

    class FakeModel:
        cfg = type("Cfg", (), {"pad_token_id": 0})()

        def __call__(self, input_ids, **kwargs):
            seen.setdefault("calls", []).append(
                {
                    "shape": tuple(input_ids.shape),
                    "attention_mask": kwargs.get("attention_mask"),
                    "cache": kwargs.get("cache"),
                    "position_ids": kwargs.get("position_ids"),
                }
            )
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 6), -100.0, dtype=torch.float32)
            if input_ids.shape[1] > 1:
                logits[:, -1, 2] = 0.0
                logits[:, -1, 3] = -0.1
            else:
                last = input_ids[:, -1]
                logits[last == 2, -1, 4] = 0.0
                logits[last == 3, -1, 5] = -0.5
                logits[last == 2, -1, 0] = -1.0
                logits[last == 3, -1, 0] = -1.5
            return logits

    def fake_from_model(cls, model_in, input_ids, **kwargs):
        del cls
        return runtime_generation_mod.RuntimeGenerationSession(
            model=model_in,
            seq=input_ids,
            attention_mask=kwargs.get("attention_mask"),
            cache=fake_cache,
            trace=False,
        )

    def fake_reorder(cache, row_ids):
        assert cache is fake_cache
        seen["reorder_rows"].append(row_ids.detach().cpu().tolist())
        return fake_cache

    monkeypatch.setattr(runtime_generation_mod, "create_native_model_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_generation_mod.RuntimeGenerationSession, "from_model", classmethod(fake_from_model))
    monkeypatch.setattr(runtime_generation_mod, "runtime_reorder_kv_cache_rows_", fake_reorder)
    monkeypatch.setattr(
        runtime_generation_mod,
        "_generate_with_full_forward_beam_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback")),
    )

    cfg = runtime_generation_mod.GenerationConfig(max_new_tokens=2, beam_size=2, eos_id=0)
    out = runtime_generation_mod.generate(
        FakeModel(),
        torch.tensor([[1, 1]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        config=cfg,
    )

    assert out.tolist() == [[1, 1, 2, 4]]
    assert seen["reorder_rows"][0] == [0, 0]
    assert len(seen["reorder_rows"]) == 1
    assert seen["calls"][0]["shape"] == (1, 2)
    assert seen["calls"][1]["shape"] == (2, 1)
    assert seen["calls"][1]["cache"] is fake_cache
    assert torch.equal(
        seen["calls"][1]["attention_mask"],
        torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
    )


def test_runtime_generation_session_python_append_preserves_explicit_2d_attention_mask_mode(monkeypatch):
    monkeypatch.setattr(runtime_generation_mod, "create_native_model_session", lambda *args, **kwargs: None)
    session = runtime_generation_mod.RuntimeGenerationSession(
        torch.nn.Identity(),
        torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=runtime_blocks_mod.build_prefix_lm_mask(2, 1),
    )

    session.append(torch.tensor([[3]], dtype=torch.long))
    session.append(torch.tensor([[4]], dtype=torch.long))

    assert session.attention_mask_mode == "explicit"
    assert session.attention_mask.shape == (1, 4)
    assert session.attention_mask.dtype == torch.bool
    assert torch.equal(session.attention_mask, torch.zeros((1, 4), dtype=torch.bool))


def test_runtime_generation_session_discards_python_executor_wrappers(monkeypatch):
    class FakePythonSession:
        native_executor_kind = "python"

        def __init__(self, model, seq, attention_mask=None, cache=None, trace=False):
            del model, trace
            self.seq = seq
            self.attention_mask = attention_mask
            self.cache = cache

    monkeypatch.setattr(
        runtime_generation_mod,
        "create_native_model_session",
        lambda model, seq, attention_mask=None, cache=None, trace=False: FakePythonSession(
            model,
            seq,
            attention_mask,
            cache,
            trace,
        ),
    )

    session = runtime_generation_mod.RuntimeGenerationSession(
        torch.nn.Identity(),
        torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
    )

    assert session.uses_native_session is False
    assert session.native_executor_kind == "python"
    assert torch.equal(session.seq, torch.tensor([[1, 2]], dtype=torch.long))


def test_prefix_causal_lm_accepts_attention_mask_alias():
    model = runtime_modeling_mod.build_model(_cfg(), task="prefix-lm")
    out = model(
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=runtime_blocks_mod.build_prefix_lm_mask(3, 1),
    )
    assert out.shape == (1, 3, model.cfg.vocab_size)


def test_runtime_generation_session_native_prefix_executor_skips_python_forward():
    model = runtime_modeling_mod.build_model(_cfg(), task="prefix-lm")
    model.eval()
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=runtime_blocks_mod.build_prefix_lm_mask(2, 1),
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    assert prefill.shape == (1, model.cfg.vocab_size)

    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)

    assert session.attention_mask is not None
    assert tuple(session.attention_mask.shape) == (1, 3)

    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_native_causal_executor_supports_3d_attention_mask():
    model = causal_model_mod.CausalLM(_cfg(), block_variant="llama")
    model.eval()
    attention_mask = runtime_blocks_mod.build_prefix_lm_mask(2, 1).unsqueeze(0)
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=attention_mask,
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)

    assert session.attention_mask is not None
    assert tuple(session.attention_mask.shape) == (1, 1, 3)

    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_native_causal_executor_supports_4d_attention_mask():
    model = causal_model_mod.CausalLM(_cfg(), block_variant="llama")
    model.eval()
    bool_mask = runtime_blocks_mod.build_prefix_lm_mask(2, 1)
    attention_mask = torch.where(
        bool_mask,
        torch.full(bool_mask.shape, float("-inf"), dtype=torch.float32),
        torch.zeros(bool_mask.shape, dtype=torch.float32),
    ).view(1, 1, 2, 2)
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=attention_mask,
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)

    assert session.attention_mask is not None
    assert tuple(session.attention_mask.shape) == (1, 1, 1, 3)

    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_native_moe_executor_skips_python_forward():
    model = causal_model_mod.CausalLM(_cfg(), block_variant="moe", num_experts=3, k=2)
    model.eval()
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    assert prefill.shape == (1, model.cfg.vocab_size)

    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)

    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_native_causal_executor_supports_alibi_bias():
    model = causal_model_mod.CausalLM(_cfg(), block_variant="gpt", use_alibi=True)
    model.eval()
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)
    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_native_causal_executor_supports_rpb_table():
    model = causal_model_mod.CausalLM(_cfg(), block_variant="gpt", use_rpb=True, rpb_max_distance=4)
    model.eval()
    assert model.blocks[0].rpb_table is not None
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)
    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_native_parallel_executor_skips_python_forward():
    model = causal_model_mod.CausalLM(_cfg(), block_variant="parallel")
    model.eval()
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if session.native_executor_kind != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    assert prefill.shape == (1, model.cfg.vocab_size)

    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)

    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_native_causal_executor_skips_python_forward():
    model = causal_model_mod.CausalLM(_cfg(), block_variant="llama")
    model.eval()
    session = runtime_generation_mod.RuntimeGenerationSession.from_model(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        cache_pagesize=32,
        cache_backend="native-paged",
    )

    if not session.uses_native_session:
        pytest.skip("native model session unavailable")
    if getattr(session.native_session, "native_executor_kind", "python") != "causal_lm":
        pytest.skip("native causal executor unavailable")

    def boom(*args, **kwargs):
        raise AssertionError("python model forward should not be called")

    model.forward = boom

    prefill = session.prefill_next_logits()
    assert prefill is not None
    assert prefill.shape == (1, model.cfg.vocab_size)

    next_id = torch.argmax(prefill, dim=-1, keepdim=True)
    session.append(next_id)

    decode = session.decode_next_logits()
    assert decode is not None
    assert decode.shape == (1, model.cfg.vocab_size)

    full = session.full_next_logits()
    assert full.shape == (1, model.cfg.vocab_size)


def test_runtime_generation_session_python_decode_uses_step_token_attention_mask():
    seen = {}

    class FakeModel:
        def __call__(
            self,
            input_ids,
            *,
            attn_mask=None,
            attention_mask=None,
            cache=None,
            position_ids=None,
            cache_position=None,
            return_dict=False,
        ):
            seen["input_shape"] = tuple(input_ids.shape)
            seen["attn_mask"] = attn_mask
            seen["attention_mask"] = None if attention_mask is None else attention_mask.clone()
            seen["cache"] = cache
            seen["position_ids"] = None if position_ids is None else position_ids.clone()
            seen["cache_position"] = None if cache_position is None else cache_position.clone()
            seen["return_dict"] = return_dict
            return torch.zeros((input_ids.shape[0], input_ids.shape[1], 7), dtype=torch.float32)

    session = runtime_generation_mod.RuntimeGenerationSession(
        model=FakeModel(),
        seq=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        cache=object(),
        trace=False,
    )

    out = session.decode_next_logits()

    assert out is not None
    assert out.shape == (1, 7)
    assert seen["input_shape"] == (1, 1)
    assert seen["attn_mask"] is None
    assert seen["attention_mask"] is not None
    assert tuple(seen["attention_mask"].shape) == (1, 1)
    assert torch.equal(seen["attention_mask"], torch.ones((1, 1), dtype=torch.long))
    assert seen["cache"] is session.cache
    assert torch.equal(seen["position_ids"], torch.tensor([[2]], dtype=torch.long))
    assert torch.equal(seen["cache_position"], torch.tensor([2], dtype=torch.long))
    assert seen["return_dict"] is False
    assert tuple(session.attention_mask.shape) == (1, 3)


def test_runtime_generation_session_advances_beam_via_native_session(monkeypatch):
    seen = {}

    class FakeNativeSession:
        def __init__(self, model, seq, attention_mask, cache, trace):
            self.model = model
            self.seq = seq
            self.attention_mask = attention_mask
            self.cache = cache
            self.trace = trace

        @property
        def batch_size(self):
            return int(self.seq.shape[0])

        @property
        def seq_len(self):
            return int(self.seq.shape[1])

        def advance_beam_decode(
            self,
            next_beams,
            cache_row_ids,
            mask_row_ids,
            source_attention_mask,
            source_cache,
            max_tokens,
            policy,
        ):
            seen["next_beams"] = next_beams.clone()
            seen["cache_row_ids"] = cache_row_ids.clone()
            seen["mask_row_ids"] = None if mask_row_ids is None else mask_row_ids.clone()
            seen["source_attention_mask"] = source_attention_mask.clone()
            seen["source_cache"] = source_cache
            seen["max_tokens"] = max_tokens
            seen["policy"] = policy
            self.seq = next_beams
            self.attention_mask = source_attention_mask
            self.cache = source_cache
            return torch.zeros(next_beams.shape[0], 7, dtype=torch.float32)

    monkeypatch.setattr(
        runtime_generation_mod,
        "create_native_model_session",
        lambda model, seq, attention_mask=None, cache=None, trace=False: FakeNativeSession(
            model,
            seq,
            attention_mask,
            cache,
            trace,
        ),
    )

    source_mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    source_cache = object()
    session = runtime_generation_mod.RuntimeGenerationSession(
        model=object(),
        seq=torch.tensor([[1, 2], [1, 3]], dtype=torch.long),
        attention_mask=None,
        cache=None,
        trace=False,
    )

    out = session.advance_beam_decode(
        torch.tensor([[1, 2, 4], [1, 3, 5]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
        mask_row_ids=torch.tensor([0, 1], dtype=torch.long),
        source_attention_mask=source_mask,
        source_cache=source_cache,
        max_tokens=9,
        policy="sliding-window",
    )

    assert session.uses_native_session is True
    assert out.shape == (2, 7)
    assert seen["next_beams"].tolist() == [[1, 2, 4], [1, 3, 5]]
    assert seen["cache_row_ids"].tolist() == [0, 1]
    assert seen["mask_row_ids"].tolist() == [0, 1]
    assert torch.equal(seen["source_attention_mask"], source_mask)
    assert seen["source_cache"] is source_cache
    assert seen["max_tokens"] == 9
    assert seen["policy"] == "sliding-window"


def test_runtime_generate_beam_search_falls_back_to_full_forward_without_cache(monkeypatch):
    seen = {}

    def fake_from_model(cls, model_in, input_ids, **kwargs):
        del cls, model_in, kwargs
        return runtime_generation_mod.RuntimeGenerationSession(
            model=object(),
            seq=input_ids,
            attention_mask=None,
            cache=None,
            trace=False,
        )

    def fake_full_forward(model, input_ids, **kwargs):
        seen["kwargs"] = kwargs
        return torch.full((input_ids.shape[0], input_ids.shape[1] + 1), 8, dtype=torch.long)

    monkeypatch.setattr(runtime_generation_mod.RuntimeGenerationSession, "from_model", classmethod(fake_from_model))
    monkeypatch.setattr(runtime_generation_mod, "_generate_with_full_forward_beam_search", fake_full_forward)

    out = runtime_generation_mod.generate(
        object(),
        torch.tensor([[1, 2]], dtype=torch.long),
        config=runtime_generation_mod.GenerationConfig(max_new_tokens=3, beam_size=2, eos_id=4),
    )

    assert torch.equal(out, torch.full((1, 3), 8, dtype=torch.long))
    assert seen["kwargs"]["config"].beam_size == 2


def test_runtime_generate_rejects_sampling_with_beam_search():
    cfg = runtime_generation_mod.GenerationConfig(max_new_tokens=2, beam_size=2, do_sample=True)

    with pytest.raises(ValueError, match="beam search does not support sampling"):
        runtime_generation_mod.generate(object(), torch.tensor([[1, 2]], dtype=torch.long), config=cfg)


def test_model_runtime_generate_token_ids_skips_cache_for_beam_search(monkeypatch):
    cfg = _cfg()
    model = torch.nn.Linear(16, 16, bias=False)
    rt = ModelRuntime(cfg, model, torch.device("cpu"), torch.float32, kv_pagesize=160)
    seen = {}

    def fail_allocate(*args, **kwargs):
        raise AssertionError("beam search should not allocate cache")

    def fake_generate(input_ids, **kwargs):
        seen["input_ids"] = input_ids
        seen["kwargs"] = kwargs
        return torch.tensor([[7, 8, 9]], dtype=torch.long)

    monkeypatch.setattr(rt, "allocate_cache", fail_allocate)
    monkeypatch.setattr(rt, "generate", fake_generate)

    out = rt.generate_token_ids(
        [[1, 2, 3]],
        config=serve_runtime_mod.GenerationConfig(max_new_tokens=2, beam_size=3),
        cache_backend="paged",
    )

    assert torch.equal(out, torch.tensor([[7, 8, 9]], dtype=torch.long))
    assert torch.equal(seen["input_ids"], torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert seen["kwargs"]["cache"] is None
    assert seen["kwargs"]["cache_backend"] == "paged"


def test_runtime_generate_defaults_config(monkeypatch):
    seen = {}

    def fake_session_from_model(cls, model_in, input_ids, **kwargs):
        del cls
        seen["kwargs"] = kwargs

        class FakeSession:
            seq = input_ids
            seq_len = int(input_ids.shape[1])
            cache = None

            def full_next_logits(self):
                return torch.zeros(input_ids.shape[0], 8)

            def append(self, next_id):
                self.seq = torch.cat([self.seq, next_id], dim=1)
                self.seq_len = int(self.seq.shape[1])

        return FakeSession()

    monkeypatch.setattr(runtime_generation_mod.RuntimeGenerationSession, "from_model", classmethod(fake_session_from_model))
    monkeypatch.setattr(
        runtime_generation_mod,
        "runtime_sample_with_policies",
        lambda logits, seq, **kwargs: torch.zeros(seq.shape[0], 1, dtype=torch.long),
    )
    out = runtime_generation_mod.generate(object(), torch.tensor([[1, 2]], dtype=torch.long))
    assert out.shape == (1, 2 + runtime_generation_mod.GenerationConfig().max_new_tokens)
    assert seen["kwargs"]["cache_pagesize"] == 128


def test_runtime_generation_resolve_sampling_mode_infers_from_sampling_knobs():
    assert runtime_generation_mod.resolve_generation_sampling_mode(do_sample=None, temperature=1.0, top_k=None, top_p=None) is False
    assert runtime_generation_mod.resolve_generation_sampling_mode(do_sample=None, temperature=0.8, top_k=None, top_p=None) is True
    assert runtime_generation_mod.resolve_generation_sampling_mode(do_sample=None, temperature=1.0, top_k=5, top_p=None) is True
    assert runtime_generation_mod.resolve_generation_sampling_mode(do_sample=None, temperature=1.0, top_k=None, top_p=0.9) is True
    assert runtime_generation_mod.resolve_generation_sampling_mode(do_sample=False, temperature=0.8, top_k=5, top_p=0.9) is False
