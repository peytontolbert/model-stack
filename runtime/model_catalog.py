from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

DEFAULT_MODEL_INDEX_PATH = "/data/staticpeytonsite/src/research_library/data/model_index.json"

PRIMARY_INTEGRATION_LANES = {
    "video_diffusion_bridge",
    "diffusers_cuda_bridge",
    "nemo_asr_bridge",
    "transformers_causal_lm_bridge",
    "peft_adapter_bridge",
    "encoder_classifier_bridge",
}

_TRANSFORMERS_CAUSAL_ARCH_SUFFIX = "ForCausalLM"
_TRANSFORMERS_ENCODER_TASKS = {
    "feature-extraction",
    "embedding",
    "text-classification",
    "classification",
    "text-ranking",
    "reranking",
}
_DIFFUSION_TASKS = {
    "text-to-video",
    "image-to-video",
    "video-to-video",
    "video-generation",
    "text-to-image",
    "image-to-image",
    "image-generation",
    "image-editing",
}
_ASR_TASKS = {"automatic-speech-recognition", "speech-recognition", "audio-text-to-text"}
_HF_PROVIDER_ALIASES_BY_ID = {
    "RMBG-2.0": ("briaai",),
}


@dataclass(frozen=True)
class ModelCatalogRecord:
    id: str
    path: str
    relative_path: str
    library_name: str | None
    pipeline_tag: str | None
    model_type: str | None
    architectures: tuple[str, ...]
    class_name: str | None
    tasks: tuple[str, ...]
    integration_lane: str
    status: str
    reasons: tuple[str, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class ModelIntegrationPlan:
    model_id: str
    local_path: str
    lane: str
    backend: str
    runnable: bool
    loader: str
    performance_path: str
    notes: tuple[str, ...]


class ModelStackRuntimeUnsupported(RuntimeError):
    def __init__(self, plan: ModelIntegrationPlan) -> None:
        notes = "; ".join(plan.notes)
        super().__init__(
            f"{plan.model_id} is not runnable by the current model-stack loader "
            f"(lane={plan.lane}, backend={plan.backend}). {notes}"
        )
        self.plan = plan


def _tuple_of_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _task_set(record: ModelCatalogRecord) -> set[str]:
    return {task.lower() for task in (*record.tasks, record.pipeline_tag or "") if task}


def _record_from_model(
    model: Mapping[str, Any],
    *,
    audit_by_id: Mapping[str, Mapping[str, Any]],
) -> ModelCatalogRecord:
    model_id = str(model.get("id") or model.get("name") or model.get("relative_path") or "")
    audit = audit_by_id.get(model_id, {})
    lane = str(audit.get("integration_lane") or infer_integration_lane(model))
    status = str(audit.get("status") or "candidate_for_runtime_integration")
    reasons = _tuple_of_strings(audit.get("reasons") or ())
    relative_path = str(model.get("relative_path") or model.get("path") or model_id)
    return ModelCatalogRecord(
        id=model_id,
        path=str(model.get("path") or relative_path),
        relative_path=relative_path,
        library_name=model.get("library_name"),
        pipeline_tag=model.get("pipeline_tag"),
        model_type=model.get("model_type"),
        architectures=_tuple_of_strings(model.get("architectures")),
        class_name=model.get("class_name"),
        tasks=_tuple_of_strings(model.get("tasks")),
        integration_lane=lane,
        status=status,
        reasons=reasons,
        raw=model,
    )


def load_model_catalog(index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH) -> list[ModelCatalogRecord]:
    with open(index_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    audit_items = data.get("model_stack_runtime_audit", {}).get("not_runnable_today", [])
    audit_by_id = {str(item.get("id")): item for item in audit_items if item.get("id")}
    return [_record_from_model(model, audit_by_id=audit_by_id) for model in data.get("models", [])]


def find_catalog_record(
    model_id: str,
    *,
    index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH,
) -> ModelCatalogRecord:
    records = load_model_catalog(index_path)
    for record in records:
        if model_id in {record.id, record.relative_path}:
            return record
    for record in records:
        if record.id.endswith(f"/{model_id}") or record.relative_path.endswith(f"/{model_id}"):
            return record
    raise KeyError(f"Model {model_id!r} was not found in {index_path}")


def primary_lane_records(
    records: Iterable[ModelCatalogRecord],
    *,
    lanes: set[str] | None = None,
) -> list[ModelCatalogRecord]:
    selected = lanes or PRIMARY_INTEGRATION_LANES
    return [record for record in records if record.integration_lane in selected]


def infer_integration_lane(model: Mapping[str, Any]) -> str:
    library = model.get("library_name")
    class_name = str(model.get("class_name") or "")
    model_type = str(model.get("model_type") or "").lower()
    architectures = _tuple_of_strings(model.get("architectures"))
    tasks = {task.lower() for task in (*_tuple_of_strings(model.get("tasks")), str(model.get("pipeline_tag") or "")) if task}

    if library == "peft":
        return "peft_adapter_bridge"
    if library == "nemo" or tasks & _ASR_TASKS:
        return "nemo_asr_bridge" if library == "nemo" else "asr_backend_bridge"
    if library == "diffusers":
        return "diffusers_cuda_bridge"
    if class_name.endswith(_TRANSFORMERS_CAUSAL_ARCH_SUFFIX) or any(
        arch.endswith(_TRANSFORMERS_CAUSAL_ARCH_SUFFIX) for arch in architectures
    ):
        if model_type == "llama":
            return "candidate_runnable_today"
        return "transformers_causal_lm_bridge"
    if tasks & _DIFFUSION_TASKS:
        return "video_diffusion_bridge"
    if tasks & _TRANSFORMERS_ENCODER_TASKS:
        return "encoder_classifier_bridge"
    return "manual_runtime_triage"


def plan_model_integration(
    record: ModelCatalogRecord,
    *,
    model_root: str | os.PathLike[str] | None = None,
) -> ModelIntegrationPlan:
    local_path = _resolve_local_model_path(record, model_root=model_root)
    lane = record.integration_lane
    notes: list[str] = list(record.reasons)

    if lane == "candidate_runnable_today":
        return ModelIntegrationPlan(
            model_id=record.id,
            local_path=local_path,
            lane=lane,
            backend="model_stack_native_llama",
            runnable=True,
            loader="runtime.checkpoint.build_local_llama_from_snapshot",
            performance_path="Existing model-stack LLaMA blocks, KV cache, quantization, and native CUDA gates.",
            notes=tuple(notes or ["HF LLaMA causal LM snapshot candidate."]),
        )
    if lane == "transformers_causal_lm_bridge":
        return ModelIntegrationPlan(
            model_id=record.id,
            local_path=local_path,
            lane=lane,
            backend="transformers",
            runnable=True,
            loader="transformers.AutoModelForCausalLM.from_pretrained",
            performance_path="Start with PyTorch CUDA, then port compatible blocks and kernels into runtime.factory/runtime.native.",
            notes=tuple(notes or ["Custom causal LM architecture needs a Transformers bridge before native promotion."]),
        )
    if lane in {"encoder_classifier_bridge", "vlm_vision_bridge", "vlm_transformers_bridge"}:
        return ModelIntegrationPlan(
            model_id=record.id,
            local_path=local_path,
            lane=lane,
            backend="transformers",
            runnable=True,
            loader="transformers AutoModel family",
            performance_path="Start with PyTorch CUDA encoders/classifier heads, then share runtime encoder kernels where shapes match.",
            notes=tuple(notes or ["Encoder/classifier snapshot needs a Transformers bridge."]),
        )
    if lane in {"video_diffusion_bridge", "diffusers_cuda_bridge", "image_diffusion_bridge"}:
        return ModelIntegrationPlan(
            model_id=record.id,
            local_path=local_path,
            lane=lane,
            backend="diffusers",
            runnable=True,
            loader="diffusers.DiffusionPipeline.from_pretrained",
            performance_path="Start with Diffusers on CUDA/BF16, then move transformer/attention/VAE hot paths behind runtime.native.",
            notes=tuple(notes or ["Diffusion pipeline needs Diffusers first-run support."]),
        )
    if lane in {"nemo_asr_bridge", "asr_backend_bridge"}:
        return ModelIntegrationPlan(
            model_id=record.id,
            local_path=local_path,
            lane=lane,
            backend="nemo_or_transformers_asr",
            runnable=True,
            loader="nemo.collections.asr.models.ASRModel or transformers ASR AutoModel",
            performance_path="Use runtime.asr for preprocessing/streaming, with CUDA model backend as the next replaceable layer.",
            notes=tuple(notes or ["ASR model needs a concrete backend plugged into runtime.asr.AsrRuntime."]),
        )
    if lane == "peft_adapter_bridge":
        return ModelIntegrationPlan(
            model_id=record.id,
            local_path=local_path,
            lane=lane,
            backend="peft",
            runnable=False,
            loader="peft.PeftModel.from_pretrained(base_model, adapter_path)",
            performance_path="Load adapter onto an already runnable base model, then merge or compile hot paths after parity checks.",
            notes=tuple(notes or ["PEFT adapters require an explicit base model."]),
        )
    return ModelIntegrationPlan(
        model_id=record.id,
        local_path=local_path,
        lane=lane,
        backend="manual",
        runnable=False,
        loader="manual integration required",
        performance_path="Add a bridge loader first, then decide whether kernels map to existing runtime ops.",
        notes=tuple(notes or ["No automatic model-stack integration lane is available yet."]),
    )


def diffusers_catalog_status(
    record: ModelCatalogRecord | str,
    *,
    index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH,
    model_root: str | os.PathLike[str] | None = None,
) -> Any:
    selected = _resolve_catalog_record(record, index_path=index_path)
    plan = plan_model_integration(selected, model_root=model_root)
    _ensure_diffusers_plan(plan)
    from runtime.diffusers_bridge import diffusers_snapshot_status

    return diffusers_snapshot_status(plan.local_path)


def diffusers_catalog_adapter_status(
    record: ModelCatalogRecord | str,
    *,
    index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH,
    model_root: str | os.PathLike[str] | None = None,
) -> Any:
    selected = _resolve_catalog_record(record, index_path=index_path)
    plan = plan_model_integration(selected, model_root=model_root)
    _ensure_diffusers_or_adapter_plan(plan)
    from runtime.diffusers_bridge import diffusers_adapter_status

    return diffusers_adapter_status(plan.local_path)


def load_catalog_diffusers_lora_adapter(
    pipeline: Any,
    record: ModelCatalogRecord | str,
    *,
    index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH,
    model_root: str | os.PathLike[str] | None = None,
    weight_name: str | None = None,
    adapter_name: str | None = None,
    **kwargs: Any,
) -> Any:
    selected = _resolve_catalog_record(record, index_path=index_path)
    plan = plan_model_integration(selected, model_root=model_root)
    _ensure_diffusers_or_adapter_plan(plan)
    from runtime.diffusers_bridge import load_diffusers_lora_adapter

    return load_diffusers_lora_adapter(
        pipeline,
        plan.local_path,
        weight_name=weight_name,
        adapter_name=adapter_name,
        **kwargs,
    )


def load_catalog_diffusers_component(
    record: ModelCatalogRecord | str,
    component: str,
    *,
    index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH,
    model_root: str | os.PathLike[str] | None = None,
    options: Any | None = None,
    component_class: Any | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    selected = _resolve_catalog_record(record, index_path=index_path)
    plan = plan_model_integration(selected, model_root=model_root)
    _ensure_diffusers_plan(plan)
    from runtime.diffusers_bridge import DiffusersBridgeOptions, load_diffusers_component

    resolved_options = options or DiffusersBridgeOptions(
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        variant=kwargs.pop("variant", None),
        use_safetensors=kwargs.pop("use_safetensors", None),
        low_cpu_mem_usage=bool(kwargs.pop("low_cpu_mem_usage", True)),
    )
    return load_diffusers_component(
        plan.local_path,
        component,
        options=resolved_options,
        component_class=component_class,
        **kwargs,
    )


def load_catalog_diffusers_pipeline(
    record: ModelCatalogRecord | str,
    *,
    index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH,
    model_root: str | os.PathLike[str] | None = None,
    options: Any | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    selected = _resolve_catalog_record(record, index_path=index_path)
    plan = plan_model_integration(selected, model_root=model_root)
    _ensure_diffusers_plan(plan)
    from runtime.diffusers_bridge import DiffusersBridgeOptions, load_diffusers_pipeline

    resolved_options = options or DiffusersBridgeOptions(
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        enable_attention_slicing=bool(kwargs.pop("enable_attention_slicing", False)),
        enable_vae_slicing=bool(kwargs.pop("enable_vae_slicing", True)),
        enable_vae_tiling=bool(kwargs.pop("enable_vae_tiling", False)),
        enable_xformers=bool(kwargs.pop("enable_xformers", False)),
        channels_last=bool(kwargs.pop("channels_last", True)),
        compile_transformer=bool(kwargs.pop("compile_transformer", False)),
        compile_unet=bool(kwargs.pop("compile_unet", False)),
        variant=kwargs.pop("variant", None),
        use_safetensors=kwargs.pop("use_safetensors", None),
        device_map=kwargs.pop("device_map", None),
        max_memory=kwargs.pop("max_memory", None),
        low_cpu_mem_usage=bool(kwargs.pop("low_cpu_mem_usage", True)),
        enable_model_cpu_offload=bool(kwargs.pop("enable_model_cpu_offload", False)),
        enable_sequential_cpu_offload=bool(kwargs.pop("enable_sequential_cpu_offload", False)),
        skip_components=tuple(kwargs.pop("skip_components", ())),
    )
    return load_diffusers_pipeline(plan.local_path, options=resolved_options, **kwargs)


def load_catalog_model(
    record: ModelCatalogRecord | str,
    *,
    index_path: str | os.PathLike[str] = DEFAULT_MODEL_INDEX_PATH,
    model_root: str | os.PathLike[str] | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    base_model: Any | None = None,
    local_files_only: bool = True,
    trust_remote_code: bool = True,
    eval_mode: bool = True,
    **kwargs: Any,
) -> Any:
    selected = _resolve_catalog_record(record, index_path=index_path)
    plan = plan_model_integration(selected, model_root=model_root)
    if not plan.runnable and not (plan.backend == "peft" and base_model is not None):
        raise ModelStackRuntimeUnsupported(plan)

    if plan.backend == "diffusers":
        return load_catalog_diffusers_pipeline(
            selected,
            model_root=model_root,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    from runtime.prep import RuntimeModelArtifacts, prepare_model_for_runtime, resolve_model_dtype

    resolved_dtype = resolve_model_dtype(dtype)

    if plan.backend == "model_stack_native_llama":
        from runtime.checkpoint import build_local_llama_from_snapshot

        model, cfg = build_local_llama_from_snapshot(
            plan.local_path,
            device=str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
            torch_dtype=resolved_dtype,
            device_map=kwargs.pop("device_map", None),
            gpu_ids=kwargs.pop("gpu_ids", None),
        )
        torch_device = next(model.parameters()).device
        torch_dtype = next(model.parameters()).dtype
        return RuntimeModelArtifacts(cfg=cfg, model=model, device=torch_device, dtype=torch_dtype)
    if plan.backend == "transformers":
        model = _load_transformers_model(selected, plan, resolved_dtype, local_files_only, trust_remote_code, **kwargs)
    elif plan.backend == "nemo_or_transformers_asr":
        model = _load_asr_backend(selected, plan, resolved_dtype, local_files_only, trust_remote_code, **kwargs)
    elif plan.backend == "peft":
        if base_model is None:
            raise ModelStackRuntimeUnsupported(plan)
        model = _load_peft_adapter(base_model, plan, **kwargs)
    else:
        raise ModelStackRuntimeUnsupported(plan)

    model, resolved_device, resolved_dtype = prepare_model_for_runtime(
        model,
        device=device,
        dtype=resolved_dtype,
        eval_mode=eval_mode,
    )
    return RuntimeModelArtifacts(cfg=getattr(model, "config", None), model=model, device=resolved_device, dtype=resolved_dtype)


def _resolve_catalog_record(
    record: ModelCatalogRecord | str,
    *,
    index_path: str | os.PathLike[str],
) -> ModelCatalogRecord:
    return find_catalog_record(record, index_path=index_path) if isinstance(record, str) else record


def _ensure_diffusers_plan(plan: ModelIntegrationPlan) -> None:
    if plan.backend != "diffusers":
        raise ModelStackRuntimeUnsupported(plan)


def _ensure_diffusers_or_adapter_plan(plan: ModelIntegrationPlan) -> None:
    if plan.backend not in {"diffusers", "peft"}:
        raise ModelStackRuntimeUnsupported(plan)


def _resolve_local_model_path(
    record: ModelCatalogRecord,
    *,
    model_root: str | os.PathLike[str] | None,
) -> str:
    candidate = Path(record.path)
    if candidate.is_absolute():
        return str(candidate)
    env_root = os.environ.get("MODEL_STACK_MODEL_ROOT")
    if model_root is None and env_root:
        model_root = env_root
    if model_root is not None:
        return str(Path(model_root) / record.relative_path)
    if _has_explicit_hf_cache_root():
        hf_snapshot = _resolve_hf_cache_snapshot(record)
        if hf_snapshot is not None:
            return hf_snapshot
    arxiv_path = _resolve_arxiv_model_path(record)
    if arxiv_path is not None:
        return arxiv_path
    hf_snapshot = _resolve_hf_cache_snapshot(record)
    if hf_snapshot is not None:
        return hf_snapshot
    return record.relative_path


def _resolve_arxiv_model_path(record: ModelCatalogRecord) -> str | None:
    roots = _arxiv_model_roots()
    candidates: list[Path] = []
    for root in roots:
        for relpath in _local_model_relpaths(record):
            path = root / relpath
            if path.exists():
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (_local_model_path_score(path), path.stat().st_mtime), reverse=True)
    return str(candidates[0])


def _local_model_relpaths(record: ModelCatalogRecord) -> tuple[Path, ...]:
    values = [record.id, record.relative_path, record.path]
    relpaths: list[Path] = []
    providers = _record_provider_aliases(record)
    for value in values:
        if not value:
            continue
        normalized = str(value).removeprefix("models--")
        relpaths.append(Path(normalized))
        relpaths.append(Path(normalized.replace("--", "/")))
        if "/" not in normalized and "--" not in normalized:
            for provider in providers:
                relpaths.append(Path(provider) / normalized)
    return tuple(dict.fromkeys(relpaths))


def _local_model_path_score(path: Path) -> int:
    if path.is_file():
        return 1 if path.suffix in {".safetensors", ".pth", ".pt", ".bin"} else 0
    if not path.is_dir():
        return 0
    if (path / "model_index.json").is_file():
        return _snapshot_complete_score(path)
    if (path / "adapter_config.json").is_file() or any(path.glob("*.safetensors")):
        return 2
    if (path / "config.json").is_file():
        return 2
    return 1


def _arxiv_model_roots() -> tuple[Path, ...]:
    explicit = os.environ.get("MODEL_STACK_ARXIV_MODEL_ROOT")
    if explicit:
        return (Path(explicit),)
    return (Path("/arxiv/models"),)



def _resolve_hf_cache_snapshot(record: ModelCatalogRecord) -> str | None:
    slugs = _hf_cache_slugs(record)
    cache_roots = _hf_cache_roots()
    for root in cache_roots:
        candidates: list[Path] = []
        for slug in slugs:
            snapshot_root = root / f"models--{slug}" / "snapshots"
            if snapshot_root.is_dir():
                candidates.extend(path for path in snapshot_root.iterdir() if path.is_dir())
        if candidates:
            candidates.sort(key=lambda path: (_snapshot_complete_score(path), path.stat().st_mtime), reverse=True)
            return str(candidates[0])
    return None


def _snapshot_complete_score(path: Path) -> int:
    index_path = path / "model_index.json"
    if not index_path.is_file():
        return 1 if any(path.glob("*.safetensors")) else 0
    try:
        from runtime.diffusers_bridge import diffusers_snapshot_status

        return 4 if diffusers_snapshot_status(str(path)).complete else 2
    except Exception:
        return _fallback_snapshot_complete_score(path)


def _fallback_snapshot_complete_score(path: Path) -> int:
    index_path = path / "model_index.json"
    try:
        with index_path.open("r", encoding="utf-8") as fh:
            model_index = json.load(fh)
    except Exception:
        return 1
    required = [
        key
        for key, value in model_index.items()
        if not key.startswith("_") and isinstance(value, list) and len(value) >= 2 and value != [None, None]
    ]
    if required and all((path / key).exists() for key in required):
        return 3
    return 2


def _hf_cache_slugs(record: ModelCatalogRecord) -> tuple[str, ...]:
    values = [record.id, record.relative_path, record.path]
    providers = _record_provider_aliases(record)
    slugs: list[str] = []
    for value in values:
        if not value:
            continue
        if value.startswith("models--"):
            value = value[len("models--") :]
        slug = value.replace("/", "--")
        slugs.append(slug)
        if "/" not in value and "--" not in value:
            for provider in providers:
                slugs.append(f"{provider}--{slug}")
    return tuple(dict.fromkeys(slugs))


def _record_provider_aliases(record: ModelCatalogRecord) -> tuple[str, ...]:
    providers = [record.raw.get("provider"), *_HF_PROVIDER_ALIASES_BY_ID.get(record.id, ()), "nvidia"]
    out: list[str] = []
    for provider in providers:
        if not provider:
            continue
        out.append(str(provider).replace("/", "--"))
    return tuple(dict.fromkeys(out))


def _has_explicit_hf_cache_root() -> bool:
    return bool(os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME"))


def _hf_cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        roots.append(Path(explicit))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.extend([Path("/data/huggingface/hub"), Path.home() / ".cache" / "huggingface" / "hub"])
    return tuple(dict.fromkeys(roots))

def _load_transformers_model(
    record: ModelCatalogRecord,
    plan: ModelIntegrationPlan,
    dtype: torch.dtype,
    local_files_only: bool,
    trust_remote_code: bool,
    **kwargs: Any,
) -> torch.nn.Module:
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install 'transformers' to load this model-stack catalog entry") from exc

    common = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    tasks = _task_set(record)
    if record.class_name and record.class_name.endswith(_TRANSFORMERS_CAUSAL_ARCH_SUFFIX):
        return AutoModelForCausalLM.from_pretrained(plan.local_path, **common)
    if record.pipeline_tag == "text-classification" or "text-classification" in tasks:
        return AutoModelForSequenceClassification.from_pretrained(plan.local_path, **common)
    return AutoModel.from_pretrained(plan.local_path, **common)


def _load_asr_backend(
    record: ModelCatalogRecord,
    plan: ModelIntegrationPlan,
    dtype: torch.dtype,
    local_files_only: bool,
    trust_remote_code: bool,
    **kwargs: Any,
) -> torch.nn.Module:
    if record.library_name == "nemo":
        try:
            from nemo.collections.asr.models import ASRModel  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install 'nemo_toolkit[asr]' to load NeMo ASR catalog entries") from exc
        return ASRModel.restore_from(plan.local_path, map_location=kwargs.pop("map_location", None))
    try:
        from transformers import AutoModel, AutoModelForCTC, AutoModelForSpeechSeq2Seq  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install 'transformers' to load ASR catalog entries without NeMo") from exc
    common = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    model_type = (record.model_type or "").lower()
    if "ctc" in model_type or "wav2vec2" in model_type:
        return AutoModelForCTC.from_pretrained(plan.local_path, **common)
    try:
        return AutoModelForSpeechSeq2Seq.from_pretrained(plan.local_path, **common)
    except Exception:
        return AutoModel.from_pretrained(plan.local_path, **common)


def _load_peft_adapter(base_model: Any, plan: ModelIntegrationPlan, **kwargs: Any) -> torch.nn.Module:
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install 'peft' to load PEFT adapter catalog entries") from exc
    return PeftModel.from_pretrained(base_model, plan.local_path, **kwargs)
