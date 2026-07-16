from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
from pathlib import Path

import torch


@dataclass(frozen=True)
class DiffusersBridgeOptions:
    device: str | torch.device | None = None
    dtype: str | torch.dtype | None = None
    local_files_only: bool = True
    trust_remote_code: bool = True
    variant: str | None = None
    use_safetensors: bool | None = None
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = False
    enable_xformers: bool = False
    channels_last: bool = True
    compile_transformer: bool = False
    compile_unet: bool = False
    validate_snapshot: bool = True
    device_map: str | dict[str, Any] | None = None
    max_memory: dict[int | str, str | int] | None = None
    low_cpu_mem_usage: bool = True
    enable_model_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    skip_components: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiffusersSnapshotStatus:
    model_path: str
    class_name: str | None
    required_components: tuple[str, ...]
    present_components: tuple[str, ...]
    missing_components: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_components


@dataclass(frozen=True)
class DiffusersComponentSpec:
    name: str
    library: str | None
    class_name: str | None
    present: bool


@dataclass(frozen=True)
class DiffusersComponentArtifacts:
    component: Any
    name: str
    class_name: str
    device: torch.device
    dtype: torch.dtype
    parameter_count: int | None


@dataclass(frozen=True)
class DiffusersAdapterStatus:
    model_path: str
    config_files: tuple[str, ...]
    weight_files: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return bool(self.weight_files)


@dataclass(frozen=True)
class DiffusersAdapterArtifacts:
    pipeline: Any
    adapter_path: str
    weight_name: str | None
    config_files: tuple[str, ...]
    weight_files: tuple[str, ...]


@dataclass(frozen=True)
class DiffusersRuntimeArtifacts:
    pipeline: Any
    device: torch.device
    dtype: torch.dtype
    options: DiffusersBridgeOptions
    enabled_optimizations: tuple[str, ...]

    @property
    def model(self) -> Any:
        return self.pipeline


def resolve_diffusers_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def resolve_diffusers_dtype(dtype: str | torch.dtype | None = None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype or "bfloat16").lower()
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def build_diffusers_load_kwargs(options: DiffusersBridgeOptions, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": resolve_diffusers_dtype(options.dtype),
        "local_files_only": options.local_files_only,
        "trust_remote_code": options.trust_remote_code,
    }
    if options.variant is not None:
        kwargs["variant"] = options.variant
    if options.use_safetensors is not None:
        kwargs["use_safetensors"] = options.use_safetensors
    if options.device_map is not None:
        kwargs["device_map"] = options.device_map
    if options.max_memory is not None:
        kwargs["max_memory"] = options.max_memory
    if options.low_cpu_mem_usage is not None:
        kwargs["low_cpu_mem_usage"] = options.low_cpu_mem_usage
    for component in options.skip_components:
        kwargs[str(component)] = None
    kwargs.update(overrides)
    return kwargs


def load_diffusers_pipeline(
    model_path: str,
    *,
    options: DiffusersBridgeOptions | None = None,
    pipeline_class: Any | None = None,
    **kwargs: Any,
) -> DiffusersRuntimeArtifacts:
    selected = options or DiffusersBridgeOptions()
    if selected.validate_snapshot:
        status = diffusers_snapshot_status(model_path)
        if not status.complete:
            missing = ", ".join(status.missing_components)
            raise FileNotFoundError(f"Diffusers snapshot is incomplete for {model_path}: missing {missing}")

    try:
        from diffusers import DiffusionPipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install 'diffusers' to load diffusers_cuda_bridge entries") from exc

    cls = pipeline_class or DiffusionPipeline
    pipeline = cls.from_pretrained(model_path, **build_diffusers_load_kwargs(selected, **kwargs))
    return prepare_diffusers_pipeline(pipeline, options=selected)


def prepare_diffusers_pipeline(
    pipeline: Any,
    *,
    options: DiffusersBridgeOptions | None = None,
) -> DiffusersRuntimeArtifacts:
    selected = options or DiffusersBridgeOptions()
    device = resolve_diffusers_device(selected.device)
    dtype = resolve_diffusers_dtype(selected.dtype)
    enabled: list[str] = []

    uses_managed_placement = bool(
        selected.device_map is not None or selected.enable_model_cpu_offload or selected.enable_sequential_cpu_offload
    )
    if not uses_managed_placement and hasattr(pipeline, "to"):
        pipeline = pipeline.to(device=device, dtype=dtype)
        enabled.append(f"to:{device}:{str(dtype).replace('torch.', '')}")
    elif selected.device_map is not None:
        enabled.append(f"device_map:{selected.device_map}")

    if selected.enable_model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
        try:
            pipeline.enable_model_cpu_offload(device=device)
        except TypeError:
            pipeline.enable_model_cpu_offload()
        enabled.append(f"model_cpu_offload:{device}")
    if selected.enable_sequential_cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
        try:
            pipeline.enable_sequential_cpu_offload(device=device)
        except TypeError:
            pipeline.enable_sequential_cpu_offload()
        enabled.append(f"sequential_cpu_offload:{device}")

    if selected.enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()
        enabled.append("attention_slicing")
    if selected.enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        enabled.append("vae_slicing")
    if selected.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
        enabled.append("vae_tiling")
    if selected.enable_xformers and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        pipeline.enable_xformers_memory_efficient_attention()
        enabled.append("xformers_memory_efficient_attention")

    if selected.channels_last and device.type == "cuda" and not uses_managed_placement:
        for name in _diffusers_component_names(pipeline):
            module = getattr(pipeline, name, None)
            if hasattr(module, "to"):
                try:
                    module.to(memory_format=torch.channels_last)
                    enabled.append(f"channels_last:{name}")
                except Exception:
                    pass

    _apply_diffusers_compatibility_patches(pipeline, enabled)

    if selected.compile_transformer:
        _compile_component(pipeline, "transformer", enabled)
    if selected.compile_unet:
        _compile_component(pipeline, "unet", enabled)

    for name in _diffusers_component_names(pipeline):
        module = getattr(pipeline, name, None)
        if hasattr(module, "eval"):
            module.eval()

    return DiffusersRuntimeArtifacts(
        pipeline=pipeline,
        device=device,
        dtype=dtype,
        options=selected,
        enabled_optimizations=tuple(enabled),
    )


def _diffusers_component_names(pipeline: Any) -> tuple[str, ...]:
    config = getattr(pipeline, "config", None)
    names: list[str] = []
    if isinstance(config, dict):
        names.extend(str(key) for key in config if not str(key).startswith("_"))
    components = getattr(pipeline, "components", None)
    if isinstance(components, dict):
        names.extend(str(key) for key in components)
    names.extend(["transformer", "unet", "vae", "text_encoder", "text_encoder_2", "image_encoder"])
    return tuple(dict.fromkeys(names))


def _apply_diffusers_compatibility_patches(pipeline: Any, enabled: list[str]) -> None:
    _patch_anyflow_far_return_tuple(pipeline, enabled)


def _patch_anyflow_far_return_tuple(pipeline: Any, enabled: list[str]) -> None:
    if type(pipeline).__name__ != "AnyFlowFARPipeline":
        return
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or getattr(transformer, "_model_stack_anyflow_far_tuple_patch", False):
        return
    forward = getattr(transformer, "forward", None)
    if forward is None:
        return

    def forward_with_tuple_padding(*args: Any, **kwargs: Any) -> Any:
        result = forward(*args, **kwargs)
        if kwargs.get("return_dict") is False and isinstance(result, tuple) and len(result) == 1:
            return result[0], None
        return result

    transformer.forward = forward_with_tuple_padding
    transformer._model_stack_anyflow_far_tuple_patch = True
    enabled.append("compat:anyflow_far_return_tuple_padding")


def _compile_component(pipeline: Any, name: str, enabled: list[str]) -> None:
    module = getattr(pipeline, name, None)
    if module is None or not hasattr(torch, "compile"):
        return
    try:
        setattr(pipeline, name, torch.compile(module, mode="reduce-overhead", fullgraph=False))
        enabled.append(f"torch_compile:{name}")
    except Exception:
        pass


def diffusers_snapshot_status(model_path: str) -> DiffusersSnapshotStatus:
    root = Path(model_path)
    index_path = root / "model_index.json"
    if not index_path.is_file():
        return DiffusersSnapshotStatus(
            model_path=str(root),
            class_name=None,
            required_components=(),
            present_components=(),
            missing_components=("model_index.json",),
        )
    with index_path.open("r", encoding="utf-8") as fh:
        model_index = json.load(fh)
    required = tuple(
        key
        for key, value in model_index.items()
        if _is_diffusers_component_entry(key, value)
    )
    present = tuple(key for key in required if _diffusers_component_present(root, key))
    present_set = set(present)
    missing_components = [key for key in required if key not in present_set]
    specs_by_name = {spec.name: spec for spec in diffusers_component_specs(str(root))}
    for key in present:
        missing_components.extend(
            f"{key}/{filename}" for filename in _missing_component_weight_files(root / key, specs_by_name.get(key))
        )
    missing = tuple(missing_components)
    return DiffusersSnapshotStatus(
        model_path=str(root),
        class_name=model_index.get("_class_name"),
        required_components=required,
        present_components=present,
        missing_components=missing,
    )


def _is_diffusers_component_entry(key: str, value: Any) -> bool:
    return not key.startswith("_") and isinstance(value, list) and len(value) >= 2 and value != [None, None]


def _diffusers_component_present(root: Path, component: str) -> bool:
    path = root / component
    if path.is_dir():
        return True
    if path.is_file():
        return True
    return False


def _missing_component_weight_files(component_path: Path, spec: DiffusersComponentSpec | None = None) -> tuple[str, ...]:
    if not component_path.is_dir():
        return ()
    missing: list[str] = []
    index_names = ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
    for index_name in index_names:
        index_path = component_path / index_name
        if not index_path.is_file():
            continue
        try:
            with index_path.open("r", encoding="utf-8") as fh:
                index = json.load(fh)
        except Exception:
            continue
        weight_map = index.get("weight_map", {})
        filenames = sorted({str(filename) for filename in weight_map.values()})
        missing.extend(filename for filename in filenames if not (component_path / filename).is_file())
    if missing or any((component_path / name).is_file() for name in index_names):
        return tuple(dict.fromkeys(missing))
    expected = _expected_component_weight_names(spec)
    has_component_config = (component_path / "config.json").is_file()
    if has_component_config and expected and not any((component_path / name).is_file() for name in expected):
        missing.append(expected[0])
    return tuple(dict.fromkeys(missing))


def _expected_component_weight_names(spec: DiffusersComponentSpec | None) -> tuple[str, ...]:
    if spec is None or not spec.class_name:
        return ()
    class_name = spec.class_name.lower()
    if spec.library == "diffusers" and any(token in class_name for token in ("autoencoder", "transformer", "unet", "model")):
        return ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    if spec.library == "transformers" and any(token in class_name for token in ("model", "encoder", "decoder", "for")):
        return ("model.safetensors", "pytorch_model.bin", "model.bin")
    return ()


def diffusers_component_specs(model_path: str) -> tuple[DiffusersComponentSpec, ...]:
    root = Path(model_path)
    index_path = root / "model_index.json"
    if not index_path.is_file():
        return ()
    with index_path.open("r", encoding="utf-8") as fh:
        model_index = json.load(fh)
    specs: list[DiffusersComponentSpec] = []
    for key, value in model_index.items():
        if not _is_diffusers_component_entry(str(key), value):
            continue
        library = None
        class_name = None
        if isinstance(value, list) and len(value) >= 2:
            library = value[0]
            class_name = value[1]
        specs.append(
            DiffusersComponentSpec(
                name=str(key),
                library=None if library is None else str(library),
                class_name=None if class_name is None else str(class_name),
                present=_diffusers_component_present(root, str(key)),
            )
        )
    return tuple(specs)


def load_diffusers_component(
    model_path: str,
    component: str,
    *,
    options: DiffusersBridgeOptions | None = None,
    component_class: Any | None = None,
    **kwargs: Any,
) -> DiffusersComponentArtifacts:
    selected = options or DiffusersBridgeOptions()
    spec = _find_component_spec(model_path, component)
    if spec is None:
        raise KeyError(f"Diffusers component {component!r} is not declared in {model_path}")
    if not spec.present:
        raise FileNotFoundError(f"Diffusers component {component!r} is missing from {model_path}")
    cls = component_class or _resolve_component_class(spec)
    if cls is None:
        raise RuntimeError(f"Could not resolve loader class for Diffusers component {component!r}: {spec}")

    dtype = resolve_diffusers_dtype(selected.dtype)
    device = resolve_diffusers_device(selected.device)
    load_kwargs = build_diffusers_load_kwargs(selected, **kwargs)
    obj = cls.from_pretrained(model_path, subfolder=component, **load_kwargs)
    uses_managed_placement = selected.device_map is not None
    if not uses_managed_placement and hasattr(obj, "to"):
        obj = obj.to(device=device, dtype=dtype)
    if hasattr(obj, "eval"):
        obj.eval()
    return DiffusersComponentArtifacts(
        component=obj,
        name=component,
        class_name=type(obj).__name__,
        device=device,
        dtype=dtype,
        parameter_count=_parameter_count(obj),
    )


def _find_component_spec(model_path: str, component: str) -> DiffusersComponentSpec | None:
    for spec in diffusers_component_specs(model_path):
        if spec.name == component:
            return spec
    return None


def _resolve_component_class(spec: DiffusersComponentSpec) -> Any | None:
    if not spec.class_name:
        return None
    if spec.library == "diffusers":
        try:
            import diffusers  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install 'diffusers' to load Diffusers components") from exc
        return getattr(diffusers, spec.class_name, None)
    if spec.library == "transformers":
        try:
            import transformers  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install 'transformers' to load Transformers components") from exc
        return getattr(transformers, spec.class_name, None)
    return None


def _parameter_count(obj: Any) -> int | None:
    if not hasattr(obj, "parameters"):
        return None
    try:
        return int(sum(param.numel() for param in obj.parameters()))
    except Exception:
        return None


def diffusers_adapter_status(model_path: str) -> DiffusersAdapterStatus:
    root = Path(model_path)
    if not root.is_dir():
        return DiffusersAdapterStatus(str(root), (), ())
    config_files = tuple(sorted(path.name for path in root.glob("*.json")))
    root_weights = sorted(path.name for path in root.glob("*.safetensors"))
    has_adapter_config = "adapter_config.json" in config_files
    weight_files = tuple(
        name
        for name in root_weights
        if has_adapter_config or "lora" in name.lower() or "adapter" in name.lower()
    )
    return DiffusersAdapterStatus(str(root), config_files, weight_files)


def load_diffusers_lora_adapter(
    pipeline: Any,
    adapter_path: str,
    *,
    weight_name: str | None = None,
    adapter_name: str | None = None,
    **kwargs: Any,
) -> DiffusersAdapterArtifacts:
    status = diffusers_adapter_status(adapter_path)
    if not status.complete:
        raise FileNotFoundError(f"Diffusers adapter snapshot is incomplete for {adapter_path}: no .safetensors weights found")
    selected_weight = weight_name or (status.weight_files[0] if len(status.weight_files) == 1 else None)
    if not hasattr(pipeline, "load_lora_weights"):
        raise TypeError(f"Pipeline {type(pipeline).__name__} does not expose load_lora_weights")
    load_kwargs = dict(kwargs)
    if selected_weight is not None:
        load_kwargs["weight_name"] = selected_weight
    if adapter_name is not None:
        load_kwargs["adapter_name"] = adapter_name
    pipeline.load_lora_weights(adapter_path, **load_kwargs)
    return DiffusersAdapterArtifacts(
        pipeline=pipeline,
        adapter_path=adapter_path,
        weight_name=selected_weight,
        config_files=status.config_files,
        weight_files=status.weight_files,
    )


def diffusers_component_report(pipeline: Any) -> dict[str, str]:
    report: dict[str, str] = {}
    for name in _diffusers_component_names(pipeline):
        component = getattr(pipeline, name, None)
        if component is not None:
            report[name] = type(component).__name__
    return report
