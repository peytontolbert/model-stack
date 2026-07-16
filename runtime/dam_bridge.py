from __future__ import annotations

import gc
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_SOURCE = Path("/arxiv/models/DAM-3B-Self-Contained")
_COMPONENTS = ("vision_tower", "mm_projector", "context_provider", "llm")


@dataclass(frozen=True)
class DAMStatus:
    model_id: str
    model_path: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str
    detail: str
    blockers: tuple[str, ...] = ()
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    self_contained: bool = False
    runtime_source: str = str(DEFAULT_RUNTIME_SOURCE)


@dataclass(frozen=True)
class DAMComponentResult:
    component: str
    status: str
    elapsed_sec: float
    class_name: str | None = None
    param_count: int | None = None
    missing_keys: int | None = None
    unexpected_keys: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class DAMProbe:
    status: DAMStatus
    imports: dict[str, bool]
    errors: dict[str, str]
    components: tuple[DAMComponentResult, ...] = ()


def dam_status(model_path: str | Path, *, model_id: str | None = None, runtime_source: str | Path = DEFAULT_RUNTIME_SOURCE) -> DAMStatus:
    path = Path(model_path)
    runtime = Path(runtime_source)
    resolved_id = model_id or path.name
    self_contained = (path / "llava_llama.py").is_file()
    expected = {
        "config.json": path / "config.json",
        "vision_tower/config.json": path / "vision_tower" / "config.json",
        "vision_tower/model.safetensors": path / "vision_tower" / "model.safetensors",
        "mm_projector/config.json": path / "mm_projector" / "config.json",
        "mm_projector/model.safetensors": path / "mm_projector" / "model.safetensors",
        "context_provider/config.json": path / "context_provider" / "config.json",
        "context_provider/model.safetensors": path / "context_provider" / "model.safetensors",
        "llm/config.json": path / "llm" / "config.json",
        "llm/model index": path / "llm" / "model.safetensors.index.json",
        "llm/tokenizer.model": path / "llm" / "tokenizer.model",
        "DAM runtime llava_llama.py": runtime / "llava_llama.py",
    }
    present = tuple(name for name, item in expected.items() if item.exists())
    missing = tuple(name for name, item in expected.items() if not item.exists())
    llm_shards = tuple(sorted(str(item) for item in (path / "llm").glob("model-*.safetensors")))
    blockers = []
    if missing:
        blockers.append("missing required DAM artifacts: " + ", ".join(missing))
    if not llm_shards:
        blockers.append(f"missing LLM safetensors shards under {path / 'llm'}")
    blockers.extend(
        [
            "top-level LlavaLlamaModel constructor eagerly loads the full LLM; use the DAM bridge to lazy/place submodules instead of generic AutoModel construction",
            "mm_projector and context_provider need parent LlavaLlamaConfig during construction, so plain AutoModel.from_pretrained fails for those components",
        ]
    )
    status = "candidate_dam_lazy_submodule_bridge" if not missing and llm_shards else "incomplete_dam_checkpoint"
    detail = (
        "DAM assets are local and the vendored llava_llama.py runtime is available. Component-level bridge validation can load "
        "vision_tower with AutoModel and load mm_projector/context_provider by manually passing the parent LlavaLlamaConfig. "
        "Full caption generation still needs lazy LLM assembly/placement."
    )
    return DAMStatus(
        model_id=resolved_id,
        model_path=str(path),
        status=status,
        runnable=False,
        preferred_env="ai",
        loader="runtime.dam_bridge",
        recommended_dtype="bfloat16",
        detail=detail,
        blockers=tuple(blockers),
        present_artifacts=present,
        missing_artifacts=missing,
        self_contained=self_contained,
        runtime_source=str(runtime),
    )


def probe_dam_components(
    model_path: str | Path,
    *,
    model_id: str | None = None,
    runtime_source: str | Path = DEFAULT_RUNTIME_SOURCE,
    load_components: bool = False,
) -> DAMProbe:
    status = dam_status(model_path, model_id=model_id, runtime_source=runtime_source)
    runtime = Path(runtime_source)
    _register_runtime(runtime)
    imports: dict[str, bool] = {}
    errors: dict[str, str] = {}
    try:
        importlib.import_module("llava_llama")
        imports["llava_llama"] = True
    except Exception as exc:
        imports["llava_llama"] = False
        errors["llava_llama"] = f"{type(exc).__name__}:{exc}"
    components: tuple[DAMComponentResult, ...] = ()
    if load_components and imports.get("llava_llama"):
        components = tuple(_load_component(Path(model_path), name) for name in ("vision_tower", "mm_projector", "context_provider", "llm_tokenizer"))
    return DAMProbe(status=status, imports=imports, errors=errors, components=components)


def _load_component(model_path: Path, component: str) -> DAMComponentResult:
    started = time.monotonic()
    try:
        import torch
        import llava_llama
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        parent = llava_llama.LlavaLlamaConfig.from_pretrained(model_path)
        if component == "vision_tower":
            model = AutoModel.from_pretrained(model_path / component, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            result = DAMComponentResult(component, "loaded", round(time.monotonic() - started, 3), type(model).__name__, sum(p.numel() for p in model.parameters()))
            del model
        elif component == "mm_projector":
            cfg = AutoConfig.from_pretrained(model_path / component, trust_remote_code=True)
            model = llava_llama.MultimodalProjector(cfg, parent).to(torch.bfloat16)
            state = load_file(str(model_path / component / "model.safetensors"), device="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            result = DAMComponentResult(component, "loaded_manual_parent_config", round(time.monotonic() - started, 3), type(model).__name__, sum(p.numel() for p in model.parameters()), len(missing), len(unexpected))
            del model, state
        elif component == "context_provider":
            cfg = AutoConfig.from_pretrained(model_path / component, trust_remote_code=True)
            model = llava_llama.ContextProvider(cfg, parent).to(torch.bfloat16)
            state = load_file(str(model_path / component / "model.safetensors"), device="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            result = DAMComponentResult(component, "loaded_manual_parent_config", round(time.monotonic() - started, 3), type(model).__name__, sum(p.numel() for p in model.parameters()), len(missing), len(unexpected))
            del model, state
        elif component == "llm_tokenizer":
            tokenizer = AutoTokenizer.from_pretrained(model_path / "llm", use_fast=False, local_files_only=True)
            result = DAMComponentResult(component, "loaded", round(time.monotonic() - started, 3), type(tokenizer).__name__, len(tokenizer))
            del tokenizer
        else:
            raise ValueError(f"unknown DAM component {component}")
        gc.collect()
        return result
    except Exception as exc:
        return DAMComponentResult(component, "failed", round(time.monotonic() - started, 3), error=f"{type(exc).__name__}:{exc}")


def _register_runtime(runtime_source: Path) -> None:
    value = str(runtime_source.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def status_to_json(status: DAMStatus) -> dict[str, Any]:
    return asdict(status)


def probe_to_json(probe: DAMProbe) -> dict[str, Any]:
    return asdict(probe)
