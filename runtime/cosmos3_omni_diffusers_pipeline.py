from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_UPSTREAM_SNAPSHOT_ROOT = Path("/data/tmp/model-stack-artifacts/cosmos3")
UPSTREAM_CLASS_NAME = "Cosmos3OmniPipeline"
LEGACY_CLASS_NAME = "Cosmos3OmniDiffusersPipeline"
IGNORED_UPSTREAM_COMPONENTS = {"vision_encoder"}


def patch_diffusers_cosmos3() -> type[Any]:
    """Register the local Cosmos3 snapshot's legacy class name as the real upstream pipeline.

    Local Cosmos3-Nano snapshots advertise ``Cosmos3OmniDiffusersPipeline`` in
    ``model_index.json``. The installed upstream Diffusers implementation in
    ``ai`` exposes the same pipeline as ``Cosmos3OmniPipeline``. This patch is
    intentionally only an alias; generation stays inside Diffusers.
    """
    import diffusers
    from diffusers.pipelines import cosmos as cosmos_pipelines
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline

    diffusers.Cosmos3OmniDiffusersPipeline = Cosmos3OmniPipeline
    cosmos_pipelines.Cosmos3OmniDiffusersPipeline = Cosmos3OmniPipeline
    return Cosmos3OmniPipeline


def prepare_cosmos3_upstream_diffusers_snapshot(
    source: str | Path,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = True,
) -> Path:
    """Create a symlink snapshot whose metadata matches upstream Diffusers.

    The checkpoint data remains in the original directory. Only ``model_index.json``
    is rewritten in the output directory so ``DiffusionPipeline.from_pretrained``
    resolves ``Cosmos3OmniPipeline`` and skips metadata-only components that the
    upstream constructor does not accept.
    """
    source_path = _resolve_local_model_path(source)
    model_index_path = source_path / "model_index.json"
    if not model_index_path.is_file():
        raise FileNotFoundError(f"Cosmos3 model_index.json not found: {model_index_path}")

    target = Path(output_dir) if output_dir is not None else _default_upstream_snapshot_dir(source_path)
    target.mkdir(parents=True, exist_ok=True)

    for entry in source_path.iterdir():
        if entry.name == "model_index.json":
            continue
        link = target / entry.name
        if link.exists() or link.is_symlink():
            if not overwrite:
                continue
            if link.is_dir() and not link.is_symlink():
                raise IsADirectoryError(f"Refusing to replace real directory in adapter snapshot: {link}")
            link.unlink()
        os.symlink(entry, link, target_is_directory=entry.is_dir())

    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    original_class_name = str(model_index.get("_class_name") or LEGACY_CLASS_NAME)
    model_index["_class_name"] = UPSTREAM_CLASS_NAME
    model_index["_model_stack_original_class_name"] = original_class_name
    removed = sorted(name for name in IGNORED_UPSTREAM_COMPONENTS if name in model_index)
    for name in removed:
        model_index.pop(name, None)
    if removed:
        model_index["_model_stack_removed_components"] = removed
    (target / "model_index.json").write_text(json.dumps(model_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_cosmos3_upstream_diffusers_pipeline(
    pretrained_model_name_or_path: str | Path,
    *,
    adapter_dir: str | Path | None = None,
    local_files_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Load Cosmos3 through the real upstream Diffusers pipeline.

    This helper applies the class-name alias and prepares a symlink snapshot before
    delegating to ``DiffusionPipeline.from_pretrained``. Pass normal Diffusers
    loading kwargs such as ``torch_dtype``, ``device_map``, or
    ``low_cpu_mem_usage`` through ``kwargs``.
    """
    patch_diffusers_cosmos3()
    snapshot = prepare_cosmos3_upstream_diffusers_snapshot(pretrained_model_name_or_path, adapter_dir)

    from diffusers import DiffusionPipeline

    kwargs.setdefault("local_files_only", local_files_only)
    return DiffusionPipeline.from_pretrained(snapshot, **kwargs)


def _default_upstream_snapshot_dir(source_path: Path) -> Path:
    return DEFAULT_UPSTREAM_SNAPSHOT_ROOT / f"{source_path.name}-upstream-diffusers"


def _resolve_local_model_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    model_id = str(value).strip("/")
    candidates = [
        Path("/arxiv/models") / model_id,
        Path("/arxiv/models") / model_id.replace("/", "--"),
        Path("/arxiv/models/nvidia") / model_id.removeprefix("nvidia/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve local Cosmos3 model path for {value!r}")
