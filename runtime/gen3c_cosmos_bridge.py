from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GEN3CCosmosStatus:
    model_id: str
    model_path: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str | None
    detail: str
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    config: dict[str, Any] | None = None
    checkpoint_format: str | None = None
    checkpoint_size_bytes: int | None = None
    checkpoint_entries: int | None = None
    checkpoint_prefix: str | None = None
    checkpoint_storage_entries: int | None = None


def _read_config(path: Path) -> dict[str, Any] | None:
    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _inspect_pytorch_zip(path: Path) -> tuple[str | None, int | None, int | None]:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            prefix = None
            if entries:
                first = entries[0].filename
                prefix = first.split("/", 1)[0] if "/" in first else None
            storage_entries = sum(1 for item in entries if "/data/" in item.filename)
            return prefix, len(entries), storage_entries
    except zipfile.BadZipFile:
        return None, None, None


def gen3c_cosmos_status(model_path: str | Path, model_id: str | None = None) -> GEN3CCosmosStatus:
    path = Path(model_path)
    resolved_id = model_id or path.name
    present: list[str] = []
    missing: list[str] = []

    config = _read_config(path)
    if config is None:
        missing.append("config.json")
    else:
        present.append("config.json")

    readme = path / "README.md"
    if readme.is_file():
        present.append("README.md")
    else:
        missing.append("README.md")

    checkpoint = path / "model.pt"
    checkpoint_format = None
    checkpoint_size = None
    checkpoint_prefix = None
    checkpoint_entries = None
    checkpoint_storage_entries = None
    if checkpoint.is_file():
        present.append("model.pt")
        checkpoint_size = checkpoint.stat().st_size
        with checkpoint.open("rb") as handle:
            magic = handle.read(4)
        if magic == b"PK\x03\x04":
            checkpoint_format = "pytorch_zip"
            checkpoint_prefix, checkpoint_entries, checkpoint_storage_entries = _inspect_pytorch_zip(checkpoint)
        else:
            checkpoint_format = "unknown_pt_pickle_or_binary"
    else:
        missing.append("model.pt")

    expected_input = bool(config and "Cosmos_GEN3C" in config.get("input_types", []))
    if not expected_input:
        missing.append("config.input_types:Cosmos_GEN3C")

    runnable = False
    status = "needs_gen3c_cosmos_predict1_runtime"
    blockers = [
        "GEN3C requires the nv-tlabs/Gen3C runtime on top of Cosmos-Predict1; no local runtime checkout is registered in model-stack yet.",
        "Do not torch.load model.pt during status probing; it is a pickle-backed PyTorch archive and should only be loaded inside the trusted GEN3C runtime bridge.",
        "Full camera-pose + seed-image video generation has not been run from model-stack.",
    ]
    if missing:
        status = "incomplete_gen3c_cosmos_snapshot"
        blockers.insert(0, "missing required local artifacts: " + ", ".join(missing))

    return GEN3CCosmosStatus(
        model_id=resolved_id,
        model_path=str(path),
        status=status,
        runnable=runnable,
        preferred_env="gen3c_cosmos_predict1_or_custom_bridge",
        loader="runtime.gen3c_cosmos_bridge",
        recommended_dtype="bfloat16",
        detail=(
            "GEN3C-Cosmos is a camera-controlled video generator based on Cosmos Predict 1. "
            "The local snapshot has a repo-format config plus a single model.pt checkpoint, not a Diffusers model_index.json."
        ),
        present_artifacts=tuple(present),
        missing_artifacts=tuple(missing),
        blockers=tuple(blockers),
        config=config,
        checkpoint_format=checkpoint_format,
        checkpoint_size_bytes=checkpoint_size,
        checkpoint_entries=checkpoint_entries,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_storage_entries=checkpoint_storage_entries,
    )


def status_to_json(status: GEN3CCosmosStatus) -> dict[str, Any]:
    return asdict(status)
