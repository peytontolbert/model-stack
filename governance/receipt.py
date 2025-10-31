from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from importlib import metadata as importlib_metadata

from .utils import ensure_directory, now_utc_iso, read_git_info, safe_env_snapshot, system_fingerprint, write_json


def _installed_packages_snapshot(limit: int = 2000) -> list[dict[str, str]]:
    pkgs: list[dict[str, str]] = []
    count = 0
    for dist in importlib_metadata.distributions():
        pkgs.append({"name": dist.metadata.get("Name", "unknown"), "version": dist.version or "0.0.0"})
        count += 1
        if count >= limit:
            break
    return pkgs


def generate_reproducibility_receipt(
    output_path: str | os.PathLike[str],
    artifact_paths: Iterable[str | os.PathLike[str]],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    receipt: Dict[str, Any] = {}
    receipt["generated_at"] = now_utc_iso()
    receipt["artifacts"] = [str(Path(p).resolve()) for p in artifact_paths]
    receipt["system"] = system_fingerprint()
    receipt["env"] = safe_env_snapshot()
    receipt["git"] = read_git_info()
    receipt["packages"] = _installed_packages_snapshot()
    if extra_metadata:
        receipt["metadata"] = extra_metadata

    outp = Path(output_path)
    ensure_directory(outp)
    write_json(outp, receipt)
    return outp


