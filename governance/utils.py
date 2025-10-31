from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(args: list[str], cwd: Optional[str] = None, timeout: int = 15) -> Optional[str]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def read_git_info(cwd: Optional[str] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=cwd)
    if not commit:
        return info
    info["commit"] = commit
    info["branch"] = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd) or ""
    info["dirty"] = bool(run_cmd(["git", "status", "--porcelain"], cwd=cwd))
    info["remote_url"] = run_cmd(["git", "config", "--get", "remote.origin.url"], cwd=cwd) or ""
    return info


def write_json(path: str | os.PathLike[str], data: Any) -> Path:
    p = Path(path)
    ensure_directory(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def safe_env_snapshot(allow_prefixes: Iterable[str] = ("CI", "CUDA", "CUDNN", "PYTORCH", "HF_", "TRANSFORMERS_", "NCCL", "OMPI")) -> Dict[str, str]:
    snapshot: Dict[str, str] = {}
    for key, val in os.environ.items():
        if any(key.startswith(prefix) for prefix in allow_prefixes):
            snapshot[key] = val
    return snapshot


def system_fingerprint() -> Dict[str, Any]:
    return {
        "python": sys.version.split("\n")[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def try_import(module_name: str) -> Optional[Any]:
    try:
        return __import__(module_name)
    except Exception:
        return None


