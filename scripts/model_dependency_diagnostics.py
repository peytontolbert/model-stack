#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import importlib.metadata as metadata
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runtime.model_catalog import DEFAULT_MODEL_INDEX_PATH, find_catalog_record, plan_model_integration
from runtime.diffusers_bridge import diffusers_adapter_status, diffusers_snapshot_status

BASE_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "huggingface_hub",
    "safetensors",
    "nemo_toolkit",
    "pytorch_lightning",
    "hydra_core",
    "omegaconf",
    "soundfile",
    "librosa",
    "sentencepiece",
    "protobuf",
    "einops",
    "flash_attn",
    "xformers",
    "decord",
)

IMPORT_TO_PACKAGE = {
    "PIL": "pillow",
    "cv2": "opencv_python",
    "diffusers": "diffusers",
    "einops": "einops",
    "librosa": "librosa",
    "nemo": "nemo_toolkit",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "peft": "peft",
    "safetensors": "safetensors",
    "sentencepiece": "sentencepiece",
    "soundfile": "soundfile",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "xformers": "xformers",
}

KNOWN_BRIDGE_MINIMUMS = {
    "diffusers": {"torch": "2.0", "diffusers": "0.32", "transformers": "4.40", "accelerate": "0.25", "safetensors": "0.4"},
    "transformers": {"torch": "2.0", "transformers": "4.40", "safetensors": "0.4"},
    "peft": {"torch": "2.0", "transformers": "4.40", "peft": "0.10", "safetensors": "0.4"},
    "nemo_or_transformers_asr": {"python": "3.12", "torch": "2.7", "nemo_toolkit": "2.0"},
}


@dataclass(frozen=True)
class DependencyCheck:
    package: str
    installed: str
    required: str
    source: str
    status: str
    detail: str


def _env_name(explicit: str | None) -> str:
    return explicit or os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name or "unknown"


def _installed_versions(packages: set[str]) -> dict[str, str]:
    installed = {dist.metadata["Name"].lower().replace("-", "_"): dist.version for dist in metadata.distributions()}
    versions = {package: installed.get(package, "missing") for package in sorted(packages)}
    versions["python"] = sys.version.split()[0]
    try:
        import torch

        versions["torch_import"] = torch.__version__
        versions["torch_cuda"] = str(torch.cuda.is_available())
        versions["torch_cuda_version"] = str(getattr(torch.version, "cuda", None))
    except Exception as exc:  # pragma: no cover - diagnostic path
        versions["torch_import"] = f"failed:{type(exc).__name__}:{exc}"
        versions["torch_cuda"] = "False"
        versions["torch_cuda_version"] = "None"
    return versions


def _parse_version(value: str) -> tuple[int, ...]:
    if not value or value == "missing":
        return ()
    match = re.match(r"(\d+(?:\.\d+)*)", value)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _version_satisfies(installed: str, required: str) -> bool | None:
    if required in {"present", "review"}:
        return installed != "missing" if required == "present" else None
    if installed == "missing":
        return False
    left = _parse_version(installed)
    right = _parse_version(required)
    if not left or not right:
        return None
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _model_declared_requirements(model_path: Path) -> dict[str, tuple[str, str]]:
    requirements: dict[str, tuple[str, str]] = {}
    model_index = _read_json(model_path / "model_index.json")
    diffusers_version = model_index.get("_diffusers_version")
    if diffusers_version:
        requirements["diffusers"] = (str(diffusers_version), "model_index._diffusers_version")
    config_paths = [model_path / "config.json", *(path for path in model_path.glob("*/config.json"))]
    for config_path in config_paths:
        config = _read_json(config_path)
        transformers_version = config.get("transformers_version")
        if transformers_version and "transformers" not in requirements:
            requirements["transformers"] = (str(transformers_version), f"{config_path.relative_to(model_path)}:transformers_version")
    return requirements


def _enclosing_scope(tree: ast.AST, target: ast.AST) -> str:
    best: tuple[int, str] | None = None
    target_line = getattr(target, "lineno", -1)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = getattr(node, "lineno", -1)
        end = getattr(node, "end_lineno", start)
        if start <= target_line <= end:
            span = end - start
            name = f"{type(node).__name__}:{node.name}"
            if best is None or span < best[0]:
                best = (span, name)
    return best[1] if best else "module"


def _scan_python_import_usages(model_path: Path, max_files: int = 64) -> list[dict[str, Any]]:
    usages: list[dict[str, Any]] = []
    if not model_path.is_dir():
        return usages
    for py_file in sorted(model_path.glob("**/*.py"))[:max_files]:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        relpath = str(py_file.relative_to(model_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".", 1)[0]
                    package = IMPORT_TO_PACKAGE.get(module)
                    if package:
                        usages.append(
                            {
                                "package": package,
                                "module": alias.name,
                                "file": relpath,
                                "line": getattr(node, "lineno", None),
                                "scope": _enclosing_scope(tree, node),
                            }
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split(".", 1)[0]
                package = IMPORT_TO_PACKAGE.get(module)
                if package:
                    usages.append(
                        {
                            "package": package,
                            "module": node.module,
                            "file": relpath,
                            "line": getattr(node, "lineno", None),
                            "scope": _enclosing_scope(tree, node),
                        }
                    )
    return usages


def _scan_python_imports(model_path: Path, max_files: int = 64) -> dict[str, str]:
    imports: dict[str, str] = {}
    for usage in _scan_python_import_usages(model_path, max_files=max_files):
        imports.setdefault(str(usage["package"]), f"{usage['file']}:{usage['line']}:{usage['scope']}")
    return imports


def _readme_dependency_hints(model_path: Path) -> tuple[str, ...]:
    hints: list[str] = []
    for readme in [model_path / "README.md", model_path / "readme.md"]:
        if not readme.is_file():
            continue
        text = readme.read_text(encoding="utf-8", errors="ignore")
        for pattern in (r"pip install[^`\n]+", r"torch ?[>=]=? ?[0-9][^,\n )]*", r"diffusers ?[>=]=? ?[0-9][^,\n )]*", r"transformers ?[>=]=? ?[0-9][^,\n )]*"):
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                hints.append(" ".join(match.split()))
    return tuple(dict.fromkeys(hints[:20]))


def _local_layout_status(backend: str, model_path: Path) -> tuple[str, str]:
    if not model_path.exists():
        return "missing_local_path", "path does not exist"
    if backend == "diffusers":
        snapshot = diffusers_snapshot_status(str(model_path))
        adapter = diffusers_adapter_status(str(model_path))
        if snapshot.complete:
            return "diffusers_snapshot_ok", f"class={snapshot.class_name}; components={','.join(snapshot.present_components)}"
        if adapter.complete:
            return "diffusers_adapter_ok", f"weights={','.join(adapter.weight_files)}"
        if snapshot.missing_components == ("model_index.json",):
            return "custom_layout", "no model_index.json"
        return "incomplete_diffusers_snapshot", "missing=" + ",".join(snapshot.missing_components[:8])
    if backend == "nemo_or_transformers_asr":
        archives = sorted(model_path.glob("*.nemo")) if model_path.is_dir() else []
        return ("nemo_archive_present", f"archive={archives[0].name}") if archives else ("nemo_archive_missing", "no .nemo archive")
    if backend == "transformers":
        return ("transformers_config_present", "config.json present") if (model_path / "config.json").is_file() else ("custom_layout", "no config.json")
    if backend == "peft":
        return ("adapter_config_present", "adapter_config.json present") if (model_path / "adapter_config.json").is_file() else ("adapter_needs_review", "adapter config not found")
    return "manual", "manual backend"


def _collect_requirements(backend: str, model_path: Path) -> dict[str, tuple[str, str]]:
    requirements = {package: (version, "bridge_minimum") for package, version in KNOWN_BRIDGE_MINIMUMS.get(backend, {}).items()}
    requirements.update(_model_declared_requirements(model_path))
    for package, source in _scan_python_imports(model_path).items():
        requirements.setdefault(package, ("present", f"python_import:{source}"))
    if backend == "nemo_or_transformers_asr":
        requirements.setdefault("nemo_toolkit", ("2.0", "nemo_asr_bridge"))
        requirements.setdefault("python", ("3.12", "nemo_speech_target"))
    return requirements


def _checks(requirements: dict[str, tuple[str, str]], installed: dict[str, str]) -> list[DependencyCheck]:
    rows: list[DependencyCheck] = []
    for package, (required, source) in sorted(requirements.items()):
        value = installed.get(package, "missing")
        result = _version_satisfies(value, required)
        if result is True:
            status = "ok"
            detail = "installed satisfies requirement"
        elif result is False:
            status = "blocked"
            detail = "missing or below required version"
        else:
            status = "review"
            detail = "present check or non-standard version comparison"
        rows.append(DependencyCheck(package, value, required, source, status, detail))
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Dependency Diagnostics: {payload['model_id']}",
        "",
        f"Env: `{payload['env_name']}`",
        f"Backend: `{payload['backend']}`",
        f"Local path: `{payload['local_path']}`",
        f"Layout: `{payload['layout_status']}` - {payload['layout_detail']}",
        "",
        "## Checks",
        "",
        "| Package | Installed | Required | Source | Status | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["checks"]:
        lines.append(f"| `{row['package']}` | `{row['installed']}` | `{row['required']}` | `{row['source']}` | `{row['status']}` | {row['detail']} |")
    if payload["readme_hints"]:
        lines.extend(["", "## README Hints", ""])
        lines.extend(f"- `{hint}`" for hint in payload["readme_hints"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose coarse and model-specific dependency compatibility for one catalog model.")
    parser.add_argument("model_id")
    parser.add_argument("--index-path", default=DEFAULT_MODEL_INDEX_PATH)
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    env_name = _env_name(args.env_name)
    record = find_catalog_record(args.model_id, index_path=args.index_path)
    plan = plan_model_integration(record)
    model_path = Path(plan.local_path)
    requirements = _collect_requirements(plan.backend, model_path)
    installed = _installed_versions(set(BASE_PACKAGES) | set(requirements))
    layout_status, layout_detail = _local_layout_status(plan.backend, model_path)
    checks = _checks(requirements, installed)
    payload: dict[str, Any] = {
        "model_id": record.id,
        "lane": record.integration_lane,
        "backend": plan.backend,
        "env_name": env_name,
        "local_path": str(model_path),
        "layout_status": layout_status,
        "layout_detail": layout_detail,
        "installed": installed,
        "requirements": {package: {"required": req, "source": source} for package, (req, source) in requirements.items()},
        "checks": [asdict(row) for row in checks],
        "import_usages": _scan_python_import_usages(model_path),
        "readme_hints": _readme_dependency_hints(model_path),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(_markdown(payload), encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
