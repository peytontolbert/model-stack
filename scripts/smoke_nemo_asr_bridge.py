#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from runtime.model_catalog import DEFAULT_MODEL_INDEX_PATH, find_catalog_record, plan_model_integration


def _installed_versions() -> dict[str, str]:
    names = {dist.metadata["Name"].lower().replace("-", "_"): dist.version for dist in metadata.distributions()}
    out = {
        "python": sys.version.split()[0],
        "nemo_toolkit": names.get("nemo_toolkit", "missing"),
        "torch": names.get("torch", "missing"),
        "pytorch_lightning": names.get("pytorch_lightning", "missing"),
        "hydra_core": names.get("hydra_core", "missing"),
        "omegaconf": names.get("omegaconf", "missing"),
        "soundfile": names.get("soundfile", "missing"),
        "librosa": names.get("librosa", "missing"),
    }
    try:
        import torch

        out["torch_import"] = torch.__version__
        out["torch_cuda"] = str(torch.cuda.is_available())
        out["torch_cuda_version"] = str(getattr(torch.version, "cuda", None))
    except Exception as exc:  # pragma: no cover - diagnostic path
        out["torch_import"] = f"failed:{type(exc).__name__}:{exc}"
        out["torch_cuda"] = "False"
        out["torch_cuda_version"] = "None"
    return out


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in value.replace("+", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts)


def _env_status(versions: dict[str, str]) -> tuple[bool, tuple[str, ...]]:
    problems: list[str] = []
    if _version_tuple(versions["python"]) < (3, 12):
        problems.append("python<3.12")
    if versions["torch"] == "missing" or _version_tuple(versions["torch"]) < (2, 7):
        problems.append("torch<2.7")
    if versions["nemo_toolkit"] == "missing" or not importlib.util.find_spec("nemo"):
        problems.append("nemo_toolkit_missing")
    return not problems, tuple(problems)


def _find_nemo_archive(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.suffix == ".nemo":
        return model_path
    if not model_path.is_dir():
        return None
    direct = sorted(model_path.glob("*.nemo"))
    if direct:
        return direct[0]
    nested = sorted(model_path.glob("**/*.nemo"))
    return nested[0] if nested else None


def _nemo_model_target(nemo_path: Path) -> str | None:
    try:
        with tarfile.open(nemo_path) as archive:
            config_name = next((name for name in archive.getnames() if name.endswith("model_config.yaml")), None)
            if config_name is None:
                return None
            config_file = archive.extractfile(config_name)
            if config_file is None:
                return None
            for raw_line in config_file:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("target:"):
                    return line.split(":", 1)[1].strip().strip("'\"")
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"failed:{type(exc).__name__}:{exc}"
    return None


def _target_import_status(target: str | None) -> tuple[bool | None, str]:
    if not target:
        return None, "missing_target_metadata"
    if target.startswith("failed:"):
        return False, target
    module_name, _, class_name = target.rpartition(".")
    if not module_name or not class_name:
        return False, "invalid_target"
    try:
        module = importlib.import_module(module_name)
        getattr(module, class_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"
    return True, "ok"


def _restore_model(nemo_path: Path, map_location: str | None) -> Any:
    try:
        from nemo.collections.asr.models import ASRModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install a NeMo ASR-capable environment before restoring .nemo archives") from exc
    return ASRModel.restore_from(str(nemo_path), map_location=map_location)


def _exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _restore_failure_status(exc: BaseException) -> tuple[str, str]:
    chain = _exception_chain(exc)
    detail = " <- ".join(f"{type(item).__name__}:{item}" for item in chain)
    lowered = detail.lower()
    transformers_markers = (
        "transformers",
        "automodel",
        "auto_model",
        "generationconfig",
        "tokenization",
        "tokenizer",
    )
    if any(marker in lowered for marker in transformers_markers):
        return "needs_nemo_model_specific_env", detail
    return "nemo_restore_failed", detail


def _resolve_target(args: argparse.Namespace) -> tuple[str, str, str, Path, Path | None]:
    if args.archive_path:
        archive_path = Path(args.archive_path)
        model_path = archive_path.parent if archive_path.suffix == ".nemo" else archive_path
        return archive_path.stem, "nemo_asr_bridge", "direct_nemo_archive", model_path, _find_nemo_archive(archive_path)

    if not args.model_id:
        raise SystemExit("model_id is required unless --archive-path is provided")
    record = find_catalog_record(args.model_id, index_path=args.index_path)
    plan = plan_model_integration(record, model_root=args.model_root)
    model_path = Path(plan.local_path)
    return record.id, record.integration_lane, plan.backend, model_path, _find_nemo_archive(model_path)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Smoke local NeMo ASR catalog entries through model-stack.")
    parser.add_argument("model_id", nargs="?")
    parser.add_argument("--archive-path", default=None, help="Direct .nemo archive or directory to smoke outside the catalog.")
    parser.add_argument("--index-path", default=DEFAULT_MODEL_INDEX_PATH)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--restore", action="store_true", help="Actually restore the .nemo archive; status-only by default.")
    parser.add_argument("--map-location", default="cpu", help="Map location for ASRModel.restore_from when --restore is set.")
    parser.add_argument("--transcribe-audio", action="append", default=[], help="Audio file to transcribe after restore; implies --restore.")
    parser.add_argument("--force-restore", action="store_true", help="Attempt restore even when the .nemo target class is not importable.")
    args = parser.parse_args()

    started_at = time.perf_counter()
    model_id, lane, backend, model_path, nemo_path = _resolve_target(args)
    versions = _installed_versions()
    env_ok, problems = _env_status(versions)
    target = _nemo_model_target(nemo_path) if nemo_path else None
    target_importable, target_import_detail = _target_import_status(target)
    preflight_seconds = time.perf_counter() - started_at

    print(f"model_id={model_id}")
    print(f"lane={lane} backend={backend}")
    print(f"model_path={model_path}")
    print(f"model_path_exists={model_path.exists()}")
    print(f"nemo_archive={nemo_path if nemo_path else 'missing'}")
    print(f"nemo_model_target={target if target else 'missing'}")
    print(f"nemo_model_target_importable={target_importable}")
    print(f"nemo_model_target_import_detail={target_import_detail}")
    print(f"env_ok={env_ok}")
    print(f"env_problems={problems}")
    print(f"preflight_seconds={preflight_seconds:.3f}")
    for key in sorted(versions):
        print(f"{key}={versions[key]}")

    if args.transcribe_audio:
        args.restore = True
    if not args.restore:
        return
    if nemo_path is None:
        print("restore_status=missing_nemo_archive")
        raise FileNotFoundError(f"No .nemo archive found under {model_path}")
    if not env_ok:
        print("restore_status=env_not_ready")
        raise RuntimeError(f"NeMo ASR environment is not ready: {problems}")
    if target_importable is False and not args.force_restore:
        print("restore_status=needs_nemo_model_specific_env")
        print(f"restore_error=target_not_importable:{target}:{target_import_detail}")
        raise ImportError(f"NeMo archive target is not importable: {target}: {target_import_detail}")
    restore_started_at = time.perf_counter()
    try:
        model = _restore_model(nemo_path, args.map_location)
    except Exception as exc:
        status, detail = _restore_failure_status(exc)
        print(f"restore_status={status}")
        print(f"restore_error={detail}")
        print(f"restore_seconds={time.perf_counter() - restore_started_at:.3f}")
        raise
    restore_seconds = time.perf_counter() - restore_started_at
    print("restore_status=restored")
    print(f"restore_seconds={restore_seconds:.3f}")
    print(f"restored_class={type(model).__name__}")
    if args.transcribe_audio:
        audio_files = [str(Path(item)) for item in args.transcribe_audio]
        transcribe_started_at = time.perf_counter()
        try:
            outputs = model.transcribe(audio_files)
        except Exception as exc:
            status, detail = _restore_failure_status(exc)
            print(f"transcribe_status={status}")
            print(f"transcribe_error={detail}")
            print(f"transcribe_seconds={time.perf_counter() - transcribe_started_at:.3f}")
            raise
        print("transcribe_status=ok")
        print(f"transcribe_seconds={time.perf_counter() - transcribe_started_at:.3f}")
        for index, item in enumerate(outputs):
            text = getattr(item, "text", item)
            print(f"transcribe_output_{index}={text}")


if __name__ == "__main__":
    main()
