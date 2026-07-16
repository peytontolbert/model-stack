from __future__ import annotations

import importlib
import json
import struct
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class CompatibilityPatch:
    id: str
    family: str
    package: str
    expected_api: str
    current_probe: str
    status: str
    patch: str
    reason: str
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompatibilityReport:
    model_id: str
    model_path: str
    installed_versions: Mapping[str, str]
    patches: tuple[CompatibilityPatch, ...]
    blockers: tuple[str, ...]


def installed_versions(packages: tuple[str, ...] = ("python", "torch", "transformers", "diffusers", "nemo_toolkit")) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        if package == "python":
            import sys

            versions[package] = sys.version.split()[0]
            continue
        try:
            versions[package] = metadata.version(package.replace("_", "-"))
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _has_attr(module_name: str, dotted_attr: str) -> bool:
    try:
        obj: Any = importlib.import_module(module_name)
        for part in dotted_attr.split("."):
            obj = getattr(obj, part)
        return True
    except Exception:
        return False


def _classic_mobilellm(model_id: str, model_path: Path, config: Mapping[str, Any]) -> bool:
    auto_map = config.get("auto_map") if isinstance(config.get("auto_map"), Mapping) else {}
    return (
        str(config.get("model_type")) == "mobilellm"
        or model_id.startswith("MobileLLM-")
        and (model_path / "modeling_mobilellm.py").is_file()
        or "modeling_mobilellm.MobileLLMForCausalLM" in str(auto_map)
    )


def _sequence_classifier_config_mismatch(model_path: Path, config: Mapping[str, Any]) -> tuple[int | None, int | None]:
    architectures = tuple(config.get("architectures") or ())
    if not any(str(arch).endswith("ForSequenceClassification") for arch in architectures):
        return (None, None)
    configured = config.get("num_labels")
    try:
        configured_int = int(configured) if configured is not None else None
    except Exception:
        configured_int = None
    safetensors_path = model_path / "model.safetensors"
    if not safetensors_path.is_file():
        return (configured_int, None)
    try:
        with safetensors_path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len).decode("utf-8"))
        tensor = header.get("classifier.weight") or header.get("score.weight")
        shape = tensor.get("shape") if isinstance(tensor, Mapping) else None
        inferred = int(shape[0]) if shape else None
    except Exception:
        inferred = None
    return (configured_int, inferred)


def _cosmos_embed_anomaly(model_id: str, model_path: Path, config: Mapping[str, Any]) -> bool:
    return model_id == "Cosmos-Embed1-448p-anomaly-detection" or model_path.name == "Cosmos-Embed1-448p-anomaly-detection"


def _nemo_archive_path(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.suffix == ".nemo":
        return model_path
    if not model_path.is_dir():
        return None
    direct = sorted(model_path.glob("*.nemo"))
    return direct[0] if direct else None


def _read_config(model_path: Path) -> dict[str, Any]:
    path = model_path / "config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compatibility_report(model_path: str | Path, *, model_id: str | None = None) -> CompatibilityReport:
    path = Path(model_path)
    resolved_id = model_id or path.name
    config = _read_config(path)
    versions = installed_versions()
    patches: list[CompatibilityPatch] = []
    blockers: list[str] = []

    if _classic_mobilellm(resolved_id, path, config):
        has_cache_api = _has_attr("transformers.cache_utils", "DynamicCache.get_max_length")
        patches.append(
            CompatibilityPatch(
                id="transformers_mobilellm_legacy_cache",
                family="classic_mobilellm",
                package="transformers",
                expected_api="transformers.cache_utils.DynamicCache.get_max_length",
                current_probe=f"present={has_cache_api}",
                status="not_needed" if has_cache_api else "patch_available",
                patch="set use_cache=False for generate/forward until model code is patched to the new cache API",
                reason="classic MobileLLM remote code calls an API removed in newer Transformers DynamicCache",
                models=(resolved_id,),
            )
        )
        patches.append(
            CompatibilityPatch(
                id="transformers_mobilellm_slow_tokenizer",
                family="classic_mobilellm",
                package="transformers",
                expected_api="AutoTokenizer returns a callable tokenizer",
                current_probe="AutoTokenizer/LlamaTokenizerFast may return False in Transformers 4.57; slow LlamaTokenizer works",
                status="patch_available",
                patch="fallback to transformers.LlamaTokenizer.from_pretrained when AutoTokenizer return is not callable",
                reason="legacy tokenizer_config.json names LlamaTokenizer but fast/auto path is invalid in current env",
                models=(resolved_id,),
            )
        )


    configured_labels, checkpoint_labels = _sequence_classifier_config_mismatch(path, config)
    if checkpoint_labels is not None and configured_labels != checkpoint_labels:
        patches.append(
            CompatibilityPatch(
                id="transformers_classifier_head_num_labels_from_checkpoint",
                family="bert_sequence_classifier",
                package="transformers",
                expected_api="config.num_labels matches classifier.weight first dimension",
                current_probe=f"config.num_labels={configured_labels}; classifier.weight[0]={checkpoint_labels}",
                status="patch_available",
                patch="override config.num_labels/id2label/label2id from classifier.weight before AutoModelForSequenceClassification.from_pretrained",
                reason="checkpoint classifier head shape disagrees with config metadata, causing Linear bias/weight size mismatch during load",
                models=(resolved_id,),
            )
        )

    if _cosmos_embed_anomaly(resolved_id, path, config):
        has_chunking = _has_attr("transformers.modeling_utils", "apply_chunking_to_forward")
        patches.append(
            CompatibilityPatch(
                id="transformers_apply_chunking_to_forward_compat",
                family="cosmos_embed1_anomaly",
                package="transformers",
                expected_api="transformers.modeling_utils.apply_chunking_to_forward",
                current_probe=f"present={has_chunking}",
                status="not_needed" if has_chunking else "patch_candidate",
                patch="provide/import a local apply_chunking_to_forward shim before loading remote code",
                reason="local remote code imports a helper no longer exported from transformers.modeling_utils",
                models=(resolved_id,),
            )
        )

    nemo_archive = _nemo_archive_path(path)
    if nemo_archive is not None:
        patches.append(
            CompatibilityPatch(
                id="nemo_prefer_archive_over_external_transformers_metadata",
                family="nemo_asr_archive",
                package="nemo_toolkit",
                expected_api="nemo.collections.asr.models.ASRModel.restore_from",
                current_probe=f"archive={nemo_archive.name}; nemo_toolkit={versions.get('nemo_toolkit')}",
                status="patch_available" if versions.get("nemo_toolkit") != "missing" else "needs_env_package",
                patch="prefer *.nemo archive restore and ignore external Transformers 5.x metadata unless restore emits a Transformers API error",
                reason="validated Parakeet .nemo archives restore/transcribe under NeMo 2.7.3 despite external Transformers 5.x metadata",
                models=(resolved_id,),
            )
        )

    return CompatibilityReport(
        model_id=resolved_id,
        model_path=str(path),
        installed_versions=versions,
        patches=tuple(patches),
        blockers=tuple(blockers),
    )


def report_as_dict(report: CompatibilityReport) -> dict[str, Any]:
    return {
        "model_id": report.model_id,
        "model_path": report.model_path,
        "installed_versions": dict(report.installed_versions),
        "patches": [asdict(patch) for patch in report.patches],
        "blockers": list(report.blockers),
    }
