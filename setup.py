from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from setuptools import setup


def env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def resolve_pytorch_source_path() -> Path | None:
    candidates = [env_value("MODEL_STACK_PYTORCH_SOURCE_PATH")]
    for candidate in candidates:
        if candidate is None:
            continue
        root = Path(candidate).expanduser()
        marker = root / "aten" / "src" / "ATen" / "native" / "transformers" / "cuda" / "mem_eff_attention" / "kernel_forward.h"
        if marker.exists():
            return root
    return None


def installed_torch_has_pytorch_memeff_headers() -> bool:
    try:
        import torch
    except Exception:
        return False
    root = Path(torch.__file__).resolve().parent
    marker = root / "include" / "ATen" / "native" / "transformers" / "cuda" / "mem_eff_attention" / "kernel_forward.h"
    return marker.exists()


_CUDA_VERSION_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)(?:\.\d+)?")
_CUDA_ARCH_RE = re.compile(
    r"^(?P<major>\d+)(?:\.(?P<minor>\d+))?(?P<suffix>[a-z]*)?(?:\+ptx)?$",
    re.IGNORECASE,
)


def _read_first_version(text: str) -> tuple[int, int] | None:
    match = _CUDA_VERSION_RE.search(text)
    if match is None:
        return None
    return int(match.group("major")), int(match.group("minor"))


def detect_cuda_toolkit_version(cuda_home: str | None) -> tuple[int, int] | None:
    if not cuda_home:
        return None
    root = Path(cuda_home)
    version_json = root / "version.json"
    if version_json.exists():
        try:
            version = _read_first_version(version_json.read_text(encoding="utf-8", errors="ignore"))
            if version is not None:
                return version
        except OSError:
            pass
    version_txt = root / "version.txt"
    if version_txt.exists():
        try:
            version = _read_first_version(version_txt.read_text(encoding="utf-8", errors="ignore"))
            if version is not None:
                return version
        except OSError:
            pass
    nvcc = root / "bin" / "nvcc"
    if nvcc.exists():
        try:
            out = subprocess.check_output([str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT)
            return _read_first_version(out)
        except (OSError, subprocess.CalledProcessError):
            pass
    return None


def _normalize_arch_list(arch_list: str) -> list[str]:
    normalized = arch_list.replace(",", " ").replace(";", " ")
    return [token for token in normalized.split() if token]


def _arch_token_kind(token: str) -> tuple[int, int, str] | None:
    match = _CUDA_ARCH_RE.match(token.strip())
    if match is None:
        return None
    return int(match.group("major")), int(match.group("minor") or "0"), str(match.group("suffix") or "").lower()


def _arch_list_requests_sm90a(tokens: list[str]) -> bool:
    for token in tokens:
        parsed = _arch_token_kind(token)
        if parsed is None:
            continue
        major, minor, suffix = parsed
        if (major, minor) == (9, 0) and suffix == "a":
            return True
    return False


def _arch_list_targets_sm90(tokens: list[str]) -> bool:
    for token in tokens:
        parsed = _arch_token_kind(token)
        if parsed is None:
            continue
        major, minor, _ = parsed
        if (major, minor) == (9, 0):
            return True
    return False


def maybe_enable_sm90a_target(arch_list: str | None, cuda_version: tuple[int, int] | None) -> str | None:
    if not env_enabled("MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL", "0"):
        return arch_list
    if arch_list is None:
        return arch_list
    if cuda_version is not None and cuda_version < (12, 0):
        return arch_list
    tokens = _normalize_arch_list(arch_list)
    if not _arch_list_targets_sm90(tokens) or _arch_list_requests_sm90a(tokens):
        return arch_list
    tokens.append("9.0a")
    return ";".join(tokens)


def validate_cuda_arch_list(arch_list: str, cuda_version: tuple[int, int] | None) -> None:
    if cuda_version is None or cuda_version < (13, 0):
        return
    unsupported: list[str] = []
    for token in _normalize_arch_list(arch_list):
        match = _CUDA_ARCH_RE.match(token)
        if match is None:
            continue
        major = int(match.group("major"))
        minor = int(match.group("minor") or "0")
        if (major, minor) < (7, 5):
            unsupported.append(token)
    if unsupported:
        archs = ", ".join(unsupported)
        raise ValueError(
            "CUDA Toolkit 13.x cannot offline-compile for pre-Turing architectures "
            f"(got {archs} in TORCH_CUDA_ARCH_LIST / MODEL_STACK_CUDA_ARCH_LIST). "
            "Use CUDA 12.x for Maxwell/Pascal/Volta targets, or target 7.5+ only."
        )


def configure_cuda_build_environment(cuda_home: str | None) -> tuple[int, int] | None:
    model_stack_arches = env_value("MODEL_STACK_CUDA_ARCH_LIST")
    torch_arches = env_value("TORCH_CUDA_ARCH_LIST")
    if torch_arches is None and model_stack_arches is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = model_stack_arches
        torch_arches = model_stack_arches

    model_stack_max_jobs = env_value("MODEL_STACK_MAX_JOBS")
    if env_value("MAX_JOBS") is None and model_stack_max_jobs is not None:
        os.environ["MAX_JOBS"] = model_stack_max_jobs

    cuda_version = detect_cuda_toolkit_version(cuda_home)
    torch_arches = maybe_enable_sm90a_target(torch_arches, cuda_version)
    if torch_arches is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arches
    if torch_arches is not None:
        validate_cuda_arch_list(torch_arches, cuda_version)
    return cuda_version


def native_extensions():
    if not env_enabled("MODEL_STACK_BUILD_NATIVE", "0"):
        return [], {}

    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

    use_cuda = CUDA_HOME is not None and env_enabled("MODEL_STACK_BUILD_CUDA", "1")
    cuda_version = configure_cuda_build_environment(CUDA_HOME if use_cuda else None)
    sources = ["runtime/csrc/model_stack_native.cpp", "runtime/csrc/reference/aten_reference.cpp"]
    define_macros = [("MODEL_STACK_ABI_VERSION", "1")]
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}
    include_dirs = []

    extension_cls = CppExtension
    if use_cuda:
        extension_cls = CUDAExtension
        sources.append("runtime/csrc/backend/cuda_rms_norm.cu")
        sources.append("runtime/csrc/backend/cuda_add_rms_norm.cu")
        sources.append("runtime/csrc/backend/cuda_residual_add.cu")
        sources.append("runtime/csrc/backend/cuda_layer_norm.cu")
        sources.append("runtime/csrc/backend/cuda_embedding.cu")
        sources.append("runtime/csrc/backend/cuda_sampling.cu")
        sources.append("runtime/csrc/backend/cuda_append_tokens.cu")
        sources.append("runtime/csrc/backend/cuda_decode_positions.cu")
        sources.append("runtime/csrc/backend/cuda_attention.cu")
        sources.append("runtime/csrc/backend/attention/cuda_attention_decode_dispatch.cu")
        sources.append("runtime/csrc/backend/attention/cuda_attention_prefill_dispatch.cu")
        sources.append("runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cu")
        sources.append("runtime/csrc/backend/attention/cuda_attention_pytorch_memeff_prefill.cu")
        sources.append("runtime/csrc/backend/cuda_kv_cache.cu")
        sources.append("runtime/csrc/backend/cuda_rope.cu")
        sources.append("runtime/csrc/backend/cuda_activation.cu")
        sources.append("runtime/csrc/backend/cuda_gated_activation.cu")
        sources.append("runtime/csrc/backend/cuda_fp8_linear.cu")
        sources.append("runtime/csrc/backend/cuda_int4_linear.cu")
        sources.append("runtime/csrc/backend/cuda_nf4_linear.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_pack.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_frontend.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_linear_decode.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_linear_prefill.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_linear_dispatch.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_attention_decode_dispatch.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_attention_prefill_dispatch.cu")
        sources.append("runtime/csrc/backend/bitnet/bitnet_attention_dispatch.cu")
        sources.append("runtime/csrc/backend/cuda_int8_attention.cu")
        sources.append("runtime/csrc/backend/cuda_int8_linear.cu")
        sources.append("runtime/csrc/backend/cuda_quant_int8_frontend.cu")
        sources.append("runtime/csrc/backend/cublaslt_linear.cu")
        define_macros.append(("MODEL_STACK_WITH_CUDA", "1"))
        if cuda_version is not None:
            define_macros.append(("MODEL_STACK_CUDA_VERSION_MAJOR", str(cuda_version[0])))
            define_macros.append(("MODEL_STACK_CUDA_VERSION_MINOR", str(cuda_version[1])))
        if env_enabled("MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL", "0"):
            define_macros.append(("MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL", "1"))
        cutlass_path = env_value("MODEL_STACK_CUTLASS_PATH")
        if cutlass_path is not None:
            cutlass_root = Path(cutlass_path)
            include_dirs.extend(
                [
                    str(cutlass_root / "include"),
                    str(cutlass_root / "tools" / "util" / "include"),
                    str(cutlass_root / "examples" / "41_fused_multi_head_attention"),
                ]
            )
            define_macros.append(("MODEL_STACK_WITH_CUTLASS_FMHA", "1"))
        pytorch_source_path = resolve_pytorch_source_path()
        if pytorch_source_path is not None:
            include_dirs.append(str(pytorch_source_path / "aten" / "src"))
            define_macros.append(("MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA", "1"))
        elif installed_torch_has_pytorch_memeff_headers():
            define_macros.append(("MODEL_STACK_WITH_PYTORCH_MEMEFF_FMHA", "1"))
        extra_compile_args["nvcc"] = [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ]
    else:
        define_macros.append(("MODEL_STACK_WITH_CUDA", "0"))

    ext = extension_cls(
        "_model_stack_native",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )
    if hasattr(BuildExtension, "with_options"):
        build_ext = BuildExtension.with_options(use_ninja=env_enabled("MODEL_STACK_USE_NINJA", "1"))
    else:
        build_ext = BuildExtension
    return [ext], {"build_ext": build_ext}


extensions, cmdclass = native_extensions() if env_enabled("MODEL_STACK_BUILD_NATIVE", "0") else ([], {})
setup(ext_modules=extensions, cmdclass=cmdclass)
