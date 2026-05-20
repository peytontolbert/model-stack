from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_custom_kernel_architecture_docs_cover_cuda_sources_and_bindings() -> None:
    setup_py = _read("setup.py")
    native_cpp = _read("runtime/csrc/model_stack_native.cpp")
    docs = _read("docs/custom-kernel-architecture.md")

    cuda_sources = sorted(
        {
            source
            for source in re.findall(r'sources\.append\("([^"]+)"\)', setup_py)
            if source.endswith(".cu")
        }
    )
    assert cuda_sources
    for source in cuda_sources:
        assert source in docs

    binding_names = sorted(set(re.findall(r'm\.def\(\s*"([^"]+)"', native_cpp)))
    assert binding_names
    for name in binding_names:
        assert name in docs

